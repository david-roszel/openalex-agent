"""Synthetic acceptance checks; these are not retrieval-quality evidence."""

from dataclasses import replace
import json

import networkx as nx
import numpy as np
import pytest

from openalex_agent.actions import make_action
from openalex_agent.controller import Budget, Controller
from openalex_agent.corpus import LocalCorpus, freeze
from openalex_agent.graph import ScholarlyGraph
from openalex_agent.models import Paper
from openalex_agent.ranking import VECTORIZER
from openalex_agent.selection import Scoring, _novelty, score_candidates


@pytest.mark.parametrize("value", [True, 1.5, -1, float("nan"), "3"])
def test_path_depth_requires_nonnegative_integer(value):
    with pytest.raises(ValueError):
        Scoring(max_depth=value)


def test_block_novelty_matches_independent_dense_cosine():
    texts = [f"retrieval graph topic {i % 13}" for i in range(143)] + ["", "unrelated astronomy"]
    matrix = VECTORIZER.transform(texts)
    dense = (matrix @ matrix.T).toarray()
    np.fill_diagonal(dense, 0)
    actual = _novelty(matrix, list(range(len(texts))))
    assert np.array([actual[i] for i in range(len(texts))]) == pytest.approx(1 - dense.max(axis=1))


def test_graph_support_changes_selected_acquisition_with_equal_lexical_signals(db, tmp_path):
    selections = []
    for supported in ("W3", "W4"):
        # Exact same observed IDs/titles/topics; only OpenAlex reference metadata differs.
        papers = [Paper.from_openalex({"id": f"W{i}", "title": "SYNTHETIC retrieval graph",
                                     "referenced_works": ["W3", "W4"] if i == 1 else [supported]})
                  for i in (1, 2)]
        corpus_dir = tmp_path / supported
        freeze(papers, corpus_dir, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
        c = Controller("retrieval graph", LocalCorpus(corpus_dir), db, output=str(tmp_path / "runs"),
                       scoring=Scoring(relevance=0, novelty=0, importance=1, coverage=0, cost=0, exploration=0))
        c._execute(make_action("SEARCH_PAPERS", "retrieval graph"))
        context = c.context()
        assert {row["id"] for row in context["candidates"]} == {
            "https://openalex.org/W1", "https://openalex.org/W2", "https://openalex.org/W3", "https://openalex.org/W4"}
        selected = c.planner.decide(context)
        c.validate(make_action(selected["action"], selected["arguments"]["paper_id"]))
        selections.append(selected)
        assert selected["action"] == "INSPECT_PAPER"
        assert selected["arguments"]["paper_id"].endswith(supported)
        assert all(e["relation"] == "PAPER_CITES_PAPER" for e in context["subgraph"])
        assert any(e["target"].endswith(supported) for e in context["subgraph"])
    assert selections[0]["arguments"] != selections[1]["arguments"]


def test_pagerank_nonconvergence_uses_explicit_observed_support(monkeypatch):
    p = Paper.from_openalex({"id": "W1", "title": "retrieval", "referenced_works": ["W2"]})
    graph = ScholarlyGraph()
    graph.add(p)
    def fail(*args, **kwargs):
        raise nx.PowerIterationFailedConvergence(200)
    monkeypatch.setattr(nx, "pagerank", fail)
    rows = score_candidates("retrieval", {p.id: p}, graph, Scoring())
    assert rows and all(r["importance_method"] == "observed_incoming_support_fallback" for r in rows)
    assert next(r for r in rows if r["id"].endswith("W2"))["components"]["importance"] == 1
    assert all(0 <= v <= 1 for row in rows for v in row["components"].values())


def test_disabling_novelty_skips_similarity_work(monkeypatch):
    import openalex_agent.selection as selection
    p = Paper.from_openalex({"id": "W1", "title": "retrieval"})
    graph = ScholarlyGraph()
    graph.add(p)
    monkeypatch.setattr(selection, "_novelty", lambda *a: pytest.fail("Disabled component must not run"))
    assert score_candidates("retrieval", {p.id: p}, graph, Scoring(novelty=0))


def test_candidate_depth_and_original_question_remain_hard_guards(db, tmp_path):
    papers = [Paper.from_openalex({"id": "W1", "title": "retrieval", "referenced_works": ["W2"]}),
              Paper.from_openalex({"id": "W2", "title": "retrieval", "referenced_works": ["W3"]}),
              Paper.from_openalex({"id": "W3", "title": "retrieval"})]
    corpus_dir = tmp_path / "corpus"
    freeze(papers, corpus_dir, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    c = Controller("retrieval", LocalCorpus(corpus_dir), db, output=str(tmp_path / "runs"),
                   budget=Budget(page_size=1), scoring=Scoring(max_depth=1))
    c._execute(make_action("SEARCH_PAPERS", "retrieval"))
    c._execute(make_action("INSPECT_PAPER", "W2"))
    with pytest.raises(ValueError, match="path limit"):
        c.validate(make_action("EXPAND_REFERENCES", "W2"))
    with pytest.raises(ValueError, match="depth limit"):
        c.validate(make_action("INSPECT_PAPER", "W3"))
    c.scoring = replace(c.scoring, min_relevance=1)
    # Change only the question for this guard check: the original query cannot be replaced by a retrieval query.
    c.question = "unrelated astrophysics"
    with pytest.raises(ValueError, match="low relevance"):
        c.validate(make_action("EXPAND_CITATIONS", "W1"))
    assert len(json.dumps(c.context(), ensure_ascii=False)) <= c.budget.max_context_chars
