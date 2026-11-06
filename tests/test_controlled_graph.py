"""G1–G5 isolate graph behavior with public synthetic metadata and evaluator labels."""

import json
from pathlib import Path

import pytest

from openalex_agent.actions import ACTION_ARGUMENTS, Action, make_action
from openalex_agent.controlled import ControlledController, GRAPH_FIELDS
from openalex_agent.controlled_diagnostics import (GraphDiagnosticModel, SeededCorpus, fixture,
                                                  run_graph_case, run_retention)
from openalex_agent.context_planner import ContextPlanner
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.graph import ScholarlyGraph
from openalex_agent.selection import Scoring


@pytest.mark.parametrize("case", ["G1", "G2", "G3", "G5"])
def test_graph_diagnostics_measure_quality_without_requiring_graph_to_win(tmp_path, case):
    report = run_graph_case(case, tmp_path / case)
    assert report["G4_matched_input_and_tools"]
    assert report["labels_used_by_policy"] is False
    for row in report["rows"]:
        assert 0 <= row["precision_at_k"] <= 1 and 0 <= row["ndcg_at_k"] <= 1
        assert row["k"] == 10
        assert row["actions"] <= fixture(case)[-1]
        assert row["actions_per_relevant_paper"] is None or row["actions_per_relevant_paper"] >= 0
        signatures = [Action.parse(a).signature for a in row["history"]]
        assert len(signatures) == len(set(signatures))
    if case == "G1":
        # Both controls can traverse the same valid two-hop citation chain.
        assert all("https://openalex.org/W3" in r["retrieved_ids"] for r in report["rows"])
        graph = ScholarlyGraph.load(Path(report["rows"][1]["directory"]) / "graph.json")
        assert graph.citation_graph().has_edge("https://openalex.org/W1", "https://openalex.org/W2")
        assert graph.citation_graph().has_edge("https://openalex.org/W2", "https://openalex.org/W3")
    if case == "G2":
        assert report["behavior_changed"]  # Community support affects a declared graph-aware rule.
    if case == "G5":
        assert report["negative_control"]  # No assertion that quality must decrease.


def test_g4_no_graph_constructs_no_graph_or_scores_and_preserves_all_tools(monkeypatch, tmp_path, db):
    import openalex_agent.controller as controller_module
    import openalex_agent.controlled as controlled_module
    question, papers, seeds, _, _ = fixture("G1")
    corpus = tmp_path / "corpus"
    freeze(papers, corpus, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    monkeypatch.setattr(controller_module, "ScholarlyGraph", lambda *a, **kw: pytest.fail("No graph construction"))
    monkeypatch.setattr(controlled_module, "score_candidates", lambda *a, **kw: pytest.fail("No hidden graph scores"))
    model = ContextPlanner(GraphDiagnosticModel(), mode="conventional")
    c = ControlledController(question, SeededCorpus(corpus, seeds), db, architecture="llm", planner=model,
                             budget=Budget(max_actions=2, page_size=2), output=str(tmp_path / "runs"))
    c._execute(make_action("SEARCH_PAPERS", question))
    context = c.context()
    assert c.graph is None and not GRAPH_FIELDS.intersection(context)
    assert set(context["permitted_actions"]) == set(ACTION_ARGUMENTS)
    assert context["observed_metadata"][0]["references"] == papers[0].references
    assert not (c.directory / "graph.json").exists()


@pytest.mark.parametrize("horizon", [12, 36])
def test_c1_long_horizon_actual_context_policy_and_exact_retention(tmp_path, horizon):
    report = run_retention(tmp_path / str(horizon), horizon=horizon)
    conventional, clm = report["rows"]
    assert conventional["exact_evidence_retention"] == 0
    assert clm["exact_evidence_retention"] == 1
    assert clm["accepted_context_edits"] > 0
    assert clm["request_count"] > conventional["request_count"]  # Context management is not free.
    assert clm["correct_fact_recovery"] == 1 and conventional["correct_fact_recovery"] == 0
    assert clm["unsupported_claims"] == conventional["unsupported_claims"] == 0
    assert conventional["abstentions"] == 1
    assert all(row["every_request_within_limit"] for row in report["rows"])


@pytest.mark.parametrize("capacity", [4000, 6000])
def test_c3_hundreds_of_observations_remain_bounded(capacity, tmp_path):
    report = run_retention(tmp_path / str(capacity), horizon=205, max_chars=capacity)
    assert all(row["actions"] >= 200 for row in report["rows"])
    assert all(row["every_request_within_limit"] for row in report["rows"])
    assert all(row["active_context_chars"] <= capacity for row in report["rows"])
