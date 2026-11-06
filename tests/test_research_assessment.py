import csv
import json

import httpx
import pandas as pd
import pytest

from openalex_agent.config import Settings
from openalex_agent.context_memory import serialized
from openalex_agent.controller import Budget
from openalex_agent.controlled import GRAPH_FIELDS
from openalex_agent.corpus import freeze
from openalex_agent.experiments import run_experiments
from openalex_agent.llm import HostedJSONPlanner
from openalex_agent.models import Paper
from openalex_agent.research_assessment import build_claim_pool, score_research


def setup(tmp_path):
    p = Paper.from_openalex({"id": "W1", "title": "SYNTHETIC retrieval study"}, provenance="SYNTHETIC ASSESSMENT TEST ONLY")
    corpus = tmp_path / "corpus"
    freeze([p], corpus, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q1", "question": "retrieval study"}]))
    return p, corpus, questions


def factory(handler):
    return lambda: HostedJSONPlanner(Settings(llm_base_url="https://test.example/v1", llm_api_key="TEST-SECRET",
        llm_model="MOCK-EVALUATION", llm_max_tokens=64), transport=httpx.MockTransport(handler))


def test_independent_claim_pool_and_subtopic_scoring_with_valid_id_but_unsupported_claim(db, tmp_path):
    paper, corpus, questions = setup(tmp_path)
    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert "judgment" not in serialized(payload) and "assessor" not in serialized(payload)
        output = {"decision": {"action": "STOP_RETRIEVAL", "arguments": {}}, "evidence_records": [{
            "statement": "This paper proves a clinical benefit.", "work_id": paper.id,
            "evidence_quote": paper.title, "insufficient": False}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(output)}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 30}})
    root = tmp_path / "experiment"
    run_experiments(str(corpus), str(questions), db, factory(handler), output=str(root),
        suite="controlled-five", trials=1, action_budgets=(2,), budget=Budget(page_size=1))
    pool = tmp_path / "claims.csv"
    assert build_claim_pool(str(root), str(pool)) == 1
    frame = pd.read_csv(pool, dtype=str).fillna("")
    assert "method" not in frame.columns and not frame.assessor.iloc[0]
    with pytest.raises(ValueError, match="assessor"):
        score_research(root, claim_judgments_path=pool)
    frame.loc[0, "judgment"] = "unsupported"
    frame.loc[0, "citation_support_correct"] = "false"
    frame.loc[0, "assessor"] = "SYNTHETIC-TEST-ASSESSOR"
    frame.to_csv(pool, index=False)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"source": "curated", "scope": "exhaustive", "fixture_only": True,
        "judgments": [{"query_id": "q1", "paper_id": paper.id, "grade": 3}]}))
    topics = tmp_path / "subtopics.json"
    topics.write_text(json.dumps({"source": "curated", "queries": {"q1": {"expected": ["a", "b"], "papers": {"W1": ["a"]}}}}))
    assert score_research(root, labels_path=labels, subtopics_path=topics, claim_judgments_path=pool) == 5
    scores = pd.read_csv(root / "research-metrics.csv")
    assert (scores.subtopic_coverage == .5).all()
    agents = scores[scores.method != "standard"]
    assert (agents.citation_identifier_validity == 1).all() and (agents.unsupported_claim_rate == 1).all()
    assert agents.claim_factual_accuracy.isna().all()  # Unjudged truth is unknown, not false.
    assert (scores.precision_at_10 == .1).all()
    assert scores[scores.method == "standard"].supported_claim_rate.isna().all()
    assert (root / "research-paired-comparisons.json").exists()


@pytest.mark.parametrize("pair", [("llm", "llm_graph"), ("clm", "clm_graph")])
def test_g4_actual_wire_messages_keep_model_settings_tools_and_raw_metadata_equal(db, tmp_path, pair):
    _, corpus, questions = setup(tmp_path)
    wire = []
    def handler(request):
        wire.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "decision": {"action": "STOP_RETRIEVAL", "arguments": {}}})}}]})
    rows = run_experiments(str(corpus), str(questions), db, factory(handler), output=str(tmp_path / "experiment"),
        suite="controlled-five", methods=pair, trials=1, action_budgets=(2,), budget=Budget(page_size=1))
    assert len(wire) == 2 and wire[0]["messages"][0] == wire[1]["messages"][0]
    assert {k: v for k, v in wire[0].items() if k != "messages"} == {k: v for k, v in wire[1].items() if k != "messages"}
    flat, graph = [json.loads(w["messages"][1]["content"]) for w in wire]
    assert flat["protected"] == graph["protected"]
    if pair[0] == "llm":
        a, b = flat["observation"], graph["observation"]
    else:
        a, b = [json.loads(data["active_context"])["changes"] for data in (flat, graph)]
    assert a == {k: v for k, v in b.items() if k not in GRAPH_FIELDS}
    assert GRAPH_FIELDS <= b.keys() and not GRAPH_FIELDS.intersection(a)
    assert not rows[0]["graph_enabled"] and rows[1]["graph_enabled"]
