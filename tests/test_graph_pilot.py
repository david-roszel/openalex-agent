"""Graph removal must not remove metadata acquisition tools or change their results."""

import json
import httpx
import pytest
from openalex_agent.actions import ACTION_ARGUMENTS, make_action
from openalex_agent.config import Settings
from openalex_agent.controller import Budget
from openalex_agent.corpus import LocalCorpus, freeze
from openalex_agent.llm import HostedJSONPlanner
from openalex_agent.models import Paper
from openalex_agent.openalex import OpenAlexClient
from openalex_agent.pilot import GRAPH_FIELDS, PilotController, run_pilot, analyze_pilot


def records():
    return [
        {"id": "W1", "title": "SYNTHETIC adaptive retrieval", "referenced_works": ["W2"],
         "related_works": ["W2"], "topics": [{"id": "T1", "display_name": "retrieval"}]},
        {"id": "W2", "title": "SYNTHETIC additional evidence", "topics": [{"id": "T1", "display_name": "retrieval"}]},
        {"id": "W3", "title": "SYNTHETIC citation evidence", "referenced_works": ["W1"]},
    ]


@pytest.mark.parametrize("arm", ["graph", "no_graph"])
@pytest.mark.parametrize(("action", "target", "method", "expected"), [
    ("SEARCH_PAPERS", "alternate", "search", "W2"),
    ("EXPAND_REFERENCES", "W1", "references", "W2"),
    ("EXPAND_CITATIONS", "W1", "citations", "W3"),
    ("EXPLORE_TOPIC", "T1", "topic", "W2"),
    ("INSPECT_PAPER", "W2", "get", "W2"),
    ("STOP_RETRIEVAL", None, None, None),
])
def test_all_six_tools_dispatch_and_retrieve_in_both_arms(db, tmp_path, monkeypatch, arm, action, target, method, expected):
    raw = records()
    def handle(request):
        if request.url.path == "/works/W2":
            return httpx.Response(200, json=raw[1])
        query, filters = request.url.params.get("search", ""), request.url.params.get("filter", "")
        batch = [raw[0]] if query == "adaptive retrieval" else [raw[2]] if "cites:" in filters else [raw[1]]
        return httpx.Response(200, json={"results": batch, "meta": {"next_cursor": None}})
    with OpenAlexClient(db, Settings(min_interval=0, max_retries=0), transport=httpx.MockTransport(handle)) as client:
        c = PilotController("adaptive retrieval", client, db, arm=arm, output=str(tmp_path))
        c._execute(make_action("SEARCH_PAPERS", "adaptive retrieval"))
        assert set(c.context()["permitted_actions"]) == set(ACTION_ARGUMENTS)
        called = []
        if method:
            original = getattr(client, method)
            def track(*args, **kwargs):
                called.append(args)
                return original(*args, **kwargs)
            monkeypatch.setattr(client, method, track)
        before = client.usage.requests
        c._execute(make_action(action, target))
        if expected:
            assert "https://openalex.org/"+expected in c.papers
            assert len(called) == 1
            assert client.usage.requests > before
        else:
            assert c.stop_reason == "planner_stop" and client.usage.requests == before
        assert (c.graph is None) == (arm == "no_graph")


def test_context_intervention_changes_only_graph_fields_at_matched_state(db, tmp_path):
    corpus = tmp_path/"corpus"
    freeze([Paper.from_openalex(r) for r in records()], corpus, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    contexts = []
    for arm in ("graph", "no_graph"):
        c = PilotController("adaptive retrieval", LocalCorpus(corpus), db, arm=arm, output=str(tmp_path/"runs"))
        c._execute(make_action("SEARCH_PAPERS", "adaptive retrieval"))
        contexts.append(c.context())
    full, flat = contexts
    assert {k: v for k, v in full.items() if k not in GRAPH_FIELDS} == flat
    assert GRAPH_FIELDS <= full.keys() and not GRAPH_FIELDS.intersection(flat)
    assert flat["observed_metadata"][0]["references"]  # Raw metadata remains usable for inspection.
    assert full["proposals"] == flat["proposals"]
    assert full["permitted_actions"] == flat["permitted_actions"] == list(ACTION_ARGUMENTS)


def test_no_graph_arm_never_constructs_or_scores_a_graph(db, tmp_path, monkeypatch):
    corpus = tmp_path/"corpus"
    freeze([Paper.from_openalex(r) for r in records()], corpus, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    def forbidden(*args, **kwargs):
        pytest.fail("Graph construction/scoring is forbidden in the no-graph pilot arm")
    monkeypatch.setattr("openalex_agent.controller.ScholarlyGraph", forbidden)
    monkeypatch.setattr("openalex_agent.pilot.score_candidates", forbidden)
    c = PilotController("adaptive retrieval", LocalCorpus(corpus), db, arm="no_graph", output=str(tmp_path/"runs"))
    c._execute(make_action("SEARCH_PAPERS", "adaptive retrieval"))
    c._execute(make_action("EXPAND_REFERENCES", "W1"))
    c.context(); c._checkpoint(0)
    assert json.loads((c.directory/"state.json").read_text())["graph_data"] is None
    assert not (c.directory/"graph.json").exists()


@pytest.mark.parametrize("arm", ["graph", "no_graph"])
def test_common_guards_reject_unobserved_ids_and_repeats(db, tmp_path, arm):
    corpus = tmp_path/"corpus"
    freeze([Paper.from_openalex(r) for r in records()], corpus, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    c = PilotController("adaptive retrieval", LocalCorpus(corpus), db, arm=arm, output=str(tmp_path/"runs"))
    c._execute(make_action("SEARCH_PAPERS", "adaptive retrieval"))
    for name, value in (("INSPECT_PAPER", "W999"), ("EXPAND_REFERENCES", "W999"), ("EXPLORE_TOPIC", "T999"),
                        ("SEARCH_PAPERS", "adaptive retrieval")):
        with pytest.raises(ValueError):
            c.validate(make_action(name, value))


@pytest.mark.parametrize(("endpoint", "token_field"), [
    ("https://test.example/v1", "max_tokens"), ("https://api.openai.com/v1", "max_completion_tokens")])
def test_three_question_paired_runner_captures_real_request_contexts_without_quality_claims(tmp_path, endpoint, token_field):
    corpus = tmp_path/"corpus"
    freeze([Paper.from_openalex(r) for r in records()], corpus, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    settings = Settings(llm_base_url=endpoint, llm_api_key="TEST-PRIVATE", llm_model="mock-model")
    seen = []
    def handle(request):
        body = json.loads(request.content)
        context = json.loads(body["messages"][1]["content"])
        seen.append(context)
        action = make_action("EXPAND_REFERENCES", "W1") if context["round"] == 1 else make_action("STOP_RETRIEVAL")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(action.to_dict())}}],
                                       "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
    factory = lambda: HostedJSONPlanner(settings, transport=httpx.MockTransport(handle))
    questions = [{"id": f"q{i}", "question": f"adaptive retrieval {suffix}"}
                 for i, suffix in enumerate(("decisions", "literature", "verification"))]
    rows = run_pilot(str(corpus), questions, settings, tmp_path/"pilot", factory)
    assert len(rows) == 6 and len(seen) == 12
    assert all(r["llm_calls"] == 2 and r["failures"] == r["fallbacks"] == 0 for r in rows)
    assert all(r["all_tools_offered"] and r["graph_context_correct"] for r in rows)
    assert all(r["wire_context_correct"] and r["valid_for_comparison"] for r in rows)
    assert all(r["graph_memory_present"] == (r["arm"] == "graph") for r in rows)
    assert not any("recall" in key or "precision" in key for row in rows for key in row)
    saved = json.loads((tmp_path/"pilot"/"results.json").read_text())
    assert all(x["same_non_graph_context"] for x in saved["context_audits"])
    assert "TEST-PRIVATE" not in json.dumps(saved)
    for row in rows:
        from pathlib import Path
        requests = [json.loads(line) for line in (Path(row["directory"])/"planner_requests.jsonl").read_text().splitlines()]
        assert len(requests) == 2
        for request in requests:
            assert token_field in request
            context = json.loads(request["messages"][1]["content"])
            assert set(context["permitted_actions"]) == set(ACTION_ARGUMENTS)
            assert bool(GRAPH_FIELDS.intersection(context)) == (row["arm"] == "graph")
        assert "TEST-PRIVATE" not in json.dumps(requests)
    summary = analyze_pilot(tmp_path/"pilot")
    assert summary["input_isolation_verified"] and summary["valid_gemini_pairs"] == 3
    assert summary["valid_provider_pairs"] == 3 and summary["model"] == "mock-model"
    # A successful provider response cannot rescue an unmatched experiment.
    wire = __import__("pathlib").Path(rows[0]["directory"])/"planner_requests.jsonl"
    bodies = [json.loads(line) for line in wire.read_text().splitlines()]
    bodies[0][token_field] += 1
    wire.write_text("".join(json.dumps(b)+"\n" for b in bodies))
    broken = analyze_pilot(tmp_path/"pilot")
    assert not broken["input_isolation_verified"] and broken["valid_gemini_pairs"] == 2


def test_provider_failures_do_not_turn_fallback_into_gemini_evidence(tmp_path):
    corpus = tmp_path/"corpus"
    freeze([Paper.from_openalex(r) for r in records()], corpus, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    settings = Settings(llm_base_url="https://test.example/v1", llm_api_key="TEST-PRIVATE", llm_model="mock-model")
    factory = lambda: HostedJSONPlanner(settings, transport=httpx.MockTransport(
        lambda request: httpx.Response(503, text="PRIVATE TEST-PRIVATE")))
    questions = [{"id": f"q{i}", "question": f"adaptive retrieval {suffix}"}
                 for i, suffix in enumerate(("decisions", "literature", "verification"))]
    run_pilot(str(corpus), questions, settings, tmp_path/"pilot", factory)
    summary = analyze_pilot(tmp_path/"pilot")
    assert summary["input_isolation_verified"] and summary["valid_gemini_pairs"] == 0
    assert summary["provider_failures"] == {"HTTP 503": 12}
    assert not any(r["valid_for_comparison"] for r in summary["runs"])
    assert "unreported" in (tmp_path/"pilot"/"report.md").read_text()
