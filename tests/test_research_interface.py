"""Mock-transport integration of the UI adapter with the unchanged agent."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

import httpx
import pytest

from openalex_agent import research
from openalex_agent.actions import ACTION_ARGUMENTS
from openalex_agent.config import Settings
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.models import Paper
from openalex_agent.research import ResearchConfig, compare_ids, graph_view, paper_view, progress, start_research


def configured():
    return Settings(api_key="PRIVATE-OPENALEX-TEST-KEY", llm_api_key="PRIVATE-LLM-TEST-KEY",
                    llm_base_url="https://provider.example/v1/", llm_model="test-model",
                    min_interval=0, max_requests=10, max_retries=0)


def small_budget(**changes):
    return replace(Budget(max_actions=2, max_rounds=2, max_inspected=4, page_size=1,
                          max_api_requests=3, max_llm_calls=1), **changes)


def complete(job):
    job.join(10)
    state = job.snapshot()
    assert state["status"] not in {"starting", "running"}
    return state


def target():
    return {"id": "W2", "title": "SYNTHETIC adaptive retrieval reference", "referenced_works": []}


@pytest.fixture
def corpus(tmp_path, raw_work):
    directory = tmp_path / "data/corpus/test"
    freeze([Paper.from_openalex(raw_work), Paper.from_openalex(target())], directory,
           acquisition=[{"source": "SYNTHETIC UI TEST ONLY"}])
    return directory


def test_selected_live_configuration_reaches_existing_pipeline_and_actual_events(monkeypatch, tmp_path, raw_work):
    observed_requests, contexts = [], []
    monkeypatch.setattr(research, "pilot_reference", lambda root: ({"https://openalex.org/W1"},
        {"performed": False, "baseline_available": True, "baseline_papers": 1}))
    def openalex(request):
        observed_requests.append(request)
        raw = target() if request.url.params.get("filter") else dict(raw_work, referenced_works=["W2", "W999"])
        return httpx.Response(200, json={"results": [raw], "meta": {"next_cursor": None}})
    def planner(request):
        context = json.loads(json.loads(request.content)["messages"][1]["content"])
        contexts.append(context)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "action": "EXPAND_REFERENCES", "arguments": {"paper_id": "W1"}, "reason": "TEST recorded explanation"})}}]})
    config = ResearchConfig("adaptive retrieval", budget=small_budget())
    job = start_research(config, configured(), root=tmp_path,
                         openalex_transport=httpx.MockTransport(openalex), planner_transport=httpx.MockTransport(planner))
    state = complete(job)
    assert state["status"] == "completed"
    result = state["result"]
    assert result["budget"] == config.budget.__dict__ and result["method"] == "graph"
    assert result["model"] == "test-model" and result["actions"] == 2
    assert len(observed_requests) == result["api_usage"]["requests"] == 2
    assert set(contexts[0]["permitted_actions"]) == set(ACTION_ARGUMENTS)
    assert result["inspected_ids"] == ["https://openalex.org/W1", "https://openalex.org/W2"]
    assert state["comparison"]["outside_pilot_ids"] == ["https://openalex.org/W2"]
    assert state["comparison"]["outside_pilot_count"] == 1
    assert "https://openalex.org/W999" not in state["comparison"]["outside_pilot_ids"]
    evidence = progress(state)
    assert evidence["successful_responses"] == evidence["http_attempts"] == 2
    assert evidence["fallbacks"] == evidence["api_errors"] == 0
    # Observe exactly the committed rows, including API events absent from controller JSONL.
    with sqlite3.connect(job.directory / "metadata.sqlite") as db:
        rows = db.execute("SELECT seq,kind,timestamp,payload FROM events ORDER BY seq").fetchall()
    assert len(rows) == len(state["events"])
    for event, (seq, kind, timestamp, payload) in zip(state["events"], rows):
        assert event == {"kind": kind, "timestamp": timestamp, **json.loads(payload), "database_seq": seq}
    assert [json.loads(line) for line in (job.directory / "activity.jsonl").read_text().splitlines()] == state["events"]
    assert not (tmp_path / "data/openalex.sqlite").exists()


def test_snapshot_never_constructs_live_client_and_graph_free_mapping_is_explicit(monkeypatch, tmp_path, corpus):
    import openalex_agent.openalex as module
    monkeypatch.setattr(module, "OpenAlexClient", lambda *a, **kw: pytest.fail("Snapshot must not create an HTTP retriever"))
    config = ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(corpus), planner="deterministic",
                            graph_memory=False, budget=small_budget())
    state = complete(start_research(config, configured(), root=tmp_path))
    assert state["status"] == "completed" and state["result"]["method"] == "iterative"
    assert state["result"]["retrieval"]["live_openalex_available"] is False
    assert state["result"]["api_usage"]["requests"] == 0
    assert "unavailable" in progress(state)["evidence"]
    assert not (Path(state["result"]["directory"]) / "graph.json").exists()


def test_failed_live_http_is_explicit_and_cannot_substitute_existing_corpus(tmp_path, corpus):
    job = start_research(ResearchConfig("adaptive retrieval", planner="deterministic", budget=small_budget()), configured(),
        root=tmp_path, openalex_transport=httpx.MockTransport(lambda r: httpx.Response(401, text="PRIVATE-OPENALEX-TEST-KEY")))
    state = complete(job)
    assert state["status"] == "failed" and state["result"]["stop_reason"] == "retrieval_error"
    assert state["result"]["inspected_ids"] == [] and state["error"]["type"] == "retrieval_error"
    summary = progress(state)
    assert summary["http_attempts"] == summary["api_errors"] == 1
    assert summary["successful_responses"] == 0 and "failed" in summary["evidence"]
    assert any(e["kind"] == "request_error" and e["status"] == 401 for e in state["events"])
    assert "PRIVATE-OPENALEX-TEST-KEY" not in json.dumps(state)
    assert "PRIVATE-OPENALEX-TEST-KEY" not in (job.directory / "activity.jsonl").read_text()


@pytest.mark.parametrize("status", [500, 200])
def test_provider_and_state_validation_failures_both_report_fallback(tmp_path, corpus, status):
    def handler(request):
        if status == 500:
            return httpx.Response(500, text="PRIVATE-LLM-TEST-KEY")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "action": "INSPECT_PAPER", "arguments": {"paper_id": "W1"}})}}]})
    config = ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(corpus), budget=small_budget())
    state = complete(start_research(config, configured(), root=tmp_path, planner_transport=httpx.MockTransport(handler)))
    assert state["status"] == "completed" and progress(state)["fallbacks"] == 1
    invalid = next(e for e in state["events"] if e["kind"] == "invalid_decision")
    assert invalid["error_code"] == ("http_error" if status == 500 else "invalid_action")
    assert state["result"]["llm_usage"]["failures"] == (1 if status == 500 else 0)
    assert invalid["fallback_action"]["action"] in ACTION_ARGUMENTS
    assert "PRIVATE-LLM-TEST-KEY" not in json.dumps(state)


def test_configured_attempt_ceiling_is_never_raised_by_the_ui(tmp_path, raw_work):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"results": [raw_work]})
    state = complete(start_research(ResearchConfig("adaptive retrieval", planner="deterministic", budget=small_budget()),
        replace(configured(), max_requests=1), root=tmp_path, openalex_transport=httpx.MockTransport(handler)))
    assert len(calls) == 1 and state["result"]["stop_reason"] == "budget_exhausted"


def test_unexpected_exception_never_publishes_secret_exception_text(tmp_path):
    def fail(request):
        raise RuntimeError("PRIVATE-OPENALEX-TEST-KEY PRIVATE-LLM-TEST-KEY")
    job = start_research(ResearchConfig("retrieval", planner="deterministic", budget=small_budget()), configured(), root=tmp_path,
                         openalex_transport=httpx.MockTransport(fail))
    state = complete(job)
    assert state["status"] == "failed" and state["error"]["type"] == "RuntimeError"
    assert "PRIVATE-" not in json.dumps(state)
    for filename in ("research.json", "summary.json", "activity.jsonl"):
        assert "PRIVATE-" not in (job.directory / filename).read_text()


def test_cache_only_evidence_and_missing_comparison_are_not_live_discovery_claims():
    state = {"mode": "live", "status": "completed", "events": [{"kind": "cache_hit"}],
             "result": {"api_usage": {"requests": 0, "cache_hits": 1}, "inspected_ids": ["W1"], "retrieved_ids": ["W1"]}}
    assert "not been verified" in progress(state)["evidence"]
    assert compare_ids(["W1"], None, {"performed": False}) == {"performed": False}


def test_snapshot_reads_and_graph_exploration_preserve_frozen_files(tmp_path, raw_work):
    corpus = tmp_path / "data/frozen/pilot/corpus"
    freeze([Paper.from_openalex(raw_work)], corpus, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in corpus.rglob("*") if p.is_file()}
    before = hashes()
    state = complete(start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(corpus),
        planner="deterministic", budget=small_budget(max_actions=1)), configured(), root=tmp_path))
    graph_path = Path(state["result"]["directory"]) / "graph.json"
    graph_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    view = graph_view(state["result"]["directory"], "W1", hops=2, max_nodes=20)
    assert {n["type"] for n in view["nodes"]} >= {"paper", "author", "topic", "institution"}
    assert any(n.get("type") == "paper" and not n.get("resolved") for n in view["nodes"])
    assert any(e["relation"] == "PAPER_CITES_PAPER" for e in view["edges"])
    assert hashes() == before and hashlib.sha256(graph_path.read_bytes()).hexdigest() == graph_hash
    assert Path(state["directory"]).is_relative_to(tmp_path / "data/ui-runs")


def test_output_symlink_cannot_redirect_writes_into_a_frozen_directory(tmp_path):
    protected = tmp_path / "data/frozen"
    protected.mkdir(parents=True)
    (tmp_path / "data/ui-runs").symlink_to(protected, target_is_directory=True)
    with pytest.raises(ValueError, match="redirect"):
        start_research(ResearchConfig("retrieval", planner="deterministic"), configured(), root=tmp_path)
    assert list(protected.iterdir()) == []


def test_saved_runs_reopen_without_api_calls_or_file_changes(monkeypatch, tmp_path, corpus):
    from openalex_agent.research import load_research, saved_runs
    job = start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(corpus),
        planner="deterministic", budget=small_budget(max_actions=1)), configured(), root=tmp_path)
    complete(job)
    # Ended status is published before the final summary write, so join before reopening.
    job.join(5)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in job.directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(research, "start_research", lambda *a, **kw: pytest.fail("Reopening cannot start a new run"))
    assert saved_runs(tmp_path) == [job.directory.name]
    loaded = load_research(job.directory.name, root=tmp_path)
    assert loaded.snapshot() == job.snapshot()
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in job.directory.rglob("*") if p.is_file()} == before
    with pytest.raises(ValueError):
        load_research("../../frozen", root=tmp_path)


def test_configuration_failure_without_backend_events_can_be_reopened(tmp_path):
    job = start_research(ResearchConfig("retrieval", mode="snapshot", corpus="missing/corpus",
        planner="deterministic"), configured(), root=tmp_path)
    state = complete(job)
    job.join(5)
    assert state["status"] == "failed" and state["events"] == []
    assert (job.directory / "activity.jsonl").read_text() == ""
    assert research.load_research(job.directory.name, root=tmp_path).snapshot() == job.snapshot()


@pytest.mark.parametrize("doi,expected", [("https://doi.org/10.1/test", "https://doi.org/10.1/test"),
    ("javascript:alert(1)", None), ("https://doi.org@evil.test/10.1/test", None), ("https://[broken", None)])
def test_paper_metadata_and_links_are_backend_values(raw_work, doi, expected):
    paper = Paper.from_openalex(dict(raw_work, doi=doi))
    view = paper_view({"paper": paper.to_dict(), "algorithmic": {"relevance": .25, "reason": "Actual recorded ranking"}})
    assert view["title"] == paper.title and view["abstract"] == paper.abstract
    assert view["openalex_link"] == paper.id and view["doi_link"] == expected
    assert view["score"] == .25 and view["ranking_reason"] == "Actual recorded ranking"
    assert view["provenance"] == paper.provenance


@pytest.mark.parametrize("changes", [{"mode": "other"}, {"planner": "other"}, {"question": " "},
    {"mode": "snapshot"}, {"mode": "live", "corpus": "a/corpus"}, {"graph_memory": "yes"}])
def test_invalid_interface_configuration_is_rejected(changes):
    args = {"question": "retrieval", **changes}
    with pytest.raises(ValueError):
        ResearchConfig(**args)
