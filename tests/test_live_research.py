"""Offline acceptance tests at the hosted HTTP boundary; no paid calls."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from openalex_agent.actions import make_action
from openalex_agent.config import Settings
from openalex_agent.context_planner import InferenceLimits
from openalex_agent.controller import Budget, Controller
from openalex_agent.corpus import LocalCorpus, freeze
from openalex_agent.live_research import source_records, validate_synthesis
from openalex_agent.models import Paper
from openalex_agent.research import ResearchConfig, load_research, progress, start_research
from openalex_agent.storage import Database


def settings(**changes):
    return replace(Settings(api_key="PRIVATE-API-KEY", llm_api_key="PRIVATE-LLM-KEY",
        llm_base_url="https://provider.example/v1/", llm_model="mock-model", llm_max_tokens=1000,
        min_interval=0, max_requests=30, max_retries=0), **changes)


def budget(**changes):
    return replace(Budget(max_actions=30, max_rounds=30, max_inspected=6,
        max_api_requests=30, max_llm_calls=10, page_size=2, patience=20), **changes)


def valid_answer(payload):
    source = payload["evidence"][0]
    return {"answer": [{"text": "The supplied abstract describes adaptive retrieval, with limited coverage.",
        "citations": [{"work_id": source["id"], "quote": source["abstract_excerpt"] or source["title"]}]}],
        "foundations": [], "comparison": [], "limitations": ["These metadata excerpts do not establish a complete comparison."]}


def response(value, usage=True):
    body = {"choices": [{"message": {"content": json.dumps(value)}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 50, "completion_tokens": 20}
    return httpx.Response(200, json=body)


def completed(job):
    job.join(15)
    state = job.snapshot()
    assert state["status"] == "completed", state.get("error")
    return state


@pytest.fixture
def snapshot(tmp_path, raw_work):
    directory = tmp_path / "corpus"
    freeze([Paper.from_openalex(raw_work)], directory, acquisition=[{"source": "SYNTHETIC OFFLINE TEST"}])
    return directory


def test_metadata_exhaustion_still_synthesizes_last_batch_and_accounts_all_calls(tmp_path, raw_work):
    api, calls = [], []
    def retrieval(request):
        api.append(request)
        papers = [dict(raw_work, id=f"W{2 * len(api) + i}") for i in range(2)]
        return httpx.Response(200, json={"results": papers, "meta": {"next_cursor": None}})
    def planner(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(payload)
        if "evidence" in payload:
            return response(valid_answer(payload))
        return response(make_action("SEARCH_PAPERS", f"adaptive retrieval variant {len(calls)}").to_dict())
    config = ResearchConfig("adaptive retrieval", graph_memory=False, budget=budget(), final_synthesis=True)
    job = start_research(config, settings(), root=tmp_path, openalex_transport=httpx.MockTransport(retrieval),
        planner_transport=httpx.MockTransport(planner))
    state = completed(job)
    result = state["result"]
    assert result["stop_reason"] == "max_inspected" and result["actions"] == 3
    assert result["budget"] == config.budget.__dict__
    assert result["binding_limits"] == ["max_inspected"]
    assert result["synthesis"]["status"] == "qualified" and len(api) == 3
    assert {p["id"] for p in calls[-1]["evidence"]} == set(result["inspected_ids"])
    assert result["llm_usage"]["calls"] == len(calls) == 3
    assert progress(state)["input_tokens"] == 150
    assert progress(state)["sources_sent_to_model"] == 6
    assert result["live_accounting"]["returned_record_observations"] == 6
    assert result["seconds"] >= result["retrieval_seconds"]
    saved = json.loads((Path(result["directory"]) / "result.json").read_text())
    assert saved == result
    assert load_research(job.directory.name, root=tmp_path).snapshot() == state


@pytest.mark.parametrize("controlled,context", [(False, "conventional"), (True, "conventional"), (True, "clm")])
def test_one_call_means_initial_retrieval_and_one_final_call(tmp_path, snapshot, controlled, context):
    calls = []
    def planner(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(payload)
        return response(valid_answer(payload))
    config = ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(snapshot), budget=budget(max_llm_calls=1),
        controlled=controlled, context_management=context, final_synthesis=True)
    state = completed(start_research(config, settings(), root=tmp_path, planner_transport=httpx.MockTransport(planner)))
    assert len(calls) == 1 and "evidence" in calls[0]
    assert state["result"]["actions"] == 1
    assert state["result"]["stop_reason"] == "synthesis_call_reserved"
    assert state["result"]["synthesis"]["status"] == "qualified"


@pytest.mark.parametrize("controlled", [False, True])
def test_repeat_feedback_contains_visible_completed_operation_beyond_recent_history(tmp_path, raw_work, controlled):
    from openalex_agent.live_research import LiveController, LiveControlledController
    directory = tmp_path / "corpus"
    freeze([Paper.from_openalex(raw_work)], directory, acquisition=[])
    with Database(":memory:") as db:
        cls = LiveControlledController if controlled else LiveController
        kwargs = {"architecture": "llm_graph"} if controlled else {"method": "graph"}
        retriever = LocalCorpus(directory)
        search_count = [0]
        def search(query, limit):
            search_count[0] += 1
            return [Paper.from_openalex(dict(raw_work, id=f"W{search_count[0]}"))]
        retriever.search = search
        c = cls("adaptive retrieval", retriever, db, budget=budget(max_inspected=30), output=str(tmp_path / "out"), **kwargs)
        c._execute(make_action("SEARCH_PAPERS", "adaptive retrieval"))
        operation = make_action("EXPAND_REFERENCES", "W1")
        c._execute(operation)
        for i in range(5):
            c._execute(make_action("SEARCH_PAPERS", f"adaptive retrieval {i}"))
        context = c.context()
        assert operation.to_dict()["arguments"] not in [a["arguments"] for a in context["recent_actions"]]
        assert {"action": operation.action, "arguments": operation.arguments} in context["completed_operations"]
        with pytest.raises(ValueError, match="Repeated action"):
            c.validate(operation)
        assert all(a["action"] != operation.action or a["arguments"] != operation.arguments for a in context["proposals"])


def test_clm_edit_exhaustion_preserves_final_reservation(tmp_path, snapshot):
    calls = []
    def planner(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(payload)
        if "evidence" in payload:
            return response(valid_answer(payload))
        return response({"context_edit": {"path": "active_context.txt", "op": "replace", "text": "Retain known source evidence."}})
    state = completed(start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(snapshot),
        controlled=True, context_management="clm", final_synthesis=True, budget=budget(max_llm_calls=3)),
        settings(), root=tmp_path, planner_transport=httpx.MockTransport(planner)))
    assert len(calls) == 3 and "evidence" in calls[-1]
    assert state["result"]["actions"] == 1
    assert state["result"]["synthesis"]["status"] == "qualified"
    assert any(e["kind"] == "live_planning_stop" for e in state["events"])


@pytest.mark.parametrize("failure", ["invented_id", "invented_quote", "uncited", "http_error", "truncated"])
def test_bad_terminal_output_never_publishes_unchecked_claims_or_retries(tmp_path, snapshot, failure):
    calls = []
    def planner(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(payload)
        if failure == "http_error":
            return httpx.Response(503, text="PRIVATE-LLM-KEY")
        if failure == "truncated":
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]})
        answer = valid_answer(payload)
        if failure == "invented_id":
            answer["answer"][0]["citations"][0]["work_id"] = "https://openalex.org/W999"
        elif failure == "invented_quote":
            answer["answer"][0]["citations"][0]["quote"] = "This sentence is absent from the evidence."
        else:
            answer["answer"][0]["citations"] = []
        return response(answer)
    state = completed(start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(snapshot),
        budget=budget(max_actions=1), final_synthesis=True), settings(), root=tmp_path,
        planner_transport=httpx.MockTransport(planner)))
    assert len(calls) == 1
    assert state["result"]["synthesis"]["status"] == "evidence_only"
    assert state["result"]["synthesis"]["answer"] is None
    assert "PRIVATE-" not in json.dumps(state)


@pytest.mark.parametrize("limits,reason", [(InferenceLimits(max_total_tokens=1), "no_evidence_fits_remaining_resources"),
    (InferenceLimits(max_llm_cost_usd=0.000001), "no_evidence_fits_remaining_resources")])
def test_token_and_dollar_limits_block_inference_without_budget_increase(tmp_path, snapshot, limits, reason):
    state = completed(start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(snapshot),
        inference_limits=limits, budget=budget(), final_synthesis=True),
        settings(llm_input_price=1, llm_output_price=5), root=tmp_path,
        planner_transport=httpx.MockTransport(lambda r: pytest.fail("No call is affordable"))))
    assert state["result"]["synthesis"]["reason"] == reason
    assert state["result"]["llm_usage"]["calls"] == 0
    assert state["result"]["actions"] == 1


def test_unknown_usage_prevents_additional_inference_with_dollar_limit(tmp_path, snapshot):
    calls = []
    def planner(request):
        calls.append(request)
        return response(make_action("STOP_RETRIEVAL").to_dict(), usage=False)
    state = completed(start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(snapshot),
        final_synthesis=True, budget=budget(), inference_limits=InferenceLimits(max_llm_cost_usd=1)),
        settings(llm_input_price=1, llm_output_price=5), root=tmp_path, planner_transport=httpx.MockTransport(planner)))
    assert len(calls) == 1
    assert state["result"]["synthesis"]["reason"] == "unreported_cost_cannot_enforce_ceiling"
    assert state["result"]["llm_usage"]["estimated_cost_usd"] is None


def test_deterministic_mode_delivers_an_honest_digest_without_inference(tmp_path, snapshot):
    state = completed(start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(snapshot),
        planner="deterministic", budget=budget(max_actions=1), final_synthesis=True), settings(), root=tmp_path))
    assert state["result"]["synthesis"]["reason"] == "deterministic_mode"
    assert state["result"]["llm_usage"]["calls"] == 0


def test_source_delivery_distinguishes_excerpt_units_from_graph_ids_and_history():
    payload = {"active_context": json.dumps({"changes": {"top_papers": [{"id": "W1", "title": "Source title", "abstract_excerpt": "Evidence"}]}}),
        "subgraph": [{"source": "W2", "target": "W3"}], "recent_actions": [{"arguments": {"paper_id": "W4"}}]}
    assert set(source_records(payload)) == {"W1"}
    assert source_records({"id": ["untrusted", "invalid"], "title": "Not a valid source"}) == {}


def test_unexpected_terminal_failure_keeps_committed_retrieval_results(tmp_path, snapshot):
    def failure(request):
        raise RuntimeError("PRIVATE-LLM-KEY")
    state = completed(start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus=str(snapshot),
        final_synthesis=True, budget=budget(max_actions=1)), settings(), root=tmp_path,
        planner_transport=httpx.MockTransport(failure)))
    assert len(state["result"]["ranked"]) == 1
    assert state["result"]["synthesis"]["reason"] == "unexpected_synthesis_error"
    assert state["result"]["llm_usage"]["calls"] == 1
    assert "PRIVATE-" not in json.dumps(state)


def test_no_sources_means_no_final_inference(tmp_path):
    state = completed(start_research(ResearchConfig("adaptive retrieval", final_synthesis=True,
        budget=budget(max_actions=1)), settings(), root=tmp_path,
        openalex_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []})),
        planner_transport=httpx.MockTransport(lambda r: pytest.fail("No scholarly evidence to synthesize"))))
    assert state["result"]["synthesis"]["reason"] == "no_evidence"
    assert state["result"]["llm_usage"]["calls"] == 0


def test_old_controlled_archive_is_not_counted_as_delivery_and_new_zero_is_known():
    state = {"mode": "snapshot", "status": "completed", "result": None, "events": [
        {"kind": "context", "context": {"top_papers": [{"id": "W1"}]}},
        {"kind": "context_model_call"}]}
    assert progress(state)["sources_sent_to_model"] is None
    state["result"] = {"inspected_ids": [], "retrieved_ids": [],
        "live_accounting": {"sources_sent_to_model": 0}}
    assert progress(state)["sources_sent_to_model"] == 0


def test_original_benchmark_controller_retains_retrieval_only_behavior(tmp_path, snapshot):
    with Database(":memory:") as db:
        c = Controller("adaptive retrieval", LocalCorpus(snapshot), db, method="iterative", budget=budget(max_actions=1),
            output=str(tmp_path / "benchmark"))
        result = c.run()
    assert result["stop_reason"] == "max_actions" and result["actions"] == 1
    assert "synthesis" not in result and "live_accounting" not in result
    assert not list(Path(result["directory"]).glob("live-request-*"))


def test_shared_experiment_sources_and_saved_demonstrations_are_byte_preserved():
    root = Path(__file__).resolve().parents[1]
    if not (root / "docs/live-research-baseline-hashes.json").is_file():
        pytest.skip("Optional historical worktree receipt is excluded from the standalone submission")
    baseline = json.loads((root / "docs/live-research-baseline-hashes.json").read_text())
    # The submission task explicitly authorizes isolated local configuration,
    # CLI and diagnostics changes. External frozen V3 files remain read-only.
    changed = {root / "openalex_agent" / name for name in
        ("research.py", "web_ui.py", "config.py", "llm.py", "openalex.py", "cli.py")}
    for name, digest in baseline.items():
        path = Path(name)
        if path.is_relative_to(root) and path not in changed:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
