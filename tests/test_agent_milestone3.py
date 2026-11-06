"""Agent acceptance/recovery tests with mocked providers and synthetic metadata."""

import csv
from dataclasses import replace
import json
import httpx
import pytest
from openalex_agent import controller as controller_module
from openalex_agent.actions import Action, make_action
from openalex_agent.agent_demo import run_demo
from openalex_agent.config import Settings
from openalex_agent.controller import Budget, Controller
from openalex_agent.graph import ScholarlyGraph
from openalex_agent.llm import HostedJSONPlanner
from openalex_agent.openalex import BudgetExceeded, OpenAlexClient


def test_mock_hosted_agent_autonomously_uses_all_actions_and_logs_compact_state(tmp_path):
    report = run_demo(str(tmp_path))
    assert report["checks_passed"] and report["real_external_requests"] == 0
    directory = __import__("pathlib").Path(report["directory"])
    result = json.loads((directory / "result.json").read_text())
    state = json.loads((directory / "state.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert result["experiment_kind"] == "synthetic_mock" and result["planner"] == "mock-hosted-json"
    assert result["actions"] == 6 and result["llm_usage"]["calls"] == 6
    assert result["llm_usage"]["usage_unreported_calls"] == 6 and result["llm_usage"]["estimated_cost_usd"] is None
    assert state["finished"] and state["pending"] is None
    assert all(e["timestamp"] and e["run_id"] == report["run_id"] for e in events)
    assert len({e["event_seq"] for e in events}) == len(events)
    assert len([e for e in events if e["kind"] == "observation"]) == 6
    assert all(e["characters"] <= state["budget"]["max_context_chars"] for e in events if e["kind"] == "context")
    assert len(state["history"]) == len(state["trace"]) == 6
    assert all("SYNTHETIC" in row["paper"]["title"] and row["llm_interpretation"] is None for row in result["ranked"])


def test_interrupted_batch_never_commits_half_the_papers(db, settings, raw_work, tmp_path, monkeypatch):
    response = {"results": [raw_work, dict(raw_work, id="W2", referenced_works=[])]}
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response)))
    first = Controller("retrieval", client, db, output=str(tmp_path), budget=Budget(max_actions=1, page_size=2))
    original = ScholarlyGraph.add
    def interrupted(graph, paper, **kwargs):
        if paper.id.endswith("/W2"):
            raise KeyboardInterrupt()
        return original(graph, paper, **kwargs)
    monkeypatch.setattr(ScholarlyGraph, "add", interrupted)
    with pytest.raises(KeyboardInterrupt):
        first.run()
    assert first.actions == 1 and first.pending and not first.papers and first.graph.stats()["nodes"] == 0
    state = json.loads((first.directory / "state.json").read_text())
    assert state["papers"] == [] and not state["completed"] and not state["finished"]
    monkeypatch.setattr(ScholarlyGraph, "add", original)
    client.close()
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: pytest.fail("Completed HTTP response must be cached"))) as resumed_client:
        resumed = Controller("retrieval", resumed_client, db, output=str(tmp_path),
                             budget=Budget(max_actions=1, page_size=2), resume=str(first.directory))
        result = resumed.run()
    assert result["actions"] == 1 and len(result["inspected_ids"]) == 2
    assert len(resumed.trace) == 1 and resumed.pending is None
    assert result["api_usage"]["requests"] == 1 and result["api_usage"]["cache_hits"] == 1


@pytest.mark.parametrize("phase", ["json", "csv"])
def test_interrupted_final_export_can_resume_without_retrieval(db, settings, raw_work, tmp_path, monkeypatch, phase):
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": [raw_work]}))) as client:
        first = Controller("retrieval", client, db, budget=Budget(max_actions=1), output=str(tmp_path))
        original_json = controller_module.atomic_json
        original_row = csv.DictWriter.writerow
        row_calls = []
        def interrupt_json(path, payload):
            if path.name == "result.json":
                raise KeyboardInterrupt()
            return original_json(path, payload)
        def interrupt_row(writer, row):
            row_calls.append(row)
            if len(row_calls) == 2:
                raise KeyboardInterrupt()
            return original_row(writer, row)
        if phase == "json":
            monkeypatch.setattr(controller_module, "atomic_json", interrupt_json)
        else:
            monkeypatch.setattr(csv.DictWriter, "writerow", interrupt_row)
        with pytest.raises(KeyboardInterrupt):
            first.run()
        state = json.loads((first.directory / "state.json").read_text())
        assert not state["finished"] and state["stop_reason"] == "max_actions" and state["pending"] is None
        assert not (first.directory / "ranked.csv").exists()
        monkeypatch.setattr(controller_module, "atomic_json", original_json)
        monkeypatch.setattr(csv.DictWriter, "writerow", original_row)
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: pytest.fail("Finalization cannot retrieve again"))) as client:
        resumed = Controller("retrieval", client, db, budget=Budget(max_actions=1), resume=str(first.directory))
        result = resumed.run()
    assert result["actions"] == 1 and result["stop_reason"] == "max_actions"
    assert json.loads((first.directory / "state.json").read_text())["finished"]
    with (first.directory / "ranked.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 1


@pytest.mark.parametrize("error_code", ["rate_limited", "timeout", "transport_error"])
def test_hosted_failures_fall_back_and_call_limit_is_independent(db, settings, raw_work, tmp_path, error_code):
    configured = replace(settings, llm_base_url="https://provider.example/v1", llm_api_key="TEST-SECRET", llm_model="model")
    calls = []
    def provider(request):
        calls.append(request)
        if error_code == "timeout":
            raise httpx.ReadTimeout("PRIVATE TEST-SECRET", request=request)
        if error_code == "transport_error":
            raise httpx.ConnectError("PRIVATE TEST-SECRET", request=request)
        return httpx.Response(429, text="PRIVATE TEST-SECRET")
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": [raw_work]}))) as client:
        with HostedJSONPlanner(configured, transport=httpx.MockTransport(provider)) as planner:
            result = Controller("retrieval", client, db, planner=planner,
                budget=Budget(max_actions=5, max_llm_calls=1, page_size=1), output=str(tmp_path)).run()
    assert len(calls) == result["llm_usage"]["calls"] == 1
    assert result["actions"] == 2 and result["stop_reason"] == "max_llm_calls"
    events = db.events(result["run_id"])
    rejected = next(e for e in events if e["kind"] == "invalid_decision")
    assert rejected["error_code"] == error_code and rejected["fallback_action"]["action"]
    logged = [json.loads(line) for line in (tmp_path / result["run_id"] / "events.jsonl").read_text().splitlines()]
    assert next(e for e in logged if e["kind"] == "invalid_decision")["error_code"] == error_code
    assert "TEST-SECRET" not in json.dumps(logged) and "PRIVATE" not in json.dumps(logged)
    assert "TEST-SECRET" not in json.dumps(events) and "PRIVATE" not in json.dumps(events)


def test_query_reformulation_cannot_exceed_an_already_spent_call_budget(db, settings, tmp_path):
    configured = replace(settings, llm_base_url="https://provider.example/v1", llm_api_key="TEST-SECRET", llm_model="model")
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: pytest.fail("No retrieval after exhausted budget"))) as client:
        with HostedJSONPlanner(configured, transport=httpx.MockTransport(lambda r: pytest.fail("No hidden extra model call"))) as planner:
            planner.usage.calls = planner.usage.usage_unreported_calls = 1
            result = Controller("retrieval", client, db, method="reformulated", planner=planner,
                budget=Budget(max_llm_calls=1), output=str(tmp_path)).run()
    assert result["stop_reason"] == "budget_exhausted" and result["actions"] == 0
    assert result["llm_usage"]["calls"] == 1


def test_planner_cannot_mutate_persistent_memory_or_fallback_context(db, settings, raw_work, tmp_path):
    from openalex_agent.llm import DeterministicPlanner
    class MutatingPlanner(DeterministicPlanner):
        def decide(self, context):
            context["recent_actions"][0]["arguments"]["query"] = "tampered"
            context["proposals"].clear()
            return {"action": "INVALID"}
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": [raw_work]}))) as client:
        controller = Controller("retrieval", client, db, planner=MutatingPlanner(), budget=Budget(max_actions=2), output=str(tmp_path))
        result = controller.run()
    assert controller.history[0]["arguments"]["query"] == "retrieval"
    assert result["actions"] == 2
    with pytest.raises(ValueError, match="already ran"):
        controller.run()


def test_direct_execution_revalidates_arguments_and_enforces_limits(db, settings, raw_work, tmp_path):
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": [raw_work]}))) as client:
        controller = Controller("retrieval", client, db, budget=Budget(max_actions=1), output=str(tmp_path))
        tampered = make_action("SEARCH_PAPERS", "test")
        tampered.arguments["extra"] = True
        with pytest.raises(ValueError):
            controller._execute(tampered)
        assert client.usage.requests == 0
        controller._execute(make_action("SEARCH_PAPERS", "retrieval"))
        with pytest.raises(BudgetExceeded):
            controller._execute(make_action("SEARCH_PAPERS", "different query"))
        assert controller.actions == client.usage.requests == 1


def test_hallucinated_identifier_is_logged_as_rejected_and_never_becomes_metadata(db, settings, raw_work, tmp_path):
    configured = replace(settings, llm_base_url="https://provider.example/v1", llm_api_key="TEST-SECRET", llm_model="model")
    invalid = make_action("INSPECT_PAPER", "W999").to_dict()
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": [raw_work]}))) as client:
        with HostedJSONPlanner(configured, transport=httpx.MockTransport(lambda r: httpx.Response(200,
                json={"choices": [{"message": {"content": json.dumps(invalid)}}]}))) as planner:
            controller = Controller("retrieval", client, db, planner=planner, budget=Budget(max_actions=2), output=str(tmp_path))
            result = controller.run()
    assert "https://openalex.org/W999" not in controller.graph.graph
    rejected = next(e for e in db.events(result["run_id"]) if e["kind"] == "invalid_decision")
    assert rejected["rejected_action"] == {"action": "INSPECT_PAPER", "verification": "unverified planner proposal",
                                            "target_id": "https://openalex.org/W999"}


@pytest.mark.parametrize("method", ["graph", "iterative"])
def test_duplicate_observation_updates_metadata_without_losing_memory_or_charging_extra_inspection(db, settings, raw_work, tmp_path, method):
    configured = replace(settings, llm_base_url="https://provider.example/v1", llm_api_key="TEST-SECRET", llm_model="model")
    def retrieve(request):
        work = raw_work if request.url.params.get("search") == "retrieval" else {"id": "W1", "title": "SYNTHETIC revised retrieval title"}
        return httpx.Response(200, json={"results": [work]})
    def decide(request):
        context = json.loads(json.loads(request.content)["messages"][-1]["content"])
        action = make_action("SEARCH_PAPERS", "followup") if context["round"] == 1 else make_action("STOP_RETRIEVAL")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(action.to_dict())}}]})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(retrieve)) as client:
        with HostedJSONPlanner(configured, transport=httpx.MockTransport(decide)) as planner:
            controller = Controller("retrieval", client, db, method=method, planner=planner, output=str(tmp_path))
            result = controller.run()
    assert result["stop_reason"] == "planner_stop" and result["actions"] == 2
    assert len(result["inspected_ids"]) == 1 and result["trace"][-1]["updated_papers"] == 1
    assert controller.papers["https://openalex.org/W1"].title == "SYNTHETIC revised retrieval title"
    assert controller.papers["https://openalex.org/W1"].abstract and controller.papers["https://openalex.org/W1"].authorships
    if method == "iterative":
        assert controller.graph is None and "subgraph" not in controller.context()


def test_resume_preserves_planner_stop_and_does_not_call_hosted_model_again(db, settings, raw_work, tmp_path, monkeypatch):
    configured = replace(settings, llm_base_url="https://provider.example/v1", llm_api_key="TEST-SECRET", llm_model="model")
    original = Controller._checkpoint
    interruptions = []
    def interrupt_after_stop(controller, seconds, **kwargs):
        original(controller, seconds, **kwargs)
        if controller.stop_reason == "planner_stop" and not interruptions:
            interruptions.append(True)
            raise KeyboardInterrupt()
    monkeypatch.setattr(Controller, "_checkpoint", interrupt_after_stop)
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": [raw_work]}))) as client:
        with HostedJSONPlanner(configured, transport=httpx.MockTransport(lambda r: httpx.Response(200,
                json={"choices": [{"message": {"content": '{"action":"STOP_RETRIEVAL","arguments":{}}'}}]}))) as planner:
            first = Controller("retrieval", client, db, planner=planner, output=str(tmp_path))
            with pytest.raises(KeyboardInterrupt):
                first.run()
    monkeypatch.setattr(Controller, "_checkpoint", original)
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected retrieval after stop"))) as client:
        with HostedJSONPlanner(configured, transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected model call after stop"))) as planner:
            resumed = Controller("retrieval", client, db, planner=planner, resume=str(first.directory))
            result = resumed.run()
    assert result["stop_reason"] == "planner_stop" and result["llm_usage"]["calls"] == 1 and result["actions"] == 1


@pytest.mark.parametrize("corruption", ["version", "papers", "retrieved", "history", "completed", "trace", "pending",
                                      "graph", "provider", "usage", "finished", "budget"])
def test_invalid_or_mismatched_checkpoint_is_rejected_before_external_calls(db, settings, raw_work, tmp_path, corruption):
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": [raw_work]}))) as client:
        first = Controller("retrieval", client, db, output=str(tmp_path))
        first._execute(make_action("SEARCH_PAPERS", "retrieval"))
        first._checkpoint(0.1)
    path = first.directory / "state.json"
    state = json.loads(path.read_text())
    if corruption == "version": state["checkpoint_schema"] = 99
    elif corruption == "papers": state["papers"].append(state["papers"][0])
    elif corruption == "retrieved": state["retrieved_ids"] = []
    elif corruption == "history": state["history"][0]["arguments"]["extra"] = True
    elif corruption == "completed": state["completed"] = []
    elif corruption == "trace": state["trace"][0]["inspected_ids"] = ["https://openalex.org/W99"]
    elif corruption == "pending": state["pending"] = make_action("SEARCH_PAPERS", "different").to_dict()
    elif corruption == "graph": state["graph_data"] = None
    elif corruption == "provider": state["provider_fingerprint"] = "different-provider"
    elif corruption == "usage": state["api_usage"]["requests"] = -1
    elif corruption == "finished": state["finished"] = True
    elif corruption == "budget": state["budget"]["max_actions"] += 1
    path.write_text(json.dumps(state))
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: pytest.fail("Checkpoint must fail before network"))) as client:
        with pytest.raises(ValueError):
            Controller("retrieval", client, db, resume=str(first.directory))
