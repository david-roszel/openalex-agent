"""Regression for the first interactive run. All inference/retrieval is mocked."""
from dataclasses import replace
import json
from types import SimpleNamespace

import httpx
import pytest

from openalex_agent import live_research, research
from openalex_agent.context_planner import InferenceLimits
from openalex_agent.live_profiles import thorough_config
from openalex_agent.live_quality import fallback_plan, preserve_named_queries
from openalex_agent.live_research import LiveHostedPlanner, answer_markdown
from openalex_agent.openalex import BudgetExceeded
from openalex_agent.research import start_research
from test_submission_system import settings, envelope, raw_records, finish, research_transport

QUESTION = ("How do ReAct and Reflexion differ in how they enable large language model agents to reason, "
    "take actions, and learn from feedback? Compare their original research papers.")


def clock(monkeypatch):
    current = [1000.0]
    monkeypatch.setattr(live_research, "time", SimpleNamespace(monotonic=lambda: current[0]))
    monkeypatch.setattr(research, "time", SimpleNamespace(monotonic=lambda: current[0],
        perf_counter=research.time.perf_counter))
    return current


def observed_settings():
    return replace(settings(), llm_timeout=120, llm_max_tokens=4096,
        llm_input_price=2.5, llm_output_price=5)


def observed_config(question=QUESTION, **changes):
    return thorough_config(question, max_seconds=300,
        inference_limits=InferenceLimits(max_total_tokens=150000, max_llm_cost_usd=.75), **changes)


def api(raw_work, queries, current=None):
    def handle(request):
        queries.append(request.url.params.get("search"))
        if current:
            current[0] += 1.338
        return httpx.Response(200, json={"results": raw_records(raw_work)})
    return httpx.MockTransport(handle)


def test_observed_multiminute_configuration_permits_analysis_and_followup(monkeypatch, tmp_path, raw_work):
    current = clock(monkeypatch)
    transport, calls = research_transport(edit=True, recover=True)
    queries = []
    state = finish(start_research(observed_config("adaptive retrieval"), observed_settings(), root=tmp_path,
        openalex_transport=api(raw_work, queries, current), planner_transport=transport))
    result = state["result"]
    assert result["llm_usage"]["calls"] == len(calls) >= 4
    assert not any(e["kind"] == "research_plan_fallback" for e in state["events"])
    assert result["stop_reason"] not in ("research_time_limit", "synthesis_time_reserved")
    guards = [e for e in state["events"] if e["kind"] == "live_inference_budget"]
    analysis = next(e for e in guards if e["stage"] == "analysis")
    assert analysis["remaining_seconds"] == 300
    assert analysis["reserved_final_seconds"] == 150
    assert analysis["request_timeout_seconds"] == 120
    assert analysis["max_total_tokens"] == 150000
    assert analysis["reserved_final_tokens"] == 2 * (4000 + 4096)
    assert analysis["known_tokens"] + analysis["predicted_input_tokens"] + 4096 + analysis["reserved_final_tokens"] < 150000
    assert analysis["cost_ceiling_usd"] == .75
    assert analysis["reserved_final_cost_usd"] == pytest.approx(.5312)
    assert float(result["spending"]["retained_upper_bound_usd"]) <= .75
    mechanisms = result["mechanisms"]
    assert mechanisms["successful_followup_planning_calls"] >= 2
    assert mechanisms["graph_edges_in_successful_planning"] >= 1
    assert mechanisms["environment_context_updates"] >= 1
    assert mechanisms["clm_edits_committed"] == 1
    assert mechanisms["committed_edits_in_later_requests"] == 1
    assert mechanisms["recovered_evidence_in_later_requests"] == 1


def test_optional_analysis_budget_failure_does_not_turn_into_time_exhaustion(monkeypatch, tmp_path, raw_work):
    current = clock(monkeypatch)
    original = LiveHostedPlanner.guard
    def guard(self, system, payload):
        if self.stage == "analysis":
            raise BudgetExceeded("input_token_limit")
        return original(self, system, payload)
    monkeypatch.setattr(LiveHostedPlanner, "guard", guard)
    transport, calls = research_transport(gaps=True)
    queries = []
    state = finish(start_research(observed_config(), observed_settings(), root=tmp_path,
        openalex_transport=api(raw_work, queries, current), planner_transport=transport))
    result = state["result"]
    fallback = next(e for e in state["events"] if e["kind"] == "research_plan_fallback")
    assert fallback["code"] == "input_token_limit" and fallback["model_calls"] == 0
    assert queries[:2] == ["ReAct", "Reflexion"]
    first_planning = next(p for p in calls if "protected" in p)
    assert any(p["arguments"]["query"] == "Reflexion"
        for p in first_planning["protected"]["targeted_search_opportunities"])
    assert result["actions"] >= 2 and calls
    assert result["stop_reason"] not in ("research_time_limit", "synthesis_time_reserved")
    assert result["research_coverage"]["adequate"] is False
    assert result["synthesis"]["question_completion"] == "incomplete_evidence"
    assert "Research is incomplete" in answer_markdown(result)
    assert result["mechanisms"]["graph_edges_in_successful_planning"] > 0


@pytest.mark.parametrize("question,names", [
    (QUESTION, ["ReAct", "Reflexion"]),
    ("Compare LSTM and Transformer for sequence modeling.", ["LSTM", "Transformer"]),
    ('Compare "Dense Passage Retrieval" versus "REALM".', ["Dense Passage Retrieval", "REALM"]),
    ("How do Alpha, Beta and Gamma differ?", ["Alpha", "Beta", "Gamma"]),
])
def test_named_comparisons_have_individual_primary_source_opportunities(question, names):
    plan = fallback_plan(question)
    assert plan["named_entities"] == names
    assert [p["search_query"] for p in plan["components"][:len(names)]] == names
    assert plan["initial_query"] == names[0]
    assert all("original publication" in p["question"] for p in plan["components"][:len(names)])
    # Even a successful but broad hosted analysis cannot erase these queries.
    broad = {"initial_query": " ".join(names), "components": [
        {"id": "C1", "question": question, "search_query": " ".join(names)}]}
    assert preserve_named_queries(broad, question)["initial_query"] == names[0]


def test_noncomparison_question_does_not_invent_named_publications():
    plan = fallback_plan("What is the relationship between model size and retrieval accuracy?")
    assert "named_entities" not in plan


@pytest.mark.parametrize("mode,reason", [("expired", "research_time_limit"),
    ("reserved", "synthesis_time_reserved"), ("cost", "llm_cost_budget_exhausted"),
    ("tokens", "token_budget_exhausted")])
def test_real_resource_exhaustion_still_blocks_calls(monkeypatch, tmp_path, raw_work, mode, reason):
    current = clock(monkeypatch)
    config = observed_config()
    if mode in ("cost", "tokens"):
        config = replace(config, inference_limits=InferenceLimits(
            max_total_tokens=1 if mode == "tokens" else 150000,
            max_llm_cost_usd=.00001 if mode == "cost" else .75))
    if mode in ("expired", "reserved"):
        def analyze(self):
            self.controller.research_plan = fallback_plan(self.controller.question)
            current[0] += 301 if mode == "expired" else 151
        monkeypatch.setattr(LiveHostedPlanner, "analyze_question", analyze)
    calls = []
    def model(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(payload)
        assert "evidence" in payload, "No research call should be affordable"
        return envelope({})
    state = finish(start_research(config, observed_settings(), root=tmp_path,
        openalex_transport=api(raw_work, []), planner_transport=httpx.MockTransport(model)))
    events = state["events"]
    assert state["result"]["stop_reason"] == reason or any(e.get("code") == reason for e in events)
    assert not any("protected" in p for p in calls)
    assert state["result"]["llm_usage"]["calls"] <= len(calls)


def test_actual_wire_timeout_is_shortened_to_remaining_research_allowance(monkeypatch, tmp_path, raw_work):
    current = clock(monkeypatch)
    original = LiveHostedPlanner.analyze_question
    def analyze(self):
        original(self)
        current[0] += 100
    monkeypatch.setattr(LiveHostedPlanner, "analyze_question", analyze)
    transport, _ = research_transport()
    timeouts = []
    def model(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        if "protected" in payload:
            timeouts.append(request.extensions["timeout"])
        return transport.handle_request(request)
    state = finish(start_research(observed_config("adaptive retrieval"), observed_settings(), root=tmp_path,
        openalex_transport=api(raw_work, [], current), planner_transport=httpx.MockTransport(model)))
    assert timeouts and all(0 < t["read"] <= 50 for t in timeouts)
    assert state["result"]["synthesis"]["status"] == "qualified"


def test_cli_returns_partial_success_for_incomplete_but_traceable_answer(tmp_path, raw_work, capsys):
    from openalex_agent.live_cli import main
    transport, _ = research_transport(gaps=True)
    code = main(["adaptive retrieval", "--allow-paid"], root=tmp_path, settings=settings(),
        openalex_transport=api(raw_work, []), planner_transport=transport)
    assert code == 3
    assert "Research is incomplete" in capsys.readouterr().out


def test_clm_guard_failure_reports_its_actual_limit_and_no_phantom_call(monkeypatch, tmp_path, raw_work):
    clock(monkeypatch)
    original = LiveHostedPlanner.guard
    def guard(self, system, payload):
        if self.stage == "planning":
            raise BudgetExceeded("input_token_limit")
        return original(self, system, payload)
    monkeypatch.setattr(LiveHostedPlanner, "guard", guard)
    transport, calls = research_transport()
    state = finish(start_research(observed_config("adaptive retrieval"), observed_settings(), root=tmp_path,
        openalex_transport=api(raw_work, []), planner_transport=transport))
    assert state["result"]["stop_reason"] == "input_token_limit"
    failure = next(e for e in state["events"] if e["kind"] == "context_model_call" and e["operation_kind"] == "failure")
    assert failure["usage_delta"]["calls"] == 0
    assert failure["request_not_sent"] is True
    assert failure["http_status"] is None
    assert failure["error_code"] == "input_token_limit"
    assert len(calls) == 2  # Analysis and final synthesis only.
