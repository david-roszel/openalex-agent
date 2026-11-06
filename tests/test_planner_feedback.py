"""Feedback uses committed observations, and distinguishes empty from duplicate results."""

from dataclasses import replace
import json
import httpx
import pytest
from openalex_agent.actions import make_action
from openalex_agent.controller import Controller, Budget
from openalex_agent.corpus import LocalCorpus, freeze
from openalex_agent.llm import HostedJSONPlanner
from openalex_agent.models import Paper
from openalex_agent.config import Settings


@pytest.fixture
def local(tmp_path):
    directory = tmp_path / "corpus"
    freeze([Paper.from_openalex({"id": "W1", "title": "SYNTHETIC adaptive retrieval",
                                "referenced_works": ["W2"]}),
            Paper.from_openalex({"id": "W2", "title": "SYNTHETIC citation evidence"})],
           directory, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    return LocalCorpus(directory)


def seed(controller):
    controller._execute(make_action("SEARCH_PAPERS", "adaptive"))
    controller._execute(make_action("SEARCH_PAPERS", "retrieval"))
    controller._execute(make_action("SEARCH_PAPERS", "nooverlappingtermsxyz"))


def test_empty_and_duplicate_searches_are_distinct_and_trigger_a_change_of_action(local, db, tmp_path):
    c = Controller("adaptive retrieval augmented generation", local, db, output=str(tmp_path / "runs"))
    seed(c)
    context = c.context()
    feedback = context["search_feedback"]
    assert feedback["consecutive_searches_without_new_papers"] == 2
    assert feedback["recent_searches"][-2]["returned_papers"] == 1
    assert feedback["recent_searches"][-2]["new_papers"] == 0
    assert feedback["recent_searches"][-1]["returned_papers"] == 0
    assert "SEARCH_PAPERS" not in context["permitted_actions"]
    assert all(a["action"] != "SEARCH_PAPERS" for a in context["proposals"])
    with pytest.raises(ValueError, match="unproductive searches"):
        c.validate(make_action("SEARCH_PAPERS", "another redundant query"))
    c._execute(c._fallback(context))
    assert c.search_feedback()["consecutive_searches_without_new_papers"] == 0
    assert "SEARCH_PAPERS" in c.context()["permitted_actions"]


def test_graph_free_search_can_broaden_without_graph_information(local, db, tmp_path):
    c = Controller("adaptive retrieval augmented generation", local, db, method="iterative", output=str(tmp_path / "runs"))
    seed(c)
    context = c.context()
    assert "SEARCH_PAPERS" in context["permitted_actions"]
    assert context["proposals"][0]["arguments"]["query"] == "adaptive retrieval"
    assert "candidates" not in context and "subgraph" not in context


def test_old_and_pending_observations_do_not_acquire_invented_outcomes(local, db, tmp_path):
    c = Controller("adaptive retrieval", local, db, output=str(tmp_path / "runs"))
    c._execute(make_action("SEARCH_PAPERS", "adaptive"))
    c.trace[0].pop("returned_papers")  # Pre-change checkpoints lack this count.
    pending = make_action("SEARCH_PAPERS", "unknown")
    c._memory = replace(c._memory, history=c.history+[pending.to_dict()], pending=pending.to_dict())
    feedback = c.search_feedback()
    assert len(feedback["recent_searches"]) == 1
    assert feedback["recent_searches"][0]["returned_papers"] is None
    assert feedback["consecutive_searches_without_new_papers"] == 0


def test_feedback_survives_restart_and_remains_in_small_context(local, db, tmp_path):
    budget = Budget(max_context_chars=4000)
    c = Controller("adaptive retrieval", local, db, output=str(tmp_path / "runs"), budget=budget)
    seed(c)
    c._checkpoint(0)
    resumed = Controller(c.question, local, db, resume=str(c.directory), budget=budget)
    context = resumed.context()
    assert context["search_feedback"]["consecutive_searches_without_new_papers"] == 2
    assert len(json.dumps(context, ensure_ascii=False)) <= 4000
    assert context["retrieval"]["live_openalex_available"] is False
    assert context["retrieval"]["corpus_papers"] == 2


def test_pending_search_from_before_policy_change_can_finish_on_resume(local, db, tmp_path):
    budget = Budget(max_actions=4)
    c = Controller("adaptive retrieval", local, db, output=str(tmp_path / "runs"), budget=budget)
    seed(c)
    pending = make_action("SEARCH_PAPERS", "adaptive retrieval")
    c._memory = replace(c._memory, actions=4, rounds=4, history=c.history+[pending.to_dict()],
                        pending=pending.to_dict())
    c._checkpoint(0)
    resumed = Controller(c.question, local, db, resume=str(c.directory), budget=budget)
    result = resumed.run()
    assert result["stop_reason"] == "max_actions" and result["actions"] == 4
    assert resumed.pending is None and len(resumed.trace) == 4
    assert resumed.trace[-1]["returned_papers"] == 1
    assert not any(e["kind"] == "controller_error" for e in db.events(result["run_id"]))


def test_successful_hosted_call_records_safe_diagnostics_and_corpus_scope(local, db, tmp_path):
    settings = Settings(llm_base_url="https://provider.example/v1", llm_api_key="PRIVATE-KEY", llm_model="test")
    def handle(request):
        context = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert context["retrieval"]["live_openalex_available"] is False
        assert "search_feedback" in context
        return httpx.Response(200, json={"choices": [{"message": {"content":
            '{"action":"STOP_RETRIEVAL","arguments":{}}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4}})
    with HostedJSONPlanner(settings, transport=httpx.MockTransport(handle)) as planner:
        c = Controller("adaptive retrieval", local, db, planner=planner, output=str(tmp_path / "runs"))
        result = c.run()
    calls = [e for e in db.events(result["run_id"]) if e["kind"] == "planner_call"]
    assert len(calls) == 1 and calls[0]["http_status"] == 200
    assert calls[0]["prompt_tokens"] == 10 and calls[0]["error_code"] is None
    assert "PRIVATE-KEY" not in json.dumps(calls)
    assert result["retrieval"]["mode"] == "frozen_corpus"
