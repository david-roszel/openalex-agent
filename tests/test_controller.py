from dataclasses import replace
import json
import httpx
import pytest
from openalex_agent.actions import Action, make_action
from openalex_agent.config import Settings
from openalex_agent.controller import Budget, Controller
from openalex_agent.corpus import LocalCorpus, freeze
from openalex_agent.llm import DeterministicPlanner, HostedJSONPlanner
from openalex_agent.models import Paper
from openalex_agent.openalex import OpenAlexClient
from openalex_agent.selection import Scoring


@pytest.fixture
def corpus(raw_work, tmp_path):
    p = Paper.from_openalex(raw_work)
    # The reference lacks the search word, but is reachable through observed metadata.
    target = Paper(id="https://openalex.org/W2", title="SYNTHETIC: adaptive mechanisms")
    third = Paper.from_openalex({"id": "W3", "title": "SYNTHETIC: retrieval", "referenced_works": [p.id]},
                               provenance="SYNTHETIC TEST ONLY")
    directory = tmp_path / "corpus"
    freeze([p, target, third], directory, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    return LocalCorpus(directory)


def test_strict_action_schemas():
    for value in [{}, {"action": "INVENT", "arguments": {}},
                  {"action": "SEARCH_PAPERS", "arguments": {"query": "x", "paper_id": "W1"}},
                  {"action": "EXPAND_REFERENCES", "arguments": {"paper_id": "madeup"}},
                  {"action": "STOP_RETRIEVAL", "arguments": {}, "extra": True}]:
        with pytest.raises(ValueError):
            Action.parse(value)
    assert make_action("INSPECT_PAPER", "W1").arguments["paper_id"] == "https://openalex.org/W1"
    assert make_action("SEARCH_PAPERS", "A  QUERY").signature == make_action("SEARCH_PAPERS", "a query").signature


def test_autonomous_graph_loop_and_budget(corpus, db, tmp_path):
    controller = Controller("adaptive retrieval", corpus, db, output=str(tmp_path / "runs"),
                            budget=Budget(max_actions=2, page_size=1, max_inspected=10),
                            scoring=Scoring(min_relevance=0))
    result = controller.run()
    assert result["actions"] == 2
    assert result["stop_reason"] == "max_actions"
    assert result["ranked"]
    assert controller.graph.stats()["edges"] > 0
    assert any(a["action"].startswith("EXPAND") or a["action"] == "INSPECT_PAPER" for a in controller.history[1:])
    assert (controller.directory / "ranked.csv").exists()
    assert len(json.dumps(controller.context(), ensure_ascii=False)) <= controller.budget.max_context_chars


def test_graph_free_context_and_single_pass(corpus, db, tmp_path):
    for method in ("standard", "reformulated", "iterative"):
        controller = Controller("retrieval", corpus, db, method=method, output=str(tmp_path / "runs"),
                                budget=Budget(max_actions=2, page_size=1))
        result = controller.run()
        context = controller.context()
        assert controller.graph is None
        assert "subgraph" not in context and "candidates" not in context and "topic_coverage" not in context
        assert "references" not in json.dumps(context)
        if method in ("standard", "reformulated"):
            assert result["actions"] == 1
            assert result["stop_reason"] == "single_pass"
        with pytest.raises(ValueError):
            controller.validate(make_action("EXPAND_CITATIONS", "W1"))


def test_invalid_planner_falls_back_and_repeated_action_rejected(corpus, db, tmp_path):
    class BrokenPlanner(DeterministicPlanner):
        def decide(self, context):
            return {"action": "FABRICATE"}
    controller = Controller("retrieval", corpus, db, planner=BrokenPlanner(), output=str(tmp_path),
                            budget=Budget(max_actions=2, page_size=1))
    result = controller.run()
    assert result["actions"] == 2
    assert any(e["kind"] == "invalid_decision" for e in db.events(result["run_id"]))
    with pytest.raises(ValueError, match="Repeated"):
        controller.validate(make_action("SEARCH_PAPERS", "retrieval"))
    with pytest.raises(ValueError):
        controller.validate(make_action("INSPECT_PAPER", "W999"))


@pytest.mark.parametrize(("budget", "reason"), [
    (Budget(max_inspected=1, page_size=1), "max_inspected"),
    (Budget(max_rounds=1, page_size=1), "max_rounds"),
    (Budget(patience=1, min_new_relevant=100, page_size=1), "minimal_new_relevant_information"),
])
def test_stopping_conditions(corpus, db, tmp_path, budget, reason):
    result = Controller("retrieval", corpus, db, budget=budget, output=str(tmp_path)).run()
    assert result["stop_reason"] == reason
    assert len(result["inspected_ids"]) <= budget.max_inspected


def test_api_attempt_budget_is_enforced_inside_multi_page_action(db, settings, raw_work, tmp_path):
    def handle(request):
        return httpx.Response(200, json={"results": [raw_work], "meta": {"next_cursor": "second"}})
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(handle))
    result = Controller("retrieval", client, db, method="standard", budget=Budget(max_api_requests=1), output=str(tmp_path)).run()
    assert client.usage.requests == 1
    assert result["stop_reason"] == "budget_exhausted"
    client.close()


def test_hosted_provider_usage_json_and_invalid_response():
    settings = Settings(llm_base_url="https://provider.example/v1", llm_api_key="TEST-SECRET", llm_model="test-model")
    def handle(request):
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"query": "adaptive retrieval"}'}}],
                                       "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
    provider = HostedJSONPlanner(settings, transport=httpx.MockTransport(handle))
    assert provider.reformulate("question") == "adaptive retrieval"
    assert provider.usage.calls == 1 and provider.usage.input_tokens == 20
    provider.close()


def test_graph_changes_available_actions_and_drift_guards(corpus, db, tmp_path):
    controller = Controller("adaptive retrieval", corpus, db, output=str(tmp_path), scoring=Scoring(min_relevance=0.01))
    controller._execute(make_action("SEARCH_PAPERS", "adaptive retrieval"))
    assert any(p["action"] == "EXPAND_REFERENCES" for p in controller.context()["proposals"])
    controller.scoring = Scoring(min_relevance=1)
    with pytest.raises(ValueError, match="low relevance"):
        controller.validate(make_action("EXPAND_REFERENCES", "W1"))
    controller.scoring = Scoring(min_relevance=0, max_depth=0)
    with pytest.raises(ValueError, match="path limit"):
        controller.validate(make_action("EXPAND_REFERENCES", "W1"))
