"""Streamlit AppTest executes the real UI against injected HTTP mocks."""

import json
from pathlib import Path
import threading

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from openalex_agent import web_ui
from openalex_agent.config import Settings
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.models import Paper
from openalex_agent.research import ResearchConfig, start_research

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def settings():
    return Settings(api_key="PRIVATE-API-KEY", llm_api_key="PRIVATE-LLM-KEY", llm_model="test-model",
        llm_base_url="https://provider.example/v1/", min_interval=0, max_requests=4, max_retries=0)


def widget(app, kind, label):
    return next(element for element in app.get(kind) if element.label == label)


@pytest.fixture
def app(monkeypatch, tmp_path, raw_work):
    monkeypatch.setattr(web_ui, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web_ui.Settings, "from_env", settings)
    corpus = tmp_path / "data/corpus/example"
    freeze([Paper.from_openalex(raw_work)], corpus, acquisition=[{"source": "SYNTHETIC UI TEST ONLY"}])
    ui=AppTest.from_file(APP, default_timeout=15).run()
    widget(ui,"radio","Workspace").set_value("Individual research").run()
    # Existing execution tests explicitly consent to their mocked hosted calls.
    widget(ui,"checkbox","Confirm paid hosted retrieval").check().run()
    return ui


def test_page_defaults_to_configured_hosted_profile_without_publishing_credentials(app):
    assert not app.exception
    assert widget(app, "selectbox", "Retrieval source").value == "Live OpenAlex"
    assert widget(app, "selectbox", "Planner configuration").value == "Hosted · test-model (.env)"
    assert widget(app, "checkbox", "Enable graph memory").value is True
    assert not widget(app, "button", "Start retrieval").disabled
    assert "PRIVATE-" not in str(app)


def test_widgets_pass_configuration_to_existing_controller_and_show_results(app, monkeypatch):
    def execute(config, selected, **kwargs):
        return start_research(config, selected, **kwargs, planner_transport=httpx.MockTransport(lambda r:
            httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
                "action": "STOP_RETRIEVAL", "arguments": {}, "reason": "TEST recorded stop"})}}]})))
    monkeypatch.setattr(web_ui, "start_research", execute)
    widget(app, "selectbox", "Retrieval source").set_value("Cached corpus / snapshot").run()
    widget(app, "text_area", "Research question").set_value("adaptive retrieval")
    widget(app, "number_input", "Maximum actions").set_value(3)
    widget(app, "number_input", "Maximum inspected papers").set_value(6)
    widget(app, "number_input", "Maximum HTTP attempts").set_value(2)
    widget(app, "number_input", "Maximum LLM calls").set_value(1)
    widget(app, "number_input", "Papers per action").set_value(1)
    widget(app, "checkbox", "Enable graph memory").uncheck()
    widget(app, "button", "Start retrieval").click().run()
    job = app.session_state["research_job"]
    job.join(10)
    app.run()
    assert not app.exception
    state = job.snapshot()
    assert job.config.mode == "snapshot" and job.config.planner == "hosted" and not job.config.graph_memory
    assert job.config.budget.max_actions == 3 and job.config.budget.max_inspected == 6
    assert job.config.budget.max_api_requests == 2 and job.config.budget.max_llm_calls == 1
    assert state["result"]["method"] == "iterative"
    assert state["result"]["stop_reason"] == "planner_stop"
    assert any("Live OpenAlex retrieval is unavailable" in e.value or "live OpenAlex retrieval is unavailable" in e.value
               for e in list(app.info) + list(app.markdown))
    assert widget(app, "metric", "HTTP attempts").value == "0"
    assert widget(app, "metric", "Papers inspected").value == "1"
    assert any("SYNTHETIC TEST: adaptive retrieval" in e.value for e in app.text)
    assert any("TEST recorded stop" in e.value for e in app.json)
    assert "PRIVATE-" not in str(app)


def test_recorded_fallback_is_visible_and_graph_explorer_is_read_only(app, tmp_path):
    job = start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus="data/corpus/example",
        budget=Budget(max_actions=2, max_rounds=2, max_llm_calls=1, page_size=1)), settings(), root=tmp_path,
        planner_transport=httpx.MockTransport(lambda r: httpx.Response(503, text="PRIVATE-LLM-KEY")))
    job.join(10)
    app.session_state["research_job"] = job
    app.run()
    assert not app.exception
    assert widget(app, "metric", "Fallbacks").value == "1"
    assert any("fallback" in warning.value.lower() for warning in app.warning)
    assert any("http_error" in event.value for event in app.json)
    assert widget(app, "selectbox", "Neighborhood seed").value == "https://openalex.org/W1"
    assert any("retrieval_state" in frame.value.columns for frame in app.dataframe)
    assert "PRIVATE-" not in str(app)


def test_live_failure_is_not_rendered_as_success_or_cached_retrieval(app, tmp_path):
    job = start_research(ResearchConfig("unfamiliar quantum question", planner="deterministic",
        budget=Budget(max_actions=1)), settings(), root=tmp_path,
        openalex_transport=httpx.MockTransport(lambda r: httpx.Response(401, text="PRIVATE-API-KEY")))
    job.join(10)
    app.session_state["research_job"] = job
    app.run()
    assert not app.exception
    assert any("Retrieval failed" in e.value for e in app.error)
    assert not app.success
    assert widget(app, "metric", "API errors").value == "1"
    assert widget(app, "metric", "Successful responses").value == "0"
    assert "PRIVATE-" not in str(app)


def test_empty_question_does_not_launch_a_worker(app):
    widget(app, "button", "Start retrieval").click().run()
    assert not app.exception and app.error
    assert "research_job" not in app.session_state


def test_running_ui_observes_committed_progress_before_planner_returns(app, tmp_path):
    entered, release = threading.Event(), threading.Event()
    def planning(request):
        entered.set()
        assert release.wait(5)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "action": "STOP_RETRIEVAL", "arguments": {}})}}]})
    job = start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus="data/corpus/example",
        budget=Budget(max_actions=2, max_rounds=2, max_llm_calls=1, page_size=1)), settings(), root=tmp_path,
        planner_transport=httpx.MockTransport(planning))
    try:
        assert entered.wait(3)
        app.session_state["research_job"] = job
        app.run()
        assert not app.exception and job.snapshot()["status"] == "running"
        assert widget(app, "metric", "Papers inspected").value == "1"
        assert widget(app, "button", "Start retrieval").disabled
        assert not app.success
        assert job.snapshot()["result"] is None
    finally:
        release.set()
        job.join(10)
    app.run()
    assert not app.exception and app.success
    assert not widget(app, "button", "Start retrieval").disabled


def test_saved_run_can_be_reopened_in_the_interface(app, monkeypatch, tmp_path):
    job = start_research(ResearchConfig("adaptive retrieval", mode="snapshot", corpus="data/corpus/example",
        planner="deterministic", budget=Budget(max_actions=1)), settings(), root=tmp_path)
    job.join(10)
    monkeypatch.setattr(web_ui, "start_research", lambda *a, **kw: pytest.fail("Opening cannot run retrieval"))
    app.run()
    widget(app, "button", "Open saved run").click().run()
    assert not app.exception
    assert app.session_state["research_job"].snapshot()["run_id"] == job.snapshot()["run_id"]
    assert widget(app, "metric", "HTTP attempts").value == "0"


def test_clm_ui_executes_real_context_edits_and_displays_actual_snapshots(app, monkeypatch):
    calls = []
    def execute(config, selected, **kwargs):
        def handler(request):
            calls.append(json.loads(request.content))
            output = ({"context_edit": {"path": "active_context.txt", "op": "replace", "text": "Keep observed evidence W1"}}
                      if len(calls) == 1 else {"decision": {"action": "STOP_RETRIEVAL", "arguments": {}}})
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(output)}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20}})
        return start_research(config, selected, **kwargs, planner_transport=httpx.MockTransport(handler))
    monkeypatch.setattr(web_ui, "start_research", execute)
    widget(app, "selectbox", "Retrieval source").set_value("Cached corpus / snapshot").run()
    widget(app, "selectbox", "Context and experiment policy").set_value("CLM-inspired (matched six tools)")
    widget(app, "text_area", "Research question").set_value("adaptive retrieval")
    widget(app, "button", "Start retrieval").click().run()
    job = app.session_state["research_job"]
    job.join(10)
    app.run()
    assert not app.exception
    assert job.config.controlled and job.config.context_management == "clm"
    assert widget(app, "metric", "Accepted context edits").value == "1"
    assert widget(app, "metric", "Reported input tokens").value == "100"
    assert json.loads(calls[1]["messages"][1]["content"])["active_context"] == "Keep observed evidence W1"
    assert any("Keep observed evidence W1" in block.value for block in app.code)
    assert widget(app, "selectbox", "Actual inference input/output snapshot")
    assert "PRIVATE-" not in str(app)
