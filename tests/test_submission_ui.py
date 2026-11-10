"""Real Streamlit script execution with mocked research transports only."""
from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from openalex_agent import live_ui
from openalex_agent.research import start_research
from test_submission_system import settings, raw_records, research_transport

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def widget(app, kind, label):
    return next(x for x in app.get(kind) if x.label == label)


def test_simple_entry_has_question_research_and_optional_advanced(monkeypatch, tmp_path):
    monkeypatch.setattr(live_ui, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(live_ui, "load_live_settings", settings)
    monkeypatch.setattr(live_ui, "start_research", lambda *a, **k: pytest.fail("Opening the UI must not start inference"))
    app = AppTest.from_file(APP, default_timeout=15).run()
    assert not app.exception
    assert widget(app, "button", "Research")
    assert widget(app, "text_area", "Research question")
    assert app.expander[0].label == "Advanced Settings"
    assert widget(app, "selectbox", "Research configuration").value == "combined"
    assert widget(app, "selectbox", "Source").value == "Live OpenAlex"
    assert not any("actions" in x.label.lower() or "context" in x.label.lower() for x in app.number_input)
    assert "SUBMISSION_LLM_SECRET" not in str(app)


def test_research_button_uses_shared_backend_and_prominent_answer(monkeypatch, tmp_path, raw_work):
    monkeypatch.setattr(live_ui, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(live_ui, "load_live_settings", settings)
    calls = []
    def execute(config, selected, **kwargs):
        calls.append(config)
        transport, _ = research_transport(edit=True)
        return start_research(config, selected, **kwargs,
            openalex_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": raw_records(raw_work)})),
            planner_transport=transport)
    monkeypatch.setattr(live_ui, "start_research", execute)
    app = AppTest.from_file(APP, default_timeout=15).run()
    widget(app, "text_area", "Research question").set_value("adaptive retrieval")
    widget(app, "button", "Research").click().run()
    app.session_state["research_job"].join(10)
    app.run()
    assert not app.exception and len(calls) == 1
    assert calls[0].final_synthesis and calls[0].quality_aware and calls[0].context_management == "clm"
    assert app.subheader[0].value == "Research answer"
    assert any("SYNTHETIC MODEL SCRIPT" in e.value for e in app.markdown)
    assert any(e.value == "Supporting publications" for e in app.subheader)
    assert any(e.label == "Research trace and resource use" for e in app.expander)
    assert "SUBMISSION_LLM_SECRET" not in str(app)


def test_incomplete_evidence_is_prominent_even_when_synthesis_is_traceable(monkeypatch, tmp_path, raw_work):
    monkeypatch.setattr(live_ui, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(live_ui, "load_live_settings", settings)
    transport, _ = research_transport(gaps=True)
    def execute(config, selected, **kwargs):
        return start_research(config, selected, **kwargs,
            openalex_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": raw_records(raw_work)})),
            planner_transport=transport)
    monkeypatch.setattr(live_ui, "start_research", execute)
    app = AppTest.from_file(APP, default_timeout=15).run()
    widget(app, "text_area", "Research question").set_value("adaptive retrieval")
    widget(app, "button", "Research").click().run()
    app.session_state["research_job"].join(10)
    app.run()
    assert not app.exception
    assert any("Partial answer" in w.value for w in app.warning)
