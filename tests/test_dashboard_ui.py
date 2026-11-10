"""Real Streamlit renderer, offline experiment workers and artifact polling."""

from pathlib import Path
import json

import pytest
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from openalex_agent import web_ui
from openalex_agent import dashboard_ui
from openalex_agent.config import Settings
from openalex_agent.experiment_dashboard import DashboardConfig,start_dashboard

APP=str(Path(__file__).resolve().parents[1]/"legacy_app.py")


def widget(app,kind,label):return next(e for e in app.get(kind) if e.label==label)


@pytest.fixture
def ui(monkeypatch,tmp_path):
    monkeypatch.setattr(web_ui,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(web_ui.Settings,"from_env",lambda:Settings(llm_model="MOCK",llm_api_key="PRIVATE-KEY",llm_base_url="https://test.example/v1"))
    app=AppTest.from_file(APP,default_timeout=20).run()
    widget(app,"radio","Workspace").set_value("Individual research").run()
    return app


def test_individual_paid_confirmation_required(ui):
    assert not ui.exception
    assert widget(ui,"button","Start retrieval").disabled
    widget(ui,"checkbox","Confirm paid hosted retrieval").check().run()
    assert not widget(ui,"button","Start retrieval").disabled
    assert "PRIVATE-KEY" not in str(ui)


def test_comparison_executes_once_then_displays_actual_results(ui,tmp_path):
    widget(ui,"radio","Workspace").set_value("Experiment Dashboard").run()
    assert not ui.exception
    assert widget(ui,"selectbox","Scientific category").value=="Diagnostics"
    widget(ui,"button","Run Comparison").click().run()
    job=ui.session_state["dashboard_job"];job.join(20);ui.run()
    assert not ui.exception and job.snapshot()["status"]=="completed"
    assert widget(ui,"button","Run Comparison").disabled
    assert any("evaluation_status" in f.value and f.value["evaluation_status"].str.contains("SYNTHETIC DIAGNOSTIC").any() for f in ui.dataframe)
    assert any(f.value.astype(str).eq("NOT EVALUATED").any().any() for f in ui.dataframe)
    directory=job.directory
    ui.run();assert ui.session_state["dashboard_job"].directory==directory
    assert len(list((tmp_path/"data/experiment-dashboard").iterdir()))==1
    assert "PRIVATE-KEY" not in str(ui)


def test_development_and_final_categories_cannot_bypass_readiness(ui):
    widget(ui,"radio","Workspace").set_value("Experiment Dashboard").run()
    widget(ui,"selectbox","Scientific category").set_value("Development pilot").run()
    widget(ui,"checkbox","I confirm this hosted run may consume paid API credits").check().run()
    assert not ui.exception
    assert widget(ui,"button","Run Comparison").disabled
    assert any("NOT READY" in w.value for w in ui.warning)
    widget(ui,"selectbox","Scientific category").set_value("Final research evaluation").run()
    assert not ui.exception and widget(ui,"button","Run Comparison").disabled
    assert "dashboard_job" not in ui.session_state


def test_long_horizon_controls_and_invalid_condition_label(ui):
    widget(ui,"radio","Workspace").set_value("Long-Horizon Testing").run()
    widget(ui,"selectbox","Context pressure").set_value("high").run()
    widget(ui,"number_input","Observation sequence length").set_value(4)
    widget(ui,"button","Run Comparison").click().run()
    job=ui.session_state["dashboard_job"];job.join(20);ui.run()
    assert not ui.exception
    assert any("INVALID PRESSURE CONDITION" in e.value for e in ui.error)
    assert job.snapshot()["last_pressure_valid"] is False
    rows=list(job.directory.glob("pressure-trial-*/pressure.json"))
    assert len(rows)==1 and not json.loads(rows[0].read_text())["pressure_valid"]


def test_saved_multiple_experiments_open_without_new_execution(ui,tmp_path,monkeypatch):
    a=start_dashboard(DashboardConfig(methods=("llm",)),Settings(),root=tmp_path,submission_id="a");a.join(15)
    b=start_dashboard(DashboardConfig(methods=("clm",)),Settings(),root=tmp_path,submission_id="b");b.join(15)
    dashboard_ui.catalog.clear()
    monkeypatch.setattr(dashboard_ui,"start_dashboard",lambda *a,**kw:pytest.fail("Saved loading cannot execute a job"))
    widget(ui,"radio","Workspace").set_value("Saved Experiments").run()
    selected=widget(ui,"multiselect","Saved experiments to compare")
    selected.set_value([str(a.directory.relative_to(tmp_path)),str(b.directory.relative_to(tmp_path))]).run()
    assert not ui.exception
    assert len(list((tmp_path/"data/experiment-dashboard").iterdir()))==2


def test_reserved_question_cannot_bypass_approval_through_individual_page(ui,tmp_path,monkeypatch):
    directory=tmp_path/"benchmarks";directory.mkdir()
    (directory/"research_heldout.proposed.json").write_text(json.dumps([dict(question="Reserved final question")]))
    monkeypatch.setattr(web_ui,"start_research",lambda *a,**kw:pytest.fail("Reserved questions cannot launch here"))
    widget(ui,"text_area","Research question").set_value("Reserved final question")
    widget(ui,"checkbox","Confirm paid hosted retrieval").check().run()
    widget(ui,"button","Start retrieval").click().run()
    assert not ui.exception and any("Reserved final" in e.value for e in ui.error)
    assert "research_job" not in ui.session_state
