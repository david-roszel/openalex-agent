"""Read-only presentation regressions. No inference or scholarly HTTP requests."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from openalex_agent import live_ui
from openalex_agent.config import Settings
from openalex_agent.corpus import atomic_json
from openalex_agent.live_profiles import thorough_config
from openalex_agent.research import ResearchJob

APP = str(Path(__file__).resolve().parents[1] / "app.py")
RECORDED = "7444c61c-d62f-4dac-9e30-354e6154b423"


def widget(app, kind, label):
    return next(e for e in app.get(kind) if e.label == label)


def hashes(directory):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob("*") if p.is_file()}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(live_ui, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(live_ui, "load_live_settings", lambda: Settings())
    monkeypatch.setattr(live_ui, "start_research", lambda *a, **k: pytest.fail("Presentation must not start research"))
    return tmp_path


def test_missing_credentials_disable_hosted_action_but_page_renders(offline):
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert widget(app, "button", "Research").disabled
    assert any("LLM_API_KEY" in e.value for e in app.warning)


def test_demo_mode_disables_both_execution_choices(monkeypatch, offline):
    monkeypatch.setenv("OPENALEX_DEMO_READ_ONLY", "1")
    app = AppTest.from_file(APP).run()
    assert widget(app, "button", "Research").disabled
    widget(app, "selectbox", "Execution").set_value("deterministic").run()
    assert widget(app, "button", "Research").disabled
    assert any("Read-only demonstration" in e.value for e in app.info)


@pytest.mark.parametrize("variant,label", [("combined", "LLM + citation graph + CLM"),
    ("conventional", "Conventional LLM agent"), ("graph", "LLM + citation graph"), ("clm", "LLM + CLM")])
def test_friendly_labels_keep_backend_identifiers(offline, variant, label):
    app = AppTest.from_file(APP).run()
    widget(app, "selectbox", "Research configuration").set_value(variant).run()
    assert not app.exception
    assert widget(app, "selectbox", "Research configuration").value == variant
    assert label in widget(app, "selectbox", "Research configuration").options
    profile = thorough_config("An example question", variant=variant)
    assert profile.graph_memory == (variant in ("combined", "graph"))
    assert (profile.context_management == "clm") == (variant in ("combined", "clm"))


def test_bad_saved_configuration_has_specific_nonfatal_error(offline):
    directory = offline / "data/ui-runs" / RECORDED
    directory.mkdir(parents=True)
    atomic_json(directory / "summary.json", {"status": "completed"})
    atomic_json(directory / "research.json", {"configuration": None})
    app = AppTest.from_file(APP).run()
    widget(app, "button", "Open saved research").click().run()
    assert not app.exception
    assert any("saved run could not be read" in e.value for e in app.error)


def test_recorded_answer_loads_with_invalid_current_configuration_without_mutation(monkeypatch):
    # Actual historical trajectory, read only. The source is optional in the
    # portable archive, so its absence is explicit rather than a fake fixture.
    root = Path(APP).parent
    directory = root / "data/ui-runs" / RECORDED
    if not directory.exists():
        pytest.skip("Local historical demo trajectory is not packaged")
    before = hashes(directory)
    def invalid():
        raise ValueError("Invalid numeric configuration: LLM_TIMEOUT")
    monkeypatch.setattr(live_ui, "load_live_settings", invalid)
    monkeypatch.setattr(live_ui, "start_research", lambda *a, **k: pytest.fail("Saved display cannot start research"))
    monkeypatch.setenv("OPENALEX_DEMO_READ_ONLY", "1")
    app = AppTest.from_file(APP, default_timeout=20).run()
    assert not app.exception and widget(app, "button", "Research").disabled
    assert any("Saved research can still be opened" in e.value for e in app.error)
    widget(app, "selectbox", "Saved research").set_value(RECORDED).run()
    assert any("7444c61c" in option and "Reflexion" in option for option in widget(app, "selectbox", "Saved research").options)
    widget(app, "button", "Open saved research").click().run()
    assert not app.exception
    assert any("Recorded research result" in e.value for e in app.caption)
    assert any("Recorded architecture: LLM + citation graph + CLM" in e.value and "300-second" in e.value for e in app.caption)
    assert any("ReAct and Reflexion" in e.value for e in app.text)
    assert any("DOI:" in e.value and "https://doi.org/" in e.value for e in app.markdown)
    assert any(e.value == "Supporting publications" for e in app.subheader)
    receipts = [json.loads(e.value) if isinstance(e.value, str) else e.value for e in app.json]
    assert any(e.get("kind") == "live_model_input" and e.get("graph_edge_pairs") for e in receipts)
    assert any(e.get("kind") == "context_edit" and e.get("source") == "model" for e in receipts)
    assert any(e.get("kind") == "context_observation" for e in receipts)
    assert hashes(directory) == before


def test_display_deduplicates_links_without_removing_evidence_quotes():
    claim = {"text": "Example", "citations": [{"work_id": "https://openalex.org/W1", "quote": "one"},
        {"work_id": "https://openalex.org/W1", "quote": "two"}]}
    before = deepcopy(claim)
    assert live_ui.citation_links(claim) == "[W1](https://openalex.org/W1)"
    assert claim == before and len(claim["citations"]) == 2


def test_loading_phases_and_disabled_controls_are_visible(offline):
    job = ResearchJob(offline, thorough_config("SYNTHETIC loading indicator check"))
    job.update(status="running", phase="research")
    app = AppTest.from_file(APP).run()
    app.session_state["research_job"] = job
    app.run()
    assert not app.exception
    assert widget(app, "button", "Research").disabled
    assert app.status[0].label == "Investigating the literature…"
    job.update(phase="synthesis")
    app.run()
    assert not app.exception
    assert app.status[0].label == "Synthesizing the research answer…"


def test_long_answer_and_source_are_not_truncated_by_presentation(offline):
    long_text = "EXPLICIT_SYNTHETIC_UI_FIXTURE " * 1200
    config = thorough_config("Synthetic layout test")
    job = ResearchJob(offline, config)
    source = {"id": "https://openalex.org/W1", "title": "SYNTHETIC layout fixture", "year": 2024,
        "authors": ["SYNTHETIC"], "abstract_excerpt": long_text, "doi": None}
    job.update(status="completed", result={"question": config.question, "architecture": "clm_graph",
        "inspected_ids": [source["id"]], "retrieved_ids": [source["id"]],
        "synthesis": {"answer": {"answer": [{"text": long_text, "citations": [{"work_id": source["id"], "quote": "EXPLICIT_SYNTHETIC_UI_FIXTURE"}]}],
            "limitations": ["Synthetic layout check, not research evidence"]}, "sources": [source]}})
    app = AppTest.from_file(APP).run()
    app.session_state["research_job"] = job
    app.run()
    assert not app.exception
    assert any(long_text in e.value for e in app.markdown)
    assert any(long_text.strip() == e.value for e in app.text)
