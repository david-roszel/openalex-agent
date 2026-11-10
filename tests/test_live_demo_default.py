"""Normal launch remains live-capable. No server or API requests are made."""
import json
import os
from pathlib import Path
import subprocess

import pytest
from streamlit.testing.v1 import AppTest
from openalex_agent import live_ui
from test_submission_system import settings

ROOT = Path(__file__).resolve().parents[1]


def test_live_page_accepts_new_question_without_saved_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENALEX_DEMO_READ_ONLY", "0")
    monkeypatch.setattr(live_ui, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(live_ui, "load_live_settings", settings)
    monkeypatch.setattr(live_ui, "start_research", lambda *a, **k: pytest.fail("Verification must not start research"))
    app = AppTest.from_file(str(ROOT / "live_app.py")).run()
    assert not app.exception
    question = next(e for e in app.text_area if e.label == "Research question")
    question.set_value("A newly entered scholarly question").run()
    assert not question.disabled
    assert not next(e for e in app.button if e.label == "Research").disabled
    assert next(e for e in app.selectbox if e.label == "Source").value == "Live OpenAlex"
    assert not any(e.label == "Saved research" for e in app.selectbox)
    assert not any("Read-only demonstration" in e.value for e in app.info)


@pytest.mark.parametrize("explicit,expected", [(None, "0"), ("0", "0"), ("1", "1")])
def test_launcher_is_live_by_default_and_read_only_only_when_explicit(tmp_path, explicit, expected):
    receipt = tmp_path / "launch.json"
    stub = tmp_path / "python-stub"
    stub.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
        "with open(os.environ['OPENALEX_LAUNCH_TEST_OUTPUT'],'w') as f:\n"
        " json.dump({'read_only':os.getenv('OPENALEX_DEMO_READ_ONLY'),'arguments':sys.argv[1:]},f)\n")
    stub.chmod(0o700)
    env = dict(os.environ, OPENALEX_PYTHON=str(stub), OPENALEX_DASHBOARD_PORT="8532",
        OPENALEX_LAUNCH_TEST_OUTPUT=str(receipt))
    env.pop("OPENALEX_DEMO_READ_ONLY", None)
    if explicit is not None:
        env["OPENALEX_DEMO_READ_ONLY"] = explicit
    subprocess.run([str(ROOT / "launch-live-research.sh")], env=env, cwd=tmp_path, check=True)
    result = json.loads(receipt.read_text())
    assert result["read_only"] == expected
    assert result["arguments"][:3] == ["-m", "streamlit", "run"]
    assert result["arguments"][3] == str(ROOT / "live_app.py")
