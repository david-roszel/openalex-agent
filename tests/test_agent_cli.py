"""CLI acceptance is offline; hosted credentials and internet are never needed."""

import json
from pathlib import Path
import sys
import httpx
import pytest
from openalex_agent import cli
from openalex_agent.config import Settings
from openalex_agent.corpus import freeze
from openalex_agent.models import Paper
from openalex_agent.openalex import OpenAlexClient


def setup_cli(monkeypatch, handler):
    monkeypatch.setattr(cli.Settings, "from_env", lambda: Settings(min_interval=0, max_retries=0))
    monkeypatch.setattr(cli, "OpenAlexClient", lambda db, settings: OpenAlexClient(db, settings, transport=httpx.MockTransport(handler)))


def invoke(monkeypatch, database, *args):
    monkeypatch.setattr(sys, "argv", ["main.py", "--db", str(database), "agent", *args])
    cli.main()


def test_cli_offline_agent_is_runnable_and_identifies_actual_planner(monkeypatch, raw_work, tmp_path, capsys):
    setup_cli(monkeypatch, lambda r: pytest.fail("Offline agent cannot access the network"))
    corpus = tmp_path / "corpus"
    freeze([Paper.from_openalex(raw_work), Paper.from_openalex({"id": "W2", "title": "SYNTHETIC auxiliary work"})],
           corpus, acquisition=[{"source": "SYNTHETIC CLI TEST"}])
    invoke(monkeypatch, tmp_path / "metadata.sqlite", "retrieval", "--corpus", str(corpus),
           "--max-actions", "2", "--page-size", "1", "--output", str(tmp_path / "runs"))
    captured = capsys.readouterr()
    summary = json.loads(captured.out.splitlines()[-1])
    assert "Live OpenAlex retrieval is unavailable" in captured.err
    assert summary["retrieval"]["mode"] == "frozen_corpus"
    assert summary["retrieval"]["live_openalex_available"] is False
    assert summary["planner"] == "deterministic" and summary["model"] == "none" and summary["actions"] == 2
    assert summary["llm_usage"]["calls"] == 0 and summary["api_usage"]["requests"] == 0
    assert json.loads((Path(summary["directory"]) / "state.json").read_text())["finished"]


def test_cli_missing_hosted_configuration_is_actionable_and_makes_no_request(monkeypatch, tmp_path, capsys):
    setup_cli(monkeypatch, lambda r: pytest.fail("Missing configuration must fail before any request"))
    with pytest.raises(SystemExit) as exited:
        invoke(monkeypatch, tmp_path / "metadata.sqlite", "retrieval", "--planner", "hosted")
    error = capsys.readouterr().err
    assert exited.value.code == 1 and "LLM_BASE_URL" in error and "Traceback" not in error


def test_cli_retrieval_failure_returns_nonzero_and_can_resume_same_action(monkeypatch, raw_work, tmp_path, capsys):
    ready = {"ok": False}
    def handle(request):
        return httpx.Response(200, json={"results": [raw_work]}) if ready["ok"] else httpx.Response(503)
    setup_cli(monkeypatch, handle)
    database = tmp_path / "metadata.sqlite"
    common = ["retrieval", "--max-actions", "1", "--output", str(tmp_path / "runs")]
    with pytest.raises(SystemExit) as exited:
        invoke(monkeypatch, database, *common)
    assert exited.value.code == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["stop_reason"] == "retrieval_error"
    assert summary["retrieval"]["live_openalex_available"] is True
    directory = Path(summary["directory"])
    assert not json.loads((directory / "state.json").read_text())["finished"]
    ready["ok"] = True
    invoke(monkeypatch, database, *common, "--resume", str(directory))
    resumed = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert resumed["run_id"] == summary["run_id"] and resumed["actions"] == 1
    assert resumed["api_usage"]["requests"] == 2
    assert json.loads((directory / "state.json").read_text())["finished"]
