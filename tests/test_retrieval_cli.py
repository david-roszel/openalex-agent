"""End-to-end Milestone 1 CLI tests with persistent databases and mocked HTTP."""

import json
import sys
import httpx
import pytest
from openalex_agent import cli
from openalex_agent.config import Settings
from openalex_agent.openalex import OpenAlexClient
from openalex_agent.storage import Database


def configure_cli(monkeypatch, handler):
    monkeypatch.setattr(cli.Settings, "from_env", lambda: Settings(min_interval=0))
    monkeypatch.setattr(cli, "OpenAlexClient", lambda db, settings: OpenAlexClient(db, settings, transport=httpx.MockTransport(handler)))
    def reject_advanced(*args):
        pytest.fail("Basic retrieval must not dispatch graph, agent, or corpus components")
    monkeypatch.setattr(cli, "advanced_command", reject_advanced)


def invoke(monkeypatch, database, *arguments):
    monkeypatch.setattr(sys, "argv", ["main.py", "--db", str(database), *arguments])
    cli.main()


def test_cli_search_export_inspect_and_restart_cache(monkeypatch, raw_work, tmp_path, capsys):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"results": [raw_work], "meta": {"cost_usd": 0.001}})
    configure_cli(monkeypatch, handle)
    database, output = tmp_path / "test.sqlite", tmp_path / "results.json"
    invoke(monkeypatch, database, "search", "adaptive retrieval", "--limit", "1", "--output", str(output))
    first = capsys.readouterr()
    assert "SYNTHETIC TEST" in first.out and "TEST AUTHOR" in first.out
    assert json.loads(output.read_text())[0]["id"].endswith("/W1")
    first_summary = json.loads(first.out.splitlines()[-1])
    assert first_summary["usage"]["requests"] == 1
    invoke(monkeypatch, database, "search", "adaptive retrieval", "--limit", "1")
    second_summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert second_summary["usage"]["requests"] == 0 and second_summary["usage"]["cache_hits"] == 1
    assert first_summary["run_id"] != second_summary["run_id"]
    invoke(monkeypatch, database, "get", "W1")
    inspected = capsys.readouterr().out
    assert "Adaptive retrieval reduces retrieval" in inspected
    assert "TEST INSTITUTION" in inspected and "https://openalex.org/W2" in inspected
    invoke(monkeypatch, database, "cached")
    assert "SYNTHETIC TEST" in capsys.readouterr().out
    assert len(calls) == 1
    with Database(str(database)) as db:
        events = db.events(first_summary["run_id"])
        assert events[0]["kind"] == "session_start" and events[-1]["kind"] == "session_end"
        assert events[-1]["status"] == "ok"


def test_explicit_refresh_downloads_again(monkeypatch, raw_work, tmp_path, capsys):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"results": [raw_work]})
    configure_cli(monkeypatch, handle)
    for extra in [[], ["--refresh"]]:
        invoke(monkeypatch, tmp_path / "test.sqlite", "search", "test", *extra)
    assert len(calls) == 2


def test_failed_request_has_safe_cli_error_and_persistent_usage(monkeypatch, tmp_path, capsys):
    configure_cli(monkeypatch, lambda request: httpx.Response(401, text="DO NOT LOG TEST-SECRET"))
    database = tmp_path / "test.sqlite"
    with pytest.raises(SystemExit) as exited:
        invoke(monkeypatch, database, "search", "test")
    assert exited.value.code == 1
    error = capsys.readouterr().err
    assert "HTTP 401" in error and "TEST-SECRET" not in error and "Traceback" not in error
    summary = json.loads(error.splitlines()[0])
    with Database(str(database)) as db:
        assert db.events(summary["run_id"])[-1]["usage"]["errors"] == 1


def test_corrupt_database_produces_actionable_error(monkeypatch, tmp_path, capsys):
    configure_cli(monkeypatch, lambda request: pytest.fail("Unexpected HTTP"))
    database = tmp_path / "corrupt.sqlite"
    database.write_bytes(b"not a sqlite database")
    with pytest.raises(SystemExit) as exited:
        invoke(monkeypatch, database, "cached")
    assert exited.value.code == 1
    error = capsys.readouterr().err
    assert "database operation failed" in error and "Traceback" not in error


def test_invalid_environment_configuration_has_no_traceback_or_raw_value(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("openalex_agent.config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENALEX_MAX_REQUESTS", "DO-NOT-ECHO-THIS-VALUE")
    with pytest.raises(SystemExit) as exited:
        invoke(monkeypatch, tmp_path / "test.sqlite", "search", "test")
    assert exited.value.code == 1
    error = capsys.readouterr().err
    assert "OPENALEX_MAX_REQUESTS" in error and "DO-NOT-ECHO" not in error and "Traceback" not in error
