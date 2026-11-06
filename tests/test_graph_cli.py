"""Milestone 2 CLI acceptance tests with clearly synthetic transport responses."""

import json
import sys
import httpx
import pytest
from openalex_agent import cli
from openalex_agent.config import Settings
from openalex_agent.graph import ScholarlyGraph
from openalex_agent.models import Paper
from openalex_agent.openalex import OpenAlexClient
from openalex_agent.storage import Database


@pytest.fixture
def graph_cli(monkeypatch, raw_work, tmp_path):
    database = tmp_path / "metadata.sqlite"
    with Database(str(database)) as db:
        db.save_papers([Paper.from_openalex(raw_work)])
    monkeypatch.setattr(cli.Settings, "from_env", lambda: Settings(min_interval=0, max_retries=0))
    def reject_advanced(*args):
        pytest.fail("Graph commands must not dispatch agents or corpus machinery")
    monkeypatch.setattr(cli, "advanced_command", reject_advanced)
    def invoke(*args, handler=None):
        def no_network(request):
            pytest.fail("Unexpected HTTP during graph build/inspection")
        monkeypatch.setattr(cli, "OpenAlexClient", lambda db, settings: OpenAlexClient(db, settings,
                            transport=httpx.MockTransport(handler or no_network)))
        monkeypatch.setattr(sys, "argv", ["main.py", "--db", str(database), "graph", *args])
        cli.main()
    return invoke, database, tmp_path / "graph.json"


def test_cli_build_reload_typed_neighborhood_path_and_read_only_inspection(graph_cli, capsys, monkeypatch):
    invoke, database, path = graph_cli
    invoke("--output", str(path), "--seed", "W1", "--path", "W2")
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["nodes"] == 5 and lines[0]["validated"]
    assert lines[0]["usage"]["requests"] == 0
    assert lines[-1]["discovery_path"] == ["https://openalex.org/W1", "https://openalex.org/W2"]
    def reject_save(*args):
        pytest.fail("Read-only inspection must not rewrite a snapshot")
    monkeypatch.setattr(ScholarlyGraph, "save", reject_save)
    invoke("--load", str(path), "--seed", "A1", "--direction", "out", "--relation", "AUTHOR_AFFILIATED_WITH_INSTITUTION")
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["output"] is None
    view = lines[1]["neighborhood"]
    assert {n["id"] for n in view["nodes"]} == {"https://openalex.org/A1", "https://openalex.org/I1"}


def test_cli_update_resolves_cached_placeholder_and_preserves_its_path(graph_cli, capsys):
    invoke, database, path = graph_cli
    invoke("--output", str(path))
    capsys.readouterr()
    with Database(str(database)) as db:
        db.save_papers([Paper(id="https://openalex.org/W2", title="SYNTHETIC TEST target")])
    invoke("--load", str(path), "--update")
    result = json.loads(capsys.readouterr().out.splitlines()[0])
    assert result["resolved_papers"] == 2 and result["output"] == str(path)
    assert ScholarlyGraph.load(path).path("W2") == ["https://openalex.org/W1", "https://openalex.org/W2"]


def test_cli_expands_saves_and_logs_without_agent(graph_cli, capsys, raw_work):
    invoke, database, path = graph_cli
    invoke("--output", str(path))
    capsys.readouterr()
    def handler(request):
        return httpx.Response(200, json={"results": [dict(raw_work, id="W2", referenced_works=[])]})
    invoke("--load", str(path), "--expand", "references", "--paper", "W1", "--limit", "1", handler=handler)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["resolved_papers"] == 2 and lines[0]["usage"]["requests"] == 1
    assert lines[1]["expansion"]["new_resolved_papers"] == 1
    assert ScholarlyGraph.load(path).paper("W2") is not None
    with Database(str(database)) as db:
        events = db.events(lines[0]["run_id"])
        assert any(e["kind"] == "graph_expansion" for e in events)
        assert events[-1]["kind"] == "session_end" and events[-1]["status"] == "ok"


def test_failed_cli_expansion_retains_last_valid_snapshot(graph_cli, capsys):
    invoke, database, path = graph_cli
    invoke("--output", str(path))
    capsys.readouterr()
    original = path.read_bytes()
    with pytest.raises(SystemExit) as exited:
        invoke("--load", str(path), "--expand", "references", "--paper", "W1", handler=lambda r: httpx.Response(503))
    assert exited.value.code == 1 and path.read_bytes() == original
    error = capsys.readouterr().err
    assert "Traceback" not in error
    usage = json.loads(error.splitlines()[0])
    assert usage["status"] == "error" and usage["usage"]["requests"] == 1


@pytest.mark.parametrize("options", [["--expand", "references"], ["--paper", "W1"],
    ["--seed", "W1", "--hops", "-1"], ["--seed", "W99"], ["--expand", "citations", "--paper", "W1", "--limit", "0"]])
def test_invalid_cli_graph_options_do_not_write_output(graph_cli, capsys, options):
    invoke, database, path = graph_cli
    with pytest.raises(SystemExit) as exited:
        invoke("--output", str(path), *options)
    assert exited.value.code == 1 and not path.exists()
    assert "Traceback" not in capsys.readouterr().err
