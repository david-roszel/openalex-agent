"""No-cost safeguards and complete mocked architecture block; never hosted data."""
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from openalex_agent import hour_evaluation as hour
from openalex_agent.live_profiles import load_live_settings
from openalex_agent.openalex import BudgetExceeded
from test_submission_system import settings as mock_settings, research_transport, raw_records


@pytest.fixture
def selected():
    return replace(mock_settings(), llm_base_url="https://api.openai.com/v1/", llm_model="gpt-6.1-terra",
        llm_input_price=2.5, llm_output_price=5)


@pytest.fixture
def campaign(tmp_path, selected):
    path = tmp_path / "campaign"
    hour.prepare(path, selected)
    hour.authorize(path)  # Synthetic test directory, no paid transport.
    return path


def test_manifest_preserves_all_eight_questions_and_separates_diagnostic(campaign, selected):
    _, config, questions = hour.check(campaign, selected)
    original = json.loads((hour.PROJECT_ROOT / "benchmarks/live_submission_v1/questions.json").read_text())["questions"]
    assert [q for q in questions if q["id"] != "D01"] != []
    assert {q["id"]: q for q in questions if q["id"] != "D01"} == {q["id"]: q for q in original}
    assert len(config["cells"]) == 45 and config["primary_trajectories"] == 40
    assert [c["variant"] for c in config["cells"][:5]] == list(hour.VARIANTS)
    assert len({tuple(c["variant"] for c in config["cells"] if c["block"] == b) for b in range(1,6)}) == 5
    assert next(q for q in questions if q["id"] == "D01")["previously_tested"] is True
    agents = [config["research_configurations"][v] for v in hour.VARIANTS if v != "standard"]
    for key in ("budget", "inference_limits", "max_seconds", "synthesis_revisions", "final_synthesis"):
        assert all(a[key] == agents[0][key] for a in agents)
    assert Decimal(config["inference_ledger_ceiling_usd"]) + Decimal(config["openalex_total_reserve_usd"]) == 50


def test_snapshot_and_provider_changes_fail_before_network(campaign, selected):
    with pytest.raises(ValueError, match="Provider"):
        hour.check(campaign, replace(selected, llm_max_tokens=500))
    path = campaign / "source_snapshot/openalex_agent/live_quality.py"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="implementation changed"):
        hour.check(campaign, selected)
    with pytest.raises(ValueError, match="overwrite"):
        hour.prepare(campaign, selected)


def test_no_authorization_cannot_launch_paid_worker(tmp_path, selected):
    path = tmp_path / "no-grant"
    hour.prepare(path, selected)
    with pytest.raises(FileNotFoundError):
        hour.worker(path, "Q01-conventional", selected)


def test_per_run_and_shared_holds_precede_any_inference(campaign):
    _, config, _ = hour.load(campaign)
    local = hour.DualLedger(campaign, config, "Q01-conventional")
    key = local.reserve("A", "Q01-conventional", b"{}", input_price=2.5, output_price=5, max_output=4096)
    assert local.local.summary()["uncertain_attempts"] == 1
    assert local.shared.summary()["uncertain_attempts"] == 1
    with pytest.raises(BudgetExceeded, match="Unsettled"):
        local.reserve("B", "Q01-conventional", b"{}", input_price=2.5, output_price=5, max_output=4096)
    local.settle(key, {"prompt_tokens": 100, "completion_tokens": 20}, input_price=2.5, output_price=5, max_output=4096)
    assert local.local.summary()["uncertain_attempts"] == 0
    with pytest.raises(BudgetExceeded, match="ceiling"):
        local.ensure_affordable(b"x"*1000000, input_price=2.5, output_price=5, max_output=4096)


def test_unknown_usage_stops_and_cannot_replay(campaign):
    _, config, _ = hour.load(campaign)
    ledger = hour.DualLedger(campaign, config, "Q01-conventional")
    key = ledger.reserve("A", "Q01-conventional", b"{}", input_price=2.5, output_price=5, max_output=1000)
    with pytest.raises(BudgetExceeded, match="Unknown"):
        ledger.settle(key, {}, input_price=2.5, output_price=5, max_output=1000)
    assert hour.spending(campaign, config)["inference"]["uncertain_attempts"] == 1
    with pytest.raises(BudgetExceeded, match="Unsettled"):
        ledger.ensure_affordable(b"{}", input_price=2.5, output_price=5, max_output=1000)


def test_metadata_bound_is_durable_and_standard_is_one_request(campaign):
    _, config, _ = hour.load(campaign)
    transport = hour.MetadataTransport(campaign, config, "Q01-standard", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"results": [], "meta": {"cost_usd": .0001}})))
    transport.handle_request(httpx.Request("GET", "https://api.openalex.org/works?search=test"))
    assert Decimal(hour.spending(campaign, config)["openalex_retained_upper_bound_usd"]) == Decimal('.001')
    with pytest.raises(BudgetExceeded, match="attempt cap"):
        transport.handle_request(httpx.Request("GET", "https://api.openalex.org/works?search=test"))
    with pytest.raises(BudgetExceeded, match="pre-priced"):
        transport.handle_request(httpx.Request("GET", "https://api.openalex.org/works?search=test&rerank=true"))


def test_metadata_network_failure_keeps_maximum_charge(campaign):
    _, config, _ = hour.load(campaign)
    def fail(r):
        raise httpx.ConnectError("Synthetic DNS failure")
    transport = hour.MetadataTransport(campaign, config, "preflight", transport=httpx.MockTransport(fail))
    with pytest.raises(httpx.ConnectError):
        transport.handle_request(httpx.Request("GET", "https://api.openalex.org/works?search=test"))
    assert hour.spending(campaign, config)["openalex_retained_upper_bound_usd"] == "0.001"


@pytest.mark.parametrize("seconds,money,allowed", [(3600,50,True),(1504,50,False),(3600,5,False),(0,50,False),(3600,0,False)])
def test_complete_block_admission_enforces_time_and_money(seconds, money, allowed):
    cells = [{"variant": v} for v in hour.VARIANTS]
    assert (not hour.admission(seconds, Decimal(money), cells)) is allowed


def test_mocked_matched_block_uses_shared_backend_and_preserves_all_outputs(campaign, selected, raw_work):
    _, config, _ = hour.load(campaign)
    for cell in config["cells"][:5]:
        path = campaign / "trajectories" / cell["id"] / "trajectory.json"
        row = json.loads(path.read_text());row.update(status="RUNNING", attempted=True)
        hour.atomic_json(path, row)
        transport, calls = research_transport(edit=cell["variant"] in ("clm", "combined"))
        record = hour.worker(campaign, cell["id"], selected, transports={
            "planner_transport": transport, "openalex_transport": httpx.MockTransport(lambda r: httpx.Response(200, json={"results": raw_records(raw_work)}))})
        assert record["status"] == "COMPLETED"
        assert record["evidence_mode"].startswith("MOCKED")
        if cell["variant"] == "standard":
            assert not calls and not record["result"].get("synthesis")
        else:
            assert record["result"]["synthesis"]["status"] == "qualified"
            assert (Path(record["research_directory"]) / "answer.md").exists()
            assert record["result"]["llm_usage"]["calls"] == len(calls)
        with pytest.raises(ValueError, match="RUNNING"):
            hour.worker(campaign, cell["id"], selected, transports={})
    report = hour.report(campaign)
    assert len(report["rows"]) == 45
    assert sum(r["attempted"] for r in report["rows"]) == 5
    assert all(r["question_answered_independent"] is None for r in report["rows"])
    assert all(r["target_recall_at_10"] is None for r in report["rows"][5:])
    assert (campaign / "GRAPH_CLM_ACTIVITY.csv").exists()
    assert (campaign / "EVIDENCE_REVIEW.md").exists()
    assert Decimal(report["spending"]["total_retained_upper_bound_usd"]) <= 50


def test_failed_preflight_records_blocker_without_model_worker(campaign, selected, monkeypatch):
    invocations = []
    class Failed:
        returncode = 2
        def __init__(self, args, **kwargs):
            invocations.append(args)
            hour.atomic_json(campaign / "PREFLIGHT.json", {"status": "BLOCKED_PREFLIGHT", "hosted_calls": 0, "error": "SYNTHETIC_DNS_FAILURE"})
        def wait(self, **kwargs):
            return 2
        def poll(self):
            return 2
    monkeypatch.setattr(hour.subprocess, "Popen", Failed)
    hour.supervise(campaign, selected)
    state = json.loads((campaign / "RUN_STATE.json").read_text())
    assert state["status"] == "BLOCKED_PREFLIGHT" and len(invocations) == 1
    assert all("worker" not in a for a in invocations)
    assert hour.report(campaign)["spending"]["inference"]["attempts"] == 0
    with pytest.raises(ValueError, match="automatic restart"):
        hour.supervise(campaign, selected)


def test_supervisor_terminates_only_own_timed_out_worker_and_retains_failure(campaign, selected, monkeypatch):
    instances = []
    class Child:
        returncode = None
        def __init__(self, args, **kwargs):
            self.preflight = "preflight" in args
            self.terminated = False
            instances.append(self)
            if self.preflight:
                hour.atomic_json(campaign / "PREFLIGHT.json", {"status": "PASSED"})
        def wait(self, **kwargs):
            if self.preflight or self.terminated:
                self.returncode = 0 if self.preflight else -15
                return self.returncode
            raise subprocess.TimeoutExpired("mock-own-worker", 299)
        def poll(self):
            return self.returncode
        def terminate(self):
            self.terminated = True
        def kill(self):
            self.terminated = True
    monkeypatch.setattr(hour.subprocess, "Popen", Child)
    hour.supervise(campaign, selected)
    row = json.loads((campaign / "trajectories/Q01-standard/trajectory.json").read_text())
    assert row["status"] == "TIMED_OUT" and instances[-1].terminated
    assert len(instances) == 2  # Preflight then first worker; no retry.
    assert len((campaign / "TRAJECTORY_RESULTS.jsonl").read_text().splitlines()) == 1
    assert json.loads((campaign / "RUN_STATE.json").read_text())["status"] == "STOPPED"


def test_manual_execution_enters_frozen_source_before_loading_credentials(campaign, monkeypatch):
    invocations = []
    class FrozenExec(Exception):
        pass
    monkeypatch.setattr(hour.os, "chdir", lambda p: invocations.append(("cwd", p)))
    def execve(binary, argv, env):
        invocations.append((binary, argv, env))
        raise FrozenExec()
    monkeypatch.setattr(hour.os, "execve", execve)
    monkeypatch.setattr(hour, "load_live_settings", lambda *a: pytest.fail("Must re-execute frozen code before reading credentials"))
    with pytest.raises(FrozenExec):
        hour.main(["run", "--output", str(campaign), "--resume-blocked"])
    assert invocations[0][1] == campaign / "source_snapshot"
    assert invocations[1][2]["PYTHONPATH"] == str(campaign / "source_snapshot")
    assert "--resume-blocked" in invocations[1][1]
