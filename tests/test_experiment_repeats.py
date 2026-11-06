"""Offline experiment protocol checks; no independent labels are fabricated."""

import json

import httpx
import pandas as pd
import pytest

from openalex_agent.config import Settings
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.evaluation import import_judgments
from openalex_agent.experiments import report, run_experiments
from openalex_agent.llm import HostedJSONPlanner
from openalex_agent.models import Paper


def inputs(tmp_path):
    corpus = tmp_path / "corpus"
    freeze([Paper.from_openalex({"id": "W1", "title": "SYNTHETIC retrieval"})], corpus,
           acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q1", "question": "retrieval"}]))
    csv = tmp_path / "labels.csv"
    csv.write_text("query_id,paper_id,grade,assessor\nq1,W1,3,SYNTHETIC-TEST-ONLY\n")
    labels = tmp_path / "labels.json"
    import_judgments(str(csv), str(labels))
    return corpus, questions, labels


def configured(model="test-model"):
    return Settings(llm_base_url="https://provider.example/v1", llm_api_key="PRIVATE-TEST-KEY", llm_model=model)


def handler(request):
    body = json.loads(request.content)
    reformulating = "Reformulate" in body["messages"][0]["content"]
    content = {"query": "retrieval"} if reformulating else {"action": "STOP_RETRIEVAL", "arguments": {}}
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}],
                                    "usage": {"prompt_tokens": 25, "completion_tokens": 10}})


def test_three_repeats_rotated_schedule_settings_wire_capture_and_variance(db, tmp_path):
    corpus, questions, labels = inputs(tmp_path)
    output = tmp_path / "experiment"
    rows = run_experiments(str(corpus), str(questions), db,
        lambda: HostedJSONPlanner(configured(), transport=httpx.MockTransport(handler)),
        output=str(output), labels_path=str(labels), action_budgets=(2,), budget=Budget(page_size=1))
    manifest = json.loads((output / "experiment.json").read_text())
    assert len(rows) == 12 and {r["trial"] for r in rows} == {0, 1, 2}
    assert manifest["trials"] == 3 and manifest["status"] == "complete"
    assert manifest["completed_runs"] == manifest["planned_runs"] == 12
    assert manifest["source_sha256"] and manifest["provider_configuration"]["model"] == "test-model"
    assert manifest["run_order"] == [{k: r[k] for k in ("query_id", "action_budget", "trial", "method", "variant")} for r in rows]
    assert [r["method"] for r in rows[:4]] != [r["method"] for r in rows[4:8]]
    assert all(r["valid_llm_run"] and not r["fallbacks"] and not r["llm_failures"] for r in rows)
    requests = list((output / "runs").glob("*/planner_requests.jsonl"))
    assert len(requests) == 9
    assert all("PRIVATE-TEST-KEY" not in p.read_text() and "SYNTHETIC-TEST-ONLY" not in p.read_text() for p in requests)
    assert len(report(str(output))) == 6
    repeats = pd.read_csv(output / "repeatability.csv", header=[0, 1])
    assert len(repeats) >= 4
    with pytest.raises(ValueError, match="10–20"):
        report(str(output), final=True)
    manifest["status"] = "running"
    (output / "experiment.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        report(str(output))


def test_provider_drift_is_rejected_before_another_paid_request(db, tmp_path):
    corpus, questions, _ = inputs(tmp_path)
    calls = []
    def factory():
        calls.append(1)
        return HostedJSONPlanner(configured("first" if len(calls) == 1 else "changed"),
                                 transport=httpx.MockTransport(lambda r: pytest.fail("No HTTP after drift")))
    output = tmp_path / "experiment"
    with pytest.raises(ValueError, match="configuration changed"):
        run_experiments(str(corpus), str(questions), db, factory, output=str(output), action_budgets=(2,), trials=1)
    manifest = json.loads((output / "experiment.json").read_text())
    assert manifest["status"] == "failed" and manifest["error_type"] == "ValueError"
    assert len(pd.read_csv(output / "metrics.csv")) == 1


def test_state_invalid_decisions_are_counted_separately_from_http_failures(db, tmp_path):
    corpus, questions, _ = inputs(tmp_path)
    def invalid(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "action": "INSPECT_PAPER", "arguments": {"paper_id": "W1"}})}}]})
    rows = run_experiments(str(corpus), str(questions), db,
        lambda: HostedJSONPlanner(configured(), transport=httpx.MockTransport(invalid)),
        output=str(tmp_path / "experiment"), action_budgets=(2,), trials=1, methods=("graph",))
    assert rows[0]["fallbacks"] == 1 and rows[0]["llm_failures"] == 0
    assert not rows[0]["valid_llm_run"]
