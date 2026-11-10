"""Submission workflows: scripted/mock generations are never empirical results."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path

import httpx
import pytest

from openalex_agent.config import Settings
from openalex_agent.controller import Budget
from openalex_agent.context_planner import InferenceLimits
from openalex_agent.corpus import freeze
from openalex_agent.live_profiles import load_live_settings, priced_settings, thorough_config
from openalex_agent.live_research import source_records
from openalex_agent.live_safety import SpendingLedger, diagnostic
from openalex_agent.models import Paper
from openalex_agent.openalex import BudgetExceeded
from openalex_agent.research import ResearchConfig, start_research


def settings():
    return Settings(api_key="SUBMISSION_API_SECRET", llm_api_key="SUBMISSION_LLM_SECRET",
        llm_base_url="https://mock.example/v1/", llm_model="mock-test", llm_max_tokens=1000,
        llm_input_price=1, llm_output_price=5, llm_timeout=2, min_interval=0, max_retries=0)


def envelope(value, *, usage=True):
    body = {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}], "service_tier": "default"}
    if usage:
        body["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
    return httpx.Response(200, json=body)


def raw_records(raw_work):
    return [dict(raw_work, id="W1", title="SYNTHETIC adaptive retrieval source A", referenced_works=["W2"]),
        dict(raw_work, id="W2", title="SYNTHETIC adaptive retrieval source B", referenced_works=[])]


def research_transport(*, edit=False, gaps=False, bad_first=False, recover=False):
    calls, edited, finals, recovered = [], [False], [0], [False]
    def handle(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        calls.append(payload)
        if "evidence" in payload:
            finals[0] += 1
            source = payload["evidence"][0]
            citation = {"work_id": "https://openalex.org/W999" if bad_first and finals[0] == 1 else source["id"],
                "quote": source["abstract_excerpt"][:100]}
            claim = {"text": "SYNTHETIC MODEL SCRIPT: the supplied metadata describes adaptive retrieval.", "citations": [citation]}
            return envelope({"answer": [claim], "foundations": [claim], "comparison": [claim], "conclusion": [claim],
                "limitations": ["SYNTHETIC OFFLINE SCRIPT ONLY; not a factual research assessment."]})
        if "protected" not in payload:
            return envelope({"initial_query": "adaptive retrieval", "components": [
                {"id": "C1", "question": "Methods", "search_query": "adaptive retrieval methods"},
                {"id": "C2", "question": "Limitations", "search_query": "adaptive retrieval limitations"}]})
        records = list(source_records(payload).values())
        if edit and not edited[0] and "active_context" in payload:
            edited[0] = True
            return envelope({"context_edit": {"path": "active_context.txt", "op": "replace",
                "text": "PERSISTED SCRIPTED EVIDENCE\n" + json.dumps({"top_papers": records})}})
        if recover and not recovered[0]:
            recovered[0] = True
            return envelope({"read_observation": {"index": 0, "start": 0, "chars": 128}})
        evidence = [{"component": f"C{i + 1}", "work_id": p["id"], "quote": p.get("abstract_excerpt", "")[:100]}
            for i, p in enumerate(records[:2])] if not gaps else []
        return envelope({"decision": {"action": "STOP_RETRIEVAL", "arguments": {}, "reason": "Scripted stop"},
            "research_status": {"evidence": evidence, "gaps": ["C1", "C2"] if gaps else [],
                "sufficient": not gaps, "next_query": "adaptive retrieval follow up"}})
    return httpx.MockTransport(handle), calls


def finish(job):
    job.join(15)
    state = job.snapshot()
    assert state["status"] == "completed", state.get("error")
    return state


def test_explicit_env_file_and_environment_precedence(monkeypatch, tmp_path):
    values = {"OPENALEX_API_KEY": "FILE_API_SECRET", "LLM_BASE_URL": "https://mock.example/v1/",
        "LLM_API_KEY": "FILE_LLM_SECRET", "LLM_MODEL": "file-model", "LLM_INPUT_USD_PER_MILLION": "1",
        "LLM_OUTPUT_USD_PER_MILLION": "5"}
    for key in values:
        monkeypatch.delenv(key, raising=False)
    file = tmp_path / ".env"; file.write_text("\n".join(k + "=" + v for k, v in values.items()))
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    selected = Settings.from_env(file)
    assert selected.llm_model == "environment-model" and selected.api_key == "FILE_API_SECRET"
    assert selected.llm_api_key == "FILE_LLM_SECRET"


def test_zero_price_failure_is_fixed_for_verified_terra_and_specific_for_unknown_provider(tmp_path):
    terra = replace(settings(), llm_base_url="https://api.openai.com/v1/", llm_model="gpt-6.1-terra",
        llm_input_price=0, llm_output_price=0)
    assert priced_settings(terra).llm_input_price == 2.5
    assert priced_settings(terra).llm_output_price == 5
    unknown = replace(settings(), llm_input_price=0, llm_output_price=0)
    with pytest.raises(ValueError, match="LLM_INPUT_USD_PER_MILLION"):
        start_research(thorough_config("adaptive retrieval"), unknown, root=tmp_path)
    assert not (tmp_path / "data/ui-runs").exists()


def test_missing_credentials_are_named_without_starting_worker(tmp_path):
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        start_research(thorough_config("adaptive retrieval"), replace(settings(), llm_api_key=""), root=tmp_path)


def test_thorough_live_initialization_planning_graph_clm_and_synthesis(tmp_path, raw_work):
    transport, calls = research_transport(edit=True, recover=True)
    state = finish(start_research(thorough_config("adaptive retrieval"), settings(), root=tmp_path,
        openalex_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": raw_records(raw_work)})),
        planner_transport=transport))
    result = state["result"]
    assert result["synthesis"]["status"] == "qualified"
    assert result["architecture"] == "clm_graph"
    assert result["mechanisms"]["graph_edges_stored"] >= 1
    assert result["mechanisms"]["graph_unique_edges_sent"] >= 1
    assert result["mechanisms"]["clm_edits_committed"] == 1
    assert result["mechanisms"]["committed_edits_in_later_requests"] == 1
    assert result["mechanisms"]["accepted_recoveries"] == 1
    assert result["mechanisms"]["recovered_evidence_in_later_requests"] == 1
    assert any("PERSISTED SCRIPTED EVIDENCE" in p.get("active_context", "") for p in calls[2:])
    assert result["research_coverage"]["adequate"] is True
    assert result["spending"]["uncertain_attempts"] == 0
    assert (Path(state["directory"]) / "answer.md").is_file()
    assert result["llm_usage"]["calls"] == len(calls) <= 28
    assert "SUBMISSION_LLM_SECRET" not in json.dumps(state)


def test_incomplete_question_triggers_bounded_followup_before_final_answer(tmp_path, raw_work):
    transport, calls = research_transport(gaps=True)
    state = finish(start_research(thorough_config("adaptive retrieval", variant="conventional"), settings(), root=tmp_path,
        openalex_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": raw_records(raw_work)})),
        planner_transport=transport))
    result = state["result"]
    redirects = [e for e in state["events"] if e["kind"] == "live_stop_reconsidered"]
    assert len(redirects) == 2 and result["actions"] == 3
    assert result["research_coverage"]["unresolved_components"] == ["C1", "C2"]
    assert result["synthesis"]["status"] == "qualified"
    assert len(calls) <= result["budget"]["max_llm_calls"]


def test_reserved_revision_repairs_untraceable_reference(tmp_path, raw_work):
    transport, calls = research_transport(bad_first=True)
    config = ResearchConfig("adaptive retrieval", final_synthesis=True, synthesis_revisions=1,
        budget=Budget(max_actions=1, max_rounds=1, max_llm_calls=2, page_size=2),
        inference_limits=InferenceLimits(max_llm_cost_usd=1))
    state = finish(start_research(config, settings(), root=tmp_path,
        openalex_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": raw_records(raw_work)})), planner_transport=transport))
    syn = state["result"]["synthesis"]
    assert syn["status"] == "qualified" and len(calls) == 2
    assert [a["status"] for a in syn["attempts"]] == ["failed", "valid"]
    assert "revision" in calls[1]


def test_time_limit_does_not_launch_unaffordable_long_model_call(tmp_path, raw_work):
    state = finish(start_research(thorough_config("adaptive retrieval", max_seconds=0.01), settings(), root=tmp_path,
        openalex_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": raw_records(raw_work)})),
        planner_transport=httpx.MockTransport(lambda r: pytest.fail("Time ceiling should block inference"))))
    assert state["result"]["llm_usage"]["calls"] == 0
    assert state["result"]["stop_reason"] == "research_time_limit"


def test_diagnostics_keep_traceback_but_redact_keys_and_urls():
    try:
        raise RuntimeError("Bearer SUBMISSION_LLM_SECRET https://example.test?api_key=SUBMISSION_API_SECRET")
    except RuntimeError as exc:
        result = diagnostic(exc, ("SUBMISSION_LLM_SECRET", "SUBMISSION_API_SECRET"))
    assert "Traceback" in result["traceback"] and result["type"] == "RuntimeError"
    assert "SUBMISSION_LLM_SECRET" not in json.dumps(result)
    assert "SUBMISSION_API_SECRET" not in json.dumps(result)


def test_spending_hold_survives_reopening_and_prevents_replay(tmp_path):
    path = tmp_path / "independent.sqlite"
    ledger = SpendingLedger(path, ceiling=1, identity="offline-test")
    ledger.reserve("request", "cell", b"{}", input_price=1, output_price=5, max_output=100)
    reopened = SpendingLedger(path, ceiling=1, identity="offline-test")
    assert reopened.summary()["uncertain_attempts"] == 1
    with pytest.raises(BudgetExceeded, match="Unsettled"):
        reopened.reserve("request", "cell", b"{}", input_price=1, output_price=5, max_output=100)
    with pytest.raises(ValueError, match="ceiling changed"):
        SpendingLedger(path, ceiling=2, identity="offline-test")


def test_concurrent_reservations_cannot_overspend(tmp_path):
    ledger = SpendingLedger(tmp_path / "ledger.sqlite", ceiling=0.002, identity="fixture")
    def reserve(i):
        try:
            ledger.reserve(str(i), "cell", b"{}", input_price=1, output_price=5, max_output=100)
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(reserve, range(2))) == 1
    assert float(ledger.summary()["retained_upper_bound_usd"]) <= 0.002


def test_unknown_usage_keeps_reservation_and_blocks_future_calls(tmp_path):
    ledger = SpendingLedger(tmp_path / "ledger.sqlite", ceiling=1, identity="fixture")
    ledger.reserve("A", "cell", b"{}", input_price=1, output_price=5, max_output=100)
    before = ledger.summary()["retained_upper_bound_usd"]
    with pytest.raises(BudgetExceeded, match="Unknown"):
        ledger.settle("A", {}, input_price=1, output_price=5, max_output=100)
    assert ledger.summary()["retained_upper_bound_usd"] == before
    with pytest.raises(BudgetExceeded):
        ledger.reserve("B", "other", b"{}", input_price=1, output_price=5, max_output=100)


def test_maximum_request_charge_is_reserved_before_transmission(tmp_path):
    ledger = SpendingLedger(tmp_path / "ledger.sqlite", ceiling=0.0001, identity="fixture")
    with pytest.raises(BudgetExceeded, match="ceiling"):
        ledger.reserve("A", "cell", b"{}", input_price=1, output_price=5, max_output=100)
    assert ledger.summary()["attempts"] == 0


def test_cli_runs_same_backend_and_refuses_unapproved_hosted_call(tmp_path, raw_work, capsys):
    from openalex_agent.live_cli import main
    assert main(["adaptive retrieval"], root=tmp_path, settings=settings()) == 2
    assert "not authorized" in capsys.readouterr().err
    transport, calls = research_transport()
    assert main(["adaptive retrieval", "--allow-paid", "--variant", "conventional"], root=tmp_path, settings=settings(),
        openalex_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": raw_records(raw_work)})), planner_transport=transport) == 0
    output = capsys.readouterr()
    assert "SYNTHETIC MODEL SCRIPT" in output.out and "Sources and evidence" in output.out
    assert "SUBMISSION_LLM_SECRET" not in output.out + output.err
    assert len(calls) >= 3


def fixture_questions(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": [{"id": "Q1", "domain": "SYNTHETIC TEST ONLY", "question": "adaptive retrieval",
        "baseline_query": "adaptive retrieval", "required_components": ["method"],
        "targets": [{"title": "SYNTHETIC adaptive retrieval source A", "doi": "10.0000/synthetic", "openalex_id": "https://openalex.org/W1"}]}]}))
    return path


def test_benchmark_freeze_missing_metrics_and_authorization_gate(tmp_path):
    from openalex_agent.live_benchmark import prepare, report, run
    directory = tmp_path / "campaign"
    plan = prepare(directory, settings(), questions_path=fixture_questions(tmp_path))
    assert plan["proposed_hosted_trajectories"] == 2
    assert not (directory / "authorization.json").exists()
    rows = report(directory)["rows"]
    assert all(r["target_recall_at_10"] is None and r["status"] == "NOT_RUN" for r in rows)
    assert all(r["scientifically_unsupported_claims"] is None for r in rows)
    with pytest.raises(ValueError, match="authorization"):
        run(directory, settings())
    assert not (directory / "live-spending.sqlite").exists()


def test_mock_benchmark_outputs_never_count_as_real_results_and_resume_skips_terminal(tmp_path, raw_work):
    from openalex_agent.live_benchmark import prepare, authorize, run
    directory = tmp_path / "campaign"
    prepare(directory, settings(), questions_path=fixture_questions(tmp_path))
    authorize(directory, ceiling=1, confirmation="AUTHORIZE_LIVEDEMO_HOSTED", source="OFFLINE_TEST_ONLY")
    def factory(cell):
        transport, _ = research_transport()
        return {"openalex_transport": httpx.MockTransport(lambda r: httpx.Response(200, json={"results": raw_records(raw_work)})), "planner_transport": transport}
    report = run(directory, settings(), transport_factory=factory)
    assert report["real_hosted_completed"] == 0
    assert all(r["status"] == "COMPLETED" and r["evidence_mode"] == "MOCKED_INTEGRATION" for r in report["rows"])
    assert all(r["target_recall_at_10"] == 1 for r in report["rows"])
    assert next(r for r in report["rows"] if r["variant"] == "standard")["final_answer_produced"] is None
    assert (directory / "tables/reliability.tex").exists()
    assert json.loads((directory / "claim-review.json").read_text())[0]["verdict"] == "UNVERIFIED"
    again = run(directory, settings(), transport_factory=lambda c: pytest.fail("Terminal trajectories cannot silently rerun"))
    assert again["attempted"] == 3


def test_interrupted_benchmark_is_retained_without_replay(tmp_path):
    from openalex_agent.live_benchmark import prepare, run
    directory = tmp_path / "campaign"
    prepare(directory, settings(), questions_path=fixture_questions(tmp_path), variants=("standard",))
    file = directory / "trajectories/Q1-standard/trajectory.json"
    record = json.loads(file.read_text()); record.update(status="RUNNING", attempted=True)
    file.write_text(json.dumps(record))
    output = run(directory, settings(), only=("standard",))
    assert output["rows"][0]["status"] == "INTERRUPTED"


def test_fixed_submission_manifest_has_eight_domains_and_predefined_targets():
    file = Path(__file__).resolve().parents[1] / "benchmarks/live_submission_v1/questions.json"
    questions = json.loads(file.read_text())["questions"]
    assert len(questions) == len({q["domain"] for q in questions}) == 8
    assert all(len(q["targets"]) == 2 and q["required_components"] for q in questions)


def test_submission_archive_excludes_private_files_and_refuses_known_secret(tmp_path):
    import zipfile
    from scripts.build_submission import build
    secret = "NEVER_PUBLISH_SUBMISSION_SECRET_123"
    (tmp_path / ".env").write_text("LLM_API_KEY=" + secret + "\n")
    (tmp_path / ".env.example").write_text("LLM_API_KEY=\n")
    (tmp_path / "README.md").write_text("Portable public documentation")
    (tmp_path / "data").mkdir(); (tmp_path / "data/private.json").write_text(secret)
    archive, receipt = build(tmp_path)
    assert receipt["configured_secret_scan"] == "PASS"
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        assert any(n.endswith(".env.example") for n in names)
        assert not any(n.endswith("/.env") or "/data/" in n or ".venv" in n for n in names)
    (tmp_path / "README.md").write_text("Accidental credential: " + secret)
    with pytest.raises(ValueError, match="Configured credential"):
        build(tmp_path)


def test_target_identity_verification_uses_no_model_and_does_not_mutate_labels(tmp_path, raw_work, monkeypatch):
    from openalex_agent import openalex
    from openalex_agent.live_benchmark import prepare, validate_targets, sha
    directory = tmp_path / "campaign"
    prepare(directory, settings(), questions_path=fixture_questions(tmp_path))
    before = sha(directory / "plan.json")
    original = openalex.OpenAlexClient
    monkeypatch.setattr(openalex, "OpenAlexClient", lambda db, selected: original(db, selected,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=dict(raw_work,
            title="SYNTHETIC adaptive retrieval source A", doi="https://doi.org/10.0000/synthetic")))))
    receipt = validate_targets(directory, settings())
    assert receipt["all_verified"] and receipt["paid_model_calls"] == 0
    assert receipt["targets"][0]["openalex_id"] == "https://openalex.org/W1"
    assert sha(directory / "plan.json") == before
