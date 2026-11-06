"""Dashboard uses actual backend artifacts, deterministic fixtures and HTTP mocks."""

from argparse import Namespace
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import threading

import httpx
import pandas as pd
import pytest

from openalex_agent import experiment_dashboard as dash
from openalex_agent.config import Settings
from openalex_agent.context_planner import InferenceLimits
from openalex_agent.context_pressure import run_pressure
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.experiment_spec import BenchmarkSpec
from openalex_agent.models import Paper


def config(**kw):
    defaults=dict(methods=("llm","llm_graph","clm","clm_graph"),budget=Budget(max_actions=2,max_rounds=2,max_inspected=5,
        page_size=1,output_k=10,max_llm_calls=4),action_budgets=(2,))
    defaults.update(kw);return dash.DashboardConfig(**defaults)


def run(tmp_path,**kw):
    job=dash.start_dashboard(config(**kw),Settings(),root=tmp_path,submission_id="test")
    job.join(20);assert not job.thread.is_alive()
    return job


def test_multi_method_execution_isolated_states_and_measured_context(tmp_path):
    job=run(tmp_path,trials=2,seeds=(4,5))
    assert job.snapshot()["status"]=="completed"
    loaded=dash.read_comparison(job.directory,tmp_path)
    assert len(loaded["rows"])==8 and {r["method"] for r in loaded["rows"]}==set(config().methods)
    assert len({r["artifact_directory"] for r in loaded["rows"]})==8
    assert {r["seed"] for r in loaded["rows"]}=={4,5}
    for row in loaded["rows"]:
        trace=dash.trace_records(row["artifact_directory"],tmp_path)
        assert trace["redundant_actions"]==0
        if not row["method"].endswith("_graph"):assert not trace["graph_planner_inputs"]
        context=dash.context_records(row["artifact_directory"],tmp_path)
        assert context["requests"] and context["requests"][0]["active_tokens"] is not None
        if row["method"].startswith("clm"):
            assert any(t["source"]=="model" and t["status"]=="committed" for t in context["transitions"])


def test_unknown_judgments_are_not_zero_and_custom_diagnostic_query_not_misgraded(tmp_path):
    job=run(tmp_path,question="a different question not labeled by G1")
    loaded=dash.read_comparison(job.directory,tmp_path)
    assert not loaded["relevance_evaluated"] and not loaded["factual_evaluated"]
    assert all(r["precision_at_10"] is None and r["task_overall_answer_accuracy"] is None for r in loaded["rows"])
    exported=dash.export_summary(loaded)
    assert "NOT EVALUATED" in exported and "source_experiment" in exported


def test_raw_evidence_and_frozen_source_not_mutated_by_load_export_or_assessment(tmp_path):
    job=run(tmp_path)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in job.directory.rglob("*") if p.is_file()}
    loaded=dash.read_comparison(job.directory,tmp_path)
    assert dash.figure_export(loaded,"precision_at_10").startswith(b"\x89PNG")
    labels=job.directory/"synthetic-labels.json"
    assessed=dash.assess_saved_copy(job.directory,tmp_path,labels=str(labels))
    assert assessed!=job.directory and (assessed/"benchmark/metrics.csv").exists()
    assert dash.saved_experiments(tmp_path)
    for p,h in before.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h


def test_scientific_readiness_and_paid_confirmation_are_distinct_fail_closed_gates(tmp_path,monkeypatch):
    selected=config(category="Development pilot",planner="hosted",paid_confirmed=True)
    assert not dash.readiness(selected,tmp_path,Settings())["ready"]
    with pytest.raises(ValueError,match="readiness"):
        dash.start_dashboard(selected,Settings(),root=tmp_path,submission_id="blocked")
    assert not (tmp_path/"data/experiment-dashboard").exists()
    monkeypatch.setattr(dash,"readiness",lambda *a,**kw:dict(ready=True,reasons=[]))
    with pytest.raises(ValueError,match="confirmation"):
        dash.start_dashboard(replace(selected,paid_confirmed=False),Settings(),root=tmp_path,submission_id="no-paid")
    with pytest.raises(ValueError,match="offline"):
        config(kind="Long-horizon pressure",category="Development pilot").validate()


def test_idempotent_submission_survives_reruns(tmp_path):
    first=dash.start_dashboard(config(),Settings(),root=tmp_path,submission_id="same-click")
    second=dash.start_dashboard(config(),Settings(),root=tmp_path,submission_id="same-click")
    assert first is second
    first.join(20)
    assert first.cancel() is False and first.snapshot()["status"]=="completed"
    assert len(list((tmp_path/"data/experiment-dashboard").iterdir()))==1


def test_cancellation_preserves_incomplete_experiment(tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();original=dash.run_experiments
    def paused(*a,**kw):
        entered.set();assert release.wait(5);return original(*a,**kw)
    monkeypatch.setattr(dash,"run_experiments",paused)
    job=dash.start_dashboard(config(),Settings(),root=tmp_path,submission_id="cancel")
    assert entered.wait(3);job.cancel();release.set();job.join(10)
    assert job.snapshot()["status"]=="cancelled"
    manifest=json.loads((job.directory/"benchmark/experiment.json").read_text())
    assert manifest["status"]=="cancelled" and (job.directory/"metadata.sqlite").exists()


def test_global_hosted_budget_stops_before_paid_http_attempt(tmp_path,monkeypatch):
    p=Paper.from_openalex({"id":"W1","title":"retrieval"},provenance="SYNTHETIC DASHBOARD TEST ONLY")
    freeze([p],tmp_path/"data/corpus",acquisition=[{"source":"SYNTHETIC"}])
    q=tmp_path/"data/questions.json";q.write_text(json.dumps([dict(id="q",question="retrieval")]))
    monkeypatch.setattr(dash,"readiness",lambda *a,**kw:dict(ready=True,reasons=[]))
    monkeypatch.setattr(dash,"capture_review_snapshot",lambda *a,**kw:None)
    def forbidden(request):pytest.fail("No HTTP beyond the global preflight limit")
    cfg=config(category="Development pilot",planner="hosted",paid_confirmed=True,methods=("llm",),
        corpus="data/corpus",questions="data/questions.json",global_tokens=10)
    settings=Settings(llm_api_key="PRIVATE",llm_model="MOCK",llm_base_url="https://test.example/v1")
    job=dash.start_dashboard(cfg,settings,root=tmp_path,submission_id="global",planner_transport=httpx.MockTransport(forbidden))
    job.join(10)
    assert job.snapshot()["status"]=="global_budget_exhausted"
    assert json.loads((job.directory/"benchmark/experiment.json").read_text())["status"]=="global_budget_exhausted"
    assert "PRIVATE" not in json.dumps(job.snapshot())


@pytest.mark.parametrize("name",["../frozen","/tmp/elsewhere","bad name","a;touch-file",""])
def test_output_paths_cannot_escape_or_execute_commands(tmp_path,name):
    with pytest.raises(ValueError):dash.output_directory(name,tmp_path)


def test_symlink_output_and_artifact_escape_are_rejected(tmp_path):
    outside=tmp_path/"outside";outside.mkdir();(tmp_path/"data").mkdir()
    (tmp_path/"data/experiment-dashboard").symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):dash.output_directory("safe",tmp_path)
    (tmp_path/"data/link").symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):dash.allowed_path("data/link",tmp_path)


def test_backend_spec_equal_for_cli_and_ui():
    args=Namespace(corpus="data/corpus",questions="data/questions.json",output="data/new",suite="controlled-five",
        methods=["llm","llm_graph"],action_budgets=[3],trials=1,trial_seeds=[7],labels=None,factual_questions=None,
        allow_deterministic=True,ablations=False,require_evidence=False,max_total_tokens=70000,max_llm_cost_usd=None,
        active_context_tokens=4096,input_tokens=8192,max_actions=3,max_rounds=3,max_inspected=30,max_api_requests=50,
        max_llm_calls=5,page_size=5,top_k=10,patience=3,context_chars=64000)
    cli=BenchmarkSpec.from_cli(args)
    ui=BenchmarkSpec("data/corpus","data/questions.json","data/new",methods=("llm","llm_graph"),action_budgets=(3,),trials=1,trial_seeds=(7,),
        budget=Budget(max_actions=3,max_rounds=3,max_inspected=30,max_llm_calls=5,page_size=5,output_k=10,max_context_chars=64000),
        inference_limits=InferenceLimits(70000,None,4096,8192),allow_deterministic=True)
    assert cli==ui and cli.backend_kwargs()==ui.backend_kwargs()
    assert dash.DashboardConfig.from_dict(json.loads(json.dumps(asdict(config()))))==config()


def test_invalid_pressure_is_recorded_and_external_memory_availability_is_distinct(tmp_path):
    invalid=run_pressure("high",tmp_path/"invalid",observations=4,return_invalid=True)
    assert not invalid["pressure_valid"] and (tmp_path/"invalid/pressure.json").exists()
    off=run_pressure("low",tmp_path/"off",archive_reads=False)
    assert all(not r["evidence_recoverable_externally"] and r["evidence_exists_in_audit_archive"] for r in off["rows"])
    correction=run_pressure("low",tmp_path/"correction",correction_at=2,return_invalid=True)
    assert correction["configuration"]["correction_at"]==2
    assert all("factual_errors" in r for r in correction["rows"])


def test_unapproved_saved_quality_columns_remain_not_evaluated(tmp_path):
    job=run(tmp_path)
    (job.directory/"dashboard.json").unlink()
    manifest=json.loads((job.directory/"benchmark/experiment.json").read_text())
    manifest["deterministic_smoke"]=False
    (job.directory/"benchmark/experiment.json").write_text(json.dumps(manifest))
    loaded=dash.read_comparison(job.directory/"benchmark",tmp_path)
    assert loaded["category"]=="UNVERIFIED SAVED EXPERIMENT" and not loaded["relevance_evaluated"]
    assert all(r["precision_at_10"] is None for r in loaded["rows"])


def test_refused_overwrite_does_not_relabel_an_existing_manifest(tmp_path):
    from openalex_agent.experiments import run_experiments
    from openalex_agent.storage import Database
    from openalex_agent.llm import DeterministicPlanner
    path=tmp_path/"historical";path.mkdir()
    manifest=path/"experiment.json";manifest.write_text('{"status":"complete","frozen":true}\n')
    before=manifest.read_bytes()
    with Database(":memory:") as db:
        with pytest.raises(ValueError,match="already exists"):
            run_experiments("absent-corpus","absent-questions",db,DeterministicPlanner,output=str(path))
    assert manifest.read_bytes()==before


def test_saved_boolean_ready_without_actual_review_cannot_enable_quality(tmp_path):
    job=run(tmp_path)
    p=job.directory/"dashboard.json";data=json.loads(p.read_text())
    data.update(category="Development pilot",readiness={"ready":True});p.write_text(json.dumps(data))
    loaded=dash.read_comparison(job.directory,tmp_path)
    assert loaded["category"]=="UNVERIFIED SAVED EXPERIMENT"
    assert all(r["precision_at_10"] is None for r in loaded["rows"])


def test_artifact_symlink_cannot_read_unapproved_json(tmp_path):
    private=tmp_path/"private.json";private.write_text('{"secret":"UNAPPROVED"}')
    directory=tmp_path/"data/experiment";directory.mkdir(parents=True)
    (directory/"dashboard.json").symlink_to(private)
    assert "UNAPPROVED" not in json.dumps(dash.read_comparison(directory,tmp_path))


def test_reviewed_approved_mock_run_retains_actual_review_snapshot(tmp_path):
    """Synthetic attestations test the gate; they are not actual human reviews."""
    from openalex_agent.development_review import write_csv
    from openalex_agent.evaluation import import_judgments
    from openalex_agent.corpus import atomic_json
    root=tmp_path;bundle=root/"data/review";bundle.mkdir(parents=True)
    p=Paper.from_openalex({"id":"W1","title":"retrieval"},provenance="SYNTHETIC APPROVAL TEST ONLY")
    p.abstract="The value is 73."
    cm=freeze([p],bundle/"corpus",acquisition=[{"source":"SYNTHETIC APPROVAL TEST ONLY"}])
    q=[dict(id="q",question="retrieval",split="development")];f=[dict(query_id="q",question_id="f",question="What value?")]
    atomic_json(bundle/"questions.json",q);atomic_json(bundle/"factual-questions.json",f)
    atomic_json(bundle/"sources/W1.json",p.to_dict())
    attest=dict(assessor="SYNTHETIC-TEST-NOT-A-HUMAN",judgment_provenance="synthetic gate fixture only",
        independent_human_review="true",uncertainty="false",disagreement="false",adjudication_note="")
    write_csv(bundle/"relevance-worksheet.csv",[dict(query_id="q",question="retrieval",paper_id=p.id,grade="2",
        supporting_passage=p.abstract,explanation="SYNTHETIC FIXTURE",**attest)])
    write_csv(bundle/"evidence-adequacy-worksheet.csv",[dict(query_id="q",question_id="f",factual_question="What value?",
        reviewed_adequacy="sufficient",verified_answer="73",supporting_work_id=p.id,supporting_passage=p.abstract,
        source_type="abstract",evidence_location="sources/W1.json#abstract",human_verification_status="verified",**attest)])
    immutable={str(file.relative_to(bundle)):hashlib.sha256(file.read_bytes()).hexdigest() for file in bundle.rglob('*') if file.is_file() and not file.name.endswith('worksheet.csv')}
    atomic_json(bundle/"review-manifest.json",dict(questions=q,factual_questions=f,corpus_manifest=cm,initial_immutable_sha256=immutable))
    labels=root/"data/labels.json";import_judgments(str(bundle/"relevance-worksheet.csv"),str(labels),scope="exhaustive")
    cfg=config(category="Development pilot",planner="hosted",paid_confirmed=True,methods=("llm",),corpus="data/review/corpus",
        questions="data/review/questions.json",factual_questions="data/review/factual-questions.json",labels="data/labels.json",review_bundle="data/review",approval="data/approval.json")
    settings=Settings(llm_api_key="PRIVATE",llm_model="MOCK",llm_base_url="https://test.example/v1")
    atomic_json(root/"data/approval.json",dict(approved=True,approver="SYNTHETIC COORDINATOR NOT A HUMAN",approved_at="fixture",
        category=cfg.category,questions_sha256=hashlib.sha256((bundle/"questions.json").read_bytes()).hexdigest(),corpus_sha256=cm['sha256'],
        spec_sha256=dash.approval_digest(cfg),source_sha256=dash.source_digest(),provider_profile_sha256=dash.profile_digest(settings)))
    assert dash.readiness(cfg,root,settings)["ready"]
    def handler(request):
        payload=json.loads(json.loads(request.content)["messages"][1]["content"])
        value=dict(question_id="f",answer="73",work_id="W1",evidence=p.abstract,sufficiency="sufficient",abstain=False,abstention_reason="") if 'factual_task' in payload else {'decision':{'action':'STOP_RETRIEVAL','arguments':{}}}
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(value)}}]})
    job=dash.start_dashboard(cfg,settings,root=root,submission_id="synthetic-approved",planner_transport=httpx.MockTransport(handler));job.join(15)
    assert job.snapshot()["status"]=="completed"
    loaded=dash.read_comparison(job.directory,root)
    assert loaded["category"]=="Development pilot" and loaded["relevance_evaluated"]
    assert loaded["rows"][0]["precision_at_10"]==.1
    # Different imported scores cannot borrow approval from a reviewed worksheet.
    manifest_path=job.directory/"benchmark/experiment.json"
    original=manifest_path.read_bytes()
    manifest=json.loads(original);manifest['judgments']['judgments'][0]['grade']=0
    manifest_path.write_text(json.dumps(manifest))
    assert not dash.read_comparison(job.directory,root)["relevance_evaluated"]
    manifest_path.write_bytes(original)
    # A saved 'ready' boolean must not override an actually incomplete snapshot.
    worksheet=job.directory/"review-snapshot/relevance-worksheet.csv"
    frame=pd.read_csv(worksheet,dtype=str).fillna('');frame['grade']='';frame.to_csv(worksheet,index=False)
    assert not dash.read_comparison(job.directory,root)["relevance_evaluated"]


def test_claim_review_attestation_requires_matching_hash_and_named_human_reviewers():
    digest="a"*64
    review=dict(claims_sha256=digest,review_status="verified",independent_human_review=True,
        adjudicated=True,approver="SYNTHETIC TEST COORDINATOR",reviewer_ids=["TEST REVIEWER"])
    assert dash.claim_review_valid(review,digest)
    for field,value in (("claims_sha256","b"*64),("independent_human_review",False),
                        ("adjudicated",False),("reviewer_ids",[""]),("approver"," ")):
        assert not dash.claim_review_valid({**review,field:value},digest)
    assert not dash.claim_review_valid({**review,"claims_sha256":None},None)
