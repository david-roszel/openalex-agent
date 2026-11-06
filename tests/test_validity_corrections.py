"""Known outcomes and failure injection, not research-performance experiments."""

import json
from pathlib import Path

import pandas as pd
import pytest
import httpx

from openalex_agent.context_memory import ContextStore
from openalex_agent.context_planner import ContextPlanner, EngineeringContextModel, InferenceLimits
from openalex_agent.context_pressure import run_pressure
from openalex_agent.controller import Budget
from openalex_agent.evidence_provenance import claim_access
from openalex_agent.experiment_integrity import validate_schedule
from openalex_agent.models import Paper
from openalex_agent.research_assessment import claim_id, score_research
from openalex_agent.research_metrics import paired_comparisons, validate_evidence
from openalex_agent.token_budget import count, metadata
from openalex_agent.config import Settings
from openalex_agent.controlled import ControlledController, GRAPH_FIELDS
from openalex_agent.controlled_diagnostics import fixture, SeededCorpus
from openalex_agent.corpus import freeze
from openalex_agent.llm import HostedJSONPlanner
from openalex_agent.storage import Database


def rows(values, *, question="q1", trials=3, variant="full"):
    return [dict(query_id=question, method=method, variant=variant, trial=trial,
                 action_budget=5, score=value) for method, value in values.items() for trial in range(trials)]


@pytest.mark.parametrize("values,graph,clm,interaction", [
    ({"llm":1,"llm_graph":3,"clm":5,"clm_graph":9},3,5,2),
    ({"llm":1,"llm_graph":0,"clm":2,"clm_graph":1},-1,1,0),
    ({"llm":1,"llm_graph":1,"clm":1,"clm_graph":1},0,0,0)])
def test_factorial_known_positive_negative_zero_and_interaction(values, graph, clm, interaction):
    data = rows(values)+rows(values,question="q2")
    output = paired_comparisons(data,"score")
    expected = {"graph_main_effect":graph,"clm_main_effect":clm,"interaction":interaction}
    for name, value in expected.items():
        record = next(r for r in output if r["left"]==name)
        assert record["mean_difference"]==value and record["questions"]==2
        assert record["bootstrap_95_ci"]==[value,value]
        assert record["trial_counts"]=={"q1":3,"q2":3}
    assert output==paired_comparisons(data,"score")


def test_ablation_variants_and_standard_baseline_never_enter_factorial_groups():
    data = rows({"llm":1,"llm_graph":1},trials=1)
    for variant in ("no_novelty","no_importance","no_coverage"):
        data += rows({"llm_graph":0},variant=variant,trials=1)
    data += rows({"standard":999,"graph":999},trials=1)
    output = paired_comparisons(data,"score")
    assert len(output)==1 and output[0]["mean_difference"]==0 and output[0]["variant"]=="full"


def test_duplicate_or_unbalanced_trials_reject_and_missing_metric_excludes_whole_question():
    data = rows({"llm":1,"llm_graph":2},trials=2)
    with pytest.raises(ValueError,match="Duplicate"):
        paired_comparisons(data+[data[0]],"score")
    with pytest.raises(ValueError,match="Unbalanced"):
        paired_comparisons(data[:-1],"score")
    data[-1]["score"]=None
    output=paired_comparisons(data+rows({"llm":1,"llm_graph":3},question="q2",trials=2),"score")
    assert output[0]["questions"]==1 and output[0]["mean_difference"]==2
    assert output[0]["excluded_questions"]["q1"]=="missing_or_nonfinite_metric_in_trial"
    output=paired_comparisons([dict(r,score=None) for r in data],"score")
    assert output[0]["questions"]==0 and output[0]["mean_difference"] is None
    assert output[0]["excluded_questions"]


def test_schedule_gate_rejects_incomplete_duplicate_and_wrong_identity():
    data=rows({"llm":1,"clm":2},trials=1)
    manifest=dict(suite="controlled-five",status="complete",planned_runs=2,run_order=data)
    validate_schedule(manifest,data)
    with pytest.raises(ValueError,match="incomplete"):
        validate_schedule({**manifest,"status":"running"},data)
    with pytest.raises(ValueError,match="Duplicate"):
        validate_schedule(manifest,[data[0],data[0]])
    with pytest.raises(ValueError,match="schedule"):
        validate_schedule(manifest,[data[0],{**data[1],"variant":"no_novelty"}])


def test_failed_write_is_not_an_accepted_edit_and_keeps_committed_context(tmp_path,monkeypatch):
    store=ContextStore(tmp_path/"context",4000)
    store.transition("original",source="environment",operation={})
    monkeypatch.setattr(store,"_write",lambda text: (_ for _ in ()).throw(OSError("DISK FAILURE")))
    record=store.edit(dict(path="active_context.txt",op="replace",text="unwritten"))
    assert record["requested"] and record["validated"] and not record["committed"] and not record["accepted"]
    assert record["status"]=="failed" and record["requested_after"]=="unwritten"
    assert record["after"]==store.read()=="original"
    store.verify_committed()
    assert json.loads((store.directory/"transition-00001.json").read_text())==record
    assert (store.directory/"operation-00001-requested.json").exists()
    assert (store.directory/"operation-00001-validated.json").exists()


@pytest.mark.parametrize("after_write",[False,True])
def test_interrupt_before_or_after_replace_never_claims_success(tmp_path,monkeypatch,after_write):
    store=ContextStore(tmp_path/"context",4000)
    store.transition("old",source="environment",operation={})
    write=store._write
    def interrupted(text):
        if after_write: write(text)
        raise KeyboardInterrupt()
    monkeypatch.setattr(store,"_write",interrupted)
    with pytest.raises(KeyboardInterrupt):
        store.edit(dict(path="active_context.txt",op="replace",text="new"))
    record=json.loads((store.directory/"transition-00001.json").read_text())
    assert record["status"]=="interrupted" and not record["accepted"] and not record["committed"]
    assert record["after"]==store.read()==("new" if after_write else "old")
    if after_write:
        with pytest.raises(RuntimeError,match="uncertain"):store.verify_committed()
    else:store.verify_committed()


def test_missing_commit_receipt_blocks_subsequent_inference(tmp_path,monkeypatch):
    import openalex_agent.context_memory as memory
    p=ContextPlanner(EngineeringContextModel(),mode="clm")
    p.bind(tmp_path/"run","retrieval",Budget(),lambda *a:None,lambda:{})
    atomic=memory.atomic_json
    def fail_receipt(path,value):
        if path.name.startswith("transition-"):raise OSError("receipt failure")
        atomic(path,value)
    monkeypatch.setattr(memory,"atomic_json",fail_receipt)
    with pytest.raises(OSError):p.store.transition("new",source="model",operation={})
    assert not list(p.store.directory.glob("transition-*.json"))
    assert json.loads(next(p.store.directory.glob("*validated.json")).read_text())["accepted"] is False
    with pytest.raises(RuntimeError,match="uncertain"):p._call("system",{})
    assert p.usage.calls==0


def test_invalid_and_capacity_rejected_edits_have_complete_status(tmp_path):
    store=ContextStore(tmp_path/"context",30)
    invalid=store.edit(dict(path="../outside",op="replace",text="bad"))
    rejected=store.edit(dict(path="active_context.txt",op="replace",text="x"*31))
    for record in (invalid,rejected):
        assert record["requested"] and not record["validated"] and not record["committed"]
        assert record["status"]=="rejected" and not record["accepted"] and record["after"]==""


def test_pinned_offline_tokenizer_and_unicode_accounting():
    assert count("hello world")==2
    assert count("q "*1000)>count("a"*2000)  # character counts are not token budgets
    assert count("研究证据")>0
    assert metadata()["provider_native_or_billing_equivalence"] is False


def test_four_cell_token_limited_wire_isolation_and_committed_edit(tmp_path):
    question,papers,seeds,_,_=fixture("G1")
    freeze(papers,tmp_path/"corpus",acquisition=[{"source":"SYNTHETIC ISOLATION ONLY"}])
    captured={}
    limits=InferenceLimits(max_active_tokens=4096,max_input_tokens=8192)
    for architecture in ("llm","llm_graph","clm","clm_graph"):
        wire=[]
        def handler(request):
            wire.append(json.loads(request.content))
            value={"context_edit":{"path":"active_context.txt","op":"replace","text":"COMMITTED NOTE"}} if architecture.startswith("clm") and len(wire)==1 else {"decision":{"action":"STOP_RETRIEVAL","arguments":{}}}
            return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(value)}}]})
        base=HostedJSONPlanner(Settings(llm_base_url="https://test.example/v1",llm_api_key="TEST-ONLY",llm_model="SAME-MOCK",llm_max_tokens=512),transport=httpx.MockTransport(handler))
        planner=ContextPlanner(base,mode="clm" if architecture.startswith("clm") else "conventional",limits=limits)
        with Database(":memory:") as db:
            controller=ControlledController(question,SeededCorpus(tmp_path/"corpus",seeds),db,architecture=architecture,planner=planner,budget=Budget(max_actions=2,page_size=1),output=str(tmp_path/"runs"))
            result=controller.run()
        captured[architecture]=dict(wire=wire,result=result,planner=planner)
        assert result["max_active_context_tokens"]<=4096 and result["max_full_input_tokens"]<=8192
        if architecture.startswith("clm"):
            assert json.loads(wire[1]["messages"][1]["content"])["active_context"]=="COMMITTED NOTE"
            edit=next(r for r in planner.store.edits if r["source"]=="model")
            assert edit["status"]=="committed" and edit["committed"] and edit["accepted"]
        base.close()
    for a,b in (("llm","llm_graph"),("clm","clm_graph")):
        first=[captured[m]["wire"][0] for m in (a,b)]
        assert first[0]["messages"][0]==first[1]["messages"][0]
        assert {k:v for k,v in first[0].items() if k!="messages"}=={k:v for k,v in first[1].items() if k!="messages"}
        payload=[json.loads(w["messages"][1]["content"]) for w in first]
        assert payload[0]["protected"]==payload[1]["protected"]
        bodies=[json.loads(p["active_context"])["changes"] if a.startswith("clm") else p["observation"] for p in payload]
        assert bodies[0]=={k:v for k,v in bodies[1].items() if k not in GRAPH_FIELDS}
        assert not GRAPH_FIELDS.intersection(bodies[0])
        rankings=[[(r["paper"]["id"],r["algorithmic"]["relevance"]) for r in captured[m]["result"]["ranked"]] for m in (a,b)]
        assert rankings[0]==rankings[1]


def test_claim_time_provenance_is_authoritative_and_versioned(tmp_path):
    paper,_=synthetic_sources()
    claim=dict(statement="test",work_id=paper.id,evidence_quote=paper.title,insufficient=False)
    provenance=dict(schema_version=1,model_call=2,source_observed_at_claim=True,
                    source_accessible_at_claim=True,source_record=paper.to_dict())
    enriched={**claim,"provenance":provenance}
    access=claim_access(tmp_path,[enriched])[0]
    assert access["source_observed_at_claim"] and access["source_accessible_at_claim"]
    updated={**enriched,"provenance":{**provenance,"source_record":{**paper.to_dict(),"abstract":"Changed source version"}}}
    assert claim_id("q",enriched)!=claim_id("q",updated)


@pytest.mark.parametrize("condition",["low","medium","high"])
def test_pressure_is_measured_and_shared_not_inferred_from_paper_count(tmp_path,condition):
    result=run_pressure(condition,tmp_path/condition,seed=1)
    assert result["pressure_valid"] and result["shared_observation_archives"]
    assert (result["pressure_ratio"]<1) if condition=="low" else (result["pressure_ratio"]>1)
    if condition=="high":assert result["pressure_ratio"]>8
    if condition=="medium":assert result["unique_source_pressure_ratio"]>1
    if condition=="high":assert result["unique_source_pressure_ratio"]>8
    for row in result["rows"]:
        assert row["max_active_context_tokens"]<=4096 and row["max_full_input_tokens"]<=8192
        assert row["all_stream_sources_delivered"] and row["evidence_recoverable_externally"]
        assert row["model_calls"]>=row["management_calls"]
        assert row["actual_provider_tokens"] is None and row["actual_provider_cost_usd"] is None
        assert row["final_supported_synthetic_answer"]
    if condition!="low":
        conventional=result["rows"][0]
        assert conventional["evidence_successfully_recovered"]
        # First FACT starts at offset 504 and crosses a 512-character boundary.
        assert conventional["recovery_operations"][1]["request"]["start"]==256
    # No assertion about which architecture wins; both retain recovery access.


def synthetic_sources():
    path=Path(__file__).parent / "fixtures" / "synthetic-corpus" / "papers.jsonl"
    return [Paper.from_dict(json.loads(line)) for line in path.read_text().splitlines()[:2]]


def test_synthetic_corpus_sources_access_support_wrong_attribution_contradiction_and_unknown(tmp_path):
    retrieved,unretrieved=synthetic_sources()
    base=dict(statement="This study compares six scholarly network types.",work_id=retrieved.id,
              evidence_quote=(retrieved.abstract or "").split(". ")[0]+".",insufficient=False)
    observed={retrieved.id:retrieved}
    records=[validate_evidence(base,observed),
        validate_evidence({**base,"work_id":unretrieved.id},observed),
        validate_evidence({**base,"work_id":"not-an-openalex-id"},observed),
        validate_evidence({**base,"statement":"This study only examines one scholarly network type."},observed)]
    # Controlled known-outcome labels about synthetic source text, not a
    # claim of independent research judgments for the eventual benchmark.
    judgments=[("supported","true"),("unsupported","false"),("unsupported","false"),("contradicted","false")]
    run=tmp_path/"run";run.mkdir()
    events=[dict(kind="start",event_seq=1),dict(kind="observation",event_seq=2,papers=[retrieved.to_dict()])]
    for i,record in enumerate(records):events.append(dict(kind="evidence_record",event_seq=i+3,**record))
    (run/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    (run/"result.json").write_text(json.dumps(dict(evidence_records=records)))
    cell=dict(query_id="diagnostic",method="llm",variant="full",trial=0,action_budget=5,run_id="run",directory=str(run))
    manifest=dict(suite="controlled-five",status="complete",planned_runs=1,run_order=[cell],corpus_manifest=dict(ids=[retrieved.id,unretrieved.id]))
    (tmp_path/"experiment.json").write_text(json.dumps(manifest));pd.DataFrame([cell]).to_csv(tmp_path/"metrics.csv",index=False)
    grades=tmp_path/"claims.csv"
    pd.DataFrame([dict(record_id=claim_id("diagnostic",r),judgment=j,citation_support_correct=s,assessor="CONTROLLED-TEST-ONLY")
                  for r,(j,s) in zip(records,judgments)]).to_csv(grades,index=False)
    score_research(tmp_path,claim_judgments_path=grades)
    scored=json.loads((tmp_path/"research-claim-access.json").read_text())
    assert [r["citation_exists_in_corpus"] for r in scored]==[True,True,False,True]
    assert [r["source_observed_at_claim"] for r in scored]==[True,False,False,True]
    assert scored[1]["identifier_valid"] is False and scored[1]["citation_support_correct"] is False
    assert scored[3]["judgment"]=="contradicted" and scored[0]["judgment"]=="supported"
    assert all(r["source_accessible_at_claim"] is None for r in (scored[0],scored[1],scored[3]))
    empty=tmp_path/"missing-logs";empty.mkdir()
    unknown=claim_access(empty,[records[0]])[0]
    assert unknown["source_accessible_at_claim"] is None and unknown["source_observed_at_claim"] is None
    metrics=pd.read_csv(tmp_path/"research-metrics.csv")
    assert "variant" in metrics and metrics.iloc[0].contradicted_claim_rate==.25


def test_future_observation_does_not_retroactively_validate_a_claim(tmp_path):
    p,_=synthetic_sources()
    record=dict(statement="early",work_id=p.id,evidence_quote="",insufficient=False)
    events=[dict(kind="start",event_seq=1),dict(kind="evidence_record",event_seq=2,**record),
            dict(kind="observation",event_seq=3,papers=[p.to_dict()])]
    (tmp_path/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    access=claim_access(tmp_path,[record])[0]
    assert access["source_observed_at_claim"] is False and access["source_record"] is None
