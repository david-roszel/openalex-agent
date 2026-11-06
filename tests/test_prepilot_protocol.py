"""Synthetic evaluator/protocol checks; no scientific performance claims."""

import csv
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from openalex_agent.config import Settings
from openalex_agent.context_planner import ContextPlanner, EngineeringContextModel, InferenceLimits
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.development_review import prepare, validate_review
from openalex_agent.experiments import run_experiments
from openalex_agent.factual_protocol import INSTRUCTIONS, as_claim, metrics, validate_response
from openalex_agent.llm import HostedJSONPlanner
from openalex_agent.models import Paper
from openalex_agent.research_assessment import build_claim_pool, claim_id, score_research
from openalex_agent.research_metrics import validate_evidence


def response(qid, *, abstain=False, work="W1", statement="The measured value is 73.", observed=True):
    value=dict(question_id=qid,answer=statement,work_id=work,evidence="The measured value is 73.",
        sufficiency="sufficient",abstain=False,abstention_reason="")
    if abstain:value.update(answer="",work_id=None,evidence="",sufficiency="insufficient",abstain=True,abstention_reason="Insufficient evidence")
    paper=Paper.from_openalex({"id":"W1","title":"SYNTHETIC source"},provenance="SYNTHETIC PROTOCOL TEST ONLY")
    paper.abstract="The measured value is 73."
    unrelated=Paper.from_openalex({"id":"W2","title":"SYNTHETIC unrelated source"},provenance="SYNTHETIC PROTOCOL TEST ONLY")
    unrelated.abstract="A different topic with no measurement."
    claim=validate_evidence(as_claim(value),{paper.id:paper,unrelated.id:unrelated} if observed else {})
    claim.update(source_observed_at_claim=observed and not abstain,
        source_accessible_at_claim=observed and not abstain,citation_exists_in_corpus=work in ("W1","W2","W3") and not abstain)
    return dict(question_id=qid,status="valid",response=value,evidence_record=claim)


def diagnostic_cases():
    records=[];grades={}
    definitions=[
        ("q1",{},"supported",True,True,None),
        ("q2",dict(work="W2"),"unsupported",True,False,None),
        ("q3",dict(statement="Incorrect answer"),"unsupported",False,False,None),
        ("q4",dict(statement="Unsupported claim"),"unsupported",False,False,None),
        ("q5",dict(statement="The measured value is not 73."),"contradicted",False,False,None),
        ("q6",dict(abstain=True),"abstain",None,None,True),
        ("q7",dict(abstain=True),"abstain",None,None,False),
        ("q8",dict(work="invalid-source",observed=False),"unsupported",False,False,None),
        ("q9",dict(work="W3",observed=False),"supported",True,True,None)]
    for qid,changes,judgment,truth,support,appropriate in definitions:
        record=response(qid,**changes);records.append(record)
        grade=dict(judgment=judgment)
        if truth is not None:grade["factually_correct"]=truth
        if support is not None:grade["citation_support_correct"]=support
        if appropriate is not None:grade["appropriate"]=appropriate
        grades[claim_id(qid,record["evidence_record"])]=grade
    records.append(dict(question_id="q10",status="valid",response={},evidence_record={}))
    return [f"q{i}" for i in range(1,11)],records,grades


@pytest.mark.parametrize("architecture",["llm","llm_graph","clm","clm_graph"])
def test_ten_known_outcomes_identical_for_every_architecture(architecture):
    expected,records,grades=diagnostic_cases()
    values=metrics(expected,records,grades)
    assert values["expected_questions"]==10 and values["attempted_answers"]==7
    assert values["correct_answers"]==3 and values["incorrect_answers"]==4
    assert values["unsupported_answers"]==4 and values["contradicted_answers"]==1
    assert values["incorrect_source_attribution"]==5 and values["invalid_source_identifiers"]==1
    assert values["unobserved_sources"]==2
    assert values["appropriate_abstentions"]==values["inappropriate_abstentions"]==1
    assert values["malformed_responses"]==1 and values["explicit_abstentions"]==2
    assert values["answer_coverage"]==.7 and values["response_coverage"]==.9
    assert values["overall_answer_accuracy"]==.3 and values["accuracy_on_attempted_questions"]==3/7
    assert values["overall_task_completion"]==.1


def test_abstention_missing_and_empty_outputs_cannot_inflate_overall_accuracy():
    record=response("q1")
    grade={claim_id("q1",record["evidence_record"]):dict(judgment="supported",factually_correct=True,citation_support_correct=True)}
    values=metrics([f"q{i}" for i in range(1,11)],[record],grade)
    assert values["accuracy_on_attempted_questions"]==1 and values["overall_answer_accuracy"]==.1
    assert values["answer_coverage"]==.1 and values["missing_responses"]==9
    empty=metrics(["q1"],[],{})
    assert empty["overall_answer_accuracy"]==0 and empty["accuracy_on_attempted_questions"] is None
    abstain=response("q1",abstain=True)
    values=metrics(["q1"],[abstain],{claim_id("q1",abstain["evidence_record"]):dict(judgment="abstain",appropriate=True)})
    assert values["overall_answer_accuracy"]==0 and values["answer_coverage"]==0


def test_unknown_truth_remains_unknown_and_length_has_no_scoring_bonus():
    expected,records,grades=diagnostic_cases()
    key=claim_id("q4",records[3]["evidence_record"]);grades[key].pop("factually_correct")
    values=metrics(expected,records,grades)
    assert values["overall_answer_accuracy"] is None and values["accuracy_on_attempted_questions"] is None
    assert values["truth_unknown"]==1 and values["overall_accuracy_lower_bound"]==.3
    assert values["overall_accuracy_upper_bound"]==.4
    short=response("q1");long=response("q1",statement="Supported answer. "*50)
    for r in (short,long):
        grade={claim_id("q1",r["evidence_record"]):dict(judgment="supported",factually_correct=True,citation_support_correct=True)}
        assert metrics(["q1"],[r],grade)["overall_task_completion"]==1


@pytest.mark.parametrize("value",[None,"",{}, {"question_id":"q"}])
def test_missing_or_malformed_protocol_is_rejected(value):
    with pytest.raises(ValueError):validate_response(value,"q")


def test_inconsistent_abstention_extra_claim_fields_and_tampered_records_reject():
    valid=response("q1")["response"]
    with pytest.raises(ValueError):validate_response({**valid,"extra_claims":["bonus"]},"q1")
    with pytest.raises(ValueError):validate_response({**valid,"abstain":True},"q1")
    with pytest.raises(ValueError):validate_response(valid,"wrong-question")
    record=response("q1");record["evidence_record"]["statement"]="Different claim"
    assert metrics(["q1"],[record],{})["malformed_responses"]==1


def synthetic_setup(tmp_path):
    p=Paper.from_openalex({"id":"W1","title":"retrieval experiment"},provenance="SYNTHETIC FIXED OPPORTUNITY TEST ONLY")
    p.abstract="The measured value is 73."
    corpus=tmp_path/"corpus";freeze([p],corpus,acquisition=[{"source":"SYNTHETIC TEST ONLY"}])
    questions=tmp_path/"questions.json";questions.write_text(json.dumps([dict(id="q1",question="retrieval experiment")]))
    facts=tmp_path/"facts.json";facts.write_text(json.dumps([dict(query_id="q1",question_id="fact1",question="What is the measured value?")]))
    return corpus,questions,facts


def test_common_actual_wire_fixed_opportunity_pool_and_existing_scorer(tmp_path,db):
    corpus,questions,facts=synthetic_setup(tmp_path)
    finals=[];planning=[]
    def handler(request):
        wire=json.loads(request.content);payload=json.loads(wire["messages"][1]["content"])
        if "factual_task" in payload:
            finals.append(wire);value=response("fact1")["response"]
            assert "factually_correct" not in json.dumps(payload) and "verified_answer" not in json.dumps(payload)
        else:
            planning.append(wire);value={"decision":{"action":"STOP_RETRIEVAL","arguments":{}}}
        return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(value)}}],"usage":{"prompt_tokens":40,"completion_tokens":10}})
    def factory():return HostedJSONPlanner(Settings(llm_base_url="https://test.example/v1",llm_api_key="TEST",llm_model="IDENTICAL-MOCK",llm_max_tokens=128),transport=httpx.MockTransport(handler))
    root=tmp_path/"experiment"
    rows=run_experiments(str(corpus),str(questions),db,factory,output=str(root),suite="controlled-five",
        trials=1,methods=("llm","llm_graph","clm","clm_graph"),action_budgets=(2,),budget=Budget(max_llm_calls=4,page_size=1),
        inference_limits=InferenceLimits(max_active_tokens=4096,max_input_tokens=8192),factual_questions_path=str(facts))
    assert len(finals)==4 and all(w["messages"][0]["content"]==INSTRUCTIONS for w in finals)
    assert all({k:v for k,v in w.items() if k!="messages"}=={k:v for k,v in finals[0].items() if k!="messages"} for w in finals)
    assert all(json.loads(w["messages"][1]["content"])["factual_task"]["question_id"]=="fact1" for w in finals)
    for row in rows:
        result=json.loads((Path(row["directory"])/"result.json").read_text())
        assert result["budget"]["max_llm_calls"]==3 and result["total_model_call_ceiling"]==4
        assert row["factual_measurement_calls"]==1 and row["llm_calls"]==2
        assert result["factual_responses"][0]["status"]=="valid"
    pool=tmp_path/"pool.csv";assert build_claim_pool(root,pool)==1
    frame=pd.read_csv(pool,dtype=str).fillna("")
    assert frame.question_id.iloc[0]=="fact1" and "method" not in frame
    frame.loc[0,"judgment"]="supported";frame.loc[0,"factually_correct"]="true"
    frame.loc[0,"citation_support_correct"]="true";frame.loc[0,"assessor"]="SYNTHETIC-NOT-HUMAN-REVIEW"
    frame.to_csv(pool,index=False)
    assert score_research(root,claim_judgments_path=pool)==4
    scored=pd.read_csv(root/"research-metrics.csv")
    assert (scored.task_expected_questions==1).all() and (scored.task_overall_answer_accuracy==1).all()
    assert (scored.task_overall_task_completion==1).all()
    summary=json.loads((root/"factual-opportunity-summary.json").read_text())
    assert len(summary)==4 and all(s["expected_questions"]==1 for s in summary)


def test_ground_truth_fields_cannot_enter_public_probe_manifest(tmp_path,db):
    corpus,questions,facts=synthetic_setup(tmp_path)
    data=json.loads(facts.read_text());data[0]["verified_answer"]="DO NOT SEND";facts.write_text(json.dumps(data))
    def forbidden():pytest.fail("No planner should be instantiated for a leaking public manifest")
    with pytest.raises(ValueError,match="hidden answer"):
        run_experiments(str(corpus),str(questions),db,forbidden,output=str(tmp_path/"bad"),suite="controlled-five",factual_questions_path=str(facts))


def test_malformed_final_response_never_edits_context_or_retries(tmp_path):
    class Script(EngineeringContextModel):
        def _call(self,system,payload):return self.respond(system,payload,{"context_edit":{"path":"active_context.txt","op":"replace","text":"UNCOMMITTED"}})
    planner=ContextPlanner(Script(),mode="clm")
    planner.bind(tmp_path/"run","retrieval",Budget(),lambda *a:None,lambda:{})
    planner.store.transition("ORIGINAL",source="environment",operation={})
    record=planner.factual_response("fact1","Question?")
    assert record["status"]=="malformed" and planner.store.read()=="ORIGINAL" and planner.usage.calls==1
    with pytest.raises(ValueError,match="already"):planner.factual_response("fact1","Question?")
    assert planner.usage.calls==1


def test_review_bundle_has_no_fabricated_labels_and_blocks_until_human_review(tmp_path):
    corpus=Path(__file__).parent / "fixtures" / "synthetic-corpus"
    manifest=prepare(corpus,tmp_path/"review")
    assert manifest["relevance_rows"]==117
    result=validate_review(tmp_path/"review")
    assert result["independent_judgments"]==result["evidence_adequacy"]=="UNKNOWN"
    assert result["pending_count"]==120 and not result["errors"]
    with (tmp_path/"review/relevance-worksheet.csv").open(newline="") as stream:rows=list(csv.DictReader(stream))
    assert all(not r["grade"] and not r["assessor"] for r in rows)
    adequacy=json.loads((tmp_path/"review/evidence-adequacy.json").read_text())
    assert [q["preliminary_adequacy"] for q in adequacy["questions"]]==["sufficient","sufficient","insufficient"]
    assert all(not q["usable_for_factual_evaluation"] and not q["verified_ground_truth_facts"] for q in adequacy["questions"])


def test_exhausted_measurement_budget_records_missing_without_retry(tmp_path):
    base=EngineeringContextModel()
    p=ContextPlanner(base,mode="conventional")
    p.bind(tmp_path/"run","retrieval",Budget(max_llm_calls=1),lambda *a:None,lambda:{})
    base.usage.calls=1
    record=p.factual_response("fact1","Question?")
    assert record["status"]=="missing" and record["error"]=="measurement_budget_exhausted"
    assert base.usage.calls==1
    assert metrics(["fact1"],[record],{})["overall_answer_accuracy"]==0


def test_review_question_and_sources_cannot_be_silently_replaced(tmp_path):
    corpus=Path(__file__).parent / "fixtures" / "synthetic-corpus"
    root=tmp_path/"review";prepare(corpus,root)
    path=root/"relevance-worksheet.csv"
    with path.open(newline="") as stream:rows=list(csv.DictReader(stream))
    rows[0]["question"]="Different annotation task"
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match="question differs"):validate_review(root)


def test_identical_abstentions_in_different_evidence_contexts_are_judged_separately():
    a=response("same-fact",abstain=True);b=response("same-fact",abstain=True)
    a["evidence_record"]["provenance"]={"assessment_context_sha256":"sufficient-context"}
    b["evidence_record"]["provenance"]={"assessment_context_sha256":"insufficient-context"}
    aid=claim_id("same-fact",a["evidence_record"]);bid=claim_id("same-fact",b["evidence_record"])
    assert aid!=bid
    grades={aid:dict(judgment="abstain",appropriate=False),bid:dict(judgment="abstain",appropriate=True)}
    assert metrics(["same-fact"],[a],grades)["inappropriate_abstentions"]==1
    assert metrics(["same-fact"],[b],grades)["appropriate_abstentions"]==1


def test_malformed_terminal_response_is_invalid_and_scores_zero_in_existing_pipeline(tmp_path,db):
    corpus,questions,facts=synthetic_setup(tmp_path)
    def handler(request):
        payload=json.loads(json.loads(request.content)["messages"][1]["content"])
        value={"not_a_response":True} if "factual_task" in payload else {"decision":{"action":"STOP_RETRIEVAL","arguments":{}}}
        return httpx.Response(200,json={"choices":[{"message":{"content":json.dumps(value)}}]})
    def factory():return HostedJSONPlanner(Settings(llm_base_url="https://test.example/v1",llm_api_key="TEST",llm_model="MOCK"),transport=httpx.MockTransport(handler))
    root=tmp_path/"experiment"
    row=run_experiments(str(corpus),str(questions),db,factory,output=str(root),suite="controlled-five",
        methods=("llm",),trials=1,action_budgets=(2,),budget=Budget(max_llm_calls=3,page_size=1),factual_questions_path=str(facts))[0]
    assert row["factual_response_status"]=="malformed" and not row["valid_llm_run"]
    assert row["factual_measurement_calls"]==1 and row["llm_calls"]==2
    pool=tmp_path/"empty-pool.csv";assert build_claim_pool(root,pool)==0
    assert score_research(root,claim_judgments_path=pool)==1
    scored=pd.read_csv(root/"research-metrics.csv").iloc[0]
    assert scored.task_expected_questions==1 and scored.task_malformed_responses==1
    assert scored.task_answer_coverage==0 and scored.task_overall_answer_accuracy==0
