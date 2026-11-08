"""Temporary mocked authorization/charging fixtures; no live inference or human labels."""
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import httpx
import pytest

from openalex_agent.config import Settings
from openalex_agent.controller import Budget
from openalex_agent.context_planner import InferenceLimits
from openalex_agent.corpus import atomic_json,freeze
from openalex_agent.models import Paper
from openalex_agent.development_pilot import (SpendLedger,REQUEST_RESERVE,VERSION,MODEL,
    guarded_factory,verify_authorization,approval_pins,execute,validate_spec)
from openalex_agent.execution_control import ExperimentStopped
from openalex_agent.llm import PlannerError
from openalex_agent.pooled_review import sha


def settings():
    return Settings(llm_base_url='https://api.openai.com/v1/',llm_api_key='SYNTHETIC TEST KEY',llm_model=MODEL,
        llm_timeout=120,llm_max_tokens=4096,llm_temperature=None,llm_reasoning_effort='low',llm_input_price=1,llm_output_price=5)


def test_reservation_before_request_and_reported_usage_settlement(tmp_path):
    ledger=SpendLedger(tmp_path/'ledger.json',16)
    def handler(request):
        state=json.loads(ledger.path.read_text());assert Decimal(state['retained_usd'])==REQUEST_RESERVE
        body=json.loads(request.content)
        assert body['response_format']=={'type':'json_object'} and body['max_completion_tokens']==4096
        assert body['reasoning_effort']=='low' and 'temperature' not in body
        assert request.url.path=='/v1/chat/completions'
        return httpx.Response(200,json=dict(usage=dict(prompt_tokens=1000,completion_tokens=100),choices=[dict(message={'content':'{"decision": {}}'},finish_reason='stop')]))
    planner=guarded_factory(settings(),ledger,transport=httpx.MockTransport(handler))()
    try:planner._call('Return JSON',{'fixture':'request'})
    finally:planner.close()
    assert ledger.charged==Decimal('.0015') and len(ledger.attempts)==1
    assert planner.usage.calls==1 and planner.usage.usage_reported_calls==1
    assert 'SYNTHETIC TEST KEY' not in ledger.path.read_text()


@pytest.mark.parametrize('failure',['timeout','http503','missing_usage'])
def test_unknown_cost_keeps_full_hold_and_never_retries(tmp_path,failure):
    ledger=SpendLedger(tmp_path/'ledger.json',16);calls=[]
    def handler(r):
        calls.append(r)
        if failure=='timeout':raise httpx.ReadTimeout('TEST',request=r)
        if failure=='http503':return httpx.Response(503,json={})
        return httpx.Response(200,json=dict(choices=[dict(message={'content':'{}'},finish_reason='stop')]))
    planner=guarded_factory(settings(),ledger,transport=httpx.MockTransport(handler))()
    try:
        if failure=='missing_usage':planner._call('Return JSON',{})
        else:
            with pytest.raises(PlannerError):planner._call('Return JSON',{})
    finally:planner.close()
    assert len(calls)==1 and ledger.charged==REQUEST_RESERVE and ledger.attempts[0]['usage']=='UNKNOWN'


def test_no_request_can_cross_ceiling(tmp_path):
    ledger=SpendLedger(tmp_path/'ledger.json',16)
    for _ in range(7):ledger.reserve()
    with pytest.raises(ExperimentStopped,match='spending'):ledger.reserve()
    assert len(ledger.attempts)==7 and ledger.charged<16


def test_wall_clock_and_call_caps(tmp_path):
    now=[0.0];ledger=SpendLedger(tmp_path/'clock.json',16,max_seconds=10,clock=lambda:now[0])
    now[0]=11
    with pytest.raises(ExperimentStopped,match='wall_clock'):ledger.reserve()
    assert not ledger.attempts
    other=SpendLedger(tmp_path/'calls.json',16,max_calls=1)
    i=other.reserve();other.settle(i,{'prompt_tokens':0,'completion_tokens':0})
    with pytest.raises(ExperimentStopped,match='call_limit'):other.reserve()


def fixture_spec(tmp_path):
    manifest=freeze([Paper(id='https://openalex.org/W1',title='fixture evidence')],tmp_path/'corpus',acquisition=[{'source':'SYNTHETIC TEST'}])
    questions=[dict(id=f'q{i}',question=f'retrieval {i}',split='development') for i in range(3)]
    atomic_json(tmp_path/'questions.json',questions);atomic_json(tmp_path/'dev.json',questions);atomic_json(tmp_path/'selection.json',{'fixture':True})
    s=settings();spec=dict(protocol=VERSION,purpose='UNSCORED DEVELOPMENT RELIABILITY ONLY',corpus=str(tmp_path/'corpus'),corpus_sha256=manifest['sha256'],
        questions=str(tmp_path/'questions.json'),development_source=str(tmp_path/'dev.json'),selection_sha256=sha(tmp_path/'selection.json'),output=str(tmp_path/'hosted'),
        provider=dict(model=MODEL,base_url=s.llm_base_url,timeout=120,max_output_tokens=4096,temperature=None,reasoning_effort='low',input_usd_per_million=1,output_usd_per_million=5),
        budget=asdict(Budget(max_actions=10,max_rounds=20,max_inspected=200,page_size=10,output_k=20,max_llm_calls=20)),
        limits=asdict(InferenceLimits(max_total_tokens=100000,max_active_tokens=4096,max_input_tokens=8192)),trials=2,seeds=[11,23],action_budgets=[10],spending_ceiling_usd=16,wall_seconds=7200)
    atomic_json(tmp_path/'spec.json',spec);return tmp_path/'spec.json'


def test_false_approval_blocks_before_factory_or_output(tmp_path,monkeypatch):
    path=fixture_spec(tmp_path);atomic_json(tmp_path/'approval.json',dict(approved=False))
    import openalex_agent.development_pilot as module
    monkeypatch.setattr(module,'guarded_factory',lambda *_:pytest.fail('No hosted factory before approval'))
    with pytest.raises(ValueError,match='Explicit approval'):execute(path,tmp_path/'approval.json',settings())
    assert not (tmp_path/'hosted').exists()


def test_wrong_pins_and_heldout_are_rejected(tmp_path):
    path=fixture_spec(tmp_path);pins=approval_pins(path,settings())
    a=dict(protocol=VERSION,approved=True,paid_inference_authorized=True,operational_pilot_only=True,final_evaluation_authorized=False,
        spending_ceiling_usd=16,approver='TEST FIXTURE ONLY',approved_at='TEST FIXTURE ONLY',pins=pins)
    atomic_json(tmp_path/'a.json',a);assert verify_authorization(path,tmp_path/'a.json',settings())
    a['pins']['source_sha256']='WRONG';atomic_json(tmp_path/'a.json',a)
    with pytest.raises(ValueError,match='Explicit approval'):verify_authorization(path,tmp_path/'a.json',settings())
    q=json.loads((tmp_path/'questions.json').read_text());q[0]['split']='heldout';atomic_json(tmp_path/'questions.json',q)
    with pytest.raises(ValueError,match='DEV'):validate_spec(path,settings())


def test_price_override_is_explicit_and_does_not_change_environment(tmp_path):
    from dataclasses import replace
    path=fixture_spec(tmp_path);original=replace(settings(),llm_input_price=0,llm_output_price=0)
    spec,runtime=validate_spec(path,original)
    assert original.llm_input_price==0 and runtime.llm_input_price==1 and runtime.llm_output_price==5


def test_three_dollar_checkpoint_is_actual_usage_not_reservation(tmp_path):
    ledger=SpendLedger(tmp_path/'ledger.json',16,checkpoint=3)
    first=ledger.reserve()
    assert ledger.charged==REQUEST_RESERVE and ledger.reported_spend==0
    ledger.settle(first,dict(prompt_tokens=100000,completion_tokens=0))
    # A reserve is not spent money; prior actual cost is below the checkpoint.
    second=ledger.reserve();ledger.settle(second,dict(prompt_tokens=1050000,completion_tokens=4096))
    assert ledger.reported_spend<3
    third=ledger.reserve();ledger.settle(third,dict(prompt_tokens=500000,completion_tokens=0))
    assert ledger.reported_spend>=3
    with pytest.raises(ExperimentStopped,match='three_dollar'):ledger.reserve()
    assert len(ledger.attempts)==3


def test_checkpoint_stops_on_unknown_cost_without_releasing_reserve(tmp_path):
    ledger=SpendLedger(tmp_path/'ledger.json',16,checkpoint=3)
    i=ledger.reserve();ledger.settle(i,{'http_status':503})
    assert ledger.charged==REQUEST_RESERVE and ledger.reported_spend==0
    with pytest.raises(ExperimentStopped,match='unknown_cost'):ledger.reserve()


def test_canary_passes_before_benchmark_and_is_in_same_ledger(tmp_path,monkeypatch):
    import openalex_agent.development_pilot as module
    path=fixture_spec(tmp_path);spec=json.loads(path.read_text())
    spec.update(protocol=module.CHECKPOINT_VERSION,checkpoint_usd=3,live_canary=True);atomic_json(path,spec)
    pins=approval_pins(path,settings())
    atomic_json(tmp_path/'approval.json',dict(protocol=module.CHECKPOINT_VERSION,approved=True,paid_inference_authorized=True,
        operational_pilot_only=True,final_evaluation_authorized=False,spending_ceiling_usd=16,approver='TEST FIXTURE',approved_at='TEST FIXTURE',
        pins=pins,checkpoint_usd=3,canary_included=True))
    seen=[]
    def handler(r):
        seen.append('canary')
        return httpx.Response(200,json=dict(usage=dict(prompt_tokens=30,completion_tokens=6),choices=[dict(message={'content':'{"status":"ok"}'},finish_reason='stop')]))
    def fake_runs(*args,**kwargs):
        seen.append('benchmark');assert seen==['canary','benchmark']
        kwargs['planner_factory'] if 'planner_factory' in kwargs else None
        ledger=json.loads((tmp_path/'hosted/spending-ledger.json').read_text())
        assert len(ledger['attempts'])==1 and ledger['attempts'][0]['phase']=='canary' and ledger['max_calls']==481
        return []
    monkeypatch.setattr(module,'run_experiments',fake_runs)
    execute(path,tmp_path/'approval.json',settings(),transport=httpx.MockTransport(handler))
    assert seen==['canary','benchmark']


def test_failed_canary_does_not_start_benchmark(tmp_path,monkeypatch):
    import openalex_agent.development_pilot as module
    path=fixture_spec(tmp_path);spec=json.loads(path.read_text())
    spec.update(protocol=module.CHECKPOINT_VERSION,checkpoint_usd=3,live_canary=True);atomic_json(path,spec)
    atomic_json(tmp_path/'approval.json',dict(protocol=module.CHECKPOINT_VERSION,approved=True,paid_inference_authorized=True,
        operational_pilot_only=True,final_evaluation_authorized=False,spending_ceiling_usd=16,approver='TEST FIXTURE',approved_at='TEST FIXTURE',
        pins=approval_pins(path,settings()),checkpoint_usd=3,canary_included=True))
    monkeypatch.setattr(module,'run_experiments',lambda *_a,**_k:pytest.fail('Failed canary must block benchmark'))
    with pytest.raises(PlannerError):execute(path,tmp_path/'approval.json',settings(),transport=httpx.MockTransport(lambda _:httpx.Response(503,json={})))
    assert json.loads((tmp_path/'hosted/canary.json').read_text())['status']=='failed'
    assert len(json.loads((tmp_path/'hosted/spending-ledger.json').read_text())['attempts'])==1
