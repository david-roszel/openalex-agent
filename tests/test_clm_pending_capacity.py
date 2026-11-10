"""Offline capacity/termination regressions, not hosted-model quality results."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from openalex_agent.context_memory import serialized
from openalex_agent.context_planner import ContextPlanner, InferenceLimits, CONTEXT_POLICY_VERSION
from openalex_agent.controller import Budget
from openalex_agent.controlled import ControlledController
from openalex_agent.controlled_diagnostics import SeededCorpus
from openalex_agent.llm import PlannerError
from openalex_agent.models import Paper
from openalex_agent.openalex import BudgetExceeded
from test_clm_context import planner, context, STOP
from test_gemini_adc import model, response
from test_planner_tool_contract import corpus

CASES=json.loads((Path(__file__).parent/'fixtures/clm-pending-overflows.dev.json').read_text())['cases']


@pytest.mark.parametrize('case',CASES,ids=lambda c:f"{c['architecture']}-{c['query_id']}-trial{c['trial']}-call{c['source_call']}")
def test_all_eight_historical_candidates_keep_their_limits_and_do_not_fall_back(tmp_path,case):
    original=deepcopy(case['original_output'])
    stopping=original['decision']['action']=='STOP_RETRIEVAL'
    outputs=[original] if stopping else [original,{'context_edit':dict(path='active_context.txt',op='replace',text='RETAINED DEVELOPMENT SNAPSHOT')},STOP]
    base,wire=model(lambda body,n:response(text=json.dumps(outputs[n-1]),n=n))
    p=ContextPlanner(base,mode='clm',limits=InferenceLimits(**case['limits']))
    events=[];observed={d['id']:Paper.from_dict(d) for d in case['observed_papers']}
    p.bind(tmp_path/'run',case['question'],Budget(**case['budget']),lambda k,v:events.append(dict(kind=k,**v)),lambda:observed)
    try:
        p.store.transition(case['active'],source='environment',operation={'SYNTHETIC_OFFLINE_RESTORE':True})
        for archive in case['archives_before']:p.store.archive(archive)
        p._seen=deepcopy(case['seen_before']);p._pending=case['pending_before'];p._pending_indices=case['pending_indices_before'][:]
        before={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.store.directory.glob('observation-*.json')}
        action=p.decide(deepcopy(case['current_context']))
        first=json.loads((p.store.directory/'request-00001.json').read_text())
        assert first['input']==case['original_input']
        for path in p.store.directory.glob('request-*.json'):
            saved=json.loads(path.read_text())
            assert not p.capacity_measurements(saved['input'],system=saved['system'])['violated_limits']
        assert action['action']=='STOP_RETRIEVAL' and len(wire)==(1 if stopping else 3)
        blocked=[e for e in events if e['kind']=='context_admission' and e['status']=='pending']
        assert blocked and blocked[0]['capacity']['violated_limits']==['serialized_user_payload_chars']
        cap=blocked[0]['capacity']
        assert cap['active_tokens']<=4096 and cap['full_input_tokens']<=8192
        assert cap['serialized_user_payload_chars']>16000 and cap['selected_working_context_chars']<16000
        if stopping:
            assert p.store.read()==case['active'] and len(p._pending)==case['original_input']['pending_observation_chars']
            assert not [e for e in events if e['kind']=='context_management_required']
        else:
            assert not p._pending and any(e['kind']=='context_management_required' for e in events)
            assert any(e['kind']=='context_edit' and e['committed'] for e in events)
            assert any(e['kind']=='context_admission' and e['status']=='admitted' for e in events)
        assert all(hashlib.sha256((p.store.directory/name).read_bytes()).hexdigest()==digest for name,digest in before.items())
        assert p.store.recover(len(case['archives_before']))=={k:v for k,v in case['current_context'].items() if k not in ('question','round','remaining','permitted_actions','retrieval','proposals')}
        assert p.budget.__dict__==case['budget'] and asdict(p.limits)==case['limits']
        assert p.configuration['context_policy']==CONTEXT_POLICY_VERSION
    finally:p.close()


def pressure(p,c,reserve=100):
    protected,body=p._parts(c);protected['archive_count']=len(p.store.observations)+1
    overhead=len(serialized(p._payload(protected,body,active='')))
    text='A'*(p.budget.max_context_chars-overhead-reserve)
    p.store.transition(text,source='environment',operation={'SYNTHETIC_PRESSURE':True})
    return text


@pytest.mark.parametrize('mode',['conventional','clm'])
def test_explicit_serialized_character_boundary_including_escaping(tmp_path,mode):
    p,_,_=planner(tmp_path,mode,[STOP],max_chars=4000)
    try:
        p._system=''
        payload={'protected':{},'active_context':''} if mode=='clm' else {'protected':{},'observation':''}
        field='active_context' if mode=='clm' else 'observation'
        overhead=len(serialized(payload));payload[field]='A'*(4000-overhead)
        assert p.payload_fits(payload)
        payload[field]+='A';assert not p.payload_fits(payload)
        payload[field]='"\\\n'*1000
        assert len(payload[field])<4000 and not p.payload_fits(payload)
        assert p.capacity_measurements(payload)['violated_limits']==['serialized_user_payload_chars']
    finally:p.close()


@pytest.mark.parametrize('limits,dimension',[
    (InferenceLimits(max_active_tokens=5,max_input_tokens=8192),'active_and_recovery_tokens'),
    (InferenceLimits(max_active_tokens=4096,max_input_tokens=4100),'full_input_tokens'),
])
def test_capacity_diagnostics_identify_independent_token_guards(tmp_path,limits,dimension):
    p,_,_=planner(tmp_path,'clm',[STOP],max_chars=16000,limits=limits)
    try:
        payload={'protected':{},'active_context':'distinct terms '*100}
        system='system instruction '*5000 if dimension=='full_input_tokens' else ''
        assert dimension in p.capacity_measurements(payload,system=system)['violated_limits']
        assert not p.payload_fits(payload,system=system)
    finally:p.close()


@pytest.mark.parametrize('mode',['conventional','clm'])
def test_fitting_pending_or_no_pending_stop_is_one_call(tmp_path,mode):
    p,inputs,events=planner(tmp_path,mode,[STOP],max_chars=4000)
    try:
        assert p.decide(context())['action']=='STOP_RETRIEVAL'
        assert len(inputs)==1 and not p._pending
        assert events[-1]['kind']=='context_termination' and events[-1]['reason']=='model_stop'
        assert p.store.recover(0)['top_papers'][0]['abstract_excerpt']=='FACT[early-evidence-73]'
    finally:p.close()


def test_nonfitting_pending_stop_preserves_unseen_evidence_and_never_compacts(tmp_path):
    p,inputs,events=planner(tmp_path,'clm',[STOP],max_chars=4000)
    try:
        before=pressure(p,context())
        assert p.decide(context())['action']=='STOP_RETRIEVAL'
        assert len(inputs)==1 and p.store.read()==before and p._pending
        assert p._pending_indices==[0] and p.store.recover(0)['top_papers']
        ending=next(e for e in events if e['kind']=='context_termination')
        assert ending['reason']=='model_stop_with_pending_evidence' and ending['unadmitted_observation_indices']==[0]
        assert not [e for e in events if e['kind']=='context_edit']
    finally:p.close()


@pytest.mark.parametrize('bad',[
    {}, {'action':'STOP_RETRIEVAL','arguments':{}},
    {'decision':{'action':'STOP_RETRIEVAL','arguments':{'bad':'field'}}},
    {'decision':{'action':'STOP_RETRIEVAL','arguments':{}},'unexpected':True},
    {'decision':{'action':'STOP_RETRIEVAL','arguments':{}},'evidence_records':'bad'},
    {'decision':{'action':'STOP_RETRIEVAL','arguments':{}},'evidence_records':[{'bad':'claim'}]},
])
def test_pending_does_not_turn_malformed_decisions_into_valid_stop(tmp_path,bad):
    p,_,events=planner(tmp_path,'clm',[bad],max_chars=4000)
    try:
        before=pressure(p,context())
        with pytest.raises(ValueError):p.decide(context())
        assert p._pending and p.store.read()==before and p.store.recover(0)
        assert not [e for e in events if e['kind']=='context_termination']
    finally:p.close()


@pytest.mark.parametrize('restriction',['permission','evidence'])
def test_pending_stop_still_requires_permissions_and_required_evidence(tmp_path,restriction):
    p,_,events=planner(tmp_path,'clm',[STOP],max_chars=4000,require_evidence=restriction=='evidence')
    c=context()
    if restriction=='permission':c['permitted_actions']=['SEARCH_PAPERS']
    try:
        pressure(p,c)
        with pytest.raises(PlannerError,match='invalid_action_schema|evidence_record_required'):p.decide(c)
        assert not [e for e in events if e['kind']=='context_termination']
    finally:p.close()


def test_blocked_acquisition_invalid_edit_then_admission_requires_fresh_decision(tmp_path):
    acquire={'decision':{'action':'SEARCH_PAPERS','arguments':{'query':'blocked original'}}}
    fresh={'decision':{'action':'SEARCH_PAPERS','arguments':{'query':'fresh after repair'}}}
    outputs=[acquire,{'context_edit':dict(path='../forbidden',op='replace',text='invalid')},
             {'context_edit':dict(path='active_context.txt',op='replace',text='retained')},fresh]
    p,inputs,events=planner(tmp_path,'clm',outputs,max_chars=4000,max_calls=4)
    try:
        original=pressure(p,context())
        assert p.decide(context())['arguments']['query']=='fresh after repair'
        feedback=inputs[1]['operation_feedback']
        assert len(inputs)==4 and feedback.get('error',feedback.get('detail'))=='pending_evidence_blocks_acquisition'
        edits=[e for e in events if e['kind']=='context_edit']
        assert not edits[0]['accepted'] and edits[0]['after']==original
        assert edits[1]['committed'] and not p._pending
        assert p.store.recover(0)['top_papers']
    finally:p.close()


@pytest.mark.parametrize('operation',['acquisition','invalid_edit','invalid_read','valid_read_too_large'])
def test_repeated_failed_management_is_bounded_by_original_call_budget(tmp_path,operation):
    output={'acquisition':{'decision':{'action':'SEARCH_PAPERS','arguments':{'query':'new query'}}},
        'invalid_edit':{'context_edit':dict(path='../bad',op='replace',text='x')},
        'invalid_read':{'read_observation':dict(index=0,start=0,chars=513)},
        'valid_read_too_large':{'read_observation':dict(index=0,start=0,chars=512)}}[operation]
    p,inputs,events=planner(tmp_path,'clm',lambda *_:output,max_chars=4000,max_calls=2)
    try:
        old=pressure(p,context())
        with pytest.raises(BudgetExceeded,match='llm_call_limit'):p.decide(context())
        assert len(inputs)==2 and p._pending and p.store.read()==old and p.store.recover(0)
        assert events[-1]['kind']=='context_budget_stop'
        if operation=='valid_read_too_large':
            reads=[e for e in events if e['kind']=='context_recovery']
            assert len(reads)==2 and all(not e['accepted'] and e['feedback']['detail']=='archive_read_does_not_fit' for e in reads)
    finally:p.close()


@pytest.mark.parametrize('mode',['conventional','clm'])
def test_unfit_current_request_is_resource_failure_before_any_model_request(tmp_path,mode):
    p,inputs,events=planner(tmp_path,mode,[STOP],max_chars=4000)
    try:
        c=context();c['proposals'][0]['reason']='X'*5000
        with pytest.raises(BudgetExceeded,match='context_capacity_exceeded'):p.decide(c)
        assert not inputs and any(e['kind']=='context_capacity_failure' for e in events)
    finally:p.close()


@pytest.mark.parametrize('architecture',['clm','clm_graph'])
@pytest.mark.parametrize('decision',['stop','blocked_acquisition'])
def test_controller_pending_stop_never_invokes_acquisition_fallback(tmp_path,db,architecture,decision):
    output=STOP if decision=='stop' else {'decision':{'action':'SEARCH_PAPERS','arguments':{'query':'not executed'}}}
    base,_=model(lambda body,n:response(text=json.dumps(output)))
    class Pressured(ContextPlanner):
        def _decide(self,c):
            pressure(self,c)
            return super()._decide(c)
    p=Pressured(base,mode='clm',limits=InferenceLimits(max_active_tokens=4096,max_input_tokens=8192))
    c=ControlledController('retrieval',SeededCorpus(corpus(tmp_path),[1]),db,architecture=architecture,
        planner=p,budget=Budget(max_actions=10,max_llm_calls=2,page_size=1,max_context_chars=16000),output=str(tmp_path/'runs'))
    c._fallback=lambda _:pytest.fail('A valid pending STOP must never select fallback acquisition')
    try:
        result=c.run()
        assert result['stop_reason']==('planner_stop' if decision=='stop' else 'budget_exhausted') and result['actions']==1
        assert result['pending_observation_chars']>0 and result['unadmitted_observation_indices']==[0]
        assert result['experimental_implementation_version']==CONTEXT_POLICY_VERSION
        assert result['budget']['max_llm_calls']==2 and result['inference_limits']['max_total_tokens']==100000
    finally:p.close()


def test_fitting_admission_can_omit_operational_feedback_without_dropping_evidence(tmp_path):
    p,inputs,events=planner(tmp_path,'clm',[STOP],max_chars=4000)
    c=context()
    try:
        protected,body=p._parts(c);protected['archive_count']=1
        helper=type('DeltaState',(),{'_seen':{}})()
        pending='\n'+serialized({'observation':0,'changes':ContextPlanner._delta(helper,body)})
        target=p._payload(protected,body,active=pending)
        # Account for the real pending-count field exactly as _decide will.
        target['pending_observation_chars']=len(pending)
        overhead=len(serialized(target))
        filler='A'*(4000-overhead-15)
        target['active_context']=filler+pending
        assert p.payload_fits(target)
        p._feedback={'edit_accepted':True,'error':None}
        with_feedback={**target,'operation_feedback':p._feedback}
        assert not p.payload_fits(with_feedback)
        p.store.transition(filler,source='environment',operation={'SYNTHETIC_BOUNDARY':True})
        assert p.decide(c)['action']=='STOP_RETRIEVAL'
        assert p.store.read()==filler+pending and not p._pending and len(inputs)==1
        admitted=[e for e in events if e['kind']=='context_admission' and e['status']=='admitted']
        assert admitted[0]['feedback_omitted_for_admission']
        assert p.store.recover(0)==body and p.payload_fits(inputs[0])
    finally:p.close()


def test_failed_admission_commit_stops_without_model_call_or_acquisition(tmp_path,monkeypatch):
    p,inputs,events=planner(tmp_path,'clm',[STOP],max_chars=4000)
    def fail_write(text):raise OSError('SYNTHETIC failed admission commit')
    monkeypatch.setattr(p.store,'_write',fail_write)
    try:
        with pytest.raises(BudgetExceeded,match='context_observation_write_failed'):p.decide(context())
        assert not inputs and p.usage.calls==0 and p._pending and p.store.read()==''
        assert p.store.recover(0)['top_papers']
        receipt=p.store.edits[-1]
        assert not receipt['committed'] and receipt['error']=='context_write_failed'
        assert events[-1]['code']=='context_observation_write_failed'
    finally:p.close()
