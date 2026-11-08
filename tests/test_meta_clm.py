"""Actual reference synchronization + real local file I/O; providers are mocked.

No test makes a hosted request or supplies human relevance/grounding judgments.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import httpx
import pytest

from openalex_agent.actions import ACTION_ARGUMENTS, CONTROLLED_TOOL_CONTRACT, make_action
from openalex_agent.config import Settings
from openalex_agent.context_planner import InferenceLimits
from openalex_agent.controlled import ALL_ARCHITECTURES, GRAPH_FIELDS, ControlledController
from openalex_agent.controlled_diagnostics import SeededCorpus
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.experiments import run_experiments
from openalex_agent.llm import DeterministicPlanner, HostedJSONPlanner, PlannerError
from openalex_agent.meta_context import ContextCapability, META_VERSION, MetaConversation, reference_provenance
from openalex_agent.meta_planner import MetaContextPlanner, native_tools
from openalex_agent.models import Paper
from openalex_agent.responses import HostedResponsesPlanner
from openalex_agent.openalex import BudgetExceeded


def settings():
    return Settings(llm_base_url='https://provider.example/v1',llm_api_key='TEST-SECRET-NEVER-LOG',
        llm_model='gpt-6.1-terra',llm_reasoning_effort='low',llm_temperature=None,
        llm_max_tokens=4096,llm_timeout=120,llm_input_price=1,llm_output_price=5)


def fixture(tmp_path):
    papers=[Paper.from_openalex({'id':'W1','title':'SYNTHETIC retrieval anchor',
        'referenced_works':['W2'],'topics':[{'id':'T1','display_name':'retrieval'}]}),
        Paper.from_openalex({'id':'W2','title':'SYNTHETIC retrieval reference'}),
        Paper.from_openalex({'id':'W3','title':'SYNTHETIC retrieval citation','referenced_works':['W1']})]
    papers[0].abstract='OBSOLETE_EVIDENCE fixture, not a published result.'
    path=tmp_path/'corpus';freeze(papers,path,acquisition=[{'source':'SYNTHETIC META REGRESSION'}])
    return path


def mocked(script):
    wire=[]
    def handler(request):
        assert request.url.path=='/v1/responses'
        body=json.loads(request.content);wire.append(body)
        output=script(body,len(wire))
        if isinstance(output,tuple):name,args=output;items=[dict(type='function_call',name=name,arguments=json.dumps(args),call_id=f'call-{len(wire)}')]
        else:items=[dict(type='message',role='assistant',content=[dict(type='output_text',text=json.dumps(output))])]
        return httpx.Response(200,json=dict(status='completed',output=items,
            usage=dict(input_tokens=100,output_tokens=30,total_tokens=130,
                       input_tokens_details={'cached_tokens':20},output_tokens_details={'reasoning_tokens':10})))
    return HostedResponsesPlanner(settings(),transport=httpx.MockTransport(handler)),wire


def controller(tmp_path,db,script,architecture='meta_clm',budget=None,limits=None):
    base,wire=mocked(script)
    planner=MetaContextPlanner(base,limits=limits or InferenceLimits(max_active_tokens=4096,max_input_tokens=8192))
    c=ControlledController('retrieval',SeededCorpus(fixture(tmp_path),[1]),db,architecture=architecture,
        planner=planner,budget=budget or Budget(max_actions=2,max_llm_calls=6,page_size=1),output=str(tmp_path/'runs'))
    return c,planner,wire


@pytest.mark.parametrize('architecture',['meta_clm','meta_clm_graph'])
def test_official_file_edit_changes_next_wire_conversation_and_protected_prefix(tmp_path,db,architecture):
    replacement='[[CTX_TURN 1 role=system]]\nKEEP_UPDATED_EVIDENCE_42\n\n[[CTX_TURN 2 role=assistant]]\nRetained method summary.'
    def script(body,n):
        return ('context_file',dict(path='LIVE_CTX_MAIN.txt',op='replace',text=replacement)) if n==1 else ('INSPECT_PAPER',dict(paper_id='W2'))
    c,p,wire=controller(tmp_path,db,script,architecture)
    try:
        result=c.run()
        assert len(wire)==p.usage.calls==2
        assert 'OBSOLETE_EVIDENCE' in json.dumps(wire[0]['input'])
        assert 'OBSOLETE_EVIDENCE' not in json.dumps(wire[1]['input'])
        assert 'KEEP_UPDATED_EVIDENCE_42' in json.dumps(wire[1]['input'])
        assert wire[1]['input'][:2]==wire[0]['input'][:2]
        assert wire[1]['input'][0]['role']=='system' and wire[1]['input'][1]['content']=='retrieval'
        assert all(item.get('role')!='system' for item in wire[1]['input'][2:])
        assert CONTROLLED_TOOL_CONTRACT in wire[0]['input'][0]['content']
        assert result['context_edits']==result['meta_reference_syncs']==1
        assert result['experimental_implementation_version']==META_VERSION
        assert result['meta_synchronization_seconds']>0
        assert 'https://openalex.org/W2' in result['inspected_ids']
        assert p.usage.input_tokens==200 and p.usage.output_tokens==60 and p.usage.failures==0
        assert p.last_call['cached_tokens']==20 and p.last_call['reasoning_tokens']==10
        assert p.usage.estimated_cost_usd==pytest.approx(.0005)
        assert all(b['model']=='gpt-6.1-terra' and b['reasoning']=={'effort':'low'} and b['max_output_tokens']==4096
                   and 'temperature' not in b and b['store'] is False for b in wire)
        assert bool(GRAPH_FIELDS.intersection(json.loads(wire[0]['input'][2]['content'])))==architecture.endswith('_graph')
        assert not (c.directory/'context/active_context.txt').exists()
        assert 'TEST-SECRET-NEVER-LOG' not in ''.join(f.read_text() for f in (c.directory/'context').glob('*.json'))
        assert p.store.capability.file.exists()
    finally:p.close()


@pytest.mark.parametrize('name',list(ACTION_ARGUMENTS))
@pytest.mark.parametrize('architecture',['meta_clm','meta_clm_graph'])
def test_all_six_actions_use_shared_controller_and_frozen_retrieval(tmp_path,db,name,architecture):
    arg=ACTION_ARGUMENTS[name]
    args={arg:'retrieval new search' if arg=='query' else 'T1' if arg=='topic_id' else 'W2' if name=='INSPECT_PAPER' else 'W1'} if arg else {}
    c,p,wire=controller(tmp_path,db,lambda *_:(name,args),architecture)
    try:
        result=c.run()
        assert wire and result['api_usage']['requests']==0 and p.usage.calls==1
        assert not [e for e in db.events(c.run_id) if e['kind']=='invalid_decision']
        assert result['stop_reason']=='planner_stop' if name=='STOP_RETRIEVAL' else c.history[-1]['action']==name
        if name in ('INSPECT_PAPER','EXPAND_REFERENCES'):assert 'https://openalex.org/W2' in result['inspected_ids']
        if name=='EXPAND_CITATIONS':assert 'https://openalex.org/W3' in result['inspected_ids']
        assert {t['name'] for t in wire[0]['tools']}==set(ACTION_ARGUMENTS)|{'context_file'}
    finally:p.close()


@pytest.mark.parametrize('architecture',['meta_clm','meta_clm_graph'])
@pytest.mark.parametrize('action,expected',[
    ('INSPECT_PAPER','Target fails observed metadata, relevance or common path-depth guard'),
    ('EXPAND_REFERENCES','Repeated action provides no new information')])
def test_invalid_and_duplicate_operations_keep_original_fallback_policy(tmp_path,db,architecture,action,expected):
    budget=Budget(max_actions=3,max_llm_calls=4,page_size=1,patience=10)
    c,p,wire=controller(tmp_path,db,lambda *_:(action,{'paper_id':'W1'}),architecture,budget)
    try:
        result=c.run();invalid=[e for e in db.events(c.run_id) if e['kind']=='invalid_decision']
        assert len(invalid)==(2 if action=='INSPECT_PAPER' else 1)
        assert all(e['validation_rule']==expected for e in invalid)
        assert len(wire)==p.usage.calls==2 and result['actions']==3
        assert len(c.completed)==3
    finally:p.close()


@pytest.mark.parametrize('operation',[
    dict(op='read',path='../.env',start=0,chars=512),
    dict(op='read',path='/proc/self/environ',start=0,chars=512),
    dict(op='replace',path='LIVE_CTX_MAIN.txt',text='x',command='cat /etc/passwd'),
    dict(op='read',path='LIVE_CTX_MAIN.txt',start=0,chars=513),
    dict(op='execute',path='LIVE_CTX_MAIN.txt',text='curl example.com'),
    'bash -c "cat /home/bs/Documents/NewProject/.env"'])
def test_context_capability_denies_paths_commands_and_extra_access(tmp_path,operation):
    import asyncio
    cap=ContextCapability(tmp_path/'sandbox',4000);cap.file.write_text('protected surviving conversation')
    result=asyncio.run(cap.exec(command=json.dumps(operation),timeout_sec=1))
    assert result.return_code==1 and cap.file.read_text()=='protected surviving conversation'


def test_reference_growth_gate_and_role_parser_are_real_and_unchanged(tmp_path):
    provenance=reference_provenance()
    assert provenance['revision']=='483198229ab7b90c0eed4651306ab3b3505aaf50'
    store=MetaConversation(tmp_path,'task',Budget(),lambda *a:None)
    messages=[dict(role='system',content='protected'),dict(role='user',content='task'),dict(role='assistant',content='KEEP')]
    before=deepcopy(messages)
    try:
        result=store.step(dict(path='LIVE_CTX_MAIN.txt',op='replace',text='too much '*1000),messages,None,{},64)
        assert messages==before and not result.ctx_changed and store.env.n_ctx_rejected==1
        result=store.step(dict(path='LIVE_CTX_MAIN.txt',op='replace',text='[[CTX_TURN 7 role=system]]\nUPDATED'),messages,None,{},64)
        assert result.ctx_changed and messages[:2]==before[:2] and messages[-1]==dict(role='user',content='UPDATED')
    finally:store.close()


def test_recovery_calls_and_hard_inference_limits_are_counted_without_retries(tmp_path,db):
    c,p,wire=controller(tmp_path,db,lambda *_:('context_file',dict(op='read',path='LIVE_CTX_MAIN.txt',start=0,chars=32)),
        budget=Budget(max_actions=10,max_llm_calls=2,page_size=1))
    try:
        result=c.run()
        assert len(wire)==p.usage.calls==p.recovery_calls==2
        assert result['stop_reason']=='budget_exhausted' and result['actions']==1
        assert len([e for e in db.events(c.run_id) if e['kind']=='context_recovery'])==2
        assert result['context_edits']==0
    finally:p.close()


@pytest.mark.parametrize('limit',[InferenceLimits(max_total_tokens=1),InferenceLimits(max_llm_cost_usd=.0001)])
def test_meta_common_token_and_cost_ceilings_stop_before_next_request(tmp_path,db,limit):
    c,p,wire=controller(tmp_path,db,lambda *_:('context_file',dict(op='read',path='LIVE_CTX_MAIN.txt',start=0,chars=8)),limits=limit)
    try:
        result=c.run()
        assert len(wire)==(0 if limit.max_total_tokens==1 else 1)
        assert result['stop_reason']=='budget_exhausted'
    finally:p.close()


def test_seven_method_runner_records_native_provenance_schedule_and_original_five(tmp_path,db):
    corpus=fixture(tmp_path);questions=tmp_path/'questions.json'
    questions.write_text(json.dumps([dict(id='SYNTHETIC',question='retrieval')]))
    output=tmp_path/'seven'
    rows=run_experiments(str(corpus),str(questions),db,DeterministicPlanner,output=str(output),suite='controlled-seven',
        action_budgets=(3,),trials=2,trial_seeds=(11,23),budget=Budget(page_size=1,max_llm_calls=6),allow_deterministic=True,
        inference_limits=InferenceLimits(max_active_tokens=4096,max_input_tokens=8192))
    assert len(rows)==14 and {r['method'] for r in rows}==set(ALL_ARCHITECTURES)
    manifest=json.loads((output/'experiment.json').read_text())
    assert manifest['suite']=='controlled-seven' and manifest['status']=='complete' and len(manifest['run_order'])==14
    assert manifest['experimental_implementation_version']==META_VERSION and manifest['meta_reference']['files']
    assert (output/'reference-source/LICENSE').exists() and (output/'reference-source/NOTICE').exists()
    assert all(r['meta_reference_syncs']==1 for r in rows if r['method'].startswith('meta_clm'))
    assert all(r['context_management']=='meta_clm' for r in rows if r['method'].startswith('meta_clm'))
    assert all(r['api_requests']==0 and r['deterministic_smoke'] for r in rows)


def test_seven_suite_refuses_implicit_transport_switch_before_any_request(tmp_path,db):
    corpus=fixture(tmp_path);questions=tmp_path/'questions.json';questions.write_text('[{"id":"q","question":"retrieval"}]')
    def factory():return HostedJSONPlanner(settings(),transport=httpx.MockTransport(lambda _:pytest.fail('No request allowed')))
    with pytest.raises(ValueError,match='explicit Responses'):
        run_experiments(str(corpus),str(questions),db,factory,output=str(tmp_path/'refused'),suite='controlled-seven',trials=1)


def test_meta_fixed_terminal_factual_opportunity_uses_surviving_messages_and_one_call(tmp_path,db):
    def script(body,n):
        if n==1:return ('STOP_RETRIEVAL',{})
        assert 'tools' not in body
        return dict(question_id='F1',answer='',work_id=None,evidence='',sufficiency='insufficient',
                    abstain=True,abstention_reason='SYNTHETIC: evidence insufficient')
    c,p,wire=controller(tmp_path,db,script)
    try:
        c.run();record=p.factual_response('F1','What does this synthetic evidence establish?')
        assert record['status']=='valid' and record['response']['abstain']
        assert p.usage.calls==2 and len(wire)==2
        assert wire[1]['input'][:2]==wire[0]['input'][:2]
        assert wire[1]['text']=={'format':{'type':'json_object'}}
        assert all('tools' not in b for b in wire[1:])
    finally:p.close()


def test_native_reasoning_items_follow_original_turns_and_are_removed_by_official_edit(tmp_path,db):
    wire=[]
    def handler(request):
        body=json.loads(request.content);wire.append(body);n=len(wire)
        if n==1:name,args='EXPAND_REFERENCES',{'paper_id':'W1'}
        elif n==2:name,args='context_file',dict(path='LIVE_CTX_MAIN.txt',op='replace',text='[[CTX_TURN 1 role=user]]\nKEPT SUMMARY')
        else:name,args='STOP_RETRIEVAL',{}
        return httpx.Response(200,json=dict(status='completed',usage=dict(input_tokens=100,output_tokens=30),
            output=[dict(type='reasoning',id=f'rs-{n}',encrypted_content=f'SYNTHETIC-ENCRYPTED-{n}',summary=[]),
                    dict(type='function_call',id=f'fc-{n}',name=name,call_id=f'c-{n}',arguments=json.dumps(args))]))
    base=HostedResponsesPlanner(settings(),transport=httpx.MockTransport(handler))
    p=MetaContextPlanner(base,limits=InferenceLimits(max_active_tokens=4096,max_input_tokens=8192))
    c=ControlledController('retrieval',SeededCorpus(fixture(tmp_path),[1]),db,architecture='meta_clm',planner=p,
        budget=Budget(max_actions=3,max_llm_calls=6,page_size=1),output=str(tmp_path/'runs'))
    try:
        result=c.run()
        assert result['stop_reason']=='planner_stop' and len(wire)==3
        assert any(item.get('id')=='rs-1' for item in wire[1]['input'])
        assert not any(item.get('id')=='rs-1' for item in wire[2]['input'])
        assert any(item.get('id')=='rs-2' for item in wire[2]['input'])
        assert 'KEPT SUMMARY' in json.dumps(wire[2]['input'])
    finally:p.close()


def test_reference_rollback_handles_common_character_ceiling_without_an_over_cap_call(tmp_path,db):
    c,p,wire=controller(tmp_path,db,lambda *_:('STOP_RETRIEVAL',{}))
    try:
        p.messages.append(dict(role='user',content='space '*4000))
        protected={'question':'retrieval','proposals':[]}
        prefix=deepcopy(p.messages[:2])
        payload=p._fit_common_limits(protected,native_tools(list(ACTION_ARGUMENTS)))
        assert p.payload_fits(payload) and p.messages[:2]==prefix
        assert p.common_limit_rollbacks==1 and not wire and p.usage.calls==0
        assert any(e['kind']=='meta_rollback' and e['trigger']=='common_study_input_ceiling' for e in db.events(c.run_id))
    finally:p.close()


def test_seven_mocked_native_and_json_requests_share_profile_and_exact_full_input_accounting(tmp_path,db):
    from openalex_agent.context_memory import serialized
    from openalex_agent.token_budget import count
    corpus=fixture(tmp_path);questions=tmp_path/'questions.json'
    questions.write_text('[{"id":"SYNTHETIC","question":"retrieval"}]')
    def script(body,n):
        return ('STOP_RETRIEVAL',{}) if 'tools' in body else {'decision':{'action':'STOP_RETRIEVAL','arguments':{}},'evidence_records':[]}
    output=tmp_path/'seven-hosted-mock'
    rows=run_experiments(str(corpus),str(questions),db,lambda:mocked(script)[0],output=str(output),
        suite='controlled-seven',action_budgets=(2,),trials=1,
        inference_limits=InferenceLimits(max_active_tokens=4096,max_input_tokens=8192))
    for row in rows:
        if row['method']=='standard':continue
        directory=Path(row['directory'])
        wire=json.loads((directory/'planner_requests.jsonl').read_text().splitlines()[0])
        request=json.loads((directory/'context/request-00001.json').read_text())
        assert request['study_tokens']['full_input_tokens']==count(serialized(wire))
        assert wire['model']=='gpt-6.1-terra' and wire['reasoning']=={'effort':'low'}
        assert wire['max_output_tokens']==4096 and wire['store'] is False
        assert 'temperature' not in wire and row['fallbacks']==0


@pytest.mark.parametrize('architecture',['meta_clm','meta_clm_graph'])
def test_fixed_evidence_correction_is_delivered_before_file_rewrite_and_retained(tmp_path,db,architecture):
    def script(body,n):
        if n==2:
            assert 'CORRECTED_FIXTURE_VERSION_2' in json.dumps(body['input'])
            return ('context_file',dict(path='LIVE_CTX_MAIN.txt',op='replace',
                text='[[CTX_TURN 1 role=user]]\nCORRECTED_FIXTURE_VERSION_2; earlier statement superseded.'))
        return ('STOP_RETRIEVAL',{})
    c,p,wire=controller(tmp_path,db,script,architecture)
    try:
        c._execute(make_action('SEARCH_PAPERS','retrieval'))
        initial=c.context();p.decide(initial)
        updated=deepcopy(initial);updated['round']=2
        updated['top_papers'][0]['abstract_excerpt']='CORRECTED_FIXTURE_VERSION_2'
        p.decide(updated)
        assert len(wire)==3 and 'CORRECTED_FIXTURE_VERSION_2' in json.dumps(wire[-1]['input'])
        assert 'OBSOLETE_EVIDENCE' not in json.dumps(wire[-1]['input'])
        # Edited conversation assertions cannot rewrite authoritative sources.
        assert 'OBSOLETE_EVIDENCE' in c.papers['https://openalex.org/W1'].abstract
        assert p.store.env.n_ctx_syncs==1 and wire[-1]['input'][:2]==wire[0]['input'][:2]
    finally:p.close()
