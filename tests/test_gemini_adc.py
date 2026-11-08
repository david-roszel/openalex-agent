"""Mocked Gemini generation; real shared controllers and Meta file synchronization."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from openalex_agent.actions import ACTION_ARGUMENTS, make_action
from openalex_agent.context_memory import serialized
from openalex_agent.context_planner import ContextPlanner, InferenceLimits
from openalex_agent.controlled import ALL_ARCHITECTURES, ControlledController
from openalex_agent.controlled_diagnostics import SeededCorpus
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.experiments import run_experiments
from openalex_agent.gemini import GEMINI_VERSION, GeminiADCPlanner, GeminiSettings, gemini_contents
from openalex_agent.llm import PlannerError
from openalex_agent.meta_planner import MetaContextPlanner
from openalex_agent.models import Paper


def corpus(tmp_path):
    path=tmp_path/'corpus'
    freeze([Paper.from_openalex({'id':'W1','title':'SYNTHETIC retrieval anchor','referenced_works':['W2'],
           'topics':[{'id':'T1','display_name':'retrieval'}]}),
           Paper.from_openalex({'id':'W2','title':'SYNTHETIC reference'}),
           Paper.from_openalex({'id':'W3','title':'SYNTHETIC citing paper','referenced_works':['W1']})],path,acquisition=[])
    return path


def response(*,name=None,args=None,text=None,n=1,usage=None):
    part={'function_call':{'id':f'provider-{n}','name':name,'args':args or {}},'thought_signature':f'c2lnbmF0dXJl{n}'} if name else {'text':text,'thought_signature':'c2lnbmF0dXJl'}
    return dict(http_status=200,response=dict(candidates=[dict(content=dict(role='model',parts=[part]),finish_reason='STOP',
        safety_ratings=[])],model_version='gemini-3.8-flash',usage_metadata=usage if usage is not None else dict(
        prompt_token_count=8,candidates_token_count=5,thoughts_token_count=95,total_token_count=108)))


def model(script):
    bodies=[]
    def executor(body):
        bodies.append(deepcopy(body));return script(body,len(bodies))
    return GeminiADCPlanner(GeminiSettings(project='SYNTHETIC-PROJECT'),executor=executor),bodies


@pytest.mark.parametrize('architecture',['meta_clm','meta_clm_graph'])
def test_gemini_native_signature_and_function_response_survive_actual_meta_sync(tmp_path,db,architecture):
    def script(body,n):
        if n==1:return response(name='context_file',args=dict(path='LIVE_CTX_MAIN.txt',op='replace',text='[[CTX_TURN 1 role=user]]\nGEMINI_RETAINED_42'),n=n)
        return response(name='INSPECT_PAPER',args={'paper_id':'W2'},n=n)
    base,wire=model(script);planner=MetaContextPlanner(base,limits=InferenceLimits(max_active_tokens=4096,max_input_tokens=8192))
    c=ControlledController('retrieval',SeededCorpus(corpus(tmp_path),[1]),db,architecture=architecture,planner=planner,
        budget=Budget(max_actions=2,page_size=1),output=str(tmp_path/'runs'))
    try:
        result=c.run()
        assert len(wire)==2 and result['context_edits']==1
        assert 'GEMINI_RETAINED_42' in serialized(wire[1]['contents'])
        native=next(content for content in wire[1]['contents'] if any('function_call' in part for part in content['parts']))
        assert native['parts'][0]['thought_signature']=='c2lnbmF0dXJl1'
        assert native['parts'][0]['function_call']=={'id':'provider-1','name':'context_file','args':wire[0] and script(wire[0],1)['response']['candidates'][0]['content']['parts'][0]['function_call']['args']}
        returned=next(part['function_response'] for content in wire[1]['contents'] for part in content['parts'] if 'function_response' in part)
        assert returned['id']=='provider-1' and returned['name']=='context_file'
        assert wire[1]['config']['system_instruction']==wire[0]['config']['system_instruction']
        assert result['inspected_ids']==['https://openalex.org/W1','https://openalex.org/W2']
        assert base.usage.input_tokens==16 and base.usage.output_tokens==200
        assert base.last_call['visible_output_tokens']==5 and base.last_call['reasoning_tokens']==95
        assert base.usage.estimated_cost_usd==pytest.approx((16*1.5+200*7.5)/1e6)
        assert 'provider_response' in json.loads((c.directory/'context/request-00001.json').read_text())
    finally:planner.close()


@pytest.mark.parametrize('architecture',ALL_ARCHITECTURES[1:])
def test_all_six_agent_configurations_use_same_gemini_model_and_shared_budgets(tmp_path,db,architecture):
    def script(body,n):
        return response(name='STOP_RETRIEVAL',n=n) if 'tools' in body['config'] else response(text='{"decision":{"action":"STOP_RETRIEVAL","arguments":{}}}')
    base,wire=model(script)
    planner=(MetaContextPlanner(base) if architecture.startswith('meta_clm') else
        ContextPlanner(base,mode='clm' if architecture.startswith('clm') else 'conventional'))
    budget=Budget(max_actions=10,max_llm_calls=20,page_size=1)
    c=ControlledController('retrieval',SeededCorpus(corpus(tmp_path),[1]),db,architecture=architecture,planner=planner,budget=budget,output=str(tmp_path/'runs'))
    try:
        result=c.run()
        assert result['stop_reason']=='planner_stop' and len(wire)==1
        assert wire[0]['model']=='gemini-3.8-flash'
        assert wire[0]['config']['thinking_config']=={'thinking_level':'LOW'}
        assert wire[0]['config']['max_output_tokens']==4096 and 'temperature' not in wire[0]['config']
        assert base.configuration['auth']=='ADC' and result['api_usage']['requests']==0
        assert result['budget']==budget.__dict__
        source=[]
        for content in wire[0]['contents']:
            for part in content['parts']:
                try:value=json.loads(part.get('text',''))
                except ValueError:continue
                if 'observation' in value:source.append(value['observation'])
                elif 'active_context' in value:source.append(json.loads(value['active_context'].strip())['changes'])
                elif 'top_papers' in value:source.append(value)
        assert source and ('graph' in source[0])==architecture.endswith('_graph')
        assert 'observed_metadata' in source[0]
    finally:planner.close()


@pytest.mark.parametrize('name',list(ACTION_ARGUMENTS))
def test_gemini_all_six_native_scholarly_actions_are_mediated(tmp_path,db,name):
    key=ACTION_ARGUMENTS[name];args={key:'new retrieval query' if key=='query' else 'T1' if key=='topic_id' else 'W2' if name=='INSPECT_PAPER' else 'W1'} if key else {}
    base,wire=model(lambda body,n:response(name=name,args=args,n=n));p=MetaContextPlanner(base)
    c=ControlledController('retrieval',SeededCorpus(corpus(tmp_path),[1]),db,architecture='meta_clm',planner=p,
        budget=Budget(max_actions=2,page_size=1),output=str(tmp_path/'runs'))
    try:
        c.run()
        assert len(wire)==1 and not [e for e in db.events(c.run_id) if e['kind']=='invalid_decision']
        if name!='STOP_RETRIEVAL':assert c.history[-1]['action']==name
    finally:p.close()


def test_gemini_preserves_full_parts_and_metadata_without_mutating_signatures():
    original=dict(role='model',parts=[dict(text='pre-tool text',thought_signature='dGV4dA=='),
        dict(text='SYNTHETIC thought summary',thought=True),
        dict(function_call=dict(id='native-id',name='INSPECT_PAPER',args={'paper_id':'W2'}),thought_signature='AAEC/w==')])
    messages=[dict(role='system',content='protected'),dict(role='assistant',content='pre-tool text',gemini_content=deepcopy(original),
        tool_calls=[dict(id='internal-id',function=dict(name='INSPECT_PAPER',arguments='{"paper_id":"W2"}'))]),
        dict(role='tool',tool_call_id='internal-id',content='observed result')]
    _,contents=gemini_contents(messages)
    assert contents[0]==original and messages[1]['gemini_content']==original
    assert contents[1]['parts'][0]['function_response']==dict(id='native-id',name='INSPECT_PAPER',response={'output':'observed result'})
    adapter,_=model(lambda body,n:response(text='{}'))
    adapter.last_response_content=deepcopy(original)
    editable=adapter.native_assistant_message(dict(name='INSPECT_PAPER',call_id='internal-id',arguments={'paper_id':'W2'}))
    assert editable['reasoning_content']=='SYNTHETIC thought summary' and editable['gemini_content']==original


@pytest.mark.parametrize('usage,expected',[
    (dict(prompt_token_count=8,candidates_token_count=5,total_token_count=108),(8,100)),
    (dict(prompt_token_count=8,candidates_token_count=5,thoughts_token_count=95,tool_use_prompt_token_count=2,total_token_count=110),(10,100)),
    (dict(prompt_token_count=8,candidates_token_count=5),None),
    (dict(prompt_token_count=8,candidates_token_count=5,thoughts_token_count=95,total_token_count=13),None)])
def test_gemini_reasoning_and_unknown_usage_are_accounted_conservatively(usage,expected):
    p,_=model(lambda body,n:response(text='{}',usage=usage))
    p._call('Return JSON',{})
    if expected:
        assert (p.usage.input_tokens,p.usage.output_tokens)==expected and p.usage.usage_reported_calls==1
    else:assert p.usage.usage_unreported_calls==1 and p.usage.estimated_cost_usd is None


def test_seven_method_mocked_gemini_manifest_versions_and_no_heldout_use(tmp_path,db):
    questions=tmp_path/'questions.json';questions.write_text('[{"id":"SYNTHETIC","question":"retrieval"}]')
    def script(body,n):return response(name='STOP_RETRIEVAL',n=n) if 'tools' in body['config'] else response(text='{"decision":{"action":"STOP_RETRIEVAL","arguments":{}}}')
    out=tmp_path/'seven'
    rows=run_experiments(str(corpus(tmp_path)),str(questions),db,lambda:model(script)[0],output=str(out),suite='controlled-seven',trials=1,action_budgets=(2,))
    assert len(rows)==7 and {r['method'] for r in rows}==set(ALL_ARCHITECTURES)
    manifest=json.loads((out/'experiment.json').read_text())
    assert manifest['provider_experiment_version']==GEMINI_VERSION and manifest['model']=='gemini-3.8-flash'
    assert all(r['llm_calls']==(0 if r['method']=='standard' else 1) for r in rows)


@pytest.mark.parametrize('code',['timeout','authentication_error','rate_limited','transport_error'])
def test_gemini_classified_errors_make_one_attempt(code):
    p,calls=model(lambda body,n:dict(error_code=code,http_status=None))
    with pytest.raises(PlannerError,match=code):p._call('Return JSON',{})
    assert len(calls)==p.usage.calls==p.usage.failures==1


def test_sdk_probe_loads_installed_sdk_without_credentials_or_generation():
    python=Path('~/gemini-test-env/bin/python').expanduser()
    if not python.exists():pytest.skip('Optional isolated Google SDK environment is absent')
    result=subprocess.run([str(python),'openalex_agent/gemini_sdk_worker.py'],input='{"probe":true}',
        capture_output=True,text=True,timeout=20)
    assert result.returncode==0 and json.loads(result.stdout)==dict(sdk_version='3.0.0',generation_performed=False)


def test_actual_sdk_wire_serialization_signature_bytes_and_presend_guard_are_offline():
    python=Path('~/gemini-test-env/bin/python').expanduser()
    if not python.exists():pytest.skip('Optional isolated Google SDK environment is absent')
    result=subprocess.run([str(python),'tests/fixtures/gemini_sdk_serialization_probe.py'],capture_output=True,text=True,
        timeout=40,env={**os.environ,'GEMINI_TEST_TOKENIZER_PYTHON':sys.executable,'PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode==0,result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['hosted_requests']==0 and receipt['network_blocked']
    assert receipt['signature_preserved'] and receipt['function_ids_preserved']
    assert receipt['oversize_blocked_before_send'] and receipt['no_sdk_generation_retries']


def test_gemini_reused_provider_ids_and_old_signatures_follow_reference_edit_semantics(tmp_path,db):
    def script(body,n):
        if n==1:value=response(name='EXPAND_REFERENCES',args={'paper_id':'W1'},n=n)
        elif n==2:value=response(name='context_file',args=dict(path='LIVE_CTX_MAIN.txt',op='replace',text='[[CTX_TURN 1 role=user]]\nKEPT_GEMINI_SOURCE'),n=n)
        else:value=response(name='STOP_RETRIEVAL',n=n)
        value['response']['candidates'][0]['content']['parts'][0]['function_call']['id']='provider-reused-id'
        return value
    base,wire=model(script);p=MetaContextPlanner(base)
    c=ControlledController('retrieval',SeededCorpus(corpus(tmp_path),[1]),db,architecture='meta_clm',planner=p,
        budget=Budget(max_actions=3,page_size=1),output=str(tmp_path/'runs'))
    try:
        result=c.run()
        assert result['stop_reason']=='planner_stop' and len(wire)==3
        assert 'c2lnbmF0dXJl1' in serialized(wire[1]['contents'])
        assert 'c2lnbmF0dXJl1' not in serialized(wire[2]['contents'])
        assert 'c2lnbmF0dXJl2' in serialized(wire[2]['contents'])
        assert 'KEPT_GEMINI_SOURCE' in serialized(wire[2]['contents'])
        assert not [e for e in db.events(c.run_id) if e['kind']=='invalid_decision']
    finally:p.close()


def test_gemini_missing_function_signature_is_a_failure_not_a_fake_bypass():
    def mock(body,n):
        value=response(name='INSPECT_PAPER',args={'paper_id':'W2'})
        value['response']['candidates'][0]['content']['parts'][0].pop('thought_signature')
        return value
    p,_=model(mock)
    from openalex_agent.meta_planner import native_tools
    with pytest.raises(PlannerError,match='missing_thought_signature'):
        p._call_native([dict(role='user',content='SYNTHETIC')],native_tools(list(ACTION_ARGUMENTS)))
    assert p.usage.calls==1 and p.usage.output_tokens==100


def test_gemini_meta_terminal_answer_retains_sdk_metadata_and_fixed_opportunity(tmp_path,db):
    def mock(body,n):
        if n==1:return response(name='STOP_RETRIEVAL',n=n)
        return response(text=json.dumps(dict(question_id='SYNTHETIC-F1',answer='',work_id=None,evidence='',
            sufficiency='insufficient',abstain=True,abstention_reason='SYNTHETIC: insufficient evidence')))
    base,wire=model(mock);p=MetaContextPlanner(base)
    c=ControlledController('retrieval',SeededCorpus(corpus(tmp_path),[1]),db,architecture='meta_clm',planner=p,
        budget=Budget(max_actions=2,page_size=1),output=str(tmp_path/'runs'))
    try:
        c.run();answer=p.factual_response('SYNTHETIC-F1','What does the fixture establish?')
        assert answer['status']=='valid' and len(wire)==2 and 'tools' not in wire[-1]['config']
        saved=json.loads((c.directory/'context/request-00002.json').read_text())
        assert saved['provider_response']['model_version']=='gemini-3.8-flash'
        assert saved['provider_response']['usage_metadata']['thoughts_token_count']==95
        assert p.usage.output_tokens==200
    finally:p.close()
