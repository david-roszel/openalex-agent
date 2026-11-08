"""Contract regressions: synthetic invalid argument shapes and real local Meta semantics.

Provider decisions in these tests are explicitly mocked. No hosted inference.
"""
import asyncio
from copy import deepcopy
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from jsonschema import Draft202012Validator
import pytest

from openalex_agent.actions import ACTION_ARGUMENTS
from openalex_agent.context_memory import serialized
from openalex_agent.context_planner import InferenceLimits
from openalex_agent.controlled import ARCHITECTURES, ControlledController
from openalex_agent.controlled_diagnostics import SeededCorpus
from openalex_agent.controller import Budget
from openalex_agent.gemini import GeminiADCPlanner, GeminiSettings
from openalex_agent.meta_context import ContextCapability, MetaConversation
from openalex_agent.meta_planner import MetaContextPlanner, native_tools
from openalex_agent.meta_validation import audit_canary
from test_gemini_adc import model, response
from test_meta_clm import fixture, mocked

FIXTURE=Path(__file__).parent/'fixtures/meta-context-failed-edits.json'
FAILURES=json.loads(FIXTURE.read_text())['failures']


def declaration():
    return next(t for t in native_tools(ACTION_ARGUMENTS) if t['name']=='context_file')


@pytest.mark.parametrize('failure',FAILURES,ids=lambda f:f"{f['architecture']}-call-{f['call']}")
def test_synthetic_invalid_arguments_are_not_repaired_and_never_reach_parser(tmp_path,failure):
    arguments=deepcopy(failure['arguments'])
    assert Draft202012Validator(failure['original_parameters']).is_valid(arguments)
    corrected=declaration()['parameters']
    Draft202012Validator.check_schema(corrected)
    assert not Draft202012Validator(corrected).is_valid(arguments)
    store=MetaConversation(tmp_path,'protected task',Budget(),lambda *_:None)
    messages=[dict(role='system',content='protected system'),dict(role='user',content='protected task'),
              dict(role='user',content='observed evidence')]
    before=deepcopy(messages)
    try:
        with patch.object(store.env,'_apply_edit',wraps=store.env._apply_edit) as parser:
            result=store.step(arguments,messages,None,{},4096)
            assert result.result.return_code==1 and result.result.stderr=='invalid_context_operation'
            assert not result.ctx_changed and messages==before and parser.call_count==0
        assert arguments==failure['arguments'] and store.env.n_ctx_syncs==0
    finally:store.close()


@pytest.mark.parametrize('op',['read','replace','append','splice'])
def test_schema_and_executor_agree_on_every_argument_key_combination(tmp_path,op):
    validator=Draft202012Validator(declaration()['parameters'])
    capability=ContextCapability(tmp_path/'sandbox',16000)
    values=dict(start=0,chars=32,text='REPLACEMENT',old='UNIQUE',new='UPDATED')
    keys=list(values)
    accepted=0
    for length in range(len(keys)+1):
        for subset in combinations(keys,length):
            capability.file.write_text('UNIQUE original evidence')
            arguments=dict(path='LIVE_CTX_MAIN.txt',op=op,**{k:values[k] for k in subset})
            result=asyncio.run(capability.exec(command=json.dumps(arguments),timeout_sec=1))
            assert validator.is_valid(arguments)==(result.return_code==0),arguments
            accepted+=result.return_code==0
    assert accepted==1


@pytest.mark.parametrize('arguments',[
    dict(op='read',path='LIVE_CTX_MAIN.txt',start=True,chars=2),
    dict(op='read',path='LIVE_CTX_MAIN.txt',start=-1,chars=2),
    dict(op='read',path='LIVE_CTX_MAIN.txt',start=0,chars=0),
    dict(op='read',path='LIVE_CTX_MAIN.txt',start=0,chars=513),
    dict(op='read',path='LIVE_CTX_MAIN.txt',start=0,chars='20'),
    dict(op='replace',path='LIVE_CTX_MAIN.txt',text=None),
    dict(op='replace',path='LIVE_CTX_MAIN.txt',text='ok',command='arbitrary shell'),
    dict(op='replace',path='../.env',text='ok'),
    dict(op='replace',path='/etc/passwd',text='ok'),
    dict(op='splice',path='LIVE_CTX_MAIN.txt',old='',new='ok'),
    dict(op='execute',path='LIVE_CTX_MAIN.txt',text='ok'),
    dict(path='LIVE_CTX_MAIN.txt',text='ok'),
    dict(op='append',text='ok'),
])
def test_invalid_shapes_types_and_capabilities_remain_denied(tmp_path,arguments):
    assert not Draft202012Validator(declaration()['parameters']).is_valid(arguments)
    capability=ContextCapability(tmp_path/'sandbox',16000)
    capability.file.write_text('KEEP')
    result=asyncio.run(capability.exec(command=json.dumps(arguments),timeout_sec=1))
    assert result.return_code==1 and capability.file.read_text()=='KEEP'


@pytest.mark.parametrize('before,op,expected',[
    ('ONE',dict(op='splice',old='absent',new='x'),'context_match_not_unique'),
    ('ONE ONE',dict(op='splice',old='ONE',new='x'),'context_match_not_unique'),
    ('ONE',dict(op='replace',text='x'*33),'context_file_too_large'),
    ('ONE',dict(op='append',text='x'*33),'context_file_too_large'),
])
def test_stateful_restrictions_are_separate_from_schema(tmp_path,before,op,expected):
    arguments=dict(path='LIVE_CTX_MAIN.txt',**op)
    assert Draft202012Validator(declaration()['parameters']).is_valid(arguments)
    capability=ContextCapability(tmp_path/'sandbox',32);capability.file.write_text(before)
    result=asyncio.run(capability.exec(command=json.dumps(arguments),timeout_sec=1))
    assert result.return_code==1 and result.stderr==expected and capability.file.read_text()==before


@pytest.mark.parametrize('op',['replace','append','splice'])
def test_empty_text_and_deletion_keep_existing_semantics(tmp_path,op):
    arguments=dict(path='LIVE_CTX_MAIN.txt',op=op,
                   **(dict(old='ONE',new='') if op=='splice' else dict(text='')))
    assert Draft202012Validator(declaration()['parameters']).is_valid(arguments)
    capability=ContextCapability(tmp_path/'sandbox',32);capability.file.write_text('ONE')
    result=asyncio.run(capability.exec(command=json.dumps(arguments),timeout_sec=1))
    assert result.return_code==0 and capability.file.read_text()==('ONE' if op=='append' else '')


@pytest.mark.parametrize('architecture',['meta_clm','meta_clm_graph'])
@pytest.mark.parametrize('provider',['gemini','terra'])
@pytest.mark.parametrize('op',['read','replace','append','splice'])
def test_all_advertised_operations_reach_same_official_semantics_and_next_request(tmp_path,db,architecture,provider,op):
    retained='[[CTX_TURN 1 role=system]]\nKEEP_CONTRACT_MARKER https://openalex.org/W1 SYNTHETIC retrieval anchor'
    arguments=dict(path='LIVE_CTX_MAIN.txt',op=op)
    if op=='read':arguments.update(start=0,chars=200)
    elif op in ('replace','append'):arguments['text']=('\n\n' if op=='append' else '')+retained
    else:arguments.update(old='OBSOLETE_EVIDENCE',new='KEEP_CONTRACT_MARKER')
    assert Draft202012Validator(declaration()['parameters']).is_valid(arguments)
    def script(body,n):
        if provider=='gemini':
            return response(name='context_file' if n==1 else 'EXPAND_REFERENCES',
                args=arguments if n==1 else dict(paper_id='W1'),n=n)
        return ('context_file',arguments) if n==1 else ('EXPAND_REFERENCES',dict(paper_id='W1'))
    base,wire=model(script) if provider=='gemini' else mocked(script)
    planner=MetaContextPlanner(base,limits=InferenceLimits(max_active_tokens=4096,max_input_tokens=8192))
    budget=Budget(max_actions=2,max_llm_calls=4,page_size=1)
    c=ControlledController('retrieval',SeededCorpus(fixture(tmp_path),[1]),db,architecture=architecture,
        planner=planner,budget=budget,output=str(tmp_path/'runs'))
    try:
        result=c.run()
        assert len(wire)==2 and result['actions']==2 and result['meta_common_limit_rollbacks']==0
        requests=[json.loads(p.read_text()) for p in sorted((c.directory/'context').glob('request-*.json'))]
        assert requests[0]['input']['messages'][:2]==requests[1]['input']['messages'][:2]
        assert all(m['role'] not in ('system','developer') for m in planner.messages[2:])
        assert result['context_edits']==(0 if op=='read' else 1)
        if op!='read':
            assert 'KEEP_CONTRACT_MARKER' in serialized(wire[1])
            edit=json.loads(next((c.directory/'context').glob('meta-edit-*.json')).read_text())
            assert edit['executor_accepted'] and edit['executor_return_code']==0 and edit['committed']
        assert 'https://openalex.org/W2' in result['inspected_ids']
        assert not [e for e in db.events(c.run_id) if e['kind']=='invalid_decision']
        # The new declaration is byte-identical as a schema across providers.
        transmitted=(wire[0]['config']['tools'][0]['function_declarations'] if provider=='gemini' else wire[0]['tools'])
        tool=next(t for t in transmitted if t['name']=='context_file')
        assert tool.get('parameters_json_schema',tool.get('parameters'))==declaration()['parameters']
        audit=audit_canary(c.directory)
        if op=='replace':
            assert audit['passed'] and audit['edit_proofs'][0]['accepted_acquisition_after_delivery_calls']==[2]
        if op=='read':assert not audit['passed']
        assert audit['planning_latency']['measured_calls']==2
        assert result['budget']==budget.__dict__
    finally:planner.close()


def test_executor_acceptance_is_distinct_from_synchronizer_commit(tmp_path):
    store=MetaConversation(tmp_path,'protected task',Budget(),lambda *_:None)
    messages=[dict(role='system',content='protected system'),dict(role='user',content='protected task'),
              dict(role='user',content='KEEP')]
    before=deepcopy(messages)
    try:
        result=store.step(dict(path='LIVE_CTX_MAIN.txt',op='replace',text='GROW '*1000),messages,None,{},64)
        record=store.edits[-1]
        assert record['executor_accepted'] and record['executor_return_code']==0
        assert not record['accepted'] and not record['committed'] and not result.ctx_changed
        assert store.env.n_ctx_rejected==1 and messages==before
    finally:store.close()


def test_scholarly_actions_and_original_five_are_unchanged():
    assert ARCHITECTURES==('standard','llm','llm_graph','clm','clm_graph')
    old=json.loads(FIXTURE.read_text())['failures'][0]
    trace=Path(old['source_trace'])
    if not trace.exists():pytest.skip('Private historical trace is optional; standalone fixture tests still run')
    historical=json.loads(trace.read_text())['input']['tools']
    assert [t for t in native_tools(ACTION_ARGUMENTS) if t['name']!='context_file']==[t for t in historical if t['name']!='context_file']


def test_real_sdk_preserves_new_contract_without_credentials_or_network():
    python=Path('~/gemini-test-env/bin/python').expanduser()
    if not python.exists():pytest.skip('Optional isolated Google SDK environment is absent')
    planner=GeminiADCPlanner(GeminiSettings(project='SYNTHETIC-PROJECT'),executor=lambda _:pytest.fail('No inference'))
    body=planner.body([dict(role='user',content='SYNTHETIC schema serialization')],native_tools(ACTION_ARGUMENTS))
    result=subprocess.run([str(python),'tests/fixtures/gemini_context_schema_probe.py'],input=json.dumps(body),
        capture_output=True,text=True,timeout=30,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','GEMINI_TEST_TOKENIZER_PYTHON':sys.executable})
    assert result.returncode==0,result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['generation_requests']==0 and receipt['network_blocked']
    assert receipt['schema_preserved'] and receipt['sdk_mock_http_calls']==1


@pytest.mark.parametrize('scenario',['success','acquisition_before_edit','no_delivery','not_committed',
    'executor_rejected','no_official_sync','not_model_generated','wrong_retained_title','invalid_acquisition'])
def test_canary_requires_all_stages_in_order(tmp_path,scenario):
    # Synthetic audit records exercise the success checker; no model is called.
    context=tmp_path/'context';context.mkdir()
    after=[dict(role='system',content='protected'),dict(role='user',content='task'),
           dict(role='user',content='https://openalex.org/W1 SYNTHETIC observed title')]
    arguments=dict(path='LIVE_CTX_MAIN.txt',op='replace',text=after[-1]['content'])
    edit=dict(index=0,source='model',operation=arguments,accepted=True,executor_accepted=True,
              executor_return_code=0,committed=True,reference_syncs=1,after=deepcopy(after),usage=dict(call_index=2))
    if scenario=='not_committed':edit['committed']=False
    if scenario=='executor_rejected':edit['executor_accepted']=False
    if scenario=='no_official_sync':edit['reference_syncs']=0
    if scenario=='not_model_generated':edit['source']='scripted'
    if scenario=='wrong_retained_title':edit['after'][-1]['content']='https://openalex.org/W1 WRONG TITLE'
    (context/'meta-edit-00000.json').write_text(json.dumps(edit))
    observation=dict(top_papers=[dict(id='https://openalex.org/W1',title='SYNTHETIC observed title')])
    (context/'observation-00000.json').write_text(json.dumps(observation))
    acquisition=dict(decision=dict(action='EXPAND_REFERENCES',arguments=dict(paper_id='W1')))
    requests=[dict(call=1,input=dict(messages=[]),output=acquisition,diagnostics=dict(http_status=200,seconds=1)),
              dict(call=2,input=dict(messages=[]),output=dict(context_edit=arguments),
                   native_output=dict(name='context_file',arguments=arguments),diagnostics=dict(http_status=200,seconds=2)),
              dict(call=3,input=dict(messages=deepcopy(edit['after'])),output=acquisition,
                   diagnostics=dict(http_status=200,seconds=3))]
    if scenario=='acquisition_before_edit':requests[-1]['output']=dict(read_context=dict(op='read'))
    if scenario=='no_delivery':requests[-1]['diagnostics']['http_status']=503
    for request in requests:
        (context/f'request-{request["call"]:05d}.json').write_text(json.dumps(request))
    events=[dict(kind='context_model_call',call=3)]
    if scenario=='invalid_acquisition':events.append(dict(kind='invalid_decision'))
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
    result=audit_canary(tmp_path)
    assert result['passed']==(scenario=='success')
    if scenario=='success':
        assert result['edit_proofs'][0]['accepted_acquisition_after_delivery_calls']==[3]
    assert result['planning_latency']['total_seconds']==6


def test_previous_canary_approval_cannot_authorize_corrected_protocol(tmp_path):
    from openalex_agent.gemini_validation import GeminiValidationBackend
    from openalex_agent.meta_validation import execute
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(protocol='gemini-3.8-adc-development-validation-v1',mode='canary')))
    approval=tmp_path/'approval.json';approval.write_text(json.dumps(dict(approved=True,scope='canary_development_only')))
    with pytest.raises(ValueError,match='Fresh explicit'):
        execute(spec,approval,GeminiSettings(project='SYNTHETIC'),backend=GeminiValidationBackend(),
                transport=lambda _:pytest.fail('No inference is authorized'))
