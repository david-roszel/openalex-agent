"""Only mocked-provider development-runner checks; zero hosted spending."""
import hashlib
import json

import httpx
import pytest

from openalex_agent.config import Settings
from openalex_agent.corpus import freeze
from openalex_agent.development_pilot import SpendLedger,REQUEST_RESERVE
from openalex_agent.execution_control import ExperimentStopped
from openalex_agent.meta_validation import prepare,execute,guarded_factory
from openalex_agent.models import Paper


def inputs(tmp_path,monkeypatch):
    corpus=tmp_path/'corpus';freeze([Paper.from_openalex({'id':'W1','title':'SYNTHETIC retrieval anchor','referenced_works':['W2']}),
        Paper.from_openalex({'id':'W2','title':'unrelated example'})],corpus,acquisition=[])
    checksum=hashlib.sha256((corpus/'papers.jsonl').read_bytes()).hexdigest()
    (corpus/'freeze-receipt.json').write_text(json.dumps(dict(corpus_sha256=checksum)))
    monkeypatch.setattr('openalex_agent.meta_validation.EXPECTED_CORPUS_SHA',checksum)
    questions=tmp_path/'questions.json'
    questions.write_text(json.dumps([dict(id=f'SYNTHETIC-{i}',question='retrieval',split='development') for i in range(3)]))
    settings=Settings(llm_base_url='https://api.openai.com/v1/',llm_api_key='SYNTHETIC-PRIVATE',llm_model='gpt-6.1-terra',
        llm_reasoning_effort='low',llm_temperature=None,llm_max_tokens=4096,llm_timeout=120)
    proposal=tmp_path/'proposal';spec=prepare(proposal,settings,corpus=str(corpus),questions=str(questions))
    return settings,proposal,spec


def test_preparation_is_unapproved_and_refuses_to_call_provider(tmp_path,monkeypatch):
    settings,path,spec=inputs(tmp_path,monkeypatch)
    assert spec['spending_ceiling_usd']==3 and spec['max_provider_calls']==8
    assert spec['request_reservation_usd']==str(REQUEST_RESERVE)
    with pytest.raises(ValueError,match='explicit'):
        execute(path/'spec.json',path/'approval-template.json',settings,transport=httpx.MockTransport(lambda _:pytest.fail('No request')))
    assert not (path/'execution').exists()


def test_mocked_canary_reuses_guard_for_native_calls_and_preserves_usage(tmp_path,monkeypatch):
    settings,path,spec=inputs(tmp_path,monkeypatch)
    approval=json.loads((path/'approval-template.json').read_text())
    approval.update(approved=True,approver='SYNTHETIC UNIT TEST ONLY',approved_at='SYNTHETIC UNIT TEST ONLY')
    (path/'test-approval.json').write_text(json.dumps(approval))
    calls=[]
    def handler(request):
        body=json.loads(request.content);calls.append(body)
        if len(calls)%2:
            name,args='context_file',dict(path='LIVE_CTX_MAIN.txt',op='replace',text='[[CTX_TURN 1 role=user]]\nSYNTHETIC retained observation')
        else:name,args='EXPAND_REFERENCES',dict(paper_id='W1')
        return httpx.Response(200,json=dict(status='completed',output=[dict(type='function_call',name=name,call_id=f'c{len(calls)}',arguments=json.dumps(args))],
            usage=dict(input_tokens=100,output_tokens=30)))
    result=execute(path/'spec.json',path/'test-approval.json',settings,transport=httpx.MockTransport(handler))
    assert result['status']=='complete' and result['completed_cells']==2
    assert result['provider_attempts']==len(calls)==4 and result['classification']=='MOCKED PROVIDER TEST'
    ledger=json.loads((path/'execution/spending-ledger.json').read_text())
    assert len(ledger['attempts'])==4 and all(a['usage']=='provider_reported' for a in ledger['attempts'])
    assert ledger['request_reserve_usd']==str(REQUEST_RESERVE) and ledger['checkpoint_usd']=='3'
    assert 'SYNTHETIC-PRIVATE' not in (path/'execution/spending-ledger.json').read_text()


def test_native_guard_keeps_unknown_hold_and_stops_before_any_additional_request(tmp_path):
    ledger=SpendLedger(tmp_path/'ledger.json',3,checkpoint=3,max_calls=8,max_seconds=600)
    settings=Settings(llm_base_url='https://api.openai.com/v1',llm_api_key='SYNTHETIC',llm_model='gpt-6.1-terra')
    calls=[]
    def handler(request):calls.append(1);return httpx.Response(503,json={})
    p=guarded_factory(settings,ledger,transport=httpx.MockTransport(handler))()
    try:
        with pytest.raises(ValueError):p._call_native([dict(role='user',content='SYNTHETIC')],[])
        with pytest.raises(ExperimentStopped,match='unknown_cost'):p._call_native([dict(role='user',content='SYNTHETIC')],[])
        assert len(calls)==1 and ledger.charged==REQUEST_RESERVE
    finally:p.close()
