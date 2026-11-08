"""ADC validation runner uses mocked inference and the existing spending guard."""
import hashlib
import json
from pathlib import Path

import pytest

from openalex_agent.corpus import freeze
from openalex_agent.development_pilot import REQUEST_RESERVE
from openalex_agent.execution_control import ExperimentStopped
from openalex_agent.gemini import GeminiSettings
from openalex_agent.gemini_validation import GeminiSpendLedger, GeminiValidationBackend
from openalex_agent.meta_validation import prepare,execute
from openalex_agent.models import Paper
from test_gemini_adc import response


def inputs(tmp_path,monkeypatch):
    path=tmp_path/'corpus'
    freeze([Paper.from_openalex({'id':'W1','title':'SYNTHETIC retrieval anchor','referenced_works':['W2']}),
            Paper.from_openalex({'id':'W2','title':'SYNTHETIC reference'})],path,acquisition=[])
    checksum=hashlib.sha256((path/'papers.jsonl').read_bytes()).hexdigest()
    (path/'freeze-receipt.json').write_text(json.dumps({'corpus_sha256':checksum}))
    monkeypatch.setattr('openalex_agent.meta_validation.EXPECTED_CORPUS_SHA',checksum)
    questions=tmp_path/'questions.json'
    questions.write_text(json.dumps([dict(id=f'SYNTHETIC-{i}',question='retrieval',split='development') for i in range(3)]))
    return path,questions


def test_gemini_prepared_canary_has_no_authorization_and_makes_no_request(tmp_path,monkeypatch):
    corpus,questions=inputs(tmp_path,monkeypatch);out=tmp_path/'proposal';settings=GeminiSettings(project='SYNTHETIC-PROJECT')
    backend=GeminiValidationBackend()
    spec=prepare(out,settings,corpus=str(corpus),questions=str(questions),backend=backend)
    assert spec['protocol']==backend.protocol and spec['provider']['auth']=='ADC'
    assert spec['request_reservation_usd']==str(REQUEST_RESERVE) and spec['spending_ceiling_usd']==3
    with pytest.raises(ValueError,match='explicit'):
        execute(out/'spec.json',out/'approval-template.json',settings,backend=backend,transport=lambda _:pytest.fail('No inference'))
    assert not (out/'execution').exists()


def test_mocked_gemini_canary_uses_existing_workload_and_full_reasoning_accounting(tmp_path,monkeypatch):
    corpus,questions=inputs(tmp_path,monkeypatch);out=tmp_path/'proposal';settings=GeminiSettings(project='SYNTHETIC-PROJECT')
    backend=GeminiValidationBackend()
    spec=prepare(out,settings,corpus=str(corpus),questions=str(questions),backend=backend)
    approval=json.loads((out/'approval-template.json').read_text())
    approval.update(approved=True,approver='SYNTHETIC UNIT TEST',approved_at='SYNTHETIC UNIT TEST')
    (out/'test-approval.json').write_text(json.dumps(approval))
    calls=[]
    def mock(body):
        calls.append(body);n=len(calls)
        if n%2:return response(name='context_file',args=dict(path='LIVE_CTX_MAIN.txt',op='replace',
            text='[[CTX_TURN 1 role=user]]\nhttps://openalex.org/W1 SYNTHETIC retrieval anchor'),n=n)
        return response(name='EXPAND_REFERENCES',args={'paper_id':'W1'},n=n)
    result=execute(out/'spec.json',out/'test-approval.json',settings,backend=backend,transport=mock)
    assert len(calls)==result['provider_attempts']==4 and result['completed_cells']==2
    assert result['classification']=='MOCKED PROVIDER TEST' and result['canary_mechanism_passed']
    ledger=json.loads((out/'execution/spending-ledger.json').read_text())
    assert all(a['completion_tokens']==100 and a['reasoning_tokens']==95 for a in ledger['attempts'])
    assert float(ledger['reported_usage_cost_usd'])==pytest.approx((32*1.5+400*7.5)/1e6)
    assert ledger['cloud_trial_balance_independently_verified'] is False


def test_gemini_unknown_thinking_usage_retains_full_reservation_and_blocks_next_call(tmp_path):
    ledger=GeminiSpendLedger(tmp_path/'ledger.json',6,checkpoint=3)
    calls=[]
    def mock(body):
        calls.append(1)
        return response(text='{}',usage=dict(prompt_token_count=8,candidates_token_count=5))
    p=GeminiValidationBackend.guarded_factory(GeminiSettings(project='SYNTHETIC-PROJECT'),ledger,transport=mock)()
    p._call('Return JSON',{})
    assert ledger.charged==REQUEST_RESERVE and p.usage.estimated_cost_usd is None
    with pytest.raises(ExperimentStopped,match='unknown_cost'):p._call('Return JSON',{})
    assert len(calls)==1


def test_gemini_three_dollar_checkpoint_stops_new_requests_without_trial_discount(tmp_path):
    ledger=GeminiSpendLedger(tmp_path/'ledger.json',6,checkpoint=3)
    for i in range(2):
        index=ledger.reserve()
        ledger.settle(index,dict(http_status=200,prompt_tokens=1_000_000,completion_tokens=4000,reasoning_tokens=3995,visible_output_tokens=5))
    assert float(ledger.reported_spend)==pytest.approx(3.06)
    with pytest.raises(ExperimentStopped,match='three_dollar_checkpoint'):ledger.reserve()
    assert len(ledger.attempts)==2
