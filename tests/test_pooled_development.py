"""Synthetic offline integration only: no human labels or approval files created."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import pytest

from openalex_agent import pooled_review as pooled
from openalex_agent.corpus import atomic_json, freeze
from openalex_agent.execution_control import ExperimentStopped
from openalex_agent.gemini import GeminiSettings
from openalex_agent.gemini_validation import GeminiSpendLedger, GeminiValidationBackend
from openalex_agent.models import Paper
from openalex_agent.research_completion import VERSION as COMPLETION_VERSION, SUFFIX
from openalex_agent.research_assessment import build_claim_pool
from openalex_agent.experiment_integrity import validate_schedule
from test_gemini_adc import response


def proposal(tmp_path, *, facts=False, guidance=COMPLETION_VERSION):
    corpus=tmp_path/'corpus'
    freeze([Paper(id='https://openalex.org/W1',title='SYNTHETIC retrieval evidence',abstract='SYNTHETIC evidence text.')],
        corpus,acquisition=[{'source':'SYNTHETIC UNIT TEST'}])
    questions=tmp_path/'questions.json'
    atomic_json(questions,[dict(id='SYNTHETIC-DEV',question='SYNTHETIC retrieval evidence',split='development')])
    factual=tmp_path/'facts.json'
    if facts:atomic_json(factual,[dict(query_id='SYNTHETIC-DEV',question_id='SYNTHETIC-FACT',question='What evidence was delivered?')])
    out=tmp_path/'proposal'
    result=pooled.prepare_development(corpus,questions,out,GeminiSettings(project='SYNTHETIC-PROJECT'),
        factual_questions=factual if facts else None,completion_guidance=guidance,max_llm_calls=2 if facts else 1)
    assert result['configuration_valid'] and not result['question_review_ready']
    return out


def simulated_authorization(out,monkeypatch,**overrides):
    """Exercise real approval validation through in-memory I/O, never an approval file.

    The independent-review gate is simulated for execution tests; the separate
    unreviewed test exercises its actual fail-closed behavior.
    """
    monkeypatch.setattr(pooled,'development_review_ready',lambda *_:True)
    spec=out/'spec.json';review=out/'question-review.template.csv'
    checked=pooled.check_development(spec,review)
    record=dict(protocol=pooled.VERSION,stage='collect-development',approved=True,
        approver='SYNTHETIC IN-MEMORY TEST',approved_at='SYNTHETIC TEST',pins=checked['pins'],
        development_only=True,paid_inference_authorized=True,final_evaluation_authorized=False)
    record.update(overrides)
    path=out/'never-written-authorization.json';read=Path.read_text;sha=pooled.sha
    def read_text(self,*args,**kwargs):
        return json.dumps(record) if self==path else read(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_text)
    monkeypatch.setattr(pooled,'sha',lambda p:pooled.digest(record) if Path(p)==path else sha(p))
    assert not path.exists()
    return path


def test_preparation_and_check_are_offline_unapproved_and_pin_all_identities(tmp_path,monkeypatch):
    monkeypatch.setattr(GeminiValidationBackend,'guarded_factory',lambda *a,**k:pytest.fail('No provider during preparation'))
    out=proposal(tmp_path,facts=True)
    spec=json.loads((out/'spec.json').read_text());checked=pooled.check_development(out/'spec.json')
    assert spec['methods']==list(pooled.METHODS) and spec['suite']=='controlled-five'
    assert spec['clm_policy']=='controlled-context-v2-capacity-stop'
    assert spec['policy']['k']==10 and spec['policy']['gains']=='2**grade-1'
    assert spec['source_pins'] and spec['corpus_manifest_sha256'] and spec['git']['status']
    assert {'spec_sha256','source_sha256','questions_sha256','corpus_sha256','provider_profile_sha256',
        'prompt_sha256','budget_sha256','policy_sha256','factual_questions_sha256'}<=checked['pins'].keys()
    assert checked['approval_required'] and checked['hosted_calls']==0 and checked['planned_trajectories']==5
    assert not (out/'execution').exists() and not list(out.glob('*approval*'))
    assert all(r['validity']=='UNKNOWN' for r in pooled.csv_rows(out/'question-review.template.csv'))


@pytest.mark.parametrize('change',['method','clm','provider','policy','questions','facts','source','budget','guidance'])
def test_changed_protocol_or_inputs_fail_offline(tmp_path,change):
    out=proposal(tmp_path,facts=True);path=out/'spec.json';spec=json.loads(path.read_text())
    if change=='method':spec['methods'][-1]='meta_clm'
    elif change=='clm':spec['clm_policy']='historical'
    elif change=='provider':spec['provider']['thinking_level']='HIGH'
    elif change=='policy':spec['policy']['k']=20
    elif change=='questions':
        q=json.loads(Path(spec['questions']).read_text());q[0]['split']='heldout';atomic_json(Path(spec['questions']),q)
    elif change=='facts':
        q=json.loads(Path(spec['factual_questions']).read_text());q[0]['verified_answer']='SYNTHETIC FORBIDDEN FIELD';atomic_json(Path(spec['factual_questions']),q)
    elif change=='source':spec['source_pins']={}
    elif change=='budget':spec['budget']['max_llm_calls']=21
    else:spec['prompt']['completion_suffix']='UNAPPROVED DIFFERENT INSTRUCTION'
    atomic_json(path,spec)
    with pytest.raises(ValueError):pooled.check_development(path)


def test_unreviewed_development_is_rejected_before_approval_or_provider(tmp_path,monkeypatch):
    out=proposal(tmp_path)
    monkeypatch.setattr(pooled,'approval',lambda *a:pytest.fail('Review gate must precede approval'))
    with pytest.raises(ValueError,match='review is incomplete'):
        pooled.collect_development(out/'spec.json',out/'question-review.template.csv',out/'missing.json')
    assert not (out/'execution').exists()


@pytest.mark.parametrize('overrides',[
    {'approved':False},{'stage':'collect-unscored'},{'paid_inference_authorized':False},
    {'development_only':False},{'final_evaluation_authorized':True},{'pins':{}},
])
def test_wrong_or_missing_paid_approval_cannot_construct_a_provider(tmp_path,monkeypatch,overrides):
    out=proposal(tmp_path);auth=simulated_authorization(out,monkeypatch,**overrides)
    monkeypatch.setattr(GeminiValidationBackend,'guarded_factory',lambda *a,**k:pytest.fail('No provider without exact approval'))
    with pytest.raises(ValueError):
        pooled.collect_development(out/'spec.json',out/'question-review.template.csv',auth)
    assert not (out/'execution').exists() and not auth.exists()


@pytest.mark.parametrize('facts,guidance',[(False,None),(True,COMPLETION_VERSION)])
def test_native_five_method_outputs_keep_parity_factual_opportunities_and_pool_compatibility(tmp_path,monkeypatch,facts,guidance):
    out=proposal(tmp_path,facts=facts,guidance=guidance);auth=simulated_authorization(out,monkeypatch)
    calls=[]
    def script(body):
        calls.append(deepcopy(body));payload=json.loads(body['contents'][0]['parts'][0]['text'])
        assert 'tools' not in body['config'] and body['model']=='gemini-3.8-flash'
        assert body['config']['thinking_config']=={'thinking_level':'LOW'}
        assert body['config']['max_output_tokens']==4096
        value=(dict(question_id='SYNTHETIC-FACT',answer='',work_id=None,evidence='',sufficiency='insufficient',
            abstain=True,abstention_reason='SYNTHETIC MOCK; no factual assessment') if 'factual_task' in payload else
            dict(decision=dict(action='STOP_RETRIEVAL',arguments={})))
        return response(text=json.dumps(value),n=len(calls))
    receipt=pooled.collect_development(out/'spec.json',out/'question-review.template.csv',auth,transport=script)
    root=out/'execution/benchmark';manifest=json.loads((root/'experiment.json').read_text());rows=pooled.csv_rows(root/'metrics.csv')
    validate_schedule(manifest,[{**r,'trial':int(r['trial']),'action_budget':int(r['action_budget'])} for r in rows])
    assert receipt['status']=='complete' and receipt['classification']=='MOCKED OFFLINE TEST'
    assert receipt['provider_attempts']==len(calls)==(8 if facts else 4)
    assert set(r['method'] for r in rows)==set(pooled.METHODS)
    assert manifest['legacy_context_policy_version']==pooled.DEVELOPMENT_CLM_POLICY
    assert manifest['research_completion_guidance']['enabled']==bool(guidance)
    assert manifest['has_relevance_judgments'] is False and (root/'spending-ledger.json').is_file()
    for row in rows:
        result=json.loads((Path(row['directory'])/'result.json').read_text())
        if row['method']=='standard':
            assert row['llm_calls']=='0' and row['actions']=='1' and not result.get('factual_responses')
            continue
        assert result['context_policy_version']==pooled.DEVELOPMENT_CLM_POLICY
        assert int(row['llm_calls'])<=int(manifest['base_budget']['max_llm_calls'])
        requests=[json.loads(p.read_text()) for p in sorted((Path(row['directory'])/'context').glob('request-*.json'))]
        assert requests[0]['system'].endswith(SUFFIX)==bool(guidance)
        if facts:
            assert row['factual_measurement_calls']=='1' and result['total_model_call_ceiling']==2
            assert result['factual_responses'][0]['status']=='valid'
    # Same pool preparation format and claim assessment interface; all labels
    # stay blank/UNKNOWN. No relevance or factual grades are manufactured.
    pooled.prepare_pool(tmp_path/'corpus',out/'questions.json',tmp_path/'pool')
    assert all(r['grade']=='UNKNOWN' for r in pooled.csv_rows(tmp_path/'pool/reviewer-A/relevance.csv'))
    build_claim_pool(root,tmp_path/'blank-claims.csv')
    assert all(not r['judgment'] for r in pooled.csv_rows(tmp_path/'blank-claims.csv'))
    with pytest.raises(ValueError):
        pooled.collect_development(out/'spec.json',out/'question-review.template.csv',auth,transport=script)
    assert len(calls)==receipt['provider_attempts'] and not auth.exists()


@pytest.mark.parametrize('failure',['unknown','checkpoint'])
def test_accounting_stops_immediately_and_preserves_partial_native_outputs(tmp_path,monkeypatch,failure):
    out=proposal(tmp_path);auth=simulated_authorization(out,monkeypatch);calls=[]
    def script(body):
        calls.append(body)
        usage=(dict(prompt_token_count=8,candidates_token_count=1) if failure=='unknown' else
            dict(prompt_token_count=1_000_000,candidates_token_count=1,thoughts_token_count=0,total_token_count=1_000_001))
        return response(text='{"decision":{"action":"STOP_RETRIEVAL","arguments":{}}}',usage=usage)
    result=pooled.collect_development(out/'spec.json',out/'question-review.template.csv',auth,transport=script)
    assert len(calls)==(1 if failure=='unknown' else 2)
    assert result['status']==('development_unaccountable_usage' if failure=='unknown' else 'pilot_three_dollar_checkpoint')
    ledger=json.loads((out/'execution/spending-ledger.json').read_text())
    if failure=='unknown':assert ledger['attempts'][0]['retained_usd']=='2.13072'
    else:assert Decimal(ledger['reported_usage_cost_usd'])>=3
    root=out/'execution/benchmark'
    assert (root/'development-collection.json').exists() and (root/'spending-ledger.json').exists()
    assert json.loads((root/'experiment.json').read_text())['status']!='complete'
    assert list((root/'runs').glob('*/context/request-*.json'))


@pytest.mark.parametrize('guard',['reservation','calls','runtime'])
def test_reused_gemini_guard_blocks_requests_before_transport(tmp_path,guard):
    now=[0.0];calls=[]
    ledger=GeminiSpendLedger(tmp_path/'ledger.json',3,max_calls=1 if guard=='calls' else 20,
        checkpoint=3,max_seconds=2,clock=lambda:now[0])
    def script(body):
        calls.append(body)
        usage=dict(prompt_token_count=1_000_000 if guard=='reservation' else 8,candidates_token_count=1,
            thoughts_token_count=0,total_token_count=1_000_001 if guard=='reservation' else 9)
        return response(text='{}',usage=usage)
    planner=GeminiValidationBackend.guarded_factory(GeminiSettings(project='SYNTHETIC-PROJECT'),ledger,transport=script)()
    if guard=='runtime':now[0]=3
    else:planner._call('SYNTHETIC',{})
    with pytest.raises(ExperimentStopped):planner._call('SYNTHETIC',{})
    assert len(calls)==(0 if guard=='runtime' else 1)


def test_existing_final_collection_still_rejects_development_split(tmp_path,monkeypatch):
    from openalex_agent.config import Settings
    questions=[dict(id=f'SYNTHETIC-{i}',question=f'SYNTHETIC {i}',split='development') for i in range(10)]
    atomic_json(tmp_path/'questions.json',questions)
    atomic_json(tmp_path/'spec.json',dict(corpus='unused',questions=str(tmp_path/'questions.json'),output='unused',
        action_budgets=[3],trials=3,trial_seeds=[11,23,37],budget={},inference_limits={},max_total_cost_usd=6))
    monkeypatch.setattr(pooled,'csv_rows',lambda _: [dict(query_id=q['id'],question=q['question'],validity='approved',
        taxonomy_verified='true',source_adequacy='sufficient',independent_human_review='true',
        reviewer_id='SYNTHETIC MOCK',reviewed_at='SYNTHETIC',explanation='SYNTHETIC') for q in questions])
    with pytest.raises(ValueError,match='held-out'):
        pooled.collect_unscored(tmp_path/'spec.json','in-memory-review','no-approval',Settings())
