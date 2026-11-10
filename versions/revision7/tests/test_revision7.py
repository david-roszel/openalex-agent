"""R7 measurement corrections; synthetic responses are never scientific results."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import json
import pytest
import jsonschema
from openalex_agent.memory_scoring_v3 import score_response
from openalex_agent.multihorizon_metrics import score_memory,comparisons
from openalex_agent.revision7_measurements import final_outcome,visible_records
from openalex_agent.terra_accounting import TerraLedger
from openalex_agent.execution_control import ExperimentStopped
from openalex_agent.models import Paper

ROOT=Path(__file__).resolve().parents[1]
GOLD={'q1':{'work_id':'W1'},'q2':{'year':2020},'q3':{'work_id':'W3'},'q4':{'citing_work_id':'W3','cited_work_id':'W1'},'q5':{'path':['W5','W3','W1']},'q6':{'status':'SUPPORTED'}}


@pytest.mark.parametrize('empty,primary',[([],1),(['q6'],1),(['q1'],.8),(['q1','q2'],.6),(list(GOLD),0)])
def test_item_abstention_never_erases_other_credit(empty,primary):
    answer=deepcopy(GOLD)
    for q in empty:answer[q]=None
    score=score_response(answer,GOLD)
    assert score['primary_score']==primary and score['valid_final_response']
    assert score['item_counts']['abstained']==len(empty)
    schema=json.loads((ROOT/'protocols/OpenAlex_Memory_Answer_Schema_V3_R7.json').read_text())
    jsonschema.validate(answer,schema)


@pytest.mark.parametrize('value',[{'work_id':None},None])
def test_both_advertised_abstention_forms_are_valid(value):
    answer=deepcopy(GOLD);answer['q1']=value
    assert score_response(answer,GOLD)['primary_score']==.8


@pytest.mark.parametrize('q,value,primary',[('q1',{'bad':'W1'},.8),('q2',{'year':True},.8),('q6',{'status':'INVALID'},1),('q4',{'citing_work_id':'W1','cited_work_id':'W3'},.8)])
def test_malformed_or_wrong_items_remain_independent(q,value,primary):
    answer=deepcopy(GOLD);answer[q]=value
    score=score_response(answer,GOLD)
    assert score['primary_score']==primary
    assert score['item_status'][q] in ('malformed','incorrect')
    result=score_memory(answer,GOLD,policy='r7')
    assert result['operational_itt']==primary # no second whole-response erasure


def test_parseable_missing_items_and_missing_whole_response_differ():
    answer=deepcopy(GOLD);answer.pop('q1')
    assert score_response(answer,GOLD)['primary_score']==.8
    assert score_response(answer,GOLD)['item_status']['q1']=='missing'
    for response in [None,'not JSON',[]]:
        score=score_response(response,GOLD);assert score['primary_score']==0 and score['terminal_failure']
    assert score_memory(None,GOLD,attempted=False,policy='r7')['primary_score'] is None


def test_required_final_failure_retains_raw_retrieval_but_not_operational_success():
    result=final_outcome('llm',1,[{'status':'missing'}])
    assert result['collection_status']=='COMPLETED' and result['status']=='FAILED'
    assert result['raw_primary_score']==1 and result['operational_itt_score']==0
    assert final_outcome('standard',1,[])['status']=='COMPLETED'
    assert final_outcome('clm',1,[{'status':'valid','response':{'abstain':True}}])['status']=='COMPLETED'


def test_partial_M_remains_scored_in_factorial_analysis():
    rows=[{'study':'M','task_id':'fixture','domain':'synthetic','method':m,'repetition':rep,'status':'COMPLETED','attempted':True,'primary_score':.8,'response_contract_valid':False} for m in ['llm','llm_graph','clm','clm_graph'] for rep in [0,1]]
    effects=[r for r in comparisons(rows,samples=20) if r['study']=='M' and r['policy']=='operational_itt']
    assert all(r['tasks']==1 and r['mean']==0 for r in effects)


def test_observed_direction_needs_source_provenance_and_actual_reference():
    p=Paper.from_dict({'id':'https://openalex.org/W1','references':['https://openalex.org/W2'],'metadata_source':'openalex','metadata_fields':['references']})
    papers={p.id:p}
    assert visible_records({'observation':{'observed_metadata':[{'id':p.id,'references':['W2']}]}},papers)[1]=={(p.id,'https://openalex.org/W2')}
    assert not visible_records({'protected':{'proposals':[{'id':p.id,'references':['W2']}]},'observation':{'id':p.id,'references':['W3']}},papers)[1]


def ledger(path):
    l=TerraLedger(path,profile_hash='SYNTHETIC_R7');l.activate('OFFLINE_ONLY_NOT_APPROVAL',max_runtime_seconds=3600);return l


def test_subceiling_counts_all_selected_stages_and_survives_reopen(tmp_path):
    l=ledger(tmp_path);l.register_scope('pilot',ceiling_usd='0.05',job_ids=['S','L','M'],attempt_limit=10,policy_sha256='fixture')
    i=l.reserve('S1','S:main','llm',b'{}');l.settle(i,{'prompt_tokens':4098,'completion_tokens':4096})
    assert Decimal(json.loads((tmp_path/'spending-ledger.json').read_text())['retained_usd'])<Decimal('.05')
    reopened=TerraLedger(tmp_path,profile_hash='SYNTHETIC_R7')
    with pytest.raises(ExperimentStopped,match='subceiling'):reopened.reserve('M1','M:cp24:A','clm',b'{}')
    # Unselected main does not receive a new allowance; it still uses the same68 ledger.
    n=reopened.reserve('MAIN1','OTHER:main','llm',b'{}');reopened.settle(n,{'prompt_tokens':100,'completion_tokens':20})
    assert len(json.loads((tmp_path/'spending-ledger.json').read_text())['attempts'])==2
    with pytest.raises(ExperimentStopped,match='immutable'):l.register_scope('pilot',ceiling_usd='1',job_ids=['S','L','M'],attempt_limit=10,policy_sha256='fixture')


def test_pilot_attempt_ceiling_and_unknown_usage_are_not_reset(tmp_path):
    l=ledger(tmp_path);l.register_scope('pilot',ceiling_usd='10',job_ids=['CELL'],attempt_limit=1,policy_sha256='fixture')
    i=l.reserve('once','CELL:stream','clm',b'{}');l.settle(i,{'prompt_tokens':100,'completion_tokens':20})
    with pytest.raises(ExperimentStopped,match='attempt_limit'):l.reserve('twice','CELL:cp96:A','clm',b'{}')
    j=l.reserve('unknown','OTHER','llm',b'{}')
    with pytest.raises(ExperimentStopped,match='unaccountable'):l.settle(j,{})
    with pytest.raises(ExperimentStopped):l.reserve('after','CELL','llm',b'{}')


def test_concurrent_workers_cannot_overreserve_the_pilot_subceiling(tmp_path):
    import multiprocessing as mp
    from test_multihorizon_supervisor import setup,reserve_together
    out=tmp_path/'scope';l=setup(out,0);l.session_id='scope-test';l.begin_session(2)
    l.register_scope('pilot',ceiling_usd='.05',job_ids=['0','1'],attempt_limit=10,policy_sha256='fixture')
    ctx=mp.get_context('spawn');ready=ctx.Queue();release=ctx.Event()
    children=[ctx.Process(target=reserve_together,args=(out,'scope-test',str(i),ready,release)) for i in range(2)]
    for p in children:p.start()
    values=[ready.get(timeout=20) for p in children]
    assert sorted(v[0] for v in values)==['blocked','reserved']
    assert any(v[1]=='pilot_spending_subceiling' for v in values if v[0]=='blocked')
    assert Decimal(json.loads((out/'money/spending-ledger.json').read_text())['retained_usd'])<=Decimal('.05')
    release.set()
    for p in children:p.join(20);assert p.exitcode==0
    l.end_session()


def test_subset_claims_remain_unique_and_cannot_consume_unselected_work(tmp_path):
    from test_multihorizon_supervisor import setup
    from openalex_agent.multihorizon_supervisor import claim,finish
    from openalex_agent.multihorizon import queue
    out=tmp_path/'claims';setup(out,4);allowed={'OFFLINE-1','OFFLINE-3'}
    a=claim(out,'main-S','s','w1',2,'snapshot',allowed_ids=allowed)
    b=claim(out,'main-S','s','w2',2,'snapshot',allowed_ids=allowed)
    assert {a['id'],b['id']}==allowed
    finish(out,a['id'],'s','w1','COMPLETED');finish(out,b['id'],'s','w2','FAILED')
    assert claim(out,'main-S','resumed','w',1,'snapshot',allowed_ids=allowed) is None
    with queue(out) as db:assert {r['id'] for r in db.execute("SELECT id FROM jobs WHERE state='PENDING'")}=={'OFFLINE-0','OFFLINE-2'}
