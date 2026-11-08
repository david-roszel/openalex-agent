"""All approvals/grades here are temporary synthetic fixtures, never human ground truth."""
import csv
import json
from pathlib import Path

import pytest

from openalex_agent.corpus import atomic_json,freeze
from openalex_agent.models import Paper
from openalex_agent.pooled_review import (POLICY,VERSION,prepare_pool,import_review,resolve_reviews,
    freeze_outputs,score_pool,sha,digest,inventory,csv_rows,csv_write,load_pool,approval,collect_unscored)
from openalex_agent.config import Settings
from openalex_agent.evidence_visibility import audit_passage


def attest(path,reviewer):
    atomic_json(path,dict(reviewer_id=reviewer,reviewed_at='TEST FIXTURE',independent_human_review=True,not_model_generated=True))


def authorize(path,stage,pins):
    atomic_json(path,dict(protocol=VERSION,approved=True,approver='TEST FIXTURE',approved_at='TEST FIXTURE',stage=stage,pins=pins))


@pytest.fixture
def corpus(tmp_path):
    papers=[Paper.from_openalex(dict(id=f'W{i}',title=f'retrieval evidence {i}',abstract_inverted_index={'scientific':[0],'retrieval':[1],'evidence':[2]},referenced_works=[f'W{i+1}'] if i<12 else [])) for i in range(1,13)]
    freeze(papers,tmp_path/'corpus',acquisition=[{'source':'SYNTHETIC TEST'}])
    atomic_json(tmp_path/'questions.json',[dict(id='q1',question='retrieval evidence',split='development')])
    return tmp_path


def make_pool(root,*,frozen=None,name='pool'):
    prepare_pool(root/'corpus',root/'questions.json',root/name,frozen_outputs=frozen)
    return root/name


def reviewed(root,pool,*,unknown=False,disagree=False):
    paths=[]
    for role,reviewer in [('A','reviewer-one'),('B','reviewer-two')]:
        rows=csv_rows(pool/f'reviewer-{role}/relevance.csv')
        for n,row in enumerate(rows):
            row.update(grade='UNKNOWN' if unknown else '0' if disagree and role=='B' else '2',reviewer_id=reviewer,explanation='TEST FIXTURE')
        file=root/f'{role}.csv';csv_write(file,rows);attest(root/f'{role}.json',reviewer)
        out=root/f'import-{role}';import_review(pool,file,root/f'{role}.json',reviewer,role,out);paths.append(out)
    return paths


def fake_outputs(root):
    out=root/'experiment';out.mkdir();cells=[];rows=[]
    for method in ('standard','llm','llm_graph','clm','clm_graph'):
        cell=dict(query_id='q1',method=method,variant='full',trial=0,action_budget=1);cells.append(cell)
        run=out/method;run.mkdir()
        atomic_json(run/'result.json',dict(ranked=[{'paper':{'id':f'https://openalex.org/W{i}'}} for i in range(1,11)],inspected_ids=[f'https://openalex.org/W{i}' for i in range(1,11)]))
        rows.append({**cell,'directory':str(run)})
    c=json.loads((root/'corpus/manifest.json').read_text())
    atomic_json(out/'experiment.json',dict(suite='controlled-five',has_relevance_judgments=False,judgments=None,
        status='complete',planned_runs=5,run_order=cells,corpus=c['sha256'],corpus_manifest=c,
        questions_sha256=sha(root/'questions.json'),deterministic_smoke=True))
    csv_write(out/'metrics.csv',rows)
    authorize(root/'freeze-approval.json','freeze-unscored',dict(experiment_sha256=sha(out/'experiment.json'),metrics_sha256=sha(out/'metrics.csv'),raw_output_tree_sha256=digest(inventory(out)),policy_sha256=digest(POLICY)))
    freeze_outputs(out,root/'frozen',root/'freeze-approval.json')
    return root/'frozen'


def test_blinding_dedup_and_immutable_pool(corpus):
    p=make_pool(corpus);m,items=load_pool(p)
    assert len({(i['query_id'],i['paper_id']) for i in items})==len(items)==12
    a=csv_rows(p/'reviewer-A/relevance.csv')
    assert all(r['grade']=='UNKNOWN' for r in a)
    assert not {'method','rank','origin','double_review'}&set(a[0])
    assert len(csv_rows(p/'reviewer-B/relevance.csv'))==3
    (p/'questions.json').write_text('[]')
    with pytest.raises(ValueError,match='Immutable'):load_pool(p)


def test_unknown_and_disagreement_are_never_zero(corpus):
    p=make_pool(corpus);paths=reviewed(corpus,p,disagree=True)
    labels=resolve_reviews(p,paths,corpus/'annotations')
    assert labels['disagreements']==3 and len(labels['unknown_pairs'])==3
    assert all(r['grade']==2 for r in labels['judgments'])
    assert labels['raw_agreement']==0


def test_blank_reviews_remain_unknown(corpus):
    p=make_pool(corpus);paths=reviewed(corpus,p,unknown=True)
    labels=resolve_reviews(p,paths,corpus/'annotations')
    assert not labels['judgments'] and labels['coverage']==0 and len(labels['unknown_pairs'])==12


def test_independent_adjudication_is_versioned(corpus):
    p=make_pool(corpus);paths=reviewed(corpus,p,disagree=True)
    labels=resolve_reviews(p,paths,corpus/'before')
    rows=csv_rows(corpus/'before/adjudication-needed.csv')
    for r in rows:r.update(grade='1',reason='TEST FIXTURE RESOLUTION',reviewer_id='third')
    csv_write(corpus/'adjudicated.csv',rows);attest(corpus/'third.json','third')
    result=resolve_reviews(p,paths,corpus/'after',adjudication=corpus/'adjudicated.csv',attestation=corpus/'third.json',adjudicator='third')
    assert result['coverage']==1 and result['disagreements']==3 and result['adjudication_sha256']
    assert json.loads((corpus/'before/labels.json').read_text())==labels
    with pytest.raises(ValueError,match='separate adjudicator'):
        resolve_reviews(p,paths,corpus/'bad',adjudication=corpus/'adjudicated.csv',attestation=corpus/'third.json',adjudicator='reviewer-one')


def test_import_rejects_changed_source_and_missing_attestation(corpus):
    p=make_pool(corpus);rows=csv_rows(p/'reviewer-A/relevance.csv');rows[0]['paper_id']='https://openalex.org/W999'
    csv_write(corpus/'altered.csv',rows);attest(corpus/'attest.json','one')
    with pytest.raises(ValueError,match='immutable'):import_review(p,corpus/'altered.csv',corpus/'attest.json','one','A',corpus/'bad')
    atomic_json(corpus/'attest.json',{})
    with pytest.raises(ValueError,match='attestation'):import_review(p,p/'reviewer-A/relevance.csv',corpus/'attest.json','one','A',corpus/'bad')


@pytest.mark.parametrize('unknown',[True,False])
def test_preserved_scoring_coverage_and_no_source_mutation(corpus,unknown):
    f=fake_outputs(corpus);before=sha(f/'raw-output/metrics.csv')
    p=make_pool(corpus,frozen=f);paths=reviewed(corpus,p,unknown=unknown)
    labels=resolve_reviews(p,paths,corpus/'annotations')
    pins=dict(frozen_outputs_sha256=sha(f/'unscored-freeze.json'),pool_sha256=sha(p/'pool-manifest.json'),
              annotation_sha256=sha(corpus/'annotations/annotation-manifest.json'),policy_sha256=digest(POLICY))
    authorize(corpus/'score-approval.json','score-preserved',pins)
    rows=score_pool(f,p,corpus/'annotations',corpus/'score-approval.json',corpus/'scores')
    assert len(rows)==5 and sha(f/'raw-output/metrics.csv')==before
    assert all((r['precision_at_10'] is None)==unknown for r in rows)
    assert all((r['ndcg_at_10_pooled'] is None)==unknown for r in rows)
    assert all(r['judgment_coverage_at_10']==(0 if unknown else 1) for r in rows)


def test_preparation_pool_cannot_be_scored_as_final(corpus):
    f=fake_outputs(corpus);p=make_pool(corpus);paths=reviewed(corpus,p)
    resolve_reviews(p,paths,corpus/'annotations')
    with pytest.raises(ValueError,match='versions differ'):score_pool(f,p,corpus/'annotations',corpus/'not-approved.json',corpus/'scores')


def test_hosted_collection_rejects_unreviewed_before_provider(corpus,monkeypatch):
    p=make_pool(corpus)
    atomic_json(corpus/'spec.json',dict(corpus=str(corpus/'corpus'),questions=str(corpus/'questions.json'),output=str(corpus/'runs'),
        action_budgets=[10],trials=3,trial_seeds=[11,23,37],budget={},inference_limits={},max_total_cost_usd=100))
    import openalex_agent.llm
    monkeypatch.setattr(openalex_agent.llm,'HostedJSONPlanner',lambda *_:pytest.fail('No provider may be constructed'))
    with pytest.raises(ValueError,match='incomplete'):collect_unscored(corpus/'spec.json',p/'question-review.csv',corpus/'missing-approval.json',Settings())


def test_approval_pins_are_fail_closed(tmp_path):
    authorize(tmp_path/'a.json','score-preserved',{'version':'one'})
    with pytest.raises(ValueError,match='approval'):approval(tmp_path/'a.json','score-preserved',{'version':'two'})


def test_actual_evidence_visibility_distinguishes_corpus_from_input(tmp_path):
    paper=Paper(id='https://openalex.org/W1',abstract='Intro. '+('filler '*90)+'Corrected value is 42.')
    run=tmp_path/'run';(run/'context').mkdir(parents=True)
    atomic_json(run/'result.json',dict(inspected_ids=[paper.id]))
    atomic_json(run/'context/request-00001.json',dict(call=1,input={'observation':{'top_papers':[{'id':paper.id,'abstract_excerpt':paper.abstract[:350]}]}}))
    a=audit_passage(run,paper,'Corrected value is 42.')
    assert a['evidence_exists_in_stored_abstract'] and a['source_retrieved']
    assert not a['exact_passage_delivered_to_model'] and not a['memory_failure_eligible']
    assert a['factual_answer_correct'] is None and a['cited_source_supports_answer'] is None


def test_serialized_clm_quote_is_delivery_but_protected_text_is_not(tmp_path):
    passage='The result is "42".\nSecond line.'
    paper=Paper(id='https://openalex.org/W1',abstract=passage)
    run=tmp_path/'run';(run/'context').mkdir(parents=True)
    atomic_json(run/'result.json',dict(inspected_ids=[paper.id]))
    atomic_json(run/'context/request-00001.json',dict(call=1,input={'protected':{'question':passage},'active_context':''}))
    assert not audit_passage(run,paper,passage)['exact_passage_delivered_to_model']
    atomic_json(run/'context/request-00002.json',dict(call=2,input={'active_context':json.dumps({'abstract_excerpt':passage})}))
    a=audit_passage(run,paper,passage)
    assert a['exact_passage_delivered_to_model'] and a['delivery_calls']==[2] and a['memory_failure_eligible']


def test_pooled_ndcg_waits_for_pool_even_when_top_ten_is_judged(corpus):
    f=fake_outputs(corpus);p=make_pool(corpus,frozen=f);paths=reviewed(corpus,p)
    rows=csv_rows(corpus/'A.csv')
    next(r for r in rows if r['paper_id']=='https://openalex.org/W12')['grade']='UNKNOWN'
    csv_write(corpus/'A-v2.csv',rows)
    import_review(p,corpus/'A-v2.csv',corpus/'A.json','reviewer-one','A',corpus/'import-A-v2')
    resolve_reviews(p,[corpus/'import-A-v2',paths[1]],corpus/'annotations')
    pins=dict(frozen_outputs_sha256=sha(f/'unscored-freeze.json'),pool_sha256=sha(p/'pool-manifest.json'),
              annotation_sha256=sha(corpus/'annotations/annotation-manifest.json'),policy_sha256=digest(POLICY))
    authorize(corpus/'approval.json','score-preserved',pins)
    rows=score_pool(f,p,corpus/'annotations',corpus/'approval.json',corpus/'scores')
    assert all(r['precision_at_10']==1 and r['ndcg_at_10_pooled'] is None and r['pool_coverage']<1 for r in rows)
