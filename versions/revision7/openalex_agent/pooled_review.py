"""Versioned pooled human review. Never supplies labels to a planner or invents grades.

This is a separate protocol from the legacy exhaustive dashboard gate. Preparation
does not confer scientific approval; collection and scoring need distinct pins.
"""

from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil

from .corpus import LocalCorpus, atomic_json
from .models import canonical_id, utcnow

VERSION = 'pooled-two-stage-v1'
POLICY = dict(version=VERSION, lexical_depth=20, anchor_count=3, references=5,
              citations=5, background=5, output_depth=20, inspected_per_cell=5,
              double_review_fraction=.20, shuffle_seed='pool-blind-v1', k=10,
              relevant_threshold=1, gains='2**grade-1', unknown='UNKNOWN',
              full_pool_required_for_ndcg=True)
METHODS = ('standard','llm','llm_graph','clm','clm_graph')
DEVELOPMENT_VERSION = 'five-method-development-collection-v1'
DEVELOPMENT_CLM_POLICY = 'controlled-context-v2-capacity-stop'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def fresh(path):
    path=Path(path)
    if path.exists():raise ValueError('Use a new versioned output directory')
    path.mkdir(parents=True)
    return path


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def csv_write(path,rows,fields=None):
    rows=list(rows)
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def inventory(directory):
    return {str(p.relative_to(directory)):sha(p) for p in sorted(Path(directory).rglob('*')) if p.is_file()}


def check_files(directory,files):
    directory=Path(directory).resolve()
    for name,checksum in files.items():
        p=directory/name
        if p.is_symlink() or not p.resolve().is_relative_to(directory) or sha(p)!=checksum:
            raise ValueError('Immutable protocol input changed or escaped its directory')


def approval(path,stage,pins):
    a=json.loads(Path(path).read_text())
    if (a.get('approved') is not True or a.get('protocol')!=VERSION or a.get('stage')!=stage
        or not isinstance(a.get('approver'),str) or not a['approver'].strip()
        or not isinstance(a.get('approved_at'),str) or not a['approved_at'].strip()
        or a.get('pins')!=pins):
        raise ValueError('Explicit independent coordinator approval with exact protocol pins is required')
    return a


def human_attestation(path,reviewer):
    a=json.loads(Path(path).read_text())
    if (a.get('reviewer_id')!=reviewer or not reviewer.strip() or not a.get('reviewed_at')
        or a.get('independent_human_review') is not True or a.get('not_model_generated') is not True):
        raise ValueError('Independent human attestation is missing; software cannot certify independence')
    return a


def load_pool(directory):
    directory=Path(directory)
    m=json.loads((directory/'pool-manifest.json').read_text())
    if m['protocol']!=VERSION or m['policy']!=POLICY:raise ValueError('Unsupported pooled protocol')
    check_files(directory,m['immutable_sha256'])
    return m,json.loads((directory/'coordinator/items.json').read_text())


def load_freeze(directory):
    directory=Path(directory)
    m=json.loads((directory/'unscored-freeze.json').read_text())
    if m['protocol']!=VERSION:raise ValueError('Unsupported unscored freeze')
    check_files(directory,m['files'])
    return m,json.loads((directory/'rankings.json').read_text())


def prepare_pool(corpus_path,questions_path,output,*,frozen_outputs=None,heldout_approval=None):
    questions=json.loads(Path(questions_path).read_text())
    if (not questions or len({q['id'] for q in questions})!=len(questions)
        or any(not q.get('question','').strip() or q.get('split') not in ('development','validation','heldout') for q in questions)
        or any(set(q)-{'id','question','split','discipline','stratum','review_status'} for q in questions)
        or len({q['split'] for q in questions})!=1):raise ValueError('Use one explicit question split per pool')
    corpus=LocalCorpus(corpus_path)
    if questions[0]['split']=='heldout':
        approval(heldout_approval,'release-heldout-review',dict(corpus_sha256=corpus.fingerprint,questions_sha256=sha(questions_path)))
    freeze_manifest=None;rankings=[]
    if frozen_outputs:
        freeze_manifest,rankings=load_freeze(frozen_outputs)
        if (freeze_manifest['corpus_sha256']!=corpus.fingerprint or freeze_manifest['questions_sha256']!=sha(questions_path)):
            raise ValueError('Unscored outputs do not match pool questions/corpus')
    by_query={q['id']:q for q in questions};candidates=defaultdict(lambda:defaultdict(set))
    def add(qid,ids,origin):
        for identifier in ids:
            wid=canonical_id(identifier)
            if wid not in corpus.by_id:raise ValueError('Pool candidate is outside frozen corpus')
            candidates[qid][wid].add(origin)
    for q in questions:
        seeds=corpus.search(q['question'],limit=POLICY['lexical_depth'])
        add(q['id'],[p.id for p in seeds],'predeclared_lexical')
        refs=[];citers=[]
        for p in seeds[:POLICY['anchor_count']]:
            refs.extend(r.id for r in corpus.references(p.id,limit=POLICY['references']))
            citers.extend(r.id for r in corpus.citations(p.id,limit=POLICY['citations']))
        add(q['id'],list(dict.fromkeys(refs))[:POLICY['references']],'predeclared_references')
        add(q['id'],list(dict.fromkeys(citers))[:POLICY['citations']],'predeclared_citations')
        background=sorted(corpus.by_id,key=lambda wid:digest([POLICY['shuffle_seed'],q['id'],wid]))[:POLICY['background']]
        add(q['id'],background,'predeclared_unjudged_background')
    for cell in rankings:
        if cell['query_id'] not in by_query:raise ValueError('Unexpected output query')
        origin=f"output:{cell['method']}:{cell['action_budget']}:{cell['trial']}"
        add(cell['query_id'],cell['ranking'][:POLICY['output_depth']],origin)
        sampled=sorted(set(cell['inspected_ids']),key=lambda wid:digest([origin,wid]))[:POLICY['inspected_per_cell']]
        add(cell['query_id'],sampled,origin+':inspected')
    out=fresh(output);(out/'coordinator').mkdir();items=[];provenance={}
    for q in questions:
        ids=sorted(candidates[q['id']],key=lambda wid:digest([POLICY['shuffle_seed'],q['id'],wid]))
        double=set(ids[:max(1,math.ceil(len(ids)*POLICY['double_review_fraction']))])
        for wid in ids:
            p=corpus.by_id[wid];item_id=digest([VERSION,q['id'],wid])[:24]
            items.append(dict(item_id=item_id,query_id=q['id'],paper_id=wid,double_review=wid in double))
            provenance[item_id]=sorted(candidates[q['id']][wid])
    atomic_json(out/'coordinator/items.json',items)
    atomic_json(out/'coordinator/candidate-provenance.json',provenance)
    atomic_json(out/'questions.json',questions)
    worksheet=[];access=[]
    for item in items:
        p=corpus.by_id[item['paper_id']];q=by_query[item['query_id']]
        worksheet.append(dict(item_id=item['item_id'],query_id=q['id'],question=q['question'],paper_id=p.id,
            title=p.title or '',year=p.year,doi=p.doi or '',available_source_type='abstract' if p.abstract else 'metadata_only',
            source_material=p.abstract or '',grade='UNKNOWN',supporting_passage='',explanation='',reviewer_id=''))
        access.append(dict(item_id=item['item_id'],paper_id=p.id,abstract_available=bool(p.abstract),
            model_excerpt_char_limit=350,internal_references=sum(w in corpus.by_id for w in p.references),internal_citers=len(corpus.citing.get(p.id,[])),
            source_url=p.source_url or '',live_access_verified='UNKNOWN',question_source_adequacy='UNKNOWN',reviewer_id=''))
    for role in ('A','B'):
        d=out/f'reviewer-{role}';d.mkdir()
        allowed={i['item_id'] for i in items if role=='A' or i['double_review']}
        csv_write(d/'relevance.csv',[r for r in worksheet if r['item_id'] in allowed])
    csv_write(out/'source-access.csv',access)
    csv_write(out/'question-review.csv',[dict(query_id=q['id'],split=q['split'],question=q['question'],
        discipline=q.get('discipline',''),validity='UNKNOWN',taxonomy_verified='UNKNOWN',source_adequacy='UNKNOWN',
        independent_human_review='',reviewer_id='',reviewed_at='',explanation='') for q in questions])
    csv_write(out/'factual-evidence.csv',[dict(query_id=q['id'],question_id='',public_factual_question='',verified_answer='',
        supporting_work_id='',exact_passage='',source_type='',evidence_location='',source_sha256='',
        adequacy='UNKNOWN',attribution_verified='UNKNOWN',appropriate_abstention='UNKNOWN',reviewer_id='',
        independent_human_review='',reviewed_at='',notes='') for q in questions])
    csv_write(out/'adjudication-template.csv',[dict(item_id=i['item_id'],grade='UNKNOWN',supporting_passage='',reason='',reviewer_id='') for i in items if i['double_review']])
    atomic_json(out/'human-attestation-template.json',dict(reviewer_id='',reviewed_at='',independent_human_review=False,not_model_generated=False))
    manifest=dict(protocol=VERSION,created_at=utcnow(),policy=POLICY,corpus_sha256=corpus.fingerprint,
        questions_sha256=sha(questions_path),split=questions[0]['split'],questions=len(questions),candidates=len(items),
        double_review_items=sum(i['double_review'] for i in items),frozen_outputs_sha256=sha(Path(frozen_outputs)/'unscored-freeze.json') if frozen_outputs else None,
        status='UNREVIEWED OUTPUT POOL' if frozen_outputs else 'PREPARATION ONLY; final pool must include frozen outputs',
        immutable_sha256=inventory(out))
    atomic_json(out/'pool-manifest.json',manifest)
    return manifest


def import_review(pool,worksheet,attestation,reviewer,role,output):
    m,items=load_pool(pool)
    if role not in ('A','B'):raise ValueError('Reviewer role must be A or B')
    human_attestation(attestation,reviewer)
    expected={r['item_id']:r for r in items if role=='A' or r['double_review']}
    template={r['item_id']:r for r in csv_rows(Path(pool)/f'reviewer-{role}/relevance.csv')}
    rows=csv_rows(worksheet);found={};imported=[]
    mutable={'grade','supporting_passage','explanation','reviewer_id'}
    for row in rows:
        key=row['item_id']
        if key not in expected or key in found:raise ValueError('Duplicate/unassigned annotation identity')
        if any(row.get(k)!=v for k,v in template[key].items() if k not in mutable):raise ValueError('Reviewer changed immutable source/question text')
        grade=row['grade'].strip().upper()
        if grade not in ('','UNKNOWN','0','1','2'):raise ValueError('Grades must be 0, 1, 2 or UNKNOWN')
        grade=int(grade) if grade.isdigit() else None
        if grade is not None and (row.get('reviewer_id')!=reviewer or not row.get('explanation','').strip()):raise ValueError('Known grades require named human review and explanation')
        passage=row.get('supporting_passage','')
        if passage and passage not in template[key]['source_material']:raise ValueError('Supporting quote is outside supplied source')
        found[key]=dict(**expected[key],grade=grade,reviewer_id=reviewer,explanation=row.get('explanation',''),supporting_passage=passage)
    imported=[found.get(key,dict(**item,grade=None,reviewer_id=reviewer,explanation='',supporting_passage='')) for key,item in expected.items()]
    out=fresh(output);shutil.copyfile(worksheet,out/'submitted.csv');shutil.copyfile(attestation,out/'attestation.json')
    atomic_json(out/'review.json',dict(protocol=VERSION,pool_sha256=sha(Path(pool)/'pool-manifest.json'),role=role,
        reviewer_id=reviewer,items=imported,unknown=sum(r['grade'] is None for r in imported)))
    atomic_json(out/'import-manifest.json',dict(created_at=utcnow(),files=inventory(out)))
    return imported


def resolve_reviews(pool,reviews,output,*,adjudication=None,attestation=None,adjudicator=None):
    m,items=load_pool(pool);by_role={};reviewers=set();imports=[]
    for directory in reviews:
        directory=Path(directory);check_files(directory,json.loads((directory/'import-manifest.json').read_text())['files'])
        r=json.loads((directory/'review.json').read_text())
        if r['protocol']!=VERSION or r['pool_sha256']!=sha(Path(pool)/'pool-manifest.json') or r['role'] in by_role or r['reviewer_id'] in reviewers:
            raise ValueError('Need distinct independent reviewer imports for this exact pool')
        reviewers.add(r['reviewer_id']);by_role[r['role']]={i['item_id']:i for i in r['items']};imports.append(sha(directory/'import-manifest.json'))
    if set(by_role)!={'A','B'}:raise ValueError('Both primary and double-review imports are required')
    decisions={}
    if adjudication:
        if adjudicator in reviewers:raise ValueError('Disagreements need a separate adjudicator')
        human_attestation(attestation,adjudicator)
        for row in csv_rows(adjudication):
            key=row['item_id'];grade=row['grade'].strip().upper()
            if key in decisions or key not in {i['item_id'] for i in items}:raise ValueError('Duplicate/unknown adjudication identity')
            if grade in ('','UNKNOWN'):continue
            if grade not in ('0','1','2') or row.get('reviewer_id')!=adjudicator or not row.get('reason','').strip():raise ValueError('Adjudication needs an independent reviewer and reason')
            decisions[key]=row
    labels=[];unknown=[];records=[];conflicts=0;paired=0;agree=0;needed=[]
    material={r['item_id']:r['source_material'] for r in csv_rows(Path(pool)/'reviewer-A/relevance.csv')}
    for item in items:
        key=item['item_id'];a=by_role['A'][key]['grade'];b=by_role['B'].get(key,{}).get('grade')
        grade=a;conflict=item['double_review'] and a is not None and b is not None and a!=b
        if item['double_review']:
            grade=a if a is not None and a==b else None
            paired+=a is not None and b is not None;agree+=a is not None and a==b
        conflicts+=conflict
        if key in decisions:
            if grade is not None:raise ValueError('Adjudication cannot silently overwrite agreed judgments')
            d=decisions[key]
            if d.get('supporting_passage') and d['supporting_passage'] not in material[key]:raise ValueError('Adjudication quote is not in supplied evidence')
            grade=int(d['grade'])
        if grade is None:
            unknown.append(dict(query_id=item['query_id'],paper_id=item['paper_id']))
            needed.append(dict(item_id=key,grade_A='UNKNOWN' if a is None else a,grade_B='UNKNOWN' if b is None else b,
                               grade='UNKNOWN',supporting_passage='',reason='',reviewer_id=''))
        else:
            assessors=reviewers if item['double_review'] else {by_role['A'][key]['reviewer_id']}
            labels.append(dict(query_id=item['query_id'],paper_id=item['paper_id'],grade=grade,
                               assessor=adjudicator if key in decisions else ';'.join(sorted(assessors))))
        records.append(dict(**item,grade_A=a,grade_B=b,disagreement=bool(conflict),final_grade=grade,adjudication=decisions.get(key)))
    out=fresh(output)
    result=dict(protocol=VERSION,scope='pooled',source='human_import',pool_sha256=sha(Path(pool)/'pool-manifest.json'),
        corpus_sha256=m['corpus_sha256'],questions_sha256=m['questions_sha256'],reviewer_ids=sorted(reviewers),
        import_manifest_sha256=imports,adjudication_sha256=sha(adjudication) if adjudication else None,
        adjudication_attestation_sha256=sha(attestation) if adjudication else None,
        judgments=labels,unknown_pairs=unknown,coverage=len(labels)/len(items),
        disagreements=conflicts,paired_known=paired,raw_agreement=agree/paired if paired else None,
        independently_attested=True,independence_caveat='Attested by humans; not provable by software')
    atomic_json(out/'labels.json',result);atomic_json(out/'adjudication-records.json',records)
    csv_write(out/'adjudication-needed.csv',needed,fields=['item_id','grade_A','grade_B','grade','supporting_passage','reason','reviewer_id'])
    for n,directory in enumerate(reviews):shutil.copytree(directory,out/f'review-{n}')
    if adjudication:
        shutil.copyfile(adjudication,out/'adjudication.csv');shutil.copyfile(attestation,out/'adjudicator-attestation.json')
    atomic_json(out/'annotation-manifest.json',dict(created_at=utcnow(),files=inventory(out)))
    return result


def freeze_outputs(experiment,output,approval_path):
    import pandas as pd
    from .experiment_integrity import validate_schedule
    directory=Path(experiment).resolve();manifest=json.loads((directory/'experiment.json').read_text())
    if manifest.get('has_relevance_judgments') or manifest.get('judgments') or manifest.get('suite')!='controlled-five':raise ValueError('Freeze only unscored controlled outputs')
    rows=pd.read_csv(directory/'metrics.csv').to_dict('records');validate_schedule(manifest,rows)
    if any(p.is_symlink() for p in directory.rglob('*')):raise ValueError('Unscored output cannot contain symlinks')
    pins=dict(experiment_sha256=sha(directory/'experiment.json'),metrics_sha256=sha(directory/'metrics.csv'),
              raw_output_tree_sha256=digest(inventory(directory)),policy_sha256=digest(POLICY))
    a=approval(approval_path,'freeze-unscored',pins)
    rankings=[]
    ids=set(manifest['corpus_manifest']['ids'])
    for row in rows:
        path=Path(row['directory']).resolve()
        if not path.is_relative_to(directory):raise ValueError('Run directory escapes experiment')
        r=json.loads((path/'result.json').read_text())
        ranking=[canonical_id(p['paper']['id']) for p in r['ranked']]
        inspected=[canonical_id(i) for i in r['inspected_ids']]
        if len(set(ranking))!=len(ranking) or (set(ranking)|set(inspected))-ids:raise ValueError('Invalid frozen result identities')
        rankings.append({**{k:row[k] for k in ('query_id','method','variant','trial','action_budget')},'ranking':ranking,'inspected_ids':inspected,
                         'result_sha256':sha(path/'result.json'),'source':str(path.relative_to(directory)/'result.json')})
    out=fresh(output);shutil.copytree(directory,out/'raw-output');atomic_json(out/'rankings.json',rankings)
    atomic_json(out/'approval.json',a)
    frozen=dict(protocol=VERSION,created_at=utcnow(),corpus_sha256=manifest['corpus'],questions_sha256=manifest['questions_sha256'],
        deterministic_smoke=manifest['deterministic_smoke'],source_experiment_sha256=pins['experiment_sha256'],
        policy=POLICY,cells=len(rows),files=inventory(out))
    atomic_json(out/'unscored-freeze.json',frozen)
    return frozen


def score_pool(frozen,pool,annotations,approval_path,output):
    fm,rankings=load_freeze(frozen);pm,items=load_pool(pool);annotations=Path(annotations)
    check_files(annotations,json.loads((annotations/'annotation-manifest.json').read_text())['files'])
    labels=json.loads((annotations/'labels.json').read_text())
    if (pm['frozen_outputs_sha256']!=sha(Path(frozen)/'unscored-freeze.json') or labels['pool_sha256']!=sha(Path(pool)/'pool-manifest.json')
        or labels['protocol']!=VERSION or labels.get('independently_attested') is not True
        or labels['corpus_sha256']!=fm['corpus_sha256'] or labels['questions_sha256']!=fm['questions_sha256']):
        raise ValueError('Pool, annotation and frozen-output versions differ')
    pins=dict(frozen_outputs_sha256=sha(Path(frozen)/'unscored-freeze.json'),pool_sha256=sha(Path(pool)/'pool-manifest.json'),
              annotation_sha256=sha(annotations/'annotation-manifest.json'),policy_sha256=digest(POLICY))
    a=approval(approval_path,'score-preserved',pins)
    expected={(i['query_id'],i['paper_id']) for i in items};judged={}
    for row in labels['judgments']:
        key=(row['query_id'],canonical_id(row['paper_id']))
        if key not in expected or key in judged or type(row['grade']) is not int or row['grade'] not in (0,1,2):raise ValueError('Malformed or duplicate pooled judgment')
        judged[key]=row['grade']
    unknown=[(r['query_id'],canonical_id(r['paper_id'])) for r in labels['unknown_pairs']]
    if (labels.get('source')!='human_import' or labels.get('scope')!='pooled'
        or len(unknown)!=len(set(unknown)) or set(unknown)!=expected-set(judged)):
        raise ValueError('Known and UNKNOWN annotation identities do not partition the frozen pool')
    rows=[]
    for cell in rankings:
        qid=cell['query_id'];top=cell['ranking'][:POLICY['k']];known=[judged[(qid,w)] for w in top if (qid,w) in judged]
        pool_ids={w for q,w in expected if q==qid};query_labels={w:g for (q,w),g in judged.items() if q==qid}
        complete_top=all((qid,w) in judged for w in top)
        complete_pool=set(query_labels)==pool_ids
        relevant=sum(g>=POLICY['relevant_threshold'] for g in known)
        ideal=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(sorted(query_labels.values(),reverse=True)[:POLICY['k']]))
        dcg=sum((2**judged[qid,w]-1)/math.log2(i+2) for i,w in enumerate(top)) if complete_top else None
        rows.append({**{k:cell[k] for k in ('query_id','method','variant','trial','action_budget')},'returned_at_10':len(top),
            'judged_at_10':len(known),'judgment_coverage_at_10':len(known)/len(top) if top else None,
            'unjudged_at_10':len(top)-len(known),'pool_coverage':len(query_labels)/len(pool_ids),
            'precision_at_10':relevant/10 if complete_top else None,
            'precision_at_10_lower_bound':relevant/10,'precision_at_10_upper_bound':(relevant+len(top)-len(known))/10,
            'ndcg_at_10_pooled':dcg/ideal if complete_top and complete_pool and ideal else None,
            'ndcg_scope':'complete judged pool only; not exhaustive corpus',
            'short_ranking_policy':'unreturned slots are retrieval misses; unjudged returned works remain UNKNOWN',
            'deterministic_smoke':fm['deterministic_smoke']})
    out=fresh(output);csv_write(out/'pooled-metrics.csv',rows)
    from .research_metrics import paired_comparisons
    atomic_json(out/'paired-descriptive-contrasts.json',dict(
        precision_at_10=paired_comparisons(rows,'precision_at_10'),
        ndcg_at_10_pooled=paired_comparisons(rows,'ndcg_at_10_pooled'),
        interpretation='All declared 2x2 contrasts, question-level trial means, descriptive unadjusted intervals only; no confirmatory significance claim'))
    atomic_json(out/'scores.json',dict(protocol=VERSION,rows=rows,scientific_approval=a,
        incomplete_judgments=bool(labels['unknown_pairs']),recall_corpus=None,
        limitation='Pooling is incomplete for the full corpus; no corpus recall or exhaustive-corpus nDCG claim. Annotation coverage is explicit.',
        immutable_inputs=pins,hosted_calls=0,source_outputs_modified=False))
    return rows


def collect_unscored(spec_path,review_path,approval_path,settings):
    """Explicitly gated Track-A entry point; no label/factual truth path is accepted."""
    from .controller import Budget
    from .context_planner import InferenceLimits
    from .experiment_dashboard import source_digest,profile_digest
    from .experiments import run_experiments
    from .llm import HostedJSONPlanner
    from .storage import Database
    spec=json.loads(Path(spec_path).read_text());questions=json.loads(Path(spec['questions']).read_text())
    if set(spec)!= {'corpus','questions','output','action_budgets','trials','trial_seeds','budget','inference_limits','max_total_cost_usd'}:
        raise ValueError('Collection spec contains missing or unauthorized fields')
    if (len({q['id'] for q in questions})!=len(questions)
        or any(set(q)-{'id','question','split','discipline','stratum','review_status'} for q in questions)):
        raise ValueError('Require unique public questions without evaluator-only fields')
    reviewed=csv_rows(review_path)
    if (len(reviewed)!=len(questions) or {r['query_id'] for r in reviewed}!={q['id'] for q in questions}
        or any(r['validity']!='approved' or r['taxonomy_verified']!='true' or r['source_adequacy']!='sufficient'
            or r['independent_human_review']!='true' or not r['reviewer_id'].strip() or not r['reviewed_at'].strip() or not r['explanation'].strip() for r in reviewed)):
        raise ValueError('Independent question, taxonomy and source-adequacy review is incomplete')
    text={q['id']:q['question'] for q in questions}
    if any(r['question']!=text[r['query_id']] for r in reviewed):raise ValueError('Reviewed question differs from collection question')
    if (not 10<=len(questions)<=20 or any(q.get('split')!='heldout' for q in questions) or spec['trials']!=3
        or len(spec['trial_seeds'])!=3):raise ValueError('Final Track-A collection requires 10–20 approved held-out questions and three repetitions')
    corpus=LocalCorpus(spec['corpus'])
    pins=dict(spec_sha256=sha(spec_path),question_review_sha256=sha(review_path),corpus_sha256=corpus.fingerprint,
        questions_sha256=sha(spec['questions']),source_sha256=source_digest(),provider_profile_sha256=profile_digest(settings),policy_sha256=digest(POLICY))
    a=approval(approval_path,'collect-unscored',pins)
    if a.get('heldout_independence_attested') is not True or a.get('paid_inference_authorized') is not True:
        raise ValueError('Held-out independence and paid-inference authorization are required')
    budget=Budget(**spec['budget']);limits=InferenceLimits(**spec['inference_limits'])
    if not (settings.llm_input_price>0 and settings.llm_output_price>0):raise ValueError('Configured provider prices are UNKNOWN; set verified positive prices before approval')
    calls=len(questions)*4*spec['trials']*len(spec['action_budgets'])*budget.max_llm_calls
    worst=calls*(limits.max_input_tokens or 0)*settings.llm_input_price/1e6+calls*settings.llm_max_tokens*settings.llm_output_price/1e6
    if (not limits.max_input_tokens or type(spec['max_total_cost_usd']) not in (int,float)
        or not math.isfinite(spec['max_total_cost_usd']) or not 0<worst<=spec['max_total_cost_usd']):
        raise ValueError('Approved cost reservation does not cover conservative request ceilings')
    output=Path(spec['output'])
    if output.exists():raise ValueError('Do not overwrite an existing experiment')
    # The API is called only after all independent review, authorization and cost checks.
    with Database(str(output.parent/(output.name+'-metadata.sqlite'))) as db:
        rows=run_experiments(spec['corpus'],spec['questions'],db,lambda:HostedJSONPlanner(settings),output=str(output),
            suite='controlled-five',methods=METHODS,action_budgets=tuple(spec['action_budgets']),trials=spec['trials'],
            trial_seeds=tuple(spec['trial_seeds']),budget=budget,inference_limits=limits,labels_path=None)
    atomic_json(output/'collection-approval.json',a)
    return rows


def _development_questions(path):
    questions=json.loads(Path(path).read_text())
    allowed={'id','question','split','discipline','stratum','review_status'}
    if (not isinstance(questions,list) or not 1<=len(questions)<=20
        or any(not isinstance(q,dict) or set(q)-allowed or q.get('split')!='development'
            or not isinstance(q.get('id'),str) or not q['id'].strip()
            or not isinstance(q.get('question'),str) or not 1<=len(q['question'].strip())<=1000 for q in questions)
        or len({q['id'] for q in questions})!=len(questions)):
        raise ValueError('Use 1–20 unique development questions without truth/annotation fields')
    return questions


def _development_facts(path,questions):
    if path is None:return None
    facts=json.loads(Path(path).read_text())
    if (not isinstance(facts,list) or len(facts)!=len(questions)
        or any(not isinstance(f,dict) or set(f)!={'query_id','question_id','question'}
            or any(not isinstance(v,str) or not 1<=len(v.strip())<=1000 for v in f.values()) for f in facts)
        or {f['query_id'] for f in facts}!={q['id'] for q in questions}
        or len({f['question_id'] for f in facts})!=len(facts)):
        raise ValueError('Supply one public factual task per DEV question, without verified answers or judgments')
    return facts


def development_review_ready(path,questions):
    """Question/source review is separate from later, blinded relevance grades."""
    rows=csv_rows(path)
    by_id={q['id']:q['question'] for q in questions}
    return (len(rows)==len(questions) and {r.get('query_id') for r in rows}==set(by_id)
        and all(r.get('question')==by_id[r['query_id']] and r.get('split')=='development'
            and r.get('validity')=='approved' and r.get('taxonomy_verified')=='true'
            and r.get('source_adequacy')=='sufficient' and r.get('independent_human_review')=='true'
            and all(r.get(k,'').strip() for k in ('reviewer_id','reviewed_at','explanation')) for r in rows))


def _development_prompt(completion_guidance,factual):
    from .actions import CONTROLLED_TOOL_CONTRACT
    from .research_completion import VERSION as completion_version, SUFFIX
    from .factual_protocol import VERSION as factual_version, INSTRUCTIONS
    if completion_guidance not in (None,completion_version):
        raise ValueError('Unknown shared research-completion instruction version')
    return dict(completion_guidance=completion_guidance,
        completion_suffix=SUFFIX if completion_guidance else None,
        planner_template_sha256=sha(Path(__file__).with_name('context_planner.py')),
        acquisition_contract_sha256=digest(CONTROLLED_TOOL_CONTRACT),
        factual_protocol=factual_version if factual else None,
        factual_instructions_sha256=digest(INSTRUCTIONS) if factual else None,
        claim_elicitation='fixed terminal opportunity' if factual else 'optional planning claims; unequal answer opportunities',
        baseline='single-search, no LLM or factual-response call')


def prepare_development(corpus_path,questions_path,output,settings,*,factual_questions=None,
                        completion_guidance=None,action_budgets=(3,),trials=1,max_llm_calls=6):
    """Offline proposal only; deliberately creates no approval or human judgments."""
    from .context_planner import InferenceLimits
    from .controller import Budget
    from .development_pilot import REQUEST_RESERVE
    from .gemini_validation import GeminiValidationBackend
    from .git_provenance import git_provenance
    from .meta_validation import code_pins
    questions=_development_questions(questions_path)
    facts=_development_facts(factual_questions,questions)
    prompt=_development_prompt(completion_guidance,facts is not None)
    if (type(trials) is not int or not 1<=trials<=3 or type(max_llm_calls) is not int
        or not (2 if facts is not None else 1)<=max_llm_calls<=20 or not action_budgets
        or len(set(action_budgets))!=len(action_budgets)
        or any(type(v) is not int or not 1<=v<=20 for v in action_budgets)):
        raise ValueError('Use bounded DEV actions, one to three trials and at most twenty total calls per agent')
    corpus=LocalCorpus(corpus_path)
    out=fresh(output).resolve()
    shutil.copyfile(questions_path,out/'questions.json')
    if factual_questions:shutil.copyfile(factual_questions,out/'factual-questions.json')
    csv_write(out/'question-review.template.csv',[dict(query_id=q['id'],split='development',question=q['question'],
        discipline=q.get('discipline',''),validity='UNKNOWN',taxonomy_verified='UNKNOWN',source_adequacy='UNKNOWN',
        independent_human_review='',reviewer_id='',reviewed_at='',explanation='') for q in questions])
    budget=Budget(max_actions=max(action_budgets),max_rounds=20,max_llm_calls=max_llm_calls,page_size=10)
    limits=InferenceLimits(max_total_tokens=100000,max_active_tokens=4096,max_input_tokens=8192)
    spec=dict(protocol=DEVELOPMENT_VERSION,purpose='UNSCORED DEVELOPMENT COLLECTION ONLY',suite='controlled-five',
        methods=list(METHODS),corpus=str(Path(corpus_path).resolve()),corpus_sha256=corpus.fingerprint,
        corpus_manifest_sha256=sha(Path(corpus_path)/'manifest.json'),questions=str(out/'questions.json'),
        questions_sha256=sha(out/'questions.json'),question_source_sha256=sha(questions_path),
        factual_questions=str(out/'factual-questions.json') if facts is not None else None,
        factual_questions_sha256=sha(out/'factual-questions.json') if facts is not None else None,
        question_review_template=str(out/'question-review.template.csv'),
        question_review_template_sha256=sha(out/'question-review.template.csv'),
        provider=GeminiValidationBackend.profile(settings),prompt=prompt,
        clm_policy=DEVELOPMENT_CLM_POLICY,policy=POLICY,policy_sha256=digest(POLICY),
        action_budgets=list(action_budgets),trials=trials,trial_seeds=[11,23,37][:trials],
        budget=asdict(budget),inference_limits=asdict(limits),
        max_provider_calls=len(questions)*4*trials*len(action_budgets)*max_llm_calls,
        spending_ceiling_usd=6,checkpoint_usd=3,request_reservation_usd=str(REQUEST_RESERVE),
        wall_seconds=7200,automatic_retries=0,source_pins=code_pins(),git=git_provenance(),
        output=str(out/'execution/benchmark'),final_evaluation_authorized=False)
    atomic_json(out/'spec.json',spec)
    return check_development(out/'spec.json')


def check_development(spec_path,review_path=None):
    """Check exact inputs/pins offline; never constructs a provider or approves it."""
    from .context_planner import CONTEXT_POLICY_VERSION, InferenceLimits
    from .controller import Budget
    from .development_pilot import REQUEST_RESERVE
    from .gemini import GeminiSettings
    from .gemini_validation import GeminiValidationBackend
    from .meta_validation import code_pins
    from dataclasses import fields
    spec=json.loads(Path(spec_path).read_text())
    required={'protocol','purpose','suite','methods','corpus','corpus_sha256','corpus_manifest_sha256',
        'questions','questions_sha256','question_source_sha256','factual_questions','factual_questions_sha256',
        'question_review_template','question_review_template_sha256','provider','prompt','clm_policy','policy','policy_sha256',
        'action_budgets','trials','trial_seeds','budget','inference_limits','max_provider_calls',
        'spending_ceiling_usd','checkpoint_usd','request_reservation_usd','wall_seconds','automatic_retries',
        'source_pins','git','output','final_evaluation_authorized'}
    if (set(spec)!=required or spec['protocol']!=DEVELOPMENT_VERSION or spec['purpose']!='UNSCORED DEVELOPMENT COLLECTION ONLY'
        or spec['suite']!='controlled-five' or spec['methods']!=list(METHODS)
        or spec['clm_policy']!=DEVELOPMENT_CLM_POLICY or CONTEXT_POLICY_VERSION!=DEVELOPMENT_CLM_POLICY
        or spec['policy']!=POLICY or spec['policy_sha256']!=digest(POLICY)
        or spec['source_pins']!=code_pins() or spec['final_evaluation_authorized'] is not False
        or spec['spending_ceiling_usd']!=6 or spec['checkpoint_usd']!=3 or spec['wall_seconds']!=7200
        or spec['request_reservation_usd']!=str(REQUEST_RESERVE) or spec['automatic_retries']!=0):
        raise ValueError('Development protocol, methods, CLM policy, source or safeguards differ from preparation')
    settings=GeminiSettings(**{f.name:spec['provider'][f.name] for f in fields(GeminiSettings)})
    if spec['provider']!=GeminiValidationBackend.profile(settings):raise ValueError('Provider profile changed')
    questions=_development_questions(spec['questions'])
    facts=_development_facts(spec['factual_questions'],questions)
    if (sha(spec['questions'])!=spec['questions_sha256'] or
        sha(Path(spec['corpus'])/'manifest.json')!=spec['corpus_manifest_sha256'] or
        sha(Path(spec['corpus'])/'papers.jsonl')!=spec['corpus_sha256'] or
        sha(spec['question_review_template'])!=spec['question_review_template_sha256'] or
        (sha(spec['factual_questions']) if facts is not None else None)!=spec['factual_questions_sha256']):
        raise ValueError('Frozen corpus, questions or public factual tasks changed')
    if spec['prompt']!=_development_prompt(spec['prompt']['completion_guidance'],facts is not None):
        raise ValueError('Prompt identity changed')
    budget=Budget(**spec['budget']);limits=InferenceLimits(**spec['inference_limits'])
    actions=spec['action_budgets'];trials=spec['trials']
    if (type(trials) is not int or not 1<=trials<=3 or spec['trial_seeds']!=[11,23,37][:trials]
        or not actions or len(set(actions))!=len(actions) or any(type(v) is not int or not 1<=v<=20 for v in actions)
        or not (2 if facts is not None else 1)<=budget.max_llm_calls<=20
        or spec['budget']!=asdict(Budget(max_actions=max(actions),max_rounds=20,max_llm_calls=budget.max_llm_calls,page_size=10))
        or spec['inference_limits']!=asdict(InferenceLimits(max_total_tokens=100000,max_active_tokens=4096,max_input_tokens=8192))
        or spec['max_provider_calls']!=len(questions)*4*trials*len(actions)*budget.max_llm_calls):
        raise ValueError('Shared developmental budgets differ from supported resource ceilings')
    review=Path(review_path or spec['question_review_template'])
    ready=development_review_ready(review,questions)
    pins=dict(spec_sha256=sha(spec_path),question_review_sha256=sha(review),questions_sha256=spec['questions_sha256'],
        factual_questions_sha256=spec['factual_questions_sha256'],corpus_sha256=spec['corpus_sha256'],
        source_sha256=digest(spec['source_pins']),provider_profile_sha256=digest(spec['provider']),
        prompt_sha256=digest(spec['prompt']),budget_sha256=digest([spec['budget'],spec['inference_limits'],actions,trials,spec['trial_seeds']]),
        clm_policy=DEVELOPMENT_CLM_POLICY,policy_sha256=digest(POLICY))
    return dict(protocol=DEVELOPMENT_VERSION,configuration_valid=True,question_review_ready=ready,
        approval_protocol=VERSION,approval_stage='collect-development',approval_required=True,pins=pins,
        planned_trajectories=len(questions)*5*trials*len(actions),max_provider_calls=spec['max_provider_calls'],
        grounding_measurement='fixed terminal opportunity' if facts is not None else 'planning claims only; unequal answer opportunities',
        hosted_calls=0)


def collect_development(spec_path,review_path,approval_path,*,transport=None):
    """Use the same five-method runner and Gemini ledger; never final/held-out."""
    from dataclasses import fields
    import time
    from .context_planner import InferenceLimits
    from .controller import Budget
    from .execution_control import ExperimentStopped
    from .experiments import run_experiments
    from .gemini import GeminiSettings
    from .gemini_validation import GeminiSpendLedger, GeminiValidationBackend
    from .storage import Database
    checked=check_development(spec_path,review_path)
    if not checked['question_review_ready']:raise ValueError('Independent DEV question/source review is incomplete')
    a=approval(approval_path,'collect-development',checked['pins'])
    if (a.get('development_only') is not True or a.get('paid_inference_authorized') is not True
        or a.get('final_evaluation_authorized') is not False):
        raise ValueError('Explicit paid development-only approval is required; final evaluation stays unauthorized')
    authorization_sha256=sha(approval_path)
    spec=json.loads(Path(spec_path).read_text())
    benchmark=Path(spec['output']);execution=fresh(benchmark.parent)
    atomic_json(execution/'spec.json',spec)
    snapshots={'questions':execution/'questions.json','review':execution/'question-review.csv'}
    for name,source,checksum in [('questions',spec['questions'],spec['questions_sha256']),
                                 ('review',review_path,checked['pins']['question_review_sha256'])]:
        shutil.copyfile(source,snapshots[name])
        if sha(snapshots[name])!=checksum:raise ValueError('Input changed while creating the execution snapshot')
    snapshots['facts']=None
    if spec['factual_questions']:
        snapshots['facts']=execution/'factual-questions.json'
        shutil.copyfile(spec['factual_questions'],snapshots['facts'])
        if sha(snapshots['facts'])!=spec['factual_questions_sha256']:raise ValueError('Public factual tasks changed during snapshot')
    settings=GeminiSettings(**{f.name:spec['provider'][f.name] for f in fields(GeminiSettings)})
    ledger=GeminiSpendLedger(execution/'spending-ledger.json',spec['spending_ceiling_usd'],
        checkpoint=spec['checkpoint_usd'],max_calls=spec['max_provider_calls'],max_seconds=spec['wall_seconds'])
    guarded=GeminiValidationBackend.guarded_factory(settings,ledger,transport=transport)
    def progress():
        # Request admission (including reservations/call ceiling) remains solely
        # in the existing ledger. Global stops also apply immediately after a
        # response and between trajectories, including a non-LLM baseline.
        reason=ledger.stop_reason
        if ledger.clock()-ledger.started>=ledger.max_seconds:reason='pilot_wall_clock_limit'
        elif any(row['usage']=='UNKNOWN' for row in ledger.attempts):reason='development_unaccountable_usage'
        elif ledger.reported_spend>=ledger.checkpoint:reason='pilot_three_dollar_checkpoint'
        if reason:
            ledger.stop_reason=reason;ledger.save();raise ExperimentStopped(reason)
        return False
    def factory():
        planner=guarded();request=planner._request
        def checked_request(body):
            try:return request(body)
            finally:progress()
        planner._request=checked_request
        return planner
    status='running'
    try:
        with Database(str(execution/'metadata.sqlite')) as db:
            run_experiments(spec['corpus'],str(snapshots['questions']),db,factory,output=str(benchmark),
                suite='controlled-five',methods=METHODS,action_budgets=tuple(spec['action_budgets']),
                trials=spec['trials'],trial_seeds=tuple(spec['trial_seeds']),budget=Budget(**spec['budget']),
                inference_limits=InferenceLimits(**spec['inference_limits']),labels_path=None,
                factual_questions_path=str(snapshots['facts']) if snapshots['facts'] else None,
                completion_guidance=spec['prompt']['completion_guidance'] is not None,cancel_check=progress)
        status='complete'
    except ExperimentStopped as exc:status=exc.reason
    except BaseException:
        status='failed';raise
    finally:
        receipt=dict(protocol=DEVELOPMENT_VERSION,status=status,pins=checked['pins'],
            approval_sha256=authorization_sha256,
            authorization_provenance=dict(protocol=a['protocol'],stage=a['stage'],approver=a['approver'],approved_at=a['approved_at']),
            classification='MOCKED OFFLINE TEST' if transport else 'HOSTED DEVELOPMENT ONLY',
            completed_trajectories=len(csv_rows(benchmark/'metrics.csv')) if (benchmark/'metrics.csv').exists() else 0,
            provider_attempts=len(ledger.attempts),reported_gross_cost_usd=str(ledger.reported_spend),
            retained_usd=str(ledger.charged),runtime_seconds=time.monotonic()-ledger.started,
            stop_reason=ledger.stop_reason,final_evaluation_authorized=False,
            measurement_policy=dict(relevance='existing blinded pooled 0–2 review, scored separately',
                grounding='independent claim/factual judgments required',retention='actual delivered/recovered text, not semantic truth',
                efficiency='reported/unknown usage, latency, cost and context-management overhead'))
        atomic_json(execution/'collection-status.json',receipt)
        if benchmark.is_dir():
            # Keep accounting and identities inside the native tree so the
            # existing immutable-output freeze includes them without a new format.
            shutil.copyfile(ledger.path,benchmark/'spending-ledger.json')
            shutil.copyfile(snapshots['review'],benchmark/'collection-question-review.csv')
            atomic_json(benchmark/'development-spec.json',spec)
            atomic_json(benchmark/'development-collection.json',receipt)
    return receipt
