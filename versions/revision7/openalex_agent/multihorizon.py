"""V3 S/L/M campaign using existing execution, context and Terra safeguards.

No paid call is possible through prepare/check/report/offline validation. A
fresh truthful authorization and mechanical pilot/scored-freeze gates precede
future execution. All studies reference the EXISTING one $68 Terra ledger.
"""
from pathlib import Path
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime,timezone
from copy import deepcopy
from decimal import Decimal
import argparse,csv,fcntl,hashlib,json,os,shutil,sqlite3,time
from .corpus import LocalCorpus,atomic_json
from .config import Settings
from .controller import Budget
from .context_planner import InferenceLimits
from .terra_v2 import TerraPlanner,BASE_URL,MODEL
from .terra_accounting import TerraLedger
from .execution_control import ExperimentStopped
from .storage import Database
from . import experiments as engine
from .v2_evaluation import objective_score
from .multihorizon_cases import V2,CORPUS,VERSION,order
from .multihorizon_validation import validate
from .multihorizon_metrics import comparisons,HOSTED
from . import multihorizon_long as staged
from . import multihorizon_memory as memory

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'data/openalex-multihorizon-v3-2026-11-09-v7-final'
VERSION='openalex-multihorizon-v3-scientific-r7'
METHODS=('standard',*HOSTED)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def queue(out):
    db=sqlite3.connect(Path(out)/'queue.sqlite',timeout=30);db.row_factory=sqlite3.Row;db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
    try:yield db;db.commit()
    except BaseException:db.rollback();raise
    finally:db.close()


def matrix(cases):
    jobs=[]
    prefix='V3R7' if cases['main']['L'] and cases['main']['L'][0].get('protocol_revision')=='multihorizon-scientific-r7' else 'V3'
    for phase in ['pilot','main']:
        for study in ['S','L','M']:
            methods=HOSTED if study=='M' else METHODS;tasks=cases[phase][study];domains=sorted({c['domain'] for c in tasks})
            buckets={d:sorted([c for c in tasks if c['domain']==d],key=lambda c:order('schedule',c.get('id',c.get('case_id')))) for d in domains}
            ordered=[buckets[d][i] for i in range(max(map(len,buckets.values()))) for d in domains if i<len(buckets[d])]
            for i,task in enumerate(ordered):
                for rep in range(1 if phase=='pilot' else 2):
                    offset=(i+rep)%len(methods)
                    for method in methods[offset:]+methods[:offset]:
                        tid=task.get('id',task.get('case_id'))
                        jobs.append({'id':f'{prefix}-{phase}-{study}-{tid}-r{rep}-{method}','phase':phase,'study':study,'task_id':tid,'method':method,'repetition':rep,'trial_identifier':[11,23][rep],'domain':task['domain']})
    return jobs


def _cases(out):
    proposal=Path(out)/'proposal'
    return {phase:{s:json.loads((proposal/(s+('-pilot' if phase=='pilot' else '')+'.json')).read_text()) for s in ['S','L','M']} for phase in ['pilot','main']}


def prepare(out,cases_path):
    out=Path(out).resolve();cases_path=Path(cases_path).resolve()
    if out.exists():raise ValueError('Immutable preparation already exists; choose a new version')
    validation=validate(cases_path,LocalCorpus(CORPUS))
    if validation['errors']:raise ValueError('Real-source feasibility failed: '+str(validation['errors']))
    out.mkdir(parents=True);shutil.copytree(cases_path,out/'proposal');shutil.copytree(ROOT/'protocols',out/'protocols')
    (out/'evidence').mkdir();shutil.copyfile(ROOT/'engineering/pricing-verification.json',out/'evidence/pricing-verification.json');shutil.copytree(ROOT/'engineering/provider-docs',out/'evidence/provider-docs')
    jobs=matrix(_cases(out));selected=[]
    for study in ('S','L','M'):
        task=next(j['task_id'] for j in jobs if j['phase']=='pilot' and j['study']==study)
        selected += [j['id'] for j in jobs if j['phase']=='pilot' and j['study']==study and j['task_id']==task]
    selections={}
    for name,ids,cap in [('feasibility',selected,'10.00'),('full',[j['id'] for j in jobs if j['phase']=='pilot'],'68.00')]:
        caps={'S':7,'L':78,'M':88};attempts=sum(caps[j['study']] for j in jobs if j['id'] in ids and j['method']!='standard')
        file=out/'proposal'/f'pilot-selection-{name}.json'
        atomic_json(file,{'version':'r7-explicit-pilot-selection-v1','name':name,'trajectory_ids':ids,'ceiling_usd':cap,'provider_attempt_limit':attempts,'default_workers':1,'heldout':False})
        selections[name]={'path':str(file.relative_to(out)),'sha256':sha(file),'ceiling_usd':cap,'provider_attempt_limit':attempts,'count':len(ids)}
    original=json.loads((V2/'protocol.json').read_text());ledger=json.loads((V2/'spending-ledger.json').read_text())
    if ledger['ceiling_usd']!='68.00':raise ValueError('Existing $68 cumulative accounting scope changed')
    source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'openalex_agent').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    source.update({n:sha(ROOT/n) for n in ['requirements.lock.txt','pyproject.toml']})
    source.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'protocols').iterdir() if p.is_file()})
    spec={'version':VERSION,'scope':'development-only, post-diagnostic follow-up; not human-heldout evaluation','created_utc':datetime.now(timezone.utc).isoformat(),
        'core_counts':{'S':300,'L':240,'M':96,'total':636},'repetitions':2,'trial_identifiers':[11,23],'methods':list(METHODS),'memory_methods':list(HOSTED),
        'studies':{'S':{'tasks':30,'action_ceiling':3,'model_call_ceiling':7,'total_token_ceiling':100000,'primary':'Known-target Recall@10','natural_STOP':True},
            'L':{'tasks':24,'phases':6,'substantive_actions_per_phase':6,'generation_calls_per_phase':13,'model_call_ceiling':78,'total_token_ceiling':600000,'primary':'Objective source-and-directed-edge completion','STOP':'Closes current phase; safety refusal stops project','standard':'One pinned lexical query per phase; non-LLM'},
            'M':{'cases':12,'papers':96,'batches':32,'papers_per_batch':3,'checkpoints':[24,48,72,96],'maintenance_calls_per_batch':2,'quiz_A_calls':1,'quiz_B_calls':5,'recoveries_per_B':2,'context_edits_per_B':2,'model_call_ceiling':88,'total_token_ceiling':1100000,
                 'quiz_character_headroom':4000,'quiz_token_headroom':2000,'primary':'Five-item delayed exact accuracy at96 papers, A no recovery','calibration':'Q6 separate','STOP':'Maintenance only; external schedule continues'}},
        'shared_context':{'max_active_tokens':4096,'max_input_tokens':8192,'max_user_payload_chars':16000},'context_policy':'controlled-context-v2-capacity-stop','graph_policy':'graph-visible-v2','completion_guidance':'research-completion-guidance-v1','final_response_preparation':'capacity-aware-final-response-v1',
        'scientific_revision':'r7' if jobs[0]['id'].startswith('V3R7') else 'legacy-fixture-unexecutable',
        'measurement_policies':{'memory':'memory-itemwise-abstention-v3-r7','final_response':'required-S-and-L-factual-opportunities; raw acquisition separate from operational failure','S_edges':'legacy endpoint co-discovery plus observed-directed-links-r7','L_resources':'page5/cap200 acquisition records; explicit inspections counted separately','L_workload':'six connected multi-route investigations and early-metadata reuse probes'},
        'pilot_selections':selections,
        'provider':original['provider'],'corpus':str(CORPUS),'corpus_sha256':original['corpus_sha256'],'memberships_sha256':sha(CORPUS/'memberships.json'),
        'accounting':{'shared_directory':str(V2),'profile_sha256':sha(V2/'protocol.json'),'ceiling_usd':'68.00','new_ledger_created':False,'unknown_usage_stop':True,'automatic_retries':False,'credits_verified':False},
        'max_runtime_seconds':86400,'source_pins':source,'frozen_inputs':{str(p.relative_to(out)):sha(p) for folder in ['proposal','protocols','evidence'] for p in (out/folder).rglob('*') if p.is_file()},
        'execution':{'version':'v3-detached-concurrency-v1','workers_supported':[1,2,4],'default_workers':1,'unit':'independent trajectory','within_trajectory':'sequential','interrupted_trajectory':'preserve; never automatically replay','source_isolation':'sealed source and dependency snapshot','quota_validation':'live provider headroom unverified; default one worker'},
        'original_S_sha256':sha(V2/'proposal/main-agent.json'),'reference_scorer_sha256':sha(ROOT/'protocols/openalex_memory_scoring_v2.py'),
        'revision7_memory_scorer_sha256':sha(ROOT/'openalex_agent/memory_scoring_v3.py'),'revision7_answer_schema_sha256':sha(ROOT/'protocols/OpenAlex_Memory_Answer_Schema_V3_R7.json'),
        'authorization':{'paid':False,'human_judgments_created':False,'scored_freeze':False,'external_source_review':'PENDING'},
        'freeze_requirements':['Actual unmocked disjoint S/L/M pilot receipts','S graph/CLM/final-response compatibility','L sustained meaningful phased activity and measured context competition','M96 actual exposures, pressure, snapshot independence, graph/CLM delivery','Independent objective-source audit attestation'],
        'statistics':{'independent_unit':'task/case','aggregate_repetitions_first':True,'bootstrap_samples':10000,'bootstrap_seed':73019,'fixed_domain_strata':True,'primary_contrasts':['graph','clm','interaction'],'multiplicity':'Holm three contrasts within study; nine-study-family sensitivity','attempted_failed_final':'operational ITT zero; complete-case sensitivity','never_attempted':'missing, not zero'},
        'unchanged_baseline_note':'Study S standard retains original initial retrieval capacity. Study L standard is separately pinned staged non-LLM queries. Standard absent from M.'}
    spec['studies']['L'].update(records_per_acquisition=5,acquisition_record_cap=200,maximum_unique_records_in36_acquisitions=180,primary='Mean of frozen revised source and directed-edge acquisition checks; reuse probes secondary')
    spec['studies']['M'].update(answer_schema='OpenAlex_Memory_Answer_Schema_V3_R7.json',itemwise_abstention_zero=True,parseable_partial_items_independent=True)
    atomic_json(out/'protocol.json',spec);assert sum(j['phase']=='main' for j in jobs)==636
    with queue(out) as db:
        db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,ordinal INTEGER,stage TEXT,payload TEXT,state TEXT,started REAL,ended REAL,error TEXT)')
        for i,j in enumerate(jobs):db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)',(j['id'],i,j['phase']+'-'+j['study'],json.dumps(j),'PENDING',None,None,None))
    atomic_json(out/'queue-plan.json',jobs);atomic_json(out/'source-adequacy-validation.json',validation)
    atomic_json(out/'approval-template.json',{'version':'multihorizon-v3-paid-authorization-v1','approved':False,'authorization_source':None,'approved_at':None,'protocol_sha256':sha(out/'protocol.json'),'maximum_gross_usd':'68.00','max_runtime_seconds':86400,'scopes':[],'pilot_selection_sha256':selections['feasibility']['sha256'],'pilot_subceiling_usd':'10.00','note':'No authorization created. All studies and pilots share the existing cumulative Terra ledger.'})
    atomic_json(out/'status.json',{'status':'PREPARED_UNAUTHORIZED','paid_requests':0,'scored_freeze':'NOT READY: real pilots and external objective-source review required'})
    for folder in ['proposal','protocols','evidence']:
        for p in (out/folder).rglob('*'):
            if p.is_file():p.chmod(0o444)
    (out/'protocol.json').chmod(0o444);report(out);return check(out)


def check(out):
    out=Path(out);spec=json.loads((out/'protocol.json').read_text());changed=[]
    for name,h in spec['source_pins'].items():
        if sha(ROOT/name)!=h:changed.append(name)
    for name,h in spec['frozen_inputs'].items():
        if sha(out/name)!=h:changed.append(name)
    if sha(CORPUS/'papers.jsonl')!=spec['corpus_sha256'] or sha(CORPUS/'memberships.json')!=spec['memberships_sha256']:changed.append('corpus')
    ledger_root=Path(spec['accounting']['shared_directory']);ledger=json.loads((ledger_root/'spending-ledger.json').read_text())
    if ledger['ceiling_usd']!='68.00' or ledger['profile_sha256']!=spec['accounting']['profile_sha256']:changed.append('shared_ledger_identity')
    with queue(out) as db:
        stored=list(db.execute('SELECT * FROM jobs ORDER BY ordinal'));expected=matrix(_cases(out))
        if len(stored)!=len(expected) or any(json.loads(r['payload'])!=j or r['id']!=j['id'] or r['ordinal']!=i for i,(r,j) in enumerate(zip(stored,expected))):changed.append('frozen_queue_plan')
        counts=[list(r) for r in db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state')]
    if changed:raise ValueError('Pinned source/input/accounting identity changed: '+str(changed))
    return {'status':'OFFLINE_PIN_CHECK_PASS','planned_core':636,'study_counts':spec['core_counts'],'queue_counts':counts,'source_files':len(spec['source_pins']),
        'actual_terra_exposure_usd':ledger['cumulative_retained_exposure_usd'],'ceiling_usd':'68.00','existing_shared_ledger':str(ledger_root),'paid_execution_authorized':False,
        'scored_frozen':(out/'SCORED-FREEZE.json').exists(),'live_Terra_validation':'NOT PERFORMED by engineering','no_provider_request_made':True}


class Permit:
    def __init__(self,out,approval,scope,selection='feasibility'):
        self.out=Path(out);self.approval=Path(approval);self.digest=sha(self.approval);self.scope=scope;self.selection=selection;self.check()
    def check(self):
        if sha(self.approval)!=self.digest:raise ExperimentStopped('authorization_changed')
        a=json.loads(self.approval.read_text())
        if a.get('version')!='multihorizon-v3-paid-authorization-v1' or a.get('approved') is not True or a.get('protocol_sha256')!=sha(self.out/'protocol.json') or str(a.get('maximum_gross_usd'))!='68.00' or self.scope not in a.get('scopes',[]) or not a.get('authorization_source') or not a.get('approved_at'):
            raise ExperimentStopped('fresh_explicit_V3_paid_authorization_required')
        when=datetime.fromisoformat(a['approved_at'].replace('Z','+00:00'))
        if when.tzinfo is None or when.timestamp()>time.time()+60 or type(a.get('max_runtime_seconds')) is not int or not 150<a['max_runtime_seconds']<=86400:raise ExperimentStopped('invalid_authorization_parameters')
        check(self.out);self.record=a
        spec=json.loads((self.out/'protocol.json').read_text())
        if spec.get('scientific_revision')=='legacy-fixture-unexecutable':raise ExperimentStopped('legacy_fixture_not_executable_as_revision7')
        if self.scope.startswith('pilot-') and spec.get('pilot_selections'):
            selected=spec['pilot_selections'].get(self.selection)
            if not selected or a.get('pilot_selection_sha256')!=selected['sha256'] or str(a.get('pilot_subceiling_usd'))!=selected['ceiling_usd']:raise ExperimentStopped('explicit_pilot_subset_and_subceiling_authorization_required')
        if self.scope.startswith('main-'):
            freeze=self.out/'SCORED-FREEZE.json'
            if not freeze.exists():raise ExperimentStopped('scored_freeze_requires_real_disjoint_pilots_and_external_source_audit')
            if json.loads(freeze.read_text())['protocol_sha256']!=sha(self.out/'protocol.json'):raise ExperimentStopped('scored_freeze_identity_changed')


class CellPermit:
    def __init__(self,parent,ledger,root_id,max_calls,max_tokens):self.parent=parent;self.ledger=ledger;self.root_id=root_id;self.max_calls=max_calls;self.max_tokens=max_tokens
    def check(self):
        self.parent.check()
        with self.ledger.transaction() as db:rows=[dict(r) for r in db.execute('SELECT phase,status,diagnostic FROM attempts') if r['phase']==self.root_id or r['phase'].startswith(self.root_id+':')]
        if len(rows)>=self.max_calls:raise ExperimentStopped('trajectory_total_provider_call_ceiling')
        used=0
        for row in rows:
            d=json.loads(row['diagnostic']) if row['diagnostic'] else {};used+=d.get('prompt_tokens',0)+d.get('completion_tokens',0)
        if used+8192+4096>self.max_tokens:raise ExperimentStopped('trajectory_total_inference_token_ceiling')


def short_cell(out,job,case,gold,corpus,factory):
    directory=out/'cells'/job['id'];directory.mkdir(parents=True,exist_ok=False)
    atomic_json(directory/'question.json',[{'id':case['id'],'question':case['question'],'split':'development'}])
    factual=json.loads((V2/'protocol.json').read_text())['factual_question']
    atomic_json(directory/'factual.json',[{'query_id':case['id'],'question_id':case['id']+'-META','question':factual}])
    old=engine.LocalCorpus;engine.LocalCorpus=lambda _:corpus
    try:
        with Database(str(directory/'metadata.sqlite')) as db:
            rows=engine.run_experiments(str(CORPUS),str(directory/'question.json'),db,lambda:factory('main'),output=str(directory/'benchmark'),suite='controlled-five',methods=(job['method'],),action_budgets=(3,),trials=1,trial_seeds=(job['trial_identifier'],),
                budget=Budget(max_actions=3,max_rounds=40,max_llm_calls=7,max_inspected=200,max_api_requests=50,page_size=10,output_k=20,patience=3,max_context_chars=16000),inference_limits=InferenceLimits(max_active_tokens=4096,max_input_tokens=8192),completion_guidance=True,factual_questions_path=str(directory/'factual.json'),graph_policy='graph-visible-v2')
    finally:engine.LocalCorpus=old
    row=rows[0];result=json.loads((Path(row['directory'])/'result.json').read_text());scores=objective_score(job,gold,result)
    from .revision7_measurements import final_outcome,directed_diagnostics
    outcome=final_outcome(job['method'],scores['target_recall_at_10'],result.get('factual_responses',[]),collection_ok=result.get('stop_reason')!='retrieval_error')
    record=((result.get('factual_responses') or [{}])[0].get('evidence_record') or {})
    quote=record.get('evidence_quote','');visible=(record.get('provenance') or {}).get('assessment_context','')
    outcome['factual_quote_visible_in_final_input']=bool(quote and quote in visible) if record else None
    diagnostics=directed_diagnostics(row['directory'],gold,result)
    return {**row,**scores,**outcome,**diagnostics,'study':'S','primary_score':scores['target_recall_at_10'],'attempted':True,'directory':row['directory']}


def execute_cell(out,job,corpus,settings,ledger,permit,*,transport_factory=None):
    out=Path(out);spec=json.loads((out/'protocol.json').read_text());study=job['study'];case=next(c for c in _cases(out)[job['phase']][study] if c.get('id',c.get('case_id'))==job['task_id'])
    gold=json.loads((out/'proposal/keys.private.json').read_text())[study][job['task_id']]
    cap=spec['studies'][study];guard=CellPermit(permit,ledger,job['id'],cap['model_call_ceiling'],cap['total_token_ceiling'])
    def factory(scope):
        p=TerraPlanner(settings,ledger=ledger,cell_id=job['id']+':'+scope,architecture=job['method'],permit=guard,transport=transport_factory(scope) if transport_factory else None,receipt_directory=out/'provider')
        if transport_factory:
            p.configuration['mocked_generation']=True;p.fingerprint=hashlib.sha256(json.dumps(p.configuration,sort_keys=True).encode()).hexdigest()
        return p
    directory=out/'cells'/job['id']
    if study=='S':result=short_cell(out,job,case,gold,corpus,factory)
    elif study=='L':result=staged.execute(case,job['method'],directory,corpus,factory,gold,cancel_check=ledger.check)
    else:result=memory.execute(case,job['method'],directory,corpus,factory,gold,cancel_check=ledger.check)
    result.update({**job,'mocked_generation':transport_factory is not None,'experiment_version':VERSION})
    with ledger.transaction() as db:attempts=[dict(r) for r in db.execute('SELECT * FROM attempts') if r['phase']==job['id'] or r['phase'].startswith(job['id']+':')]
    diagnostics=[json.loads(a['diagnostic']) for a in attempts if a['diagnostic']]
    result.update(provider_attempts=len(attempts),successful_http_requests=sum(d.get('http_status')==200 for d in diagnostics),known_input_tokens=sum(d.get('prompt_tokens',0) for d in diagnostics),known_output_tokens=sum(d.get('completion_tokens',0) for d in diagnostics),
        reported_reasoning_tokens=sum(d.get('reasoning_tokens',0) for d in diagnostics),conservative_cost_usd=str(sum((Decimal(a['charge']) for a in attempts if a['charge'] is not None),Decimal(0))),unknown_usage=sum(a['status']=='UNKNOWN' for a in attempts),
        per_request_latency_seconds=[d['seconds'] for d in diagnostics if isinstance(d.get('seconds'),(int,float))],independent_human_judgments=False,protocol_sha256=sha(out/'protocol.json'))
    graph_audits=[]
    for a in attempts:
        path=out/'provider'/f'attempt-{a["id"]:05d}.graph.json'
        if path.exists():graph_audits.append(json.loads(path.read_text()))
    events=list(result.get('events',[]))
    if study=='S':
        event_path=Path(result['directory'])/'events.jsonl'
        events=[json.loads(line) for line in event_path.read_text().splitlines()]
    if study=='M':events += [e for c in result['checkpoints'] for b in c['branches'].values() for e in b['events']]
    groups=[]
    if study in ('S','L'):
        trace=Path(result['directory'])/'context'
        groups=[('main',events,[json.loads(p.read_text()) for p in sorted(trace.glob('request-*.json'))])]
    else:
        trace=directory/'stream/context'
        groups=[('stream',result.get('events',[]),[json.loads(p.read_text()) for p in sorted(trace.glob('request-*.json'))])]
        for cp in result['checkpoints']:
            for branch,b in cp['branches'].items():
                paths=sorted((directory/f'probe-{cp["papers_scheduled"]:03d}-{branch}'/'context').glob('request-*.json'))
                paths=paths[-b['model_calls']:] if b['model_calls'] else []
                groups.append((str(cp['papers_scheduled'])+branch,b['events'],[json.loads(p.read_text()) for p in paths]))
    deliveries=set()
    for scope,ev,requests in groups:
        for e in ev:
            if e.get('kind')!='context_edit' or e.get('source')!='model' or not e.get('committed') or not e.get('applied'):continue
            call=(e.get('usage') or {}).get('call_index');after=e.get('after','')
            if type(call) is not int or not after:continue
            for r in requests:
                if r['call']>call and after in r.get('input',{}).get('active_context','') and r.get('diagnostics',{}).get('wire_sha256') and not r.get('diagnostics',{}).get('request_not_sent',True):deliveries.add((scope,r['call']))
    result['modified_context_delivery_calls']=len(deliveries)
    if study in ('S','L'):
        from .models import Paper
        from .revision7_measurements import visible_records
        state=json.loads((Path(result['directory'])/'state.json').read_text());papers={p.id:p for p in map(Paper.from_dict,state.get('papers',[]))}
        visible=set();retained=set();recovered=set()
        for scope,ev,requests in groups:
            for r in requests:
                if not r.get('diagnostics',{}).get('wire_sha256') or r.get('diagnostics',{}).get('request_not_sent',True):continue
                payload=r.get('input',{});visible.update(visible_records(payload,papers)[0])
                retained.update(visible_records(payload.get('active_context',{}),papers)[0]);recovered.update(visible_records(payload.get('recovered_observation',{}),papers)[0])
        result.update(unique_acquired_metadata_records=len(papers),returned_records=sum(t.get('returned_papers',0) for t in state.get('trace',[])),
            explicit_inspection_actions=sum(a['action']=='INSPECT_PAPER' for a in state.get('history',[])),publications_delivered_to_model=len(visible),
            publications_in_CLM_inputs=len(retained),publications_in_recovery_inputs=len(recovered),evidence_visibility_measurement='Complete title/abstract-prefix identity in actual sent inputs; not proof of semantic reading')
    result['context_pressure_events']=sum(e.get('kind')=='context_capacity_failure' or e.get('kind')=='context_admission' and e.get('status')=='pending' for e in events)
    result['evidence_displacement_events']=sum(a.get('displaced_common_study_tokens',0)>0 or a.get('displaced_records',{}).get('source_records',0)>0 for a in graph_audits)+sum(e.get('kind')=='context_edit' and e.get('source')=='model' and e.get('committed') is True and e.get('after_chars',0)<e.get('before_chars',0) for e in events)
    latency=result['per_request_latency_seconds']
    import numpy as np
    result.update(graph_delivery_calls=sum(a['delivered_edges']>0 for a in graph_audits),verified_graph_edges_transmitted=sum(a['delivered_edges'] for a in graph_audits),
        graph_eligible_calls=sum(a.get('eligible_edges',0)>0 for a in graph_audits),graph_capacity_exclusions=sum(a.get('excluded_by_capacity',0) for a in graph_audits),
        graph_tokens_transmitted=sum(a['delivered_graph_tokens'] for a in graph_audits),context_edits_committed=sum(e.get('kind')=='context_edit' and e.get('source')=='model' and e.get('committed') and e.get('applied') for e in events),
        accepted_recoveries=sum(e.get('kind')=='context_recovery' and e.get('accepted') is True for e in events),context_limit_events=sum(e.get('kind') in ('context_capacity_failure','context_budget_stop') or e.get('kind')=='context_admission' and e.get('status')=='pending' for e in events),
        latency_mean_seconds=float(np.mean(latency)) if latency else None,latency_median_seconds=float(np.median(latency)) if latency else None,latency_p95_seconds=float(np.percentile(latency,95)) if latency else None,latency_max_seconds=max(latency) if latency else None,
        latency_samples=len(latency),latency_interpretation='Observed model-request latency; small-n tail estimates descriptive; task-stage delays excluded',
        fallback_events=sum(e.get('kind')=='invalid_decision' for e in events),final_context_compactions=sum(e.get('kind')=='final_context_preparation' and e.get('status')!='unchanged_fits' for e in events),
        final_capacity_failures=sum(e.get('kind')=='final_context_preparation' and e.get('status') in ('required_envelope_does_not_fit','no_complete_evidence_unit_fits') for e in events))
    atomic_json(directory/'summary.json',result);return result


def run(out,approval,phase,study):
    # Route every new execution through the same isolated, bounded supervisor.
    from .multihorizon_supervisor import supervise
    return supervise(out,approval,phase,study,workers=1)


def report(out):
    from .multihorizon_reporting import export
    out=Path(out)
    with (out/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return export(out)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ['prepare','check','report','run','pilot-gates','freeze']:
        p=sub.add_parser(name);p.add_argument('--campaign',type=Path,default=DEFAULT)
        if name=='prepare':p.add_argument('--cases',type=Path,required=True)
        if name=='run':p.add_argument('--approval',type=Path,required=True);p.add_argument('--phase',choices=['pilot','main'],required=True);p.add_argument('--study',choices=['S','L','M'],required=True)
        if name=='freeze':p.add_argument('--external-review',type=Path,required=True)
    a=parser.parse_args(argv)
    try:
        if a.command=='prepare':r=prepare(a.campaign,a.cases)
        elif a.command=='check':r=check(a.campaign)
        elif a.command=='report':r=report(a.campaign)
        elif a.command=='run':r=run(a.campaign,a.approval,a.phase,a.study)
        else:
            from .multihorizon_reporting import pilot_gates,freeze
            r=pilot_gates(a.campaign) if a.command=='pilot-gates' else freeze(a.campaign,a.external_review)
    except (ValueError,ExperimentStopped,FileNotFoundError) as exc:print(json.dumps({'status':'BLOCKED','reason':getattr(exc,'reason',str(exc))}));return 2
    print(json.dumps(r,indent=2));return 2 if r.get('status') in ('BLOCKED','STOPPED') else 0

if __name__=='__main__':raise SystemExit(main())
