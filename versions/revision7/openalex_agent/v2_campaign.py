"""Prepare/check/execute separately authorized Terra V2 via the existing engine.

Preparation and checking never load credentials or issue provider requests.
"""
from pathlib import Path
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,ast,csv,fcntl,hashlib,json,os,shutil,sqlite3,time,traceback
from decimal import Decimal
from .corpus import atomic_json,LocalCorpus
from .controller import Budget
from .context_planner import InferenceLimits
from .storage import Database
from .config import Settings
from .llm import DeterministicPlanner
from .execution_control import ExperimentStopped
from .terra_accounting import TerraLedger,CEILING
from .terra_v2 import TerraPlanner,BASE_URL,MODEL
from .graph_evidence import VERSION as GRAPH_POLICY
from . import experiments as engine
from .v2_evaluation import objective_score

ROOT=Path(__file__).resolve().parents[1]
V1=Path('/home/bs/Documents/OpenAlex-Agent-Integration-2026-11-08-v1/data/comprehensive-scientific-2026-11-09-v1')
DEFAULT=ROOT/'data/terra-graph-visible-v2-2026-11-09-v3'
METHODS=('standard','llm','llm_graph','clm','clm_graph')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


@contextmanager
def queue(out):
    db=sqlite3.connect(Path(out)/'queue.sqlite',timeout=30);db.row_factory=sqlite3.Row
    db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
    try:yield db;db.commit()
    except BaseException:db.rollback();raise
    finally:db.close()


def plan(main,pilot):
    jobs=[]
    for stage,tasks,horizons,reps in [('A-pilot',pilot,(3,),1),('main',main,(3,10),3)]:
        ordered=sorted(tasks,key=lambda q:hashlib.sha256(('queue-v1|'+q['id']).encode()).hexdigest())
        for horizon in horizons:
            phase=stage if stage=='A-pilot' else 'B-main' if horizon==3 else 'C-long'
            for qi,task in enumerate(ordered):
                for rep in range(reps):
                    offset=(qi+rep)%5
                    for method in METHODS[offset:]+METHODS[:offset]:
                        jobs.append({'id':f'{phase}-{task["id"]}-a{horizon}-r{rep}-{method}','stage':phase,'task_id':task['id'],'question':task['question'],
                            'method':method,'action_budget':horizon,'repetition':rep})
    return jobs


def prepare(out):
    out=Path(out).resolve()
    if out.exists():raise ValueError('Preparation is immutable; choose a fresh output directory')
    out.mkdir(parents=True);(out/'proposal/keys').mkdir(parents=True);(out/'tools').mkdir()
    original=json.loads((V1/'protocol.json').read_text())
    for name in ['main-agent.json','pilot-agent.json']:
        shutil.copyfile(V1/'tasks'/name,out/'proposal'/name)
    shutil.copyfile(V1/'private-keys/answer-keys.json',out/'proposal/keys/answer-keys.json')
    shutil.copyfile(V1/'analysis-plan.json',out/'proposal/analysis-plan.json')
    keys=json.loads((out/'proposal/keys/answer-keys.json').read_text())
    atomic_json(out/'proposal/task-metadata.json',[{k:r[k] for k in ['id','domain','family','difficulty','split']} for r in keys])
    # Existing pure statistical function, not the V1 writer with Gemini-specific text.
    text=(V1/'tools/reporting.py').read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='task_bootstrap')
    (out/'tools/reporting.py').write_text('"""Unchanged official task-bootstrap function only; read-only consumers."""\nimport numpy as np\n\n'+ast.get_source_segment(text,node)+'\n')
    source={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'openalex_agent').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    source.update({name:sha(ROOT/name) for name in ['pyproject.toml','requirements.lock.txt']})
    history=Path('/home/bs/Documents/NewProject/data/graph-ablation-pilot-2026-11-02-openai-terra/experiment.json')
    old=json.loads(history.read_text());assert old['provider_configuration']['endpoint_hash']==hashlib.sha256((BASE_URL+'/').encode()).hexdigest()
    pricing=ROOT/'engineering/pricing-verification.json'
    if not pricing.exists():raise ValueError('Verified pricing evidence must be recorded before preparation')
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'version':'terra-graph-visible-development-v2','experimental_implementation_version':'graph-visible-terra-v2','graph_evidence_policy':GRAPH_POLICY,
        'scope':'development-only objective benchmark; no heldout or independent human judgments','provider':{'provider':'openai','base_url':BASE_URL,'model':MODEL,
            'protocol':'chat-completions-json-object','reasoning_effort':'low','temperature':None,'max_output_tokens':4096,'timeout':120,'service_tier':'default',
            'input_price':1.0,'output_price':5.0,'conservative_input_accounting_price':2.5},
        'provider_history':{'path':str(history),'sha256':sha(history),'endpoint_hash':old['provider_configuration']['endpoint_hash']},
        'pricing_evidence':{'path':str(pricing),'sha256':sha(pricing)},'maximum_gross_usd':'68.00','one_ledger_for_pilot_and_main':True,
        'credits_verified':False,'paid_execution_authorized':False,'prior_google_expenses_included':False,'max_runtime_seconds':86400,
        'methods':list(METHODS),'main_tasks':30,'repetitions':3,'horizons':[3,10],'main_cells':900,'pilot_cells':30,'trial_identifiers':[11,23,37],
        'corpus':original['corpus'],'corpus_sha256':original['corpus_sha256'],'context_policy':original['context_policy'],'completion_guidance':original['completion_guidance'],
        'budgets':original['budgets'],'factual_question':original['factual_question'],'source_pins':source,
        'immutable_files':{str(p.relative_to(out)):sha(p) for p in (out/'proposal').rglob('*') if p.is_file()},
        'analysis_plan_sha256':sha(out/'analysis-plan.json') if (out/'analysis-plan.json').exists() else sha(out/'proposal/analysis-plan.json'),
        'bootstrap_source_sha256':sha(out/'tools/reporting.py'),'frozen_v1_question_source_sha256':sha(V1/'tasks/main-agent.json'),
        'scientific_comparability':'Provider and graph evidence allocation both change; V1/V2 differences cannot isolate either intervention.'}
    atomic_json(out/'protocol.json',protocol);shutil.copyfile(out/'proposal/analysis-plan.json',out/'analysis-plan.json')
    atomic_json(out/'approval-template.json',{'version':'terra-v2-development-approval-v1','approved':False,'authorization_source':None,'approved_at':None,
        'campaign_protocol_sha256':sha(out/'protocol.json'),'maximum_gross_usd':'68.00','max_runtime_seconds':86400,'stages':[],
        'note':'Template only. Engineering phase has no paid authorization. Future user authorization must be recorded truthfully; no credit purchase is assumed.'})
    main=json.loads((out/'proposal/main-agent.json').read_text());pilot=json.loads((out/'proposal/pilot-agent.json').read_text());jobs=plan(main,pilot)
    assert sum(j['stage']!='A-pilot' for j in jobs)==900
    atomic_json(out/'queue-plan.json',jobs)
    with queue(out) as db:
        db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,ordinal INTEGER,stage TEXT,payload TEXT,state TEXT,started REAL,ended REAL,error TEXT)')
        for i,j in enumerate(jobs):db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)',(j['id'],i,j['stage'],json.dumps(j),'PENDING',None,None,None))
    TerraLedger(out,profile_hash=sha(out/'protocol.json'))
    summarize(out)
    atomic_json(out/'status.json',{'status':'prepared','stage':None,'provider':'openai','main_trajectories_planned':900,'hosted_requests':0,'execution_authorized':False})
    for p in [out/'protocol.json',*list((out/'proposal').rglob('*'))]:
        if p.is_file():p.chmod(0o444)
    return check(out)


def check(out,*,full_corpus=True):
    out=Path(out);spec=json.loads((out/'protocol.json').read_text())
    if spec['maximum_gross_usd']!='68.00' or spec['main_cells']!=900:raise ValueError('Unexpected campaign identity')
    changed=[k for k,v in spec['source_pins'].items() if sha(ROOT/k)!=v]
    changed += [k for k,v in spec['immutable_files'].items() if sha(out/k)!=v]
    if sha(out/'tools/reporting.py')!=spec['bootstrap_source_sha256']:changed.append('bootstrap')
    if sha(Path(spec['pricing_evidence']['path']))!=spec['pricing_evidence']['sha256']:changed.append('pricing')
    pricing=json.loads(Path(spec['pricing_evidence']['path']).read_text())
    changed += [r['file'] for r in pricing['sources'] if sha(ROOT/r['file'])!=r['sha256']]
    if sha(out/'analysis-plan.json')!=spec['analysis_plan_sha256']:changed.append('analysis-plan.json')
    if changed:raise ValueError('Changed pinned inputs: '+','.join(changed))
    if full_corpus and sha(Path(spec['corpus'])/'papers.jsonl')!=spec['corpus_sha256']:raise ValueError('Corpus changed')
    with queue(out) as db:
        stored=list(db.execute('SELECT id,ordinal,stage,payload FROM jobs ORDER BY ordinal'))
        expected=plan(json.loads((out/'proposal/main-agent.json').read_text()),json.loads((out/'proposal/pilot-agent.json').read_text()))
        if len(stored)!=len(expected) or any(r['id']!=j['id'] or r['ordinal']!=i or r['stage']!=j['stage'] or json.loads(r['payload'])!=j for i,(r,j) in enumerate(zip(stored,expected))):raise ValueError('Frozen queue identity changed')
        counts=[list(r) for r in db.execute('SELECT stage,state,COUNT(*) FROM jobs GROUP BY stage,state')]
    return {'version':spec['version'],'source_files_checked':len(spec['source_pins']),'frozen_inputs_checked':len(spec['immutable_files']),
        'source_and_inputs_unchanged':True,'matrix_cells':900,'queue_counts':counts,'gross_ceiling_usd':'68.00','paid_execution_authorized':False,
        'credit_balance_verified':False,'live_provider_validation':'Not performed; requires separate authorization','no_hosted_calls_during_check':True}


class ExecutionPermit:
    def __init__(self,out,approval,stage):
        self.out=Path(out);self.approval=Path(approval);self.stage=stage;self.digest=sha(self.approval)
        self.check()

    def check(self):
        if sha(self.approval)!=self.digest:raise ExperimentStopped('authorization_changed')
        spec=json.loads((self.out/'protocol.json').read_text());a=json.loads(self.approval.read_text())
        if (a.get('approved') is not True or a.get('version')!='terra-v2-development-approval-v1'
            or a.get('campaign_protocol_sha256')!=sha(self.out/'protocol.json') or str(a.get('maximum_gross_usd'))!='68.00'
            or self.stage not in a.get('stages',[]) or not a.get('authorization_source') or not a.get('approved_at')
            or type(a.get('max_runtime_seconds')) is not int or not 150<a['max_runtime_seconds']<=86400):
            raise ExperimentStopped('explicit_v2_paid_authorization_required')
        when=datetime.fromisoformat(a['approved_at'].replace('Z','+00:00'))
        if when.tzinfo is None or when.timestamp()>time.time()+60:raise ExperimentStopped('invalid_authorization_timestamp')
        check(self.out,full_corpus=False)
        self.record=a


def run_cell(out,job,corpus,ledger,permit,settings):
    directory=out/'cells'/job['id'];directory.mkdir(parents=True,exist_ok=False)
    atomic_json(directory/'agent-question.json',[{'id':job['task_id'],'question':job['question'],'split':'development'}])
    spec=json.loads((out/'protocol.json').read_text())
    atomic_json(directory/'factual-question.json',[{'query_id':job['task_id'],'question_id':job['task_id']+'-META','question':spec['factual_question']}])
    h=job['action_budget'];budget=Budget(max_actions=h,max_rounds=40,max_inspected=200,max_api_requests=50,max_llm_calls=2*h+1,page_size=10,output_k=20,patience=3,max_context_chars=16000)
    limits=InferenceLimits(max_total_tokens=100000,max_active_tokens=4096,max_input_tokens=8192)
    factory=lambda:TerraPlanner(settings,ledger=ledger,cell_id=job['id'],architecture=job['method'],permit=permit)
    original=engine.LocalCorpus;engine.LocalCorpus=lambda path:corpus if Path(path).resolve()==Path(spec['corpus']).resolve() else (_ for _ in ()).throw(ValueError('Corpus identity mismatch'))
    try:
        with Database(str(directory/'metadata.sqlite')) as db:
            rows=engine.run_experiments(spec['corpus'],str(directory/'agent-question.json'),db,factory,output=str(directory/'benchmark'),suite='controlled-five',
                methods=(job['method'],),action_budgets=(h,),trials=1,trial_seeds=(spec['trial_identifiers'][job['repetition']],),budget=budget,inference_limits=limits,
                completion_guidance=True,factual_questions_path=str(directory/'factual-question.json'),cancel_check=lambda:(ledger.check() or False),graph_policy=GRAPH_POLICY)
    finally:engine.LocalCorpus=original
    r=rows[0];result=json.loads((Path(r['directory'])/'result.json').read_text())
    key=next(k for k in json.loads((out/'proposal/keys/answer-keys.json').read_text()) if k['id']==job['task_id'])
    scores=objective_score(job,key,result)
    delivery=[json.loads(p.read_text()).get('graph_evidence_delivery') for p in (Path(r['directory'])/'context').glob('request-*.json')]
    delivery=[d for d in delivery if d is not None]
    with ledger.transaction() as db:charges=[dict(a) for a in db.execute('SELECT * FROM attempts WHERE phase=?',(job['id'],))]
    summary={**job,**r,**scores,'repetition':job['repetition'],'trial_identifier':spec['trial_identifiers'][job['repetition']],
        'domain':key['domain'],'family':key['family'],'difficulty':key['difficulty'],'terminal':True,'collection_status':'completed',
        'experimental_version':spec['version'],'graph_evidence_policy':GRAPH_POLICY,'graph_verified_delivery_calls':sum(d['delivered_edges']>0 for d in delivery),
        'graph_verified_delivered_edges':sum(d['delivered_edges'] for d in delivery),'graph_capacity_exclusions':sum(d.get('excluded_by_capacity',0) for d in delivery),
        'graph_study_tokens_delivered':sum(d['delivered_graph_tokens'] for d in delivery),'provider_attempts':len(charges),
        'ledger_gross_estimate_usd':str(sum((Decimal(a['charge']) for a in charges if a['charge'] is not None),Decimal(0))),
        'unknown_usage_attempts':sum(a['status']=='UNKNOWN' for a in charges),'objective_keys_passed_to_model':False,'independent_human_evaluation':False}
    atomic_json(directory/'summary.json',summary)
    return summary


def run(out,approval,stage):
    out=Path(out).resolve()
    # Authorization is checked before credentials/settings, corpus or networking.
    permit=ExecutionPermit(out,approval,stage)
    lock=(out/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'FINALIZED.json').exists():raise ValueError('Completed outputs are immutable')
    check(out)
    settings=Settings.from_env()
    if (settings.llm_base_url.rstrip('/')!=BASE_URL or settings.llm_model!=MODEL or settings.llm_reasoning_effort!='low' or settings.llm_temperature is not None or settings.llm_max_tokens!=4096 or settings.llm_timeout!=120 or settings.llm_input_price!=1 or settings.llm_output_price!=5 or not settings.llm_api_key):raise ValueError('Set the exact .env.terra-v2.example profile and a key before execution')
    ledger=TerraLedger(out,profile_hash=sha(out/'protocol.json'))
    ledger.activate(permit.digest,max_runtime_seconds=permit.record['max_runtime_seconds'])
    recovered=ledger.recover_interrupted()
    with queue(out) as db:db.execute("UPDATE jobs SET state='INTERRUPTED',ended=?,error='accounted_interruption_no_replay' WHERE state='RUNNING'",(time.time(),))
    if recovered:
        atomic_json(out/'status.json',{'status':'stopped_by_safeguard','stop_reason':'interrupted_provider_usage_requires_reconciliation','provider':'openai'})
        summarize(out)
        raise ExperimentStopped('interrupted_provider_usage_requires_reconciliation')
    ledger.check()
    (out/'runner-pid.txt').write_text(str(os.getpid())+'\n')
    corpus=LocalCorpus(json.loads((out/'protocol.json').read_text())['corpus'])
    state={'status':'running','stage':stage,'started_epoch':time.time(),'provider':'openai','experimental_version':'terra-graph-visible-development-v2'}
    atomic_json(out/'status.json',state)
    try:
        stages=['A-pilot'] if stage=='pilot' else ['B-main','C-long']
        for phase in stages:
            state['stage']=phase;atomic_json(out/'status.json',state)
            while True:
                permit.check();ledger.check()
                with queue(out) as db:
                    stored=db.execute("SELECT * FROM jobs WHERE stage=? AND state='PENDING' ORDER BY ordinal LIMIT 1",(phase,)).fetchone()
                    if stored is None:break
                    job=json.loads(stored['payload']);db.execute("UPDATE jobs SET state='RUNNING',started=? WHERE id=?",(time.time(),job['id']))
                terminal='FAILED';error=None
                try:run_cell(out,job,corpus,ledger,permit,settings);terminal='COMPLETED'
                except BaseException as exc:
                    error=getattr(exc,'reason',getattr(exc,'code',type(exc).__name__))
                    atomic_json(out/'cells'/job['id']/'failure.json',{'id':job['id'],'error_code':error,'failure_is_not_retried':True})
                    raise
                finally:
                    with queue(out) as db:db.execute('UPDATE jobs SET state=?,ended=?,error=? WHERE id=?',(terminal,time.time(),error,job['id']))
                    summarize(out)
            summarize(out)
        state.update(status='pilot_complete' if stage=='pilot' else 'complete_main_collection',finished_epoch=time.time())
        if stage=='main':atomic_json(out/'FINALIZED.json',{'status':'complete_main_collection','source_protocol_sha256':sha(out/'protocol.json'),'completed_epoch':time.time(),'no_independent_human_evaluation':True})
    except BaseException as exc:
        state.update(status='stopped_by_safeguard' if isinstance(exc,ExperimentStopped) else 'failed',stop_reason=getattr(exc,'reason',type(exc).__name__),finished_epoch=time.time())
    finally:atomic_json(out/'status.json',state);summarize(out)
    return state


def summarize(out):
    out=Path(out);ledger=json.loads((out/'spending-ledger.json').read_text());charges={}
    for a in ledger['attempts']:charges.setdefault(a['phase'],[]).append(a)
    with queue(out) as db:jobs=[dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY ordinal')]
    rows=[]
    for q in jobs:
        j=json.loads(q['payload']);path=out/'cells'/j['id']/'summary.json'
        value=json.loads(path.read_text()) if path.exists() else {k:v for k,v in j.items() if k!='question'}
        value.update(collection_status=q['state'],error=q['error'])
        rows.append(value)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    tmp=out/'RESULTS.csv.tmp'
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    tmp.replace(out/'RESULTS.csv')
    completed=sum(r['collection_status']=='COMPLETED' for r in rows);failed=sum(r['collection_status'] in ('FAILED','INTERRUPTED') for r in rows)
    groups=[];projected_remaining=Decimal(0);projection_covered=True
    for horizon in [3,10]:
        for method in METHODS:
            planned=[r for r in rows if r['stage']!='A-pilot' and r['action_budget']==horizon and r['method']==method]
            observed=[r for r in rows if r['action_budget']==horizon and r['method']==method and r['collection_status']=='COMPLETED']
            amounts=[sum((Decimal(a['charge']) for a in charges.get(r['id'],[]) if a.get('charge') is not None),Decimal(0)) for r in observed]
            mean=sum(amounts,Decimal(0))/len(amounts) if amounts else Decimal(0) if method=='standard' else None
            remaining=sum(r['collection_status']=='PENDING' for r in planned)
            if remaining and mean is None:projection_covered=False
            elif mean is not None:projected_remaining+=mean*remaining
            groups.append({'method':method,'action_budget':horizon,'successful_observations':len(observed),'remaining_main_cells':remaining,'mean_conservative_cost_usd':str(mean) if mean is not None else None})
    projection=Decimal(ledger['cumulative_retained_exposure_usd'])+projected_remaining if projection_covered and not any(a['status']=='UNKNOWN' for a in ledger['attempts']) else None
    atomic_json(out/'progress.json',{'planned_main':900,'successful_all_stages':completed,'failed_all_stages':failed,'cumulative_exposure_usd':ledger['cumulative_retained_exposure_usd'],
        'remaining_authorization_usd':ledger['remaining_authorization_usd'],'new_confirmed_conservative_usd':ledger['confirmed_upper_bound_gross_usd'],
        'average_known_conservative_cost_per_completed_trajectory_usd':float(Decimal(ledger['confirmed_upper_bound_gross_usd'])/completed) if completed else None,
        'observed_method_horizon_costs':groups,'projected_main_completion_including_spent_pilot_cost_usd':str(projection) if projection is not None else None,
        'full_main_collection_within_68_at_observed_mix':projection<=CEILING if projection is not None else None,
        'projection_limitation':'No projection until every unfinished hosted method/horizon has observations; includes current exposure, no future retries. Small samples, failures and changing trajectory lengths prevent guarantees.',
        'credit_balance_verified':False,'independent_grounding':'NOT EVALUATED'})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ['prepare','check','run']:
        p=sub.add_parser(name);p.add_argument('--campaign',type=Path,default=DEFAULT)
        if name=='run':p.add_argument('--approval',type=Path,required=True);p.add_argument('--stage',choices=['pilot','main'],required=True)
    args=parser.parse_args(argv)
    try:
        result=prepare(args.campaign) if args.command=='prepare' else check(args.campaign) if args.command=='check' else run(args.campaign,args.approval,args.stage)
    except (ValueError,ExperimentStopped) as exc:
        print(json.dumps({'status':'blocked','reason':getattr(exc,'reason',str(exc))}));return 2
    print(json.dumps(result,indent=2));return 2 if result.get('status') in ('failed','stopped_by_safeguard') else 0

if __name__=='__main__':raise SystemExit(main())
