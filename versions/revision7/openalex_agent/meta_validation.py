"""Prepared, approval-pinned development checks using the existing spend guard.

Nothing in prepare() calls a hosted provider. execute() requires fresh explicit
approval; original pilot approvals and frozen outputs are never reused.
"""
from dataclasses import asdict, replace
import hashlib
import csv
import json
from pathlib import Path

from .context_planner import InferenceLimits
from .context_memory import serialized
from .controlled import ALL_ARCHITECTURES, META_ARCHITECTURES, ControlledController
from .controller import Budget
from .corpus import LocalCorpus, atomic_json
from .development_pilot import SpendLedger, REQUEST_RESERVE
from .execution_control import ExperimentStopped
from .experiments import _capture_requests, run_experiments
from .meta_context import META_VERSION, reference_provenance
from .meta_planner import MetaContextPlanner
from .responses import HostedResponsesPlanner
from .storage import Database

VERSION='meta-development-validation-v2-tool-contract'
EXPECTED_CORPUS_SHA='0b3f693c8c1e64a232ed6012a26877430edf3373bb0eb6dc815618a68b7a1614'
CANARY_INSTRUCTION=(
    'This is an explicitly prompted synchronization CANARY, not a quality benchmark. '
    'Before your first scholarly acquisition, use context_file to replace the editable conversation '
    'with a short retained observation containing one actually observed paper ID and title. '
    'Then choose an eligible scholarly action relevant to the original research question. '
    'Do not invent metadata. Failure to edit or continue will be reported, not retried.'
)
CANARY_CRITERIA=(
    'Hosted model generates the native context_file edit; executor accepts its unchanged arguments.',
    'Official Meta synchronizer commits that edit and retains an actually observed paper ID and title.',
    'A later successful model request receives the modified conversation.',
    'That request or a later one generates a valid nonfallback scholarly acquisition after the edit.',
)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def code_pins():
    files=[p for root in ('openalex_agent','scripts') for p in Path(root).rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    files += [Path(name) for name in ('pyproject.toml','requirements.lock.txt')]
    return {str(p):sha(p) for p in sorted(files)}


def profile(settings):
    value=dict(model=settings.llm_model,base_url=settings.llm_base_url.rstrip('/')+'/',
        reasoning_effort=settings.llm_reasoning_effort,temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_tokens,timeout=settings.llm_timeout,
        input_usd_per_million=1,output_usd_per_million=5,transport='responses-stateless-v1')
    if value!=dict(model='gpt-6.1-terra',base_url='https://api.openai.com/v1/',reasoning_effort='low',
        temperature=None,max_output_tokens=4096,timeout=120,input_usd_per_million=1,output_usd_per_million=5,
        transport='responses-stateless-v1'):
        raise ValueError('Preserve the pinned Terra profile; no automatic provider/model/generation changes')
    return value


def prepare(path,settings,*,mode='canary',corpus='data/corpus/multidisciplinary-2026-11-06-v1',
            questions='data/hosted-development-preflight-2026-11-06-v1/questions.json',canary_status=None,backend=None):
    path=Path(path)
    if path.exists():raise ValueError('Use a fresh preparation directory')
    if mode not in ('canary','comparison'):raise ValueError('Choose canary or comparison')
    data=json.loads(Path(questions).read_text())
    if len(data)!=3 or any(q.get('split')!='development' for q in data):raise ValueError('Use the three frozen development questions; no held-out questions')
    receipt=json.loads((Path(corpus)/'freeze-receipt.json').read_text())
    if sha(Path(corpus)/'papers.jsonl')!=receipt['corpus_sha256'] or receipt['corpus_sha256']!=EXPECTED_CORPUS_SHA:
        raise ValueError('Frozen corpus mismatch')
    provider=backend.profile(settings) if backend else profile(settings)
    path.mkdir(parents=True)
    selected=data[:1] if mode=='canary' else data
    atomic_json(path/'questions.json',selected)
    budget=Budget(max_actions=2 if mode=='canary' else 10,max_rounds=20,max_inspected=200,page_size=10,
                  max_llm_calls=4 if mode=='canary' else 20)
    spec=dict(protocol=backend.protocol if backend else VERSION,mode=mode,purpose='UNSCORED DEVELOPMENT ONLY',corpus=str(corpus),
        corpus_sha256=receipt['corpus_sha256'],questions=str(path/'questions.json'),questions_sha256=sha(path/'questions.json'),
        parent_questions=str(questions),parent_questions_sha256=sha(questions),output=str(path/'execution'),
        provider=provider,budget=asdict(budget),limits=asdict(InferenceLimits(max_active_tokens=4096,max_input_tokens=8192)),
        methods=list(META_ARCHITECTURES if mode=='canary' else ALL_ARCHITECTURES),trials=1 if mode=='canary' else 2,
        seeds=[11] if mode=='canary' else [11,23],spending_ceiling_usd=3 if mode=='canary' else 6,
        checkpoint_usd=3,wall_seconds=600 if mode=='canary' else 7200,max_provider_calls=8 if mode=='canary' else 720,
        request_reservation_usd=str(REQUEST_RESERVE),canary_instruction=CANARY_INSTRUCTION if mode=='canary' else None,
        canary_success_criteria=list(CANARY_CRITERIA) if mode=='canary' else None,
        planning_latency_measurement='Per-request diagnostics.seconds; end-to-end worker/auth/tokenization/transport/generation/response processing, not provider-only compute',
        implementation_version=META_VERSION,meta_reference=reference_provenance(),source_sha256=code_pins(),
        canary_status_file=str(canary_status) if canary_status else None,
        canary_status_sha256=sha(canary_status) if canary_status else None,
        final_evaluation_authorized=False)
    atomic_json(path/'spec.json',spec)
    atomic_json(path/'approval-template.json',dict(approved=False,scope=mode+'_development_only',
        spec_sha256=sha(path/'spec.json'),approver='',approved_at='',final_evaluation_authorized=False))
    return spec


def audit_canary(directory):
    """Mechanical synchronization/action evidence, not semantic quality judgments."""
    directory=Path(directory)
    requests=[json.loads(p.read_text()) for p in sorted((directory/'context').glob('request-*.json'))]
    edits=[json.loads(p.read_text()) for p in sorted((directory/'context').glob('meta-edit-*.json'))]
    observations=[json.loads(p.read_text()) for p in sorted((directory/'context').glob('observation-*.json'))]
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    rejected=set();call=None
    for event in events:
        if event['kind']=='context_model_call':call=event['call']
        elif event['kind']=='invalid_decision':rejected.add(call)
    proofs=[];attempts=[]
    for edit in edits:
        request=next((r for r in requests if r['call']==edit['usage']['call_index']),{})
        native=request.get('native_output') or {}
        generated=(edit.get('source')=='model' and native.get('name')=='context_file'
            and native.get('arguments')==edit['operation'] and request.get('diagnostics',{}).get('http_status')==200)
        attempt=dict(edit_index=edit['index'],call=edit['usage']['call_index'],
            model_generated_edit=generated,executor_accepted=edit.get('executor_accepted',edit['accepted']),
            executor_return_code=edit.get('executor_return_code'),synchronizer_committed=edit['committed'],
            official_reference_syncs=edit.get('reference_syncs',0))
        attempts.append(attempt)
        if not (generated and attempt['executor_accepted'] and edit['committed'] and attempt['official_reference_syncs']>0):continue
        after=edit['after'];text='\n'.join(m.get('content','') for m in after[2:])
        retained=[paper['id'] for observation in observations for paper in observation.get('top_papers',[])
                  if paper['id'] in text and paper.get('title') and paper['title'] in text]
        deliveries=[r['call'] for r in requests if r['call']>edit['usage']['call_index']
                    and r.get('diagnostics',{}).get('http_status')==200
                    and r['input']['messages'][:len(after)]==after]
        proofs.append(dict(edit_index=edit['index'],retained_observed_paper_ids=sorted(set(retained)),
                           later_successful_delivery_calls=deliveries))
    acquisitions={'SEARCH_PAPERS','EXPAND_REFERENCES','EXPAND_CITATIONS','EXPLORE_TOPIC','INSPECT_PAPER'}
    accepted=[r['call'] for r in requests if (r.get('output') or {}).get('decision',{}).get('action') in acquisitions
              and r['call'] not in rejected and r.get('diagnostics',{}).get('http_status')==200]
    for proof in proofs:
        proof['accepted_acquisition_after_delivery_calls']=[call for call in accepted
            if any(call>=delivery for delivery in proof['later_successful_delivery_calls'])]
        proof['passed']=bool(proof['retained_observed_paper_ids'] and proof['later_successful_delivery_calls']
            and proof['accepted_acquisition_after_delivery_calls'])
    timings=[dict(call=r['call'],seconds=r.get('diagnostics',{}).get('seconds'),
                  operation=next(iter(r.get('output') or {}),'failure')) for r in requests]
    measured=[r['seconds'] for r in timings if isinstance(r['seconds'],(int,float))]
    return dict(run_directory=str(directory),edit_attempts=attempts,edit_proofs=proofs,
        accepted_model_acquisition_calls=accepted,planning_latency=dict(calls=timings,
            measured_calls=len(measured),total_seconds=sum(measured),
            mean_seconds=sum(measured)/len(measured) if measured else None,
            measurement='End-to-end hosted planner request; not provider-only compute'),
        passed=any(p['passed'] for p in proofs))


def guarded_factory(settings,ledger,*,transport=None):
    class GuardedResponses(HostedResponsesPlanner):
        def _request(self,body):
            attempt=ledger.reserve()
            try:return super()._request(body)
            finally:ledger.settle(attempt,self.last_call)
    return lambda:GuardedResponses(settings,transport=transport)


def execute(spec_path,approval_path,settings,*,transport=None,backend=None):
    spec=json.loads(Path(spec_path).read_text());approval=json.loads(Path(approval_path).read_text())
    if (spec.get('protocol')!=(backend.protocol if backend else VERSION) or spec.get('mode') not in ('canary','comparison') or
        approval.get('approved') is not True or approval.get('scope')!=spec['mode']+'_development_only' or
        approval.get('spec_sha256')!=sha(spec_path) or not approval.get('approver') or not approval.get('approved_at') or
        approval.get('final_evaluation_authorized') is not False):
        raise ValueError('Fresh explicit development-only approval is required; no requests issued')
    expected_profile=backend.profile(settings) if backend else profile(settings)
    if spec['provider']!=expected_profile or spec['source_sha256']!=code_pins():raise ValueError('Source/provider pins changed; prepare and review again')
    if spec['final_evaluation_authorized'] is not False or spec['checkpoint_usd']!=3 or spec['request_reservation_usd']!=str(REQUEST_RESERVE):
        raise ValueError('Preserve development scope and existing spending safeguards')
    questions=json.loads(Path(spec['questions']).read_text());parent=json.loads(Path(spec['parent_questions']).read_text())
    if sha(spec['questions'])!=spec['questions_sha256'] or sha(spec['parent_questions'])!=spec['parent_questions_sha256']:
        raise ValueError('Development question pins changed')
    if any(q.get('split')!='development' or q not in parent for q in questions):raise ValueError('Development questions only')
    corpus=LocalCorpus(spec['corpus'])
    if corpus.fingerprint!=spec['corpus_sha256'] or corpus.fingerprint!=EXPECTED_CORPUS_SHA:raise ValueError('Corpus pin mismatch')
    expected=(list(META_ARCHITECTURES),1,8,3) if spec['mode']=='canary' else (list(ALL_ARCHITECTURES),2,720,6)
    if (spec['methods'],spec['trials'],spec['max_provider_calls'],spec['spending_ceiling_usd'])!=expected:
        raise ValueError('Unexpected development workload or spending ceiling')
    expected_budget=Budget(max_actions=2 if spec['mode']=='canary' else 10,max_rounds=20,max_inspected=200,
        page_size=10,max_llm_calls=4 if spec['mode']=='canary' else 20)
    if (spec['budget']!=asdict(expected_budget) or
        spec['limits']!=asdict(InferenceLimits(max_active_tokens=4096,max_input_tokens=8192)) or
        spec['canary_instruction']!=(CANARY_INSTRUCTION if spec['mode']=='canary' else None) or
        spec['wall_seconds']!=(600 if spec['mode']=='canary' else 7200) or
        len(questions)!=(1 if spec['mode']=='canary' else 3) or spec['seeds']!=([11] if spec['mode']=='canary' else [11,23])):
        raise ValueError('Preserve the reviewed workload, generation task and inference ceilings')
    if spec['mode']=='comparison':
        prerequisite=spec.get('canary_status_file')
        if not prerequisite or sha(prerequisite)!=spec.get('canary_status_sha256'):
            raise ValueError('A successful separately authorized live Meta canary must be pinned before comparison')
        passed=json.loads(Path(prerequisite).read_text())
        if passed.get('canary_mechanism_passed') is not True or passed.get('classification')!='Development only; not final scientific results':
            raise ValueError('A mock or failed canary cannot authorize the development comparison')
        if passed.get('provider_experiment_version')!=spec['protocol']:
            raise ValueError('A canary from another provider/version cannot authorize this comparison')
    out=Path(spec['output'])
    if out.exists():raise ValueError('Never resume or overwrite a validation execution')
    out.mkdir(parents=True)
    atomic_json(out/'spec.json',spec);atomic_json(out/'approval.json',approval)
    ledger_class=backend.ledger if backend else SpendLedger
    ledger=ledger_class(out/'spending-ledger.json',spec['spending_ceiling_usd'],checkpoint=3,
        max_calls=spec['max_provider_calls'],max_seconds=spec['wall_seconds'])
    ledger.phase=spec['mode']
    actual=settings if backend else replace(settings,llm_input_price=1,llm_output_price=5)
    factory=backend.guarded_factory(actual,ledger,transport=transport) if backend else guarded_factory(actual,ledger,transport=transport)
    results=[];status='complete'
    try:
        with Database(str(out/'metadata.sqlite')) as db:
            if spec['mode']=='comparison':
                results=run_experiments(spec['corpus'],spec['questions'],db,factory,output=str(out/'benchmark'),
                    suite='controlled-seven',methods=tuple(spec['methods']),action_budgets=(10,),trials=2,
                    trial_seeds=(11,23),budget=Budget(**spec['budget']),inference_limits=InferenceLimits(**spec['limits']))
            else:
                class CanaryPlanner(MetaContextPlanner):
                    def bind(self,*args,**kwargs):
                        super().bind(*args,**kwargs)
                        self.messages[0]['content']+='\n'+spec['canary_instruction']
                        self.store.set_limit(self.messages,self.limits.max_active_tokens)
                for method in spec['methods']:
                    ledger.check()
                    p=CanaryPlanner(factory(),limits=InferenceLimits(**spec['limits']))
                    c=ControlledController(questions[0]['question'],corpus,db,architecture=method,planner=p,
                        budget=Budget(**spec['budget']),output=str(out/'runs'))
                    _capture_requests(p,c.directory/'planner_requests.jsonl')
                    try:results.append(c.run())
                    finally:p.close()
    except ExperimentStopped as exc:
        status=exc.reason
        saved=out/'benchmark/metrics.csv'
        if spec['mode']=='comparison' and saved.exists():
            with saved.open() as stream:results=list(csv.DictReader(stream))
    cell_reports=[]
    for row in results:
        directory=Path(row['directory'])
        completed=json.loads((directory/'result.json').read_text())
        events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        requests=[json.loads(p.read_text()) for p in sorted((directory/'context').glob('request-*.json'))]
        invalid=sum(e['kind']=='invalid_decision' for e in events)
        decisions=sum('decision' in (r.get('output') or {}) for r in requests)
        visibility=[]
        from .evidence_visibility import audit_passage
        for item in completed['ranked'][:3]:
            paper=corpus.by_id[item['paper']['id']]
            if paper.abstract:visibility.append(audit_passage(directory,paper,paper.abstract[:200]))
        cell_reports.append(dict(directory=str(directory),architecture=completed.get('architecture',row.get('method','standard')),
            stop_reason=completed['stop_reason'],retrieval_actions=completed['actions'],
            model_decisions=decisions,invalid_actions=invalid,accepted_model_decisions=max(0,decisions-invalid),fallbacks=invalid,
            context_edits=completed.get('context_edits',0),rejected_context_edits=completed.get('rejected_context_edits',0),
            recovery_operations=sum(e['kind']=='context_recovery' for e in events),llm_usage=completed['llm_usage'],
            elapsed_seconds=completed['seconds'],provider_failures=completed['llm_usage']['failures'],
            evidence_visibility_sample=visibility,visibility_scope='First 200 abstract characters of at most three ranked sources; not truth/support judgments'))
    result=dict(status=status,mode=spec['mode'],completed_cells=len(results),provider_attempts=len(ledger.attempts),
        provider_experiment_version=spec['protocol'],
        reported_usage_cost_usd=str(ledger.reported_spend),retained_reservations_usd=str(ledger.charged),
        final_evaluation_authorized=False,classification='MOCKED PROVIDER TEST' if transport is not None else 'Development only; not final scientific results')
    result['cells']=cell_reports
    if spec['mode']=='canary':
        proof=[audit_canary(r['directory']) for r in results]
        result.update(canary_mechanism_passed=len(proof)==2 and all(p['passed'] for p in proof),canary_evidence=proof)
    atomic_json(out/'validation-status.json',result)
    return result
