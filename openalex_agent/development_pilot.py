"""Explicitly authorized, unscored development reliability pilot; final gates unchanged."""
from dataclasses import asdict,replace
from decimal import Decimal
import json
from pathlib import Path
import time

from .config import Settings
from .context_planner import InferenceLimits
from .controlled import ARCHITECTURES
from .controller import Budget
from .corpus import atomic_json
from .execution_control import ExperimentStopped
from .experiment_dashboard import profile_digest,source_digest
from .experiments import run_experiments
from .llm import DeterministicPlanner,HostedJSONPlanner
from .pooled_review import sha,digest
from .storage import Database

VERSION='hosted-development-reliability-v1'
CHECKPOINT_VERSION='hosted-development-reliability-v2-checkpoint'
MODEL='gpt-6.1-terra'
# Model-wide long-context upper bound, deliberately much larger than study input.
# Input uses 2x and output 1.5x long-context rates; no cache discounts assumed.
REQUEST_RESERVE=Decimal('2.13072')  # 1,050,000*2/1M + 4,096*7.5/1M


class SpendLedger:
    """Serial request reservation; unknown/failed usage retains the full hold.

    A local spending ceiling relies on the pinned provider prices/token limits,
    not on unverified zero-cost errors. It is not a provider billing-account cap.
    """
    def __init__(self,path,ceiling,*,max_calls=480,max_seconds=7200,clock=time.monotonic,checkpoint=None):
        self.path=Path(path);self.ceiling=Decimal(str(ceiling));self.max_calls=max_calls
        self.max_seconds=max_seconds;self.clock=clock;self.started=clock();self.attempts=[];self.stop_reason=None
        self.checkpoint=Decimal(str(checkpoint)) if checkpoint is not None else None
        self.phase='benchmark'
        if not Decimal('0')<self.ceiling<=Decimal('16'):raise ValueError('Pilot ceiling must be positive and at most $16')
        if self.path.exists():raise ValueError('Never reuse a spending ledger or retry an existing pilot')
        self.save()

    @property
    def charged(self):return sum((Decimal(a['retained_usd']) for a in self.attempts),Decimal('0'))

    @property
    def reported_spend(self):
        return sum((Decimal(a['retained_usd']) for a in self.attempts if a['usage']=='provider_reported'),Decimal('0'))

    def save(self):
        atomic_json(self.path,dict(protocol=VERSION,ceiling_usd=str(self.ceiling),retained_usd=str(self.charged),
            request_reserve_usd=str(REQUEST_RESERVE),max_calls=self.max_calls,wall_seconds=self.max_seconds,
            checkpoint_usd=str(self.checkpoint) if self.checkpoint is not None else None,
            reported_usage_cost_usd=str(self.reported_spend),
            attempts=self.attempts,stop_reason=self.stop_reason,unknown_cost_policy='Retain entire model-wide request reservation; never infer zero'))

    def check(self):
        reason=self.stop_reason
        if self.clock()-self.started>=self.max_seconds:reason='pilot_wall_clock_limit'
        elif self.checkpoint is not None and self.reported_spend>=self.checkpoint:reason='pilot_three_dollar_checkpoint'
        elif self.checkpoint is not None and any(a['usage']=='UNKNOWN' for a in self.attempts):reason='pilot_unknown_cost_review_required'
        elif len(self.attempts)>=self.max_calls:reason='pilot_call_limit'
        elif self.charged+REQUEST_RESERVE>self.ceiling:reason='pilot_spending_limit'
        if reason:
            self.stop_reason=reason;self.save();raise ExperimentStopped(reason)

    def reserve(self):
        self.check();index=len(self.attempts)
        self.attempts.append(dict(attempt=index+1,phase=self.phase,retained_usd=str(REQUEST_RESERVE),usage='UNKNOWN',http_status=None))
        self.save();return index

    def settle(self,index,diagnostic):
        a=self.attempts[index];a['http_status']=diagnostic.get('http_status');a['error_code']=diagnostic.get('error_code')
        incoming=diagnostic.get('prompt_tokens');outgoing=diagnostic.get('completion_tokens')
        if (type(incoming) is int and 0<=incoming<=1050000 and type(outgoing) is int and 0<=outgoing<=4096):
            long=incoming>272000
            cost=(Decimal(incoming)*(2 if long else 1)+Decimal(outgoing)*(Decimal('7.5') if long else 5))/1000000
            writes=diagnostic.get('cache_write_tokens',0)
            if type(writes) is int and 0<writes<=incoming:
                # No new cache options are sent; account for a default 30m
                # write charge if the provider explicitly reports one.
                cost+=Decimal(writes)*Decimal('.25')*(2 if long else 1)/1000000
                a['cache_write_tokens']=writes
            a.update(usage='provider_reported',prompt_tokens=incoming,completion_tokens=outgoing,retained_usd=str(cost))
            cached=diagnostic.get('cached_tokens')
            if type(cached) is int and 0<=cached<=incoming:
                a['cached_tokens']=cached
                # Separate estimate only; checkpoint/hard guard retain uncached
                # prices and never assume a cache discount to permit more calls.
                a['cache_adjusted_cost_estimate_usd']=str(cost-Decimal(cached)*Decimal('.95')*(2 if long else 1)/1000000)
        if diagnostic.get('http_status') in (401,403,404):self.stop_reason='pilot_provider_access_failure'
        if incoming is not None and (type(incoming) is not int or incoming>1050000) or outgoing is not None and (type(outgoing) is not int or outgoing>4096):
            self.stop_reason='provider_usage_outside_approved_limits'
        self.save()


def settings_for_spec(spec,settings):
    expected=dict(model=settings.llm_model,base_url=settings.llm_base_url.rstrip('/')+'/',timeout=settings.llm_timeout,
        max_output_tokens=settings.llm_max_tokens,temperature=settings.llm_temperature,reasoning_effort=settings.llm_reasoning_effort,
        input_usd_per_million=1,output_usd_per_million=5)
    if spec['provider']!=expected or expected['model']!=MODEL or expected['base_url']!='https://api.openai.com/v1/':
        raise ValueError('Provider/generation settings differ from the pinned Terra proposal')
    if expected['max_output_tokens']!=4096 or expected['timeout']!=120 or expected['temperature'] is not None or expected['reasoning_effort']!='low':
        raise ValueError('Pilot must retain the proposed generation limits')
    return replace(settings,llm_input_price=1,llm_output_price=5)


def validate_spec(spec_path,settings):
    spec=json.loads(Path(spec_path).read_text())
    required={'protocol','purpose','corpus','corpus_sha256','questions','development_source','selection_sha256',
              'output','provider','budget','limits','trials','seeds','action_budgets','spending_ceiling_usd','wall_seconds'}
    if spec.get('protocol')==CHECKPOINT_VERSION:
        required|={'checkpoint_usd','live_canary'}
        if spec.get('checkpoint_usd')!=3 or spec.get('live_canary') is not True:raise ValueError('Preserve the authorized $3 checkpoint and live canary')
    if set(spec)!=required or spec['protocol'] not in (VERSION,CHECKPOINT_VERSION) or spec['purpose']!='UNSCORED DEVELOPMENT RELIABILITY ONLY':raise ValueError('Unsupported pilot specification')
    if spec['trials']!=2 or spec['seeds']!=[11,23] or spec['action_budgets']!=[10] or spec['spending_ceiling_usd']!=16 or spec['wall_seconds']!=7200:
        raise ValueError('Use the fixed three-question/two-repeat pilot proposal')
    expected_budget=asdict(Budget(max_actions=10,max_rounds=20,max_inspected=200,page_size=10,output_k=20,max_llm_calls=20))
    expected_limits=asdict(InferenceLimits(max_total_tokens=100000,max_active_tokens=4096,max_input_tokens=8192))
    if spec['budget']!=expected_budget or spec['limits']!=expected_limits:raise ValueError('Pilot budgets differ from approval proposal')
    corpus=Path(spec['corpus']);manifest=json.loads((corpus/'manifest.json').read_text())
    if manifest['sha256']!=spec['corpus_sha256'] or sha(corpus/'papers.jsonl')!=spec['corpus_sha256']:raise ValueError('Frozen corpus checksum mismatch')
    questions=json.loads(Path(spec['questions']).read_text());dev=json.loads(Path(spec['development_source']).read_text())
    if (len(questions)!=3 or len({q['id'] for q in questions})!=3
        or any(q.get('split')!='development' or q not in dev or set(q)-{'id','question','split','discipline','stratum','review_status'} for q in questions)):
        raise ValueError('Pilot requires exactly three unchanged DEV questions, no truth fields or held-out inputs')
    if sha(Path(spec['questions']).parent/'selection.json')!=spec['selection_sha256']:raise ValueError('Selection rationale changed')
    runtime=settings_for_spec(spec,settings)
    return spec,runtime


def approval_pins(spec_path,settings):
    spec,runtime=validate_spec(spec_path,settings)
    return dict(spec_sha256=sha(spec_path),questions_sha256=sha(spec['questions']),development_source_sha256=sha(spec['development_source']),
        corpus_sha256=spec['corpus_sha256'],source_sha256=source_digest(),provider_profile_sha256=profile_digest(runtime),
        entrypoint_sha256=sha(Path(__file__).resolve().parents[1]/'scripts/development_pilot.py'))


def verify_authorization(spec_path,approval_path,settings):
    pins=approval_pins(spec_path,settings);a=json.loads(Path(approval_path).read_text())
    spec=json.loads(Path(spec_path).read_text())
    if (a.get('protocol')!=spec['protocol'] or a.get('approved') is not True or a.get('paid_inference_authorized') is not True
        or a.get('operational_pilot_only') is not True or a.get('final_evaluation_authorized') is not False
        or a.get('spending_ceiling_usd')!=16 or not a.get('approver') or not a.get('approved_at') or a.get('pins')!=pins):
        raise ValueError('Explicit approval of this exact $16 development pilot is required before hosted inference')
    if spec['protocol']==CHECKPOINT_VERSION and (a.get('checkpoint_usd')!=3 or a.get('canary_included') is not True):
        raise ValueError('User authorization must include the canary and $3 stop checkpoint')
    return a


def guarded_factory(settings,ledger,*,transport=None):
    class PilotPlanner(HostedJSONPlanner):
        def _record_usage(self,value):
            super()._record_usage(value)
            if isinstance(value,dict):
                details=value.get('prompt_tokens_details') or value.get('input_tokens_details') or {}
                if isinstance(details,dict) and type(details.get('cache_write_tokens')) is int:
                    self.last_call['cache_write_tokens']=details['cache_write_tokens']
        def _call(self,system,payload):
            index=ledger.reserve()
            try:return super()._call(system,payload)
            finally:ledger.settle(index,self.last_call)
    return lambda:PilotPlanner(settings,transport=transport)


def execute(spec_path,approval_path,settings,*,offline=False,offline_output=None,transport=None):
    spec,runtime=validate_spec(spec_path,settings)
    a=None if offline else verify_authorization(spec_path,approval_path,settings)
    out=Path(offline_output) if offline and offline_output else Path(spec['output'])
    if offline and not offline_output:raise ValueError('Offline preflight needs a separate output directory')
    if out.exists():raise ValueError('Never overwrite/retry an existing pilot directory')
    out.mkdir(parents=True)
    atomic_json(out/'pilot-spec.json',spec)
    if a:atomic_json(out/'approval-snapshot.json',a)
    ledger=None if offline else SpendLedger(out/'spending-ledger.json',spec['spending_ceiling_usd'],max_seconds=spec['wall_seconds'],
        max_calls=481 if spec.get('live_canary') else 480,checkpoint=spec.get('checkpoint_usd'))
    factory=DeterministicPlanner if offline else guarded_factory(runtime,ledger,transport=transport)
    def cancelled():
        if ledger:ledger.check()
        return False
    atomic_json(out/'pilot-status.json',dict(protocol=VERSION,status='running',offline=offline,quality_evaluated=False))
    try:
        if ledger and spec.get('live_canary'):
            ledger.phase='canary'
            canary=dict(status='running',expected={'status':'ok'},included_in_pilot_ledger=True)
            atomic_json(out/'canary.json',canary)
            model=factory()
            from .experiments import _capture_requests
            _capture_requests(model,out/'canary-request.jsonl')
            try:
                response=model._call('Return exactly the JSON object {"status":"ok"}.',{'purpose':'authorized provider canary; no research data'})
                if response!={'status':'ok'}:raise ValueError('Canary response did not match the expected JSON object')
                if model.usage.usage_unreported_calls:raise ExperimentStopped('canary_usage_unreported')
                canary.update(status='passed',response=response,usage=asdict(model.usage),diagnostics=model.last_call)
            except BaseException as exc:
                canary.update(status='failed',error_code=getattr(exc,'code',getattr(exc,'reason',type(exc).__name__)),
                    usage=asdict(model.usage),diagnostics=model.last_call)
                raise
            finally:
                model.close();atomic_json(out/'canary.json',canary)
            ledger.phase='benchmark';ledger.check()
        with Database(str(out/'metadata.sqlite')) as db:
            rows=run_experiments(spec['corpus'],spec['questions'],db,factory,output=str(out/'benchmark'),
                suite='controlled-five',methods=ARCHITECTURES,action_budgets=(10,),trials=2,trial_seeds=(11,23),
                budget=Budget(**spec['budget']),inference_limits=InferenceLimits(**spec['limits']),
                allow_deterministic=offline,labels_path=None,cancel_check=cancelled)
    except BaseException as exc:
        atomic_json(out/'pilot-status.json',dict(protocol=VERSION,status=getattr(exc,'reason','failed'),
            error_type=type(exc).__name__,offline=offline,quality_evaluated=False,automatic_retry=False))
        raise
    atomic_json(out/'pilot-status.json',dict(protocol=VERSION,status='complete',rows=len(rows),offline=offline,
        quality_evaluated=False,independent_labels_present=False,automatic_retry=False))
    return rows
