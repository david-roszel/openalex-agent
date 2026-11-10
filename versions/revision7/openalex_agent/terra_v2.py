"""Pinned historical Terra JSON adapter + request-boundary evidence/accounting."""
from copy import deepcopy
import hashlib,json,time
from pathlib import Path
import httpx
from .llm import HostedJSONPlanner
from .execution_control import ExperimentStopped
from .context_memory import serialized
from .token_budget import count
from .graph_evidence import audit_wire
from .corpus import atomic_json

BASE_URL='https://api.openai.com/v1'
MODEL='gpt-6.1-terra'


class TerraPlanner(HostedJSONPlanner):
    def __init__(self,settings,*,ledger,cell_id,architecture,permit,transport=None,receipt_directory=None):
        if (settings.llm_base_url.rstrip('/')!=BASE_URL or settings.llm_model!=MODEL or settings.llm_reasoning_effort!='low'
            or settings.llm_temperature is not None or settings.llm_max_tokens!=4096 or settings.llm_timeout!=120
            or settings.llm_input_price!=1 or settings.llm_output_price!=5):raise ValueError('Terra V2 profile must match frozen endpoint/model/settings/pricing')
        super().__init__(settings,transport=transport)
        self.ledger=ledger;self.receipt_directory=Path(receipt_directory) if receipt_directory is not None else ledger.out/'provider';self.cell_id=cell_id;self.architecture=architecture;self.permit=permit
        self._attempt=None;self.last_graph_boundary_audit=None;self.graph_evidence_state=None
        self.configuration.update(experimental_provider_version='terra-json-graph-visible-v2',service_tier='default',ledger_ceiling_usd='68.00')
        self.fingerprint=hashlib.sha256(serialized(self.configuration).encode()).hexdigest()
        self.http.event_hooks['request'].insert(0,self._before_send)
        self.http.event_hooks['response'].append(self._after_response)

    def _prepare_body(self,body):
        # Explicit standard-tier request; no native/server tools or cache-retention
        # opt-in. JSON scholarly/context operations stay the original interface.
        return {**body,'service_tier':'default'}

    def study_input_tokens(self,system,payload):
        body=self._prepare_body({'model':MODEL,'messages':[{'role':'system','content':system},{'role':'user','content':serialized(payload)}],
            'max_completion_tokens':4096,'response_format':{'type':'json_object'},'reasoning_effort':'low'})
        return count(serialized(body))

    def _before_send(self,request):
        self.last_call['request_not_sent']=True
        self.permit.check()
        if self._attempt is not None:raise ExperimentStopped('automatic_generation_retry_forbidden')
        body=json.loads(request.content)
        if (str(request.url)!=BASE_URL+'/chat/completions' or body.get('model')!=MODEL or body.get('service_tier')!='default'
            or body.get('max_completion_tokens')!=4096 or body.get('reasoning_effort')!='low' or body.get('response_format')!={'type':'json_object'}
            or set(body)-{'model','messages','max_completion_tokens','response_format','reasoning_effort','service_tier'}):raise ValueError('Unpinned billable request profile')
        if [m.get('role') for m in body['messages']]!=['system','user'] or any(not isinstance(m.get('content'),str) for m in body['messages']):raise ValueError('Text-only two-message profile required')
        if count(request.content.decode())>8192:raise ExperimentStopped('wire_context_capacity_exceeded')
        audit=audit_wire(request.content,self.graph_evidence_state)
        if self.graph_evidence_state:
            audit['modeled_context_policy']='controlled-context-v2-capacity-stop'
        self._attempt=self.ledger.reserve(f'{self.cell_id}:call{self.usage.calls}',self.cell_id,self.architecture,request.content)
        self.last_call['campaign_attempt_id']=self._attempt
        self.last_call['receipt_directory']=str(self.receipt_directory)
        self.last_call['wire_sha256']=audit['wire_sha256']
        atomic_json(self.receipt_directory/f'attempt-{self._attempt:05d}.request.json',self._redact(body))
        atomic_json(self.receipt_directory/f'attempt-{self._attempt:05d}.graph.json',audit)
        self.last_graph_boundary_audit=audit
        self.last_call['request_not_sent']=False

    def _record_provider_envelope(self,payload):
        self.last_call['provider_response_id']=payload.get('id')
        self.last_call['returned_model']=payload.get('model')
        self.last_call['returned_service_tier']=payload.get('service_tier')
        self.last_call['provider_usage']=deepcopy(payload.get('usage'))
        usage=payload.get('usage') or {}
        writes=usage.get('cache_creation_input_tokens') if isinstance(usage,dict) else None
        if writes is not None:self.last_call['cache_write_tokens']=writes
        atomic_json(self.receipt_directory/f'attempt-{self._attempt:05d}.response.json',self._redact(payload))

    def _after_response(self,response):
        self.last_call['provider_request_id']=response.headers.get('x-request-id')
        self.last_call['rate_limits']={k:v for k,v in response.headers.items() if k.startswith('x-ratelimit-') or k=='retry-after'}

    def _call(self,system,context):
        self._attempt=None;self.last_graph_boundary_audit=None
        self.last_call={'call':self.usage.calls+1,'http_status':None,'seconds':0,'request_not_sent':True}
        try:self.permit.check();self.ledger.check()
        except ExperimentStopped as exc:
            self.last_call['error_code']=exc.reason
            raise
        try:return super()._call(system,context)
        finally:
            if self._attempt is not None:
                atomic_json(self.receipt_directory/f'attempt-{self._attempt:05d}.diagnostic.json',self._redact(self.last_call))
                self.ledger.settle(self._attempt,self.last_call)
