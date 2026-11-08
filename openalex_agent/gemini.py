"""Enterprise Gemini/ADC adapter, separate from Terra and its Responses protocol."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .context_memory import serialized
from .llm import LLMUsage, PlannerError, _strict_object

GEMINI_VERSION='gemini-enterprise-adc-3.8-v1'


@dataclass(frozen=True)
class GeminiSettings:
    project: str
    location: str='global'
    model: str='gemini-3.8-flash'
    sdk_python: str='~/gemini-test-env/bin/python'
    timeout: float=120.
    max_output_tokens: int=4096
    thinking_level: str='LOW'
    input_price: float=1.5  # gross Standard rate; do not spend against delayed promotional credits
    output_price: float=7.5

    def __post_init__(self):
        if not self.project or self.location!='global' or self.model!='gemini-3.8-flash':
            raise ValueError('Gemini ADC requires an explicit project, global endpoint and gemini-3.8-flash')
        if self.thinking_level!='LOW' or self.max_output_tokens!=4096 or self.timeout!=120:
            raise ValueError('Preserve the shared low-thinking, 4096-output, 120-second profile')
        if self.input_price!=1.5 or self.output_price!=7.5:
            raise ValueError('Use pinned gross Standard prices; do not assume trial/promotional credits')

    @classmethod
    def from_env(cls):
        return cls(project=os.getenv('GEMINI_ADC_PROJECT',os.getenv('GOOGLE_CLOUD_PROJECT','')),
            sdk_python=os.getenv('GEMINI_SDK_PYTHON','~/gemini-test-env/bin/python'))


def gemini_contents(messages):
    system=[];contents=[];functions={}
    for message in messages:
        role=message['role']
        if role in ('system','developer'):
            system.append(message['content']);continue
        if 'gemini_content' in message:
            original=deepcopy(message['gemini_content'])
            for call in message.get('tool_calls',[]):
                matches=[p['function_call'] for p in original.get('parts',[]) if 'function_call' in p]
                if len(matches)!=1:raise PlannerError('invalid_tool_response')
                functions[call['id']]=matches[0]
            contents.append(original);continue
        if role=='tool':
            call=functions.get(message['tool_call_id'])
            if call is None:raise PlannerError('missing_native_function_context')
            response={'name':call['name'],'response':{'output':message['content']}}
            if call.get('id') is not None:response['id']=call['id']
            contents.append({'role':'user','parts':[{'function_response':response}]});continue
        if message.get('tool_calls'):raise PlannerError('missing_gemini_native_content')
        if message.get('content'):
            contents.append({'role':'model' if role=='assistant' else 'user','parts':[{'text':message['content']}]})
    return '\n\n'.join(system),contents


class GeminiADCPlanner:
    name='hosted-gemini-adc'
    study_input_framing='complete_gemini_generate_content_request'

    def __init__(self,settings,*,executor=None):
        self.gemini=settings;self.model=settings.model;self.usage=LLMUsage();self.last_call={}
        self.executor=executor or self._sdk_request
        self.settings=SimpleNamespace(llm_max_tokens=settings.max_output_tokens,llm_input_price=settings.input_price,
                                      llm_output_price=settings.output_price)
        self.configuration=dict(protocol=GEMINI_VERSION,model=self.model,auth='ADC',sdk='google-genai==3.0.0',
            project_sha256=hashlib.sha256(settings.project.encode()).hexdigest(),location=settings.location,
            thinking_level='LOW',max_output_tokens=4096,timeout=120,temperature=None,
            input_price=settings.input_price,output_price=settings.output_price,
            native_functions='complete Content/Part replay; no signature bypass',automatic_retries=0,
            pricing_basis='gross Standard before trial/promotional credits')
        if executor is not None:self.configuration['mocked_generation']=True
        self.fingerprint=hashlib.sha256(serialized(self.configuration).encode()).hexdigest()
        self.last_response_content=None;self.last_provider_response=None;self.deadline=None
        self.request_input_limit=8192;self.last_wire_requests=[]

    def set_inference_limits(self,limits):
        self.request_input_limit=limits.max_input_tokens or 8192

    def close(self):pass
    def _redact(self,value):return value # no API keys or credentials enter the host adapter

    def body(self,messages,tools=None):
        system,contents=gemini_contents(messages)
        config=dict(system_instruction=system,max_output_tokens=4096,thinking_config={'thinking_level':'LOW'},
            automatic_function_calling={'disable':True},automatic_continuation=False)
        if tools is None:config['response_mime_type']='application/json'
        else:
            config['tools']=[{'function_declarations':[dict(name=t['name'],description=t['description'],
                    parameters_json_schema=t['parameters']) for t in tools]}]
            config['tool_config']={'function_calling_config':{'mode':'ANY','allowed_function_names':[t['name'] for t in tools]}}
        return dict(model=self.model,contents=contents,config=config)

    def study_input_tokens(self,system,payload):
        from .token_budget import count
        return count(serialized(self.body([dict(role='system',content=system),dict(role='user',content=serialized(payload))])))

    def _call(self,system,payload):
        return self._request(self.body([dict(role='system',content=system),dict(role='user',content=serialized(payload))]))

    def _call_native(self,messages,tools):return self._request(self.body(messages,tools))

    def native_assistant_message(self,call):
        original=deepcopy(self.last_response_content)
        return dict(role='assistant',content='\n'.join(p['text'] for p in original.get('parts',[]) if isinstance(p.get('text'),str) and not p.get('thought')),
            reasoning_content='\n'.join(p['text'] for p in original.get('parts',[]) if isinstance(p.get('text'),str) and p.get('thought')),
            gemini_content=original,tool_calls=[dict(id=call['call_id'],type='function',
                function=dict(name=call['name'],arguments=serialized(call['arguments'])))])

    def _sdk_request(self,body):
        python=Path(self.gemini.sdk_python).expanduser()
        if not python.is_file():return {'error_code':'sdk_interpreter_unavailable'}
        worker=Path(__file__).with_name('gemini_sdk_worker.py')
        timeout=self.gemini.timeout
        if self.deadline is not None:timeout=min(timeout,self.deadline-time.monotonic())
        if timeout<=0:raise PlannerError('wall_clock_limit')
        env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','LANG','LC_ALL','GOOGLE_APPLICATION_CREDENTIALS',
            'CLOUDSDK_CONFIG','HTTPS_PROXY','HTTP_PROXY','NO_PROXY','SSL_CERT_FILE','REQUESTS_CA_BUNDLE')}
        env['PYTHONDONTWRITEBYTECODE']='1'
        request=dict(project=self.gemini.project,location=self.gemini.location,timeout_ms=max(1,int(timeout*1000)),body=body,
                     tokenizer_python=sys.executable,input_limit=self.request_input_limit)
        try:
            result=subprocess.run([str(python),str(worker)],input=serialized(request),text=True,capture_output=True,env=env,timeout=timeout)
        except subprocess.TimeoutExpired:raise PlannerError('timeout') from None
        except OSError:raise PlannerError('sdk_interpreter_unavailable') from None
        if result.returncode or len(result.stdout)>2_000_000:raise PlannerError('sdk_worker_failed')
        try:return json.loads(result.stdout)
        except ValueError:raise PlannerError('invalid_envelope') from None

    def _request(self,body):
        self.usage.calls+=1;self.usage.usage_unreported_calls+=1;self.usage.estimated_cost_usd=None
        self.usage.estimated_input_tokens+=(len(serialized(body).encode())+3)//4
        start=time.perf_counter();self.last_call=dict(call=self.usage.calls,protocol=GEMINI_VERSION,http_status=None,error_code=None)
        self.last_response_content=None;self.last_provider_response=None
        self.last_wire_requests=[]
        try:
            receipt=self.executor(deepcopy(body))
            if not isinstance(receipt,dict):raise PlannerError('invalid_envelope')
            self.last_wire_requests=deepcopy(receipt.get('wire_requests',[]))
            if receipt.get('wire_tokens'):self.last_call['study_wire_input_tokens']=receipt['wire_tokens'][-1]
            if isinstance(receipt.get('sdk_seconds'),(int,float)):self.last_call['sdk_worker_seconds']=receipt['sdk_seconds']
            self.last_call['http_status']=receipt.get('http_status')
            if receipt.get('error_code'):raise PlannerError(receipt['error_code'])
            response=receipt.get('response')
            if not isinstance(response,dict):raise PlannerError('invalid_envelope')
            self.last_provider_response=deepcopy(response)
            usage=response.get('usage_metadata') or {}
            if not isinstance(usage,dict):raise PlannerError('invalid_envelope')
            self.last_call['provider_usage']=deepcopy(usage)
            prompt,visible,thoughts,tool,total=[usage.get(k) for k in ('prompt_token_count','candidates_token_count','thoughts_token_count','tool_use_prompt_token_count','total_token_count')]
            valid=lambda v:type(v) is int and v>=0
            # Missing reasoning can be inferred only from a consistent provider
            # total. Never silently treat 5 visible tokens as the whole output.
            inferred=thoughts is None
            if thoughts is None and valid(total) and valid(prompt) and valid(visible):thoughts=total-prompt-visible-(tool or 0)
            tool=0 if tool is None else tool
            if all(valid(v) for v in (prompt,visible,thoughts,tool,total)) and total==prompt+visible+thoughts+tool:
                incoming=prompt+tool;outgoing=visible+thoughts
                self.usage.input_tokens+=incoming;self.usage.output_tokens+=outgoing
                self.usage.usage_reported_calls+=1;self.usage.usage_unreported_calls-=1
                self.last_call.update(prompt_tokens=incoming,completion_tokens=outgoing,visible_output_tokens=visible,
                    reasoning_tokens=thoughts,tool_use_prompt_tokens=tool,total_tokens=total)
                self.last_call['reasoning_tokens_inferred_from_total']=inferred
                if valid(usage.get('cached_content_token_count')):self.last_call['cached_tokens']=usage['cached_content_token_count']
                if self.usage.usage_unreported_calls==0:
                    self.usage.estimated_cost_usd=(self.usage.input_tokens*self.gemini.input_price+self.usage.output_tokens*self.gemini.output_price)/1e6
            candidates=response.get('candidates')
            if not isinstance(candidates,list) or len(candidates)!=1:raise PlannerError('invalid_envelope')
            candidate=candidates[0]
            if candidate.get('finish_reason')=='MAX_TOKENS':raise PlannerError('truncated')
            if candidate.get('finish_reason') not in ('STOP',None):raise PlannerError('refused_or_invalid_finish')
            content=candidate.get('content')
            if not isinstance(content,dict) or not isinstance(content.get('parts'),list):raise PlannerError('invalid_content')
            self.last_response_content=deepcopy(content)
            self.usage.estimated_output_tokens+=(len(serialized(content).encode())+3)//4
            calls=[p['function_call'] for p in content['parts'] if isinstance(p,dict) and 'function_call' in p]
            if 'tools' in body['config']:
                if len(calls)!=1:raise PlannerError('invalid_tool_response')
                call=calls[0]
                if not isinstance(call,dict) or not isinstance(call.get('name'),str) or not isinstance(call.get('args'),dict):
                    raise PlannerError('invalid_tool_response')
                part=next(p for p in content['parts'] if isinstance(p,dict) and 'function_call' in p)
                if not isinstance(part.get('thought_signature'),str) or not part['thought_signature']:
                    raise PlannerError('missing_thought_signature')
                # Provider IDs are matched verbatim in FunctionResponse; the
                # controller uses a separate unique identity across fresh turns.
                return dict(name=call['name'],call_id=f'gemini-call-{self.usage.calls}',arguments=deepcopy(call['args']))
            if calls:raise PlannerError('invalid_tool_response')
            text=[p['text'] for p in content['parts'] if isinstance(p,dict) and isinstance(p.get('text'),str) and not p.get('thought')]
            return _strict_object(''.join(text))
        except PlannerError as exc:
            self.usage.failures+=1;self.last_call['error_code']=exc.code
            raise
        finally:
            self.last_call['seconds']=time.perf_counter()-start;self.usage.seconds+=self.last_call['seconds']
