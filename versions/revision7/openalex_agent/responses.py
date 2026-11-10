"""One-attempt Responses transport, preserving the existing configured model/profile.

JSON mode serves the original planners; native functions serve the Meta adapter.
No server-side conversation, previous_response_id, automatic compaction or retry.
"""
from copy import deepcopy
import hashlib
import json
import time

import httpx

from .llm import HostedJSONPlanner, PlannerError, _strict_object


def response_input(messages):
    items = []
    for message in messages:
        if 'response_items' in message:
            # Preserve complete native output, including encrypted reasoning and
            # assistant phase, until the official conversation parser edits it.
            items.extend(deepcopy(message['response_items']))
            continue
        role = message['role']
        if role == 'tool':
            items.append(dict(type='function_call_output', call_id=message['tool_call_id'], output=message['content']))
            continue
        if message.get('content'):
            items.append(dict(role=role, content=message['content']))
        for call in message.get('tool_calls', []):
            items.append(dict(type='function_call', call_id=call['id'],
                name=call['function']['name'], arguments=call['function']['arguments']))
    return items


class HostedResponsesPlanner(HostedJSONPlanner):
    name = 'hosted-responses'

    def __init__(self, settings, *, transport=None):
        super().__init__(settings, transport=transport)
        self.configuration = {**self.configuration, 'protocol':'responses-stateless-v1',
                              'token_limit_field':'max_output_tokens', 'store':False, 'automatic_retries':0}
        self.cache_write_tokens=0
        self.fingerprint = hashlib.sha256(json.dumps(self.configuration,sort_keys=True).encode()).hexdigest()

    def body(self, messages, tools=None):
        body = dict(model=self.model, input=response_input(messages),
                    max_output_tokens=self.settings.llm_max_tokens, store=False)
        if self.settings.llm_reasoning_effort:
            body['reasoning']={'effort':self.settings.llm_reasoning_effort}
        if self.settings.llm_temperature is not None:
            body['temperature']=self.settings.llm_temperature
        if tools is None:
            body['text']={'format':{'type':'json_object'}}
        else:
            body.update(tools=deepcopy(tools),tool_choice='required',parallel_tool_calls=False,
                        include=['reasoning.encrypted_content'])
        return body

    def _call(self, system, context):
        return self._request(self.body([{'role':'system','content':system},
            {'role':'user','content':json.dumps(context,ensure_ascii=False,separators=(',',':'))}]))

    def _call_native(self, messages, tools):
        return self._request(self.body(messages,tools))

    def study_input_tokens(self,system,payload):
        from .context_memory import serialized
        from .token_budget import count
        return count(serialized(self.body([{'role':'system','content':system},
            {'role':'user','content':serialized(payload)}])))

    def _request(self, body):
        self.usage.calls+=1
        self.usage.usage_unreported_calls+=1
        self.usage.estimated_cost_usd=None
        self.usage.estimated_input_tokens+=(len(json.dumps(body,ensure_ascii=False).encode())+3)//4
        start=time.perf_counter()
        self.last_call=dict(call=self.usage.calls,http_status=None,error_code=None,
                            headers_seconds=None,body_seconds=None,protocol='responses')
        self.last_response_output=None
        try:
            try:
                with self.http.stream('POST','responses',json=body) as response:
                    self.last_call.update(http_status=response.status_code,headers_seconds=time.perf_counter()-start)
                    body_start=time.perf_counter();response.read()
                    self.last_call['body_seconds']=time.perf_counter()-body_start
            except httpx.TimeoutException:
                raise PlannerError('timeout') from None
            except httpx.HTTPError:
                raise PlannerError('transport_error') from None
            if not response.is_success:
                raise PlannerError('rate_limited' if response.status_code==429 else
                                   'authentication_error' if response.status_code in (401,403) else 'http_error')
            try:
                value=response.json()
            except ValueError:
                raise PlannerError('invalid_envelope') from None
            if not isinstance(value,dict):raise PlannerError('invalid_envelope')
            usage=value.get('usage') or {}
            if not isinstance(usage,dict):usage={}
            self._record_usage(dict(prompt_tokens=usage.get('input_tokens'),completion_tokens=usage.get('output_tokens')))
            for source,target in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens')]:
                if type(usage.get(source)) is int and usage[source]>=0:self.last_call[target]=usage[source]
            for detail,key in [('input_tokens_details','cached_tokens'),('output_tokens_details','reasoning_tokens')]:
                details=usage.get(detail) or {}
                if isinstance(details,dict) and type(details.get(key)) is int and details[key]>=0:self.last_call[key]=details[key]
            details=usage.get('input_tokens_details') or {}
            writes=usage.get('cache_creation_input_tokens',details.get('cache_write_tokens') if isinstance(details,dict) else None)
            incoming=usage.get('input_tokens')
            if type(writes) is int and type(incoming) is int and 0<=writes<=incoming:
                self.last_call['cache_write_tokens']=writes;self.cache_write_tokens+=writes
            if self.usage.estimated_cost_usd is not None:
                # Conservative uncached input estimate plus explicitly reported
                # default 30-minute write surcharge; cache discounts never fund
                # extra calls. Provider invoice/unknown charges remain separate.
                self.usage.estimated_cost_usd+=self.cache_write_tokens*.25*self.settings.llm_input_price/1e6
            if value.get('status')=='incomplete':raise PlannerError('truncated')
            if value.get('status') not in ('completed',None):raise PlannerError('invalid_envelope')
            output=value.get('output')
            if not isinstance(output,list):raise PlannerError('invalid_envelope')
            if len(json.dumps(output,ensure_ascii=False).encode())>65536:raise PlannerError('response_too_large')
            self.last_response_output=deepcopy(output)
            calls=[item for item in output if isinstance(item,dict) and item.get('type')=='function_call']
            text=[]
            for item in output:
                if not isinstance(item,dict):raise PlannerError('invalid_envelope')
                content=item.get('content',[])
                if not isinstance(content,list):raise PlannerError('invalid_envelope')
                for part in content:
                    if not isinstance(part,dict):raise PlannerError('invalid_envelope')
                    if part.get('type')=='refusal':raise PlannerError('refused')
                    if part.get('type')=='output_text':
                        if not isinstance(part.get('text'),str):raise PlannerError('invalid_content')
                        text.append(part['text'])
            self.usage.estimated_output_tokens+=(len(json.dumps(output,ensure_ascii=False).encode())+3)//4
            if body.get('tools') is not None:
                if len(calls)!=1:raise PlannerError('invalid_tool_response')
                call=calls[0]
                if not all(isinstance(call.get(k),str) and call[k] for k in ('name','call_id','arguments')):
                    raise PlannerError('invalid_tool_response')
                return self._redact(dict(name=call['name'],call_id=call['call_id'],arguments=_strict_object(call['arguments'])))
            if calls:raise PlannerError('invalid_tool_response')
            return self._redact(_strict_object(''.join(text)))
        except PlannerError as exc:
            self.usage.failures+=1;self.last_call['error_code']=exc.code
            raise
        finally:
            self.last_call['seconds']=time.perf_counter()-start
            self.usage.seconds+=self.last_call['seconds']
