"""Real installed SDK serialization with an HTTP mock and anonymous credentials.
No ADC refresh, DNS, hosted request or paid inference occurs in this probe.
"""
import importlib.util
import base64
import json
import os
from pathlib import Path
import socket

import httpx2
from google import genai
from google.auth.credentials import Credentials
from google.genai import types

socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Network forbidden'))
socket.getaddrinfo=lambda *a,**k:(_ for _ in ()).throw(AssertionError('DNS forbidden'))
spec=importlib.util.spec_from_file_location('worker',Path('openalex_agent/gemini_sdk_worker.py'))
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
captured=[]
fail=False
class FixtureCredentials(Credentials):
    def __init__(self):super().__init__();self.token='SYNTHETIC-NOT-A-REAL-TOKEN'
    def refresh(self,request):self.token='SYNTHETIC-NOT-A-REAL-TOKEN'
def handler(request):
    assert request.url.host=='aiplatform.googleapis.com' or request.url.host.endswith('-aiplatform.googleapis.com')
    assert 'projects/synthetic-project/locations/global/publishers/google/models/gemini-3.8-flash:generateContent' in request.url.path
    captured.append(json.loads(request.content))
    if fail:return httpx2.Response(503,json={'error':{'code':503,'message':'SYNTHETIC unavailable','status':'UNAVAILABLE'}})
    return httpx2.Response(200,json=dict(candidates=[dict(content=dict(role='model',parts=[dict(
        functionCall=dict(id='call-original',name='INSPECT_PAPER',args=dict(paper_id='W2')),
        thoughtSignature='AAEC/w==')]),finishReason='STOP')],usageMetadata=dict(
            promptTokenCount=8,candidatesTokenCount=5,thoughtsTokenCount=95,totalTokenCount=108)))

def factory(**kwargs):
    options=kwargs['http_options']
    assert options.retry_options.attempts==1 and options.timeout==120000
    hooks=options.httpx_client.event_hooks
    options.httpx_client=httpx2.Client(transport=httpx2.MockTransport(handler),event_hooks=hooks)
    return genai.Client(**kwargs)

config=dict(system_instruction='Protected SDK fixture',max_output_tokens=4096,thinking_config=dict(thinking_level='LOW'),
    automatic_function_calling=dict(disable=True),automatic_continuation=False,
    tools=[dict(function_declarations=[dict(name='INSPECT_PAPER',parameters_json_schema=dict(type='object',
        properties=dict(paper_id=dict(type='string')),required=['paper_id']))])],
    tool_config=dict(function_calling_config=dict(mode='ANY',allowed_function_names=['INSPECT_PAPER'])))
contents=[dict(role='user',parts=[dict(text='SYNTHETIC native tool fixture')])]
request=dict(project='synthetic-project',location='global',timeout_ms=120000,
             tokenizer_python=os.environ.get('GEMINI_TEST_TOKENIZER_PYTHON',str(Path('.venv/bin/python').absolute())),input_limit=8192,
             body=dict(model='gemini-3.8-flash',contents=contents,config=config))
first=worker.run_request(request,client_factory=factory,credentials_factory=FixtureCredentials)
assert first.get('http_status')==200,first
native=first['response']['candidates'][0]['content']
request['body']['contents'] += [native,dict(role='user',parts=[dict(function_response=dict(id='call-original',
    name='INSPECT_PAPER',response=dict(output='SYNTHETIC observation')))])]
second=worker.run_request(request,client_factory=factory,credentials_factory=FixtureCredentials)
assert second.get('http_status')==200,second
assert len(captured)==2
part=captured[1]['contents'][1]['parts'][0]
assert base64.urlsafe_b64decode(part['thoughtSignature'])==base64.b64decode('AAEC/w==')
assert part['functionCall']['id']=='call-original'
assert captured[1]['contents'][2]['parts'][0]['functionResponse']['id']=='call-original'
assert first['response']['usage_metadata']['thoughts_token_count']==95
request['input_limit']=1
before=len(captured)
blocked=worker.run_request(request,client_factory=factory,credentials_factory=FixtureCredentials)
assert blocked.get('error_code')=='context_capacity_exceeded' and len(captured)==before,blocked
request['input_limit']=8192;fail=True
failed=worker.run_request(request,client_factory=factory,credentials_factory=FixtureCredentials)
assert failed.get('http_status')==503 and len(captured)==before+1,failed
print(json.dumps(dict(real_sdk_serialization_passed=True,network_blocked=True,mocked_http_calls=len(captured),
    enterprise_endpoint_verified=True,
    hosted_requests=0,signature_preserved=True,function_ids_preserved=True,
    oversize_blocked_before_send=True,no_sdk_generation_retries=True,
    usage=first['response']['usage_metadata'],wire_bodies=captured),indent=2))
