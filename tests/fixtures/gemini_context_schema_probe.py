"""Serialize the corrected declaration using SDK 3.0.0; HTTP is mocked."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

import httpx2
from google import genai
from google.auth.credentials import Credentials

socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Network forbidden'))
socket.getaddrinfo=lambda *a,**k:(_ for _ in ()).throw(AssertionError('DNS forbidden'))
spec=importlib.util.spec_from_file_location('worker',Path('openalex_agent/gemini_sdk_worker.py'))
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
captured=[]

class FixtureCredentials(Credentials):
    def __init__(self):super().__init__();self.token='SYNTHETIC-NOT-A-REAL-TOKEN'
    def refresh(self,request):raise AssertionError('No credential refresh')

def handler(request):
    captured.append(json.loads(request.content))
    return httpx2.Response(200,json=dict(candidates=[dict(content=dict(role='model',parts=[dict(
        functionCall=dict(id='contract-mock',name='context_file',args=dict(path='LIVE_CTX_MAIN.txt',op='read',start=0,chars=1)),
        thoughtSignature='AAEC/w==')]),finishReason='STOP')],
        usageMetadata=dict(promptTokenCount=8,candidatesTokenCount=5,thoughtsTokenCount=0,totalTokenCount=13)))

def factory(**kwargs):
    options=kwargs['http_options']
    assert options.retry_options.attempts==1
    options.httpx_client=httpx2.Client(transport=httpx2.MockTransport(handler),event_hooks=options.httpx_client.event_hooks)
    return genai.Client(**kwargs)

body=json.load(sys.stdin)
result=worker.run_request(dict(project='synthetic-project',location='global',timeout_ms=120000,body=body,
    tokenizer_python=os.environ['GEMINI_TEST_TOKENIZER_PYTHON'],input_limit=8192),
    client_factory=factory,credentials_factory=FixtureCredentials)
assert result.get('http_status')==200,result
assert len(captured)==1
original=next(t for t in body['config']['tools'][0]['function_declarations'] if t['name']=='context_file')['parameters_json_schema']
wire=next(t for t in captured[0]['tools'][0]['functionDeclarations'] if t['name']=='context_file')
serialized_schema=wire.get('parameters_json_schema',wire.get('parametersJsonSchema'))
assert serialized_schema==original
assert len(serialized_schema['anyOf'])==4
print(json.dumps(dict(generation_requests=0,network_blocked=True,schema_preserved=True,sdk_mock_http_calls=1,
                     wire_declaration=wire),indent=2))
