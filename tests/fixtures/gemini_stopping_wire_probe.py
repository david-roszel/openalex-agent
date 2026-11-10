"""Offline SDK wire check; anonymous credentials, mock HTTP, DNS/socket blocked."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

import httpx2
from google import genai
from google.auth.credentials import Credentials

socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Network forbidden'))
socket.getaddrinfo = lambda *a, **k: (_ for _ in ()).throw(AssertionError('DNS forbidden'))
spec = importlib.util.spec_from_file_location('worker', Path('openalex_agent/gemini_sdk_worker.py'))
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


class Anonymous(Credentials):
    def __init__(self):
        super().__init__(); self.token = 'SYNTHETIC-NOT-A-CREDENTIAL'

    def refresh(self, request):
        raise AssertionError('Credential refresh forbidden')


def handler(request):
    return httpx2.Response(200, json=dict(candidates=[dict(content=dict(role='model', parts=[dict(text='{}')]),
        finishReason='STOP')], usageMetadata=dict(promptTokenCount=8, candidatesTokenCount=2,
        thoughtsTokenCount=0, totalTokenCount=10)))


def factory(**kwargs):
    options = kwargs['http_options']
    assert options.retry_options.attempts == 1
    options.httpx_client = httpx2.Client(transport=httpx2.MockTransport(handler), event_hooks=options.httpx_client.event_hooks)
    return genai.Client(**kwargs)


proposal = json.load(sys.stdin)
rows = []
for case in proposal['cases']:
    body = json.loads(Path(case['request']).read_text())['body']
    receipt = worker.run_request(dict(project='synthetic-project', location='global', timeout_ms=120000,
        body=body, tokenizer_python=os.environ['GEMINI_TEST_TOKENIZER_PYTHON'], input_limit=8192),
        client_factory=factory, credentials_factory=Anonymous)
    assert receipt.get('http_status') == 200, receipt
    assert len(receipt['wire_requests']) == 1
    wire = receipt['wire_requests'][0]
    original = json.loads(Path(case['source']).read_text())['wire_requests'][0]
    expected = json.loads(json.dumps(original))
    if case['condition'] == 'clarified':
        # SDK has one text part for each saved system instruction in these inputs.
        expected['systemInstruction']['parts'][0]['text'] += proposal['guidance_suffix']
    assert wire == expected, case['id']
    rows.append(dict(id=case['id'], exact_original_wire_plus_optional_suffix=True,
                     study_wire_input_tokens=receipt['wire_tokens'][0]))
print(json.dumps(dict(hosted_requests=0, network_blocked=True, credentials_refreshed=False,
                     sdk_version='3.0.0', mocked_serializations=len(rows), cells=rows), indent=2))
