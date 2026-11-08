"""Isolated Google SDK bridge. Credentials stay in this process, never in JSON.

One stdin request, one generate_content invocation, one credential-free stdout
receipt. Importing/probing this module never authenticates or generates content.
"""
import importlib.metadata
import json
import os
import subprocess
import sys
import time


def run_request(request,*,client_factory=None,credentials_factory=None):
    from google import genai
    from google.genai import types
    import google.auth
    import httpx2 as httpx # google-genai 3.0.0's pinned transport namespace
    from google.auth import exceptions as auth_errors
    from google.genai import errors
    version=importlib.metadata.version('google-genai')
    if version!='3.0.0':return {'error_code':'sdk_version_mismatch','sdk_version':version}
    if request.get('probe'):
        return {'sdk_version':version,'generation_performed':False}
    client=None
    http_client=None;wire=[];wire_tokens=[];started=time.monotonic()
    def receipt(**kwargs):
        return {**kwargs,'wire_requests':wire,'wire_tokens':wire_tokens,
                'sdk_seconds':time.monotonic()-started}
    class PreflightError(Exception):pass
    def before_send(http_request):
        if wire:raise PreflightError('unexpected_generation_retry')
        body=json.loads(http_request.content)
        check=subprocess.run([request['tokenizer_python'],'-m','openalex_agent.gemini_token_check'],
            input=json.dumps({'body':body,'limit':request['input_limit']}),capture_output=True,text=True,
            env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','PYTHONDONTWRITEBYTECODE')},timeout=20)
        if check.returncode:raise PreflightError('tokenizer_preflight_failed')
        measured=json.loads(check.stdout)
        if not measured['allowed']:raise PreflightError('context_capacity_exceeded')
        if time.monotonic()-started>=request['timeout_ms']/1000:raise PreflightError('timeout')
        wire.append(body);wire_tokens.append(measured['tokens'])
    try:
        credentials=(credentials_factory() if credentials_factory else
            google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])[0])
        http_client=httpx.Client(event_hooks={'request':[before_send]})
        client=(client_factory or genai.Client)(enterprise=True,project=request['project'],location=request['location'],
            credentials=credentials,http_options=types.HttpOptions(api_version='v1',timeout=request['timeout_ms'],
                retry_options=types.HttpRetryOptions(attempts=1),httpx_client=http_client))
        # Pydantic's base64 codec round-trips bytes (including thought signatures).
        contents=[types.Content.model_validate_json(json.dumps(c)) for c in request['body']['contents']]
        config=types.GenerateContentConfig.model_validate(request['body']['config'])
        response=client.models.generate_content(model=request['body']['model'],contents=contents,config=config)
        value=response.model_dump(mode='json',exclude_none=True)
        value.pop('sdk_http_response',None) # no response headers/transport objects in research artifacts
        return receipt(response=value,http_status=200,sdk_version=version)
    except PreflightError as exc:
        return receipt(error_code=str(exc),http_status=None,sdk_version=version)
    except auth_errors.TransportError:
        return receipt(error_code='transport_error',http_status=None,sdk_version=version)
    except auth_errors.GoogleAuthError:
        return receipt(error_code='authentication_error',http_status=None,sdk_version=version)
    except errors.APIError as exc:
        code=exc.code
        return receipt(error_code='authentication_error' if code in (401,403) else 'rate_limited' if code==429 else 'timeout' if code==504 else 'http_error',
                http_status=code,sdk_version=version)
    except (TimeoutError,httpx.TimeoutException):
        return receipt(error_code='timeout',http_status=None,sdk_version=version)
    except httpx.HTTPError:
        return receipt(error_code='transport_error',http_status=None,sdk_version=version)
    except Exception:
        # No raw SDK/OAuth exception text, credentials, tokens or endpoint bodies.
        return receipt(error_code='sdk_transport_or_protocol_error',http_status=None,sdk_version=version)
    finally:
        if client is not None:client.close()
        if http_client is not None:http_client.close()


def main():
    request=json.load(sys.stdin)
    result=run_request(request)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
