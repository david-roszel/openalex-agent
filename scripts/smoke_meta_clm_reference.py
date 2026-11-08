"""Run only in Meta's isolated environment; synthetic component checks, no model API.

No model-written Bash is executed. The fake sandbox accepts one fixed fixture
operation. This verifies the unchanged official synchronization component, not
Harbor container isolation, a live provider, or a seven-method research task.
"""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import faulthandler
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch


def forbidden(*args,**kwargs):
    raise AssertionError('Network forbidden in reference smoke')


async def check(output):
    print('Importing official harness',flush=True)
    from clm_harness.clm_agent.harness import ClmAgent
    from clm_harness.utils import tokens
    print('Constructing official agent',flush=True)
    agent=ClmAgent(output/'agent-log',model_name='openai/gpt-6.1-terra',max_steps=2,lm_call_cap=4,
        context_budget_tokens=8192,max_num_retry_on_limit=1,max_tokens=4096,temperature=None,top_p=None,
        send_chat_template_kwargs=False,api_base='https://api.openai.com/v1',cost_metric='usd')
    messages=[{'role':'system','content':'Protected fixture instructions'}, {'role':'user','content':'Protected synthetic task'},
              {'role':'assistant','content':'Obsolete fixture detail. '*200}, {'role':'tool','content':'Synthetic observation'}]
    prefix=deepcopy(messages[:2]);before=deepcopy(messages)
    class FixtureSandbox:
        def __init__(self):self.files={};self.replacement='[[CTX_TURN 1 role=user]]\nRetained synthetic datum 42.'
        async def upload_file(self,source,destination):self.files[destination]=Path(source).read_text()
        async def download_file(self,source,destination):Path(destination).write_text(self.files[source])
        async def exec(self,*,command,timeout_sec):
            assert command=='FIXTURE_REWRITE_LIVE_CTX_MAIN'
            self.files[agent._ctx.paths.ctx_file]=self.replacement
            return SimpleNamespace(stdout='',stderr='',return_code=0)
    sandbox=FixtureSandbox();agent._ctx.persistent_bash=False
    print('Checking official context synchronization',flush=True)
    step=await agent._ctx.step('FIXTURE_REWRITE_LIVE_CTX_MAIN',messages,environment=sandbox)
    assert step.ctx_changed and messages[:2]==prefix and 'Retained synthetic datum 42.' in str(messages)
    assert 'Obsolete fixture detail.' not in str(messages)
    captured=[]
    def mock_completion(**kwargs):
        captured.append(deepcopy(kwargs));return SimpleNamespace(choices=[],usage=None)
    async def in_process_fixture(function,*args,**kwargs):
        # Avoid the restricted environment's blocked cross-thread self-pipe.
        # The mocked transport performs no blocking work or provider inference.
        return function(*args,**kwargs)
    with patch('litellm.completion',mock_completion),patch('asyncio.to_thread',in_process_fixture):
        print('Checking mock next requests',flush=True)
        await agent._query_with_retry('openai/gpt-6.1-terra',before)
        await agent._query_with_retry('openai/gpt-6.1-terra',messages)
    assert captured[0]['messages']!=captured[1]['messages']
    assert 'Retained synthetic datum 42.' in str(captured[1]['messages'])
    after=deepcopy(messages);sandbox.replacement='[[CTX_TURN 1 role=user]]\n'+('oversized '*20000)
    rejected=await agent._ctx.step('FIXTURE_REWRITE_LIVE_CTX_MAIN',messages,environment=sandbox)
    assert not rejected.ctx_changed and messages==after and agent._ctx.n_ctx_rejected==1
    overflow=agent._budget.evaluate(prefix+[{'role':'user','content':'overflow '*20000}])
    assert overflow.action=='rollback'
    count,tokenizer=tokens.count_tokens(messages)
    return dict(component_synchronization_passed=True,protected_prefix_unchanged=True,
        next_request_changes_after_edit=True,oversize_growth_rejected=True,overflow_action=overflow.action,
        mock_transport_in_process=True,
        token_count=count,tokenizer=tokenizer,mock_provider_calls=len(captured),real_provider_calls=0,
        request_keys=sorted(captured[-1]),output_cap_key='max_completion_tokens' if 'max_completion_tokens' in captured[-1] else 'max_tokens',
        native_tools=[t['function']['name'] for t in captured[-1]['tools']],reasoning_effort_transmitted='reasoning_effort' in captured[-1],
        model_executed_edit=False,bash_execution=False,harbor_container_task_executed=False,
        scholarly_adapter_implemented=False,global_cost_enforcement_verified=False,
        source='Unmodified official ContextEnv.step and ClmAgent._query_with_retry; fake filesystem and mocked completion transport')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--upstream',required=True,type=Path)
    p.add_argument('--tokenizer',required=True,type=Path)
    a=p.parse_args()
    faulthandler.dump_traceback_later(25,repeat=False)
    if a.output.exists():raise ValueError('Use a fresh smoke directory')
    a.output.mkdir(parents=True)
    cache=a.output/'tokenizer-cache';cache.mkdir()
    url='https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken'
    shutil.copyfile(a.tokenizer,cache/hashlib.sha1(url.encode()).hexdigest())
    # No live keys or provider environment variables are retained by this process.
    for name in list(os.environ):
        if any(word in name.upper() for word in ('KEY','SECRET','TOKEN','PASSWORD')):os.environ.pop(name,None)
    os.environ.update(TIKTOKEN_CACHE_DIR=str(cache.resolve()),LITELLM_LOCAL_MODEL_COST_MAP='True',DO_NOT_TRACK='1')
    with patch.object(socket.socket,'connect',forbidden),patch.object(socket.socket,'connect_ex',forbidden),patch('socket.create_connection',forbidden),patch('socket.getaddrinfo',forbidden):
        report=asyncio.run(check(a.output))
    safe_env={k:v for k,v in os.environ.items() if k in ('PATH','LITELLM_LOCAL_MODEL_COST_MAP','DO_NOT_TRACK','TIKTOKEN_CACHE_DIR')}
    code="import socket,sys; socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network forbidden')); sys.argv=['clm-harbor','--help']; from clm_harness.cli import main; main()"
    cli=subprocess.run([sys.executable,'-c',code],env=safe_env,capture_output=True,text=True,timeout=45)
    (a.output/'cli-help.txt').write_text(cli.stdout+cli.stderr)
    report.update(cli_starts=cli.returncode==0,cli_exit_code=cli.returncode,
        revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip(),
        python=sys.version,dependencies={name:importlib.metadata.version(name) for name in ('clm','harbor','litellm','tiktoken','pyyaml')},
        container_runtime_available=bool(shutil.which('docker') or shutil.which('podman') or shutil.which('singularity')),
        classification='EXPLORATORY ONLY; live provider and isolated scholarly adapter not validated')
    (a.output/'smoke.json').write_text(json.dumps(report,indent=2))
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(report,indent=2))
    if not report['cli_starts']:raise ValueError('Reference CLI failed to start; inspect cli-help.txt')


if __name__=='__main__':main()
