"""Restricted file capability around the unmodified Meta conversation synchronizer.

This is deliberately not a shell. Model strings are never evaluated as Python or
Bash. The only readable/writable object is this run's LIVE_CTX_MAIN.txt.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from ._vendor.meta_clm.context_env.env import ContextEnv
from ._vendor.meta_clm.context_env.paths import Paths
from ._vendor.meta_clm.utils.budget import BudgetController
from ._vendor.meta_clm.utils import tokens
from .context_memory import serialized
from .corpus import atomic_json
from .token_budget import encoding

META_VERSION='meta-conversation-reference-v3-tool-contract'
VENDOR=Path(__file__).parent/'_vendor/meta_clm'


def reference_provenance():
    provenance=json.loads((VENDOR/'PROVENANCE.json').read_text())
    for name,expected in provenance['files'].items():
        if hashlib.sha256((VENDOR/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Pinned Meta reference checksum mismatch')
    return provenance


class ContextCapability:
    """A single-file object capability; no pathname, shell, network or corpus API."""
    virtual='/context/LIVE_CTX_MAIN.txt'

    def __init__(self, directory, max_chars):
        self.directory=Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.file=self.directory/'LIVE_CTX_MAIN.txt'
        self.file.write_text('')
        self.max_chars=max_chars

    def guard(self):
        if self.directory.resolve()!=self.directory.absolute() or self.file.is_symlink():
            raise ValueError('context_path_redirected')

    async def upload_file(self, source, destination):
        self.guard()
        if destination!=self.virtual:raise ValueError('path_not_allowed')
        # source is supplied only by trusted ContextEnv, never by the model.
        self.file.write_text(Path(source).read_text())

    async def download_file(self, source, destination):
        self.guard()
        if source!=self.virtual:raise ValueError('path_not_allowed')
        Path(destination).write_text(self.file.read_text())

    async def exec(self, *, command, timeout_sec):
        try:
            self.guard()
            operation=json.loads(command)
            if not isinstance(operation,dict) or operation.get('path')!='LIVE_CTX_MAIN.txt':
                raise ValueError('path_not_allowed')
            op=operation.get('op')
            keys={'read':{'op','path','start','chars'},'replace':{'op','path','text'},
                  'append':{'op','path','text'},'splice':{'op','path','old','new'}}
            if op not in keys or set(operation)!=keys[op]:raise ValueError('invalid_context_operation')
            before=self.file.read_text()
            if op=='read':
                start,chars=operation['start'],operation['chars']
                if type(start) is not int or start<0 or type(chars) is not int or not 1<=chars<=512:
                    raise ValueError('invalid_context_slice')
                return SimpleNamespace(stdout=before[start:start+chars],stderr='',return_code=0)
            values=[operation[k] for k in keys[op]-{'op','path'}]
            if any(not isinstance(v,str) for v in values):raise ValueError('context_text_required')
            if op=='splice':
                if not operation['old'] or before.count(operation['old'])!=1:raise ValueError('context_match_not_unique')
                after=before.replace(operation['old'],operation['new'],1)
            else:
                after=before+operation['text'] if op=='append' else operation['text']
            if len(after)>self.max_chars:raise ValueError('context_file_too_large')
            with tempfile.NamedTemporaryFile(mode='w',dir=self.directory,delete=False) as stream:
                temporary=Path(stream.name);stream.write(after);stream.flush()
            temporary.replace(self.file)
            if self.file.read_text()!=after:raise RuntimeError('context_write_uncertain')
            return SimpleNamespace(stdout='',stderr='',return_code=0)
        except (ValueError,TypeError) as exc:
            # Never echo arbitrary command text, host paths or provider secrets.
            code=str(exc) if str(exc) in {'path_not_allowed','invalid_context_operation','invalid_context_slice',
                'context_text_required','context_match_not_unique','context_file_too_large','context_path_redirected'} else 'invalid_context_operation'
            return SimpleNamespace(stdout='',stderr=code,return_code=1)


class MetaConversation:
    def __init__(self,directory,question,budget,sink):
        self.directory=Path(directory)/'context'
        self.directory.mkdir(exist_ok=False)
        self.capability=ContextCapability(self.directory/'sandbox',budget.max_context_chars)
        self.observations=[];self.edits=[];self.sink=sink
        # Inject the existing checksum-verified offline o200k vocabulary into the
        # reference tokenizer cache; its counting implementation is unchanged.
        tokens._ENCODER=encoding();tokens._ENCODER_NAME='study-o200k-base-v1'
        self.budget=BudgetController(0,reserve_tokens=0,nudge_ratios=[.25,.5,.75],
            persistent_nudge_ratio='adaptive',compaction_hint='Edit LIVE_CTX_MAIN.txt with context_file.',
            max_num_retry_on_limit=50)
        paths=Paths(SimpleNamespace(_CTX_DIR='/context',_CTX_FILE=self.capability.virtual))
        self.env=ContextEnv(paths=paths,budget=self.budget,protect=2,persistent_bash=False,
            allow_edit_growth=True,context_budget_tokens=0)

    def archive(self,body):
        index=len(self.observations)
        self.observations.append(deepcopy(body))
        atomic_json(self.directory/f'observation-{index:05d}.json',body)
        return index

    def read(self):
        self.capability.guard()
        return self.capability.file.read_text()

    def mirror(self,messages):
        asyncio.run(self.env.write_mirror(self.capability,messages))

    def set_limit(self,messages,active_limit):
        prefix=tokens.count_tokens(messages[:2])[0]
        self.budget.budget=(active_limit+prefix) if active_limit else 0
        self.budget.strict_target=self.budget.budget
        self.env.context_budget_tokens=self.budget.budget

    def step(self,operation,messages,pending,usage,active_limit):
        started=time.perf_counter()
        before=deepcopy(messages)
        # Meta counts the protected prefix too. Offset its gate by exactly that
        # prefix so the retained editable text has the same active ceiling.
        self.set_limit(messages,active_limit)
        result=asyncio.run(self.env.step(serialized(operation),messages,
                         environment=self.capability,pending=pending))
        is_read=operation.get('op')=='read'
        accepted=(result.result.return_code==0 and (is_read or result.ctx_changed))
        record=dict(index=len(self.edits),source='model',operation=operation,accepted=accepted,
                    executor_accepted=result.result.return_code==0,
                    executor_return_code=result.result.return_code,
                    committed=result.ctx_changed,before=before,after=deepcopy(messages),
                    notes=result.notes,readout=result.readout,usage=usage,
                    command_seconds=result.exec_time,synchronization_seconds=time.perf_counter()-started,
                    reference_syncs=self.env.n_ctx_syncs,
                    reference_meaningful_syncs=self.env.n_ctx_syncs_real,
                    reference_rejected_edits=self.env.n_ctx_rejected)
        if is_read:
            self.sink('context_recovery',{**record,'source':'surviving_conversation_only'})
        else:
            self.edits.append(record)
            atomic_json(self.directory/f'meta-edit-{record["index"]:05d}.json',record)
            self.sink('context_edit',record)
        return result

    def close(self):
        self.env.host_mirror.unlink(missing_ok=True)
