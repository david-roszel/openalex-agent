"""Immutable selected-context snapshots and isolated CLM continuation clones."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib,json,shutil
from .context_memory import ContextStore
from .context_planner import ContextPlanner
from .llm import LLMUsage
from .corpus import atomic_json


def hashes(directory):return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(directory).rglob('*')) if p.is_file() and p.name!='SNAPSHOT.json'}


def snapshot(planner,directory,*,extra=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    planner.store.verify_committed();shutil.copytree(planner.store.directory,directory/'context')
    state={'mode':planner.mode,'question':planner.question,'budget':asdict(planner.budget),'limits':asdict(planner.limits),
        'usage':asdict(planner.usage),'last_model_input':deepcopy(planner.last_model_input),'seen':deepcopy(planner._seen),'pending':planner._pending,
        'pending_indices':list(planner._pending_indices),'feedback':deepcopy(planner._feedback),'evidence_records':deepcopy(planner.evidence_records),
        'factual_used':planner._factual_response_used,'archive_reads':planner.archive_reads,'completion_guidance':planner.completion_guidance,
        'store':{'version':planner.store.version,'observations':deepcopy(planner.store.observations),'edits':deepcopy(planner.store.edits),'committed_hash':planner.store._committed_hash},
        'extra':deepcopy(extra or {})}
    atomic_json(directory/'state.json',state);manifest={'version':'v3-isolated-context-snapshot-v1','files':hashes(directory),'read_only':True}
    atomic_json(directory/'SNAPSHOT.json',manifest)
    for p in directory.rglob('*'):
        if p.is_file():p.chmod(0o444)
    return manifest


def verify_snapshot(directory):
    directory=Path(directory);manifest=json.loads((directory/'SNAPSHOT.json').read_text())
    if hashes(directory)!=manifest['files']:raise ValueError('Immutable snapshot changed')
    return manifest


def fork(directory,branch,base,*,sink,observed_papers):
    from .controller import Budget
    from .context_planner import InferenceLimits
    directory=Path(directory);branch=Path(branch);verify_snapshot(directory)
    if branch.exists():raise ValueError('Branch already exists; no replay')
    branch.mkdir(parents=True);state=json.loads((directory/'state.json').read_text())
    shutil.copytree(directory/'context',branch/'context')
    for p in (branch/'context').rglob('*'):
        if p.is_file():p.chmod(0o644)
    base.usage=LLMUsage(**state['usage'])
    p=ContextPlanner(base,mode=state['mode'],limits=InferenceLimits(**state['limits']),archive_reads=state['archive_reads'],completion_guidance=state['completion_guidance'])
    p.budget=Budget(**state['budget']);p.question=state['question'];p.sink=sink;p.observed_papers=observed_papers
    store=ContextStore.__new__(ContextStore);store.directory=(branch/'context').absolute();store.max_chars=p.budget.max_context_chars;store.sink=sink;store.path=store.directory/'active_context.txt'
    store.version=state['store']['version'];store.observations=state['store']['observations'];store.edits=state['store']['edits'];store._committed_hash=state['store']['committed_hash'];store._uncertain=False
    store.verify_committed();p.store=store;p.last_model_input=state['last_model_input'];p._seen=state['seen'];p._pending=state['pending'];p._pending_indices=state['pending_indices'];p._feedback=state['feedback'];p.evidence_records=state['evidence_records'];p._factual_response_used=state['factual_used']
    return p,state['extra']


def checkpoint(planner,path,*,extra=None):
    """Durable per-batch state without repeatedly copying immutable archives."""
    planner.store.verify_committed();path=Path(path)
    if path.exists():raise ValueError('Checkpoint already exists; no overwrite')
    state={'version':'v3-per-batch-state-v1','question':planner.question,'mode':planner.mode,'budget':asdict(planner.budget),'limits':asdict(planner.limits),'usage':asdict(planner.usage),
        'active_context':planner.store.read(),'committed_hash':planner.store._committed_hash,'store_version':planner.store.version,
        'archive_directory':str(planner.store.directory),'archive_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(planner.store.directory.glob('observation-*.json'))},
        'pending':planner._pending,'pending_indices':list(planner._pending_indices),'seen':deepcopy(planner._seen),'feedback':deepcopy(planner._feedback),'last_model_input':deepcopy(planner.last_model_input),'extra':deepcopy(extra or {})}
    atomic_json(path,state);path.chmod(0o444);return hashlib.sha256(path.read_bytes()).hexdigest()
