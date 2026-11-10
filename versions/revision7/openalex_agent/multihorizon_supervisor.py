"""Detached, bounded V3 execution. Offline snapshot/status never request inference.

Only independent trajectories run concurrently. Graceful pause drains whole
trajectories; interrupted trajectories and ambiguous requests are never replayed.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
import venv

from .corpus import LocalCorpus, atomic_json
from .execution_control import ExperimentStopped
from .process_identity import identity, alive
from .terra_accounting import TerraLedger
from . import multihorizon as runner

WORKERS = (1, 2, 4)
CONTROL_ENTRY = ('"""Pinned control entry point; safe from any working directory."""\n'
    'from pathlib import Path\nimport sys\n'
    'sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
    'from openalex_agent.multihorizon_supervisor import main\n'
    'if __name__ == "__main__":\n    raise SystemExit(main())\n')


@contextmanager
def exclusive(path):
    with Path(path).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ExperimentStopped('another_runner_owns_execution_lock') from None
        yield


def initialize(out):
    with runner.queue(out) as db:
        db.execute('CREATE TABLE IF NOT EXISTS supervision_control(key TEXT PRIMARY KEY,value TEXT)')
        db.execute('''CREATE TABLE IF NOT EXISTS trajectory_claims(
            job_id TEXT PRIMARY KEY,session TEXT,worker TEXT,owner TEXT,
            concurrency INTEGER,snapshot_sha256 TEXT,started REAL,ended REAL,outcome TEXT)''')


def request_pause(out, reason='user_pause'):
    initialize(out)
    with runner.queue(out) as db:
        db.execute("INSERT OR REPLACE INTO supervision_control VALUES('pause',?)", (reason,))
    return {'status': 'PAUSE_REQUESTED', 'meaning': 'No new claims; active trajectories drain without interruption'}


def claim(out, stage, session, worker, workers, snapshot_hash,allowed_ids=None):
    with runner.queue(out) as db:
        if db.execute("SELECT 1 FROM supervision_control WHERE key='pause'").fetchone():
            return None
        row = next((r for r in db.execute("SELECT * FROM jobs WHERE stage=? AND state='PENDING' ORDER BY ordinal", (stage,)) if allowed_ids is None or r['id'] in allowed_ids),None)
        if row is None:
            return None
        # Never replace an orphan output, even if its queue row is inconsistent.
        if (Path(out)/'cells'/row['id']).exists():
            db.execute("UPDATE jobs SET state='INTERRUPTED',ended=?,error='orphan_output_no_replay' WHERE id=?", (time.time(), row['id']))
            raise ExperimentStopped('orphan_output_no_replay')
        now = time.time()
        db.execute('INSERT INTO trajectory_claims VALUES(?,?,?,?,?,?,?,NULL,NULL)',
                   (row['id'], session, worker, json.dumps(identity()), workers, snapshot_hash, now))
        db.execute("UPDATE jobs SET state='RUNNING',started=? WHERE id=? AND state='PENDING'", (now, row['id']))
        return json.loads(row['payload'])


def finish(out, job_id, session, worker, outcome, error=None):
    if outcome not in ('COMPLETED', 'FAILED', 'INTERRUPTED'):
        raise ValueError('Unexpected terminal trajectory status')
    with runner.queue(out) as db:
        owner = db.execute('SELECT * FROM trajectory_claims WHERE job_id=?', (job_id,)).fetchone()
        if not owner or owner['session'] != session or owner['worker'] != worker or owner['ended'] is not None:
            raise ExperimentStopped('trajectory_claim_owner_mismatch')
        now = time.time()
        db.execute('UPDATE jobs SET state=?,ended=?,error=? WHERE id=? AND state=\'RUNNING\'', (outcome, now, error, job_id))
        db.execute('UPDATE trajectory_claims SET ended=?,outcome=? WHERE job_id=?', (now, outcome, job_id))


def recover_claims(out):
    """Only dead identities may become INTERRUPTED; never requeue a claim."""
    initialize(out)
    with runner.queue(out) as db:
        for row in list(db.execute("SELECT * FROM jobs WHERE state='RUNNING'")):
            owner = db.execute('SELECT * FROM trajectory_claims WHERE job_id=?', (row['id'],)).fetchone()
            if owner and alive(json.loads(owner['owner'])):
                raise ExperimentStopped('prior_trajectory_worker_still_alive')
            now = time.time()
            db.execute("UPDATE jobs SET state='INTERRUPTED',ended=?,error='interrupted_no_automatic_replay' WHERE id=?", (now, row['id']))
            db.execute("UPDATE trajectory_claims SET ended=?,outcome='INTERRUPTED' WHERE job_id=?", (now, row['id']))


def snapshot(out, destination):
    """Copy pinned source and installed dependencies, excluding editable hooks/secrets."""
    out = Path(out).resolve(); destination = Path(destination).resolve()
    runner.check(out)
    if (out/'runtime.json').exists():raise ValueError('Campaign already has a runtime binding; do not replace it')
    if destination.exists() or destination == runner.ROOT or destination.is_relative_to(runner.ROOT):
        raise ValueError('Choose a new snapshot directory outside the development checkout')
    spec = json.loads((out/'protocol.json').read_text())
    destination.mkdir(parents=True)
    for name, digest in spec['source_pins'].items():
        source = runner.ROOT/name
        if source.is_symlink() or runner.sha(source) != digest:
            raise ValueError('Pinned source changed during snapshot: '+name)
        target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    venv.EnvBuilder(with_pip=False, symlinks=True).create(destination/'.venv')
    site = Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
    target_site = destination/'.venv/lib'/site.parent.name/'site-packages'
    shutil.copytree(site, target_site, dirs_exist_ok=True, symlinks=False,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '__editable__*'))
    # Other executable .pth hooks could redirect imports outside the snapshot.
    if list(target_site.glob('*.pth')):
        raise ValueError('Review unsupported dependency .pth hooks before sealing runtime')
    (destination/'v3ctl.py').write_text(CONTROL_ENTRY)
    files = {str(p.relative_to(destination)): runner.sha(p) for p in destination.rglob('*') if p.is_file() and not p.is_symlink()}
    interpreter = Path(sys.executable).resolve()
    manifest = {'version': 'v3-isolated-runtime-v1', 'protocol_sha256': runner.sha(out/'protocol.json'),
                'campaign': str(out), 'created_utc': datetime.now(timezone.utc).isoformat(),
                'interpreter': str(interpreter), 'interpreter_sha256': runner.sha(interpreter), 'files': files,
                'source_pins': spec['source_pins'], 'credentials_copied': False}
    atomic_json(destination/'runtime-manifest.json', manifest)
    digest = runner.sha(destination/'runtime-manifest.json')
    for p in sorted(destination.rglob('*'), reverse=True):
        if not p.is_symlink():
            p.chmod(0o555 if p.is_dir() or os.access(p, os.X_OK) else 0o444)
    destination.chmod(0o555)
    descriptor = {'snapshot': str(destination), 'snapshot_sha256': digest, 'protocol_sha256': manifest['protocol_sha256']}
    # Append-only runtime descriptors. A campaign is bound to one execution snapshot.
    if (out/'runtime.json').exists():
        raise ValueError('Campaign already has a runtime binding; do not replace it')
    atomic_json(out/'runtime.json', descriptor)
    return verify_runtime(out)


def verify_runtime(out, *, require_current=False):
    out = Path(out).resolve()
    descriptor = json.loads((out/'runtime.json').read_text()); source = Path(descriptor['snapshot'])
    if runner.sha(source/'runtime-manifest.json') != descriptor['snapshot_sha256']:
        raise ValueError('Runtime manifest identity changed')
    manifest = json.loads((source/'runtime-manifest.json').read_text())
    if manifest['campaign'] != str(out) or manifest['protocol_sha256'] != runner.sha(out/'protocol.json') or manifest['source_pins'] != json.loads((out/'protocol.json').read_text())['source_pins']:
        raise ValueError('Runtime and campaign/source pins disagree')
    if require_current and runner.ROOT != source:
        raise ValueError('Supervisor must execute from the isolated snapshot')
    for name, digest in manifest['files'].items():
        if runner.sha(source/name) != digest:
            raise ValueError('Runtime source/dependency changed: '+name)
    if runner.sha(manifest['interpreter']) != manifest['interpreter_sha256']:
        raise ValueError('System Python interpreter changed; refreeze runtime')
    return descriptor


def _settings(env_file):
    # Read credentials once, before workers start; do not discover a development .env.
    from dotenv import dotenv_values
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    if env_file:
        for key, value in dotenv_values(env_file).items():
            if key.startswith(('LLM_', 'OPENALEX_')) and value is not None:
                os.environ[key] = value
    settings = runner.Settings.from_env()
    from .terra_v2 import BASE_URL, MODEL
    if (settings.llm_base_url.rstrip('/') != BASE_URL or settings.llm_model != MODEL or
        settings.llm_reasoning_effort != 'low' or settings.llm_temperature is not None or
        settings.llm_max_tokens != 4096 or settings.llm_timeout != 120 or
        settings.llm_input_price != 1 or settings.llm_output_price != 5 or not settings.llm_api_key):
        raise ValueError('Exact pinned Terra settings and key required; no request made')
    return settings


def worker_loop(out, approval, stage, session, worker, workers, snapshot_hash, settings,
                *, diagnostic_executor=None,selection='feasibility'):
    """Separate process for each worker; no shared Controller/planner/CLM state."""
    out = Path(out)
    spec = json.loads((out/'protocol.json').read_text())
    ledger = TerraLedger(spec['accounting']['shared_directory'], profile_hash=spec['accounting']['profile_sha256'], session_id=session, worker_id=worker)
    ledger.register_worker()
    permit = runner.Permit(out, approval, stage,selection) if diagnostic_executor is None else None
    allowed=None
    if stage.startswith('pilot-') and spec.get('pilot_selections'):
        policy=spec['pilot_selections'][selection];allowed=set(json.loads((out/policy['path']).read_text())['trajectory_ids'])
    corpus = LocalCorpus(runner.CORPUS) if diagnostic_executor is None else None
    while True:
        try:
            ledger.check()
            if permit: permit.check()
            job = claim(out, stage, session, worker, workers, snapshot_hash,allowed_ids=allowed)
            if job is None: return
        except BaseException as exc:
            request_pause(out, getattr(exc, 'reason', type(exc).__name__))
            return
        start = time.time(); mono = time.monotonic(); outcome = 'FAILED'; error = None
        try:
            result = (runner.execute_cell(out, job, corpus, settings, ledger, permit) if diagnostic_executor is None
                      else diagnostic_executor(out, job, ledger))
            outcome = result['status']
        except BaseException as exc:
            error = getattr(exc, 'reason', getattr(exc, 'code', type(exc).__name__))
            directory = out/'cells'/job['id']; directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory/'failure.json', {'job': job, 'reason': str(error), 'no_automatic_replay': True})
            # Conservative fail-stop; other in-flight requests may settle, no retry.
            request_pause(out, str(error))
        finally:
            directory = out/'cells'/job['id']; directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory/'execution.json', {'version': 'v3-execution-provenance-v1', 'session': session,
                'worker': worker, 'worker_identity': identity(), 'configured_workers': workers,
                'snapshot_sha256': snapshot_hash, 'started_epoch': start, 'ended_epoch': time.time(),
                'trajectory_wall_seconds': time.monotonic()-mono, 'within_trajectory_execution': 'sequential',
                'latency_interpretation': 'Provider timing under recorded concurrency; contention is not architectural performance'})
            finish(out, job['id'], session, worker, outcome, str(error) if error else None)


def supervise(out, approval, phase, study, workers=1, env_file=None, *, diagnostic_executor=None,selection='feasibility'):
    out = Path(out).resolve()
    if workers not in WORKERS: raise ValueError('Workers must be 1, 2 or 4')
    permit = runner.Permit(out, approval, phase+'-'+study,selection) if diagnostic_executor is None else runner.Permit(out,approval,phase+'-'+study)
    runtime = verify_runtime(out, require_current=True)
    settings = _settings(env_file)
    spec = json.loads((out/'protocol.json').read_text()); ledger_root = Path(spec['accounting']['shared_directory'])
    if phase=='pilot' and selection=='feasibility' and spec.get('pilot_selections') and workers!=1:raise ExperimentStopped('feasibility_pilot_requires_one_worker')
    initialize(out)
    with exclusive(out/'runner.lock'), exclusive(ledger_root/'execution.lock'):
        recover_claims(out)
        session = uuid.uuid4().hex
        ledger = TerraLedger(ledger_root, profile_hash=spec['accounting']['profile_sha256'], session_id=session)
        for name,entry in spec.get('pilot_selections',{}).items():
            policy=json.loads((out/entry['path']).read_text())
            if runner.sha(out/entry['path'])!=entry['sha256']:raise ExperimentStopped('pilot_selection_identity_changed')
            ledger.register_scope(runner.sha(out/'protocol.json')+':'+name,ceiling_usd=entry['ceiling_usd'],job_ids=policy['trajectory_ids'],attempt_limit=entry['provider_attempt_limit'],policy_sha256=entry['sha256'])
        ledger.activate(permit.digest, max_runtime_seconds=permit.record['max_runtime_seconds'])
        if ledger.recover_interrupted(): raise ExperimentStopped('interrupted_provider_usage_requires_reconciliation')
        # Keep existing S -> L -> M main collection gates unchanged.
        if phase == 'main' and study in ('L', 'M'):
            prior = ['main-S'] if study == 'L' else ['main-S', 'main-L']
            with runner.queue(out) as db:
                if any(db.execute("SELECT 1 FROM jobs WHERE stage=? AND state IN ('PENDING','RUNNING')", (s,)).fetchone() for s in prior):
                    raise ExperimentStopped('prespecified_main_study_order_requires_prior_studies_terminal')
        ledger.check(); ledger.begin_session(workers)
        with runner.queue(out) as db: db.execute("DELETE FROM supervision_control WHERE key='pause'")
        state = {'status': 'RUNNING', 'session': session, 'owner': identity(), 'configured_workers': workers,
                 'service_unit':os.environ.get('OPENALEX_V3_UNIT'),'pilot_selection':selection if phase=='pilot' else None,
                 'snapshot_sha256': runtime['snapshot_sha256'], 'phase': phase, 'study': study,
                 'started_epoch': time.time(), 'workers': [], 'no_automatic_replay': True}
        control = out/'supervision'; control.mkdir(exist_ok=True)
        atomic_json(control/f'session-{session}.json', state)
        stopping = [False]
        def stop(signum, frame): stopping[0] = True
        previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        processes = []
        last_report = 0; last_terminal = -1
        try:
            ctx = mp.get_context('spawn')
            for n in range(workers):
                p = ctx.Process(target=worker_loop, args=(str(out), str(Path(approval).resolve()), phase+'-'+study,
                                session, f'worker-{n+1}', workers, runtime['snapshot_sha256'], settings),
                                kwargs={'diagnostic_executor':diagnostic_executor,'selection':selection})
                p.start(); processes.append(p)
            while any(p.is_alive() for p in processes):
                if stopping[0]: request_pause(out, 'supervisor_graceful_stop')
                try: ledger.check()
                except ExperimentStopped as exc: request_pause(out, exc.reason)
                if any(p.exitcode not in (None, 0) for p in processes):
                    request_pause(out, 'worker_process_failed'); ledger.recover_interrupted()
                with runner.queue(out) as db:
                    pause = db.execute("SELECT value FROM supervision_control WHERE key='pause'").fetchone()
                    terminal = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('COMPLETED','FAILED','INTERRUPTED')").fetchone()[0]
                if terminal!=last_terminal and time.time()-last_report>=30:
                    runner.report(out); last_terminal=terminal; last_report=time.time()
                state.update(status='PAUSING' if pause else 'RUNNING', reason=pause['value'] if pause else None,
                    observed_epoch=time.time(), workers=[{'pid':p.pid,'identity':identity(p.pid),'alive':p.is_alive(),'exitcode':p.exitcode} for p in processes])
                atomic_json(control/'status.json', state)
                time.sleep(1)
            for p in processes: p.join()
            ledger.recover_interrupted(); recover_claims(out)
            with runner.queue(out) as db:
                pause = db.execute("SELECT value FROM supervision_control WHERE key='pause'").fetchone()
            state.update(status='STOPPED' if any(p.exitcode for p in processes) else 'PAUSED' if pause else 'STUDY_COLLECTION_TERMINAL',
                         reason=pause['value'] if pause else None, finished_epoch=time.time(),
                         workers=[{'pid':p.pid,'alive':False,'exitcode':p.exitcode} for p in processes])
        except BaseException as exc:
            state.update(status='STOPPED',reason=getattr(exc,'reason',type(exc).__name__),finished_epoch=time.time())
            raise
        finally:
            for s, handler in previous.items(): signal.signal(s, handler)
            # On unexpected parent failure, never release ownership while children run.
            if any(p.is_alive() for p in processes):
                request_pause(out, 'supervisor_failure_drain')
                for p in processes: p.join()
            ledger.recover_interrupted(); recover_claims(out); ledger.end_session()
            atomic_json(control/'status.json', state); atomic_json(control/f'session-{session}.json', state)
            atomic_json(out/'status.json', state); runner.report(out)
        return state


def start(out, approval, phase, study, workers=1, env_file=None,*,selection='feasibility'):
    """Launch a systemd USER service, never Codex or a login-shell child."""
    if workers not in WORKERS: raise ValueError('Workers must be 1, 2 or 4')
    out = Path(out).resolve(); approval = Path(approval).resolve()
    runtime = verify_runtime(out)
    # Validate from snapshot, so editing development source cannot invalidate launch.
    source = Path(runtime['snapshot'])
    command = [str(source/'.venv/bin/python'), str(source/'v3ctl.py'), 'supervise',
               '--campaign', str(out), '--approval', str(approval), '--phase', phase, '--study', study, '--workers', str(workers),'--pilot-selection',selection]
    if env_file: command += ['--env-file', str(Path(env_file).resolve())]
    unit = 'openalex-v3-'+uuid.uuid4().hex[:12]
    argv = ['systemd-run', '--user', '--unit='+unit, '--collect', '--service-type=exec', '--expand-environment=no',
            '--working-directory='+str(source), '--property=KillMode=mixed', '--property=TimeoutStopSec=infinity',
            '--property=Restart=no', '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=PYTHONUNBUFFERED=1',
            '--setenv=OPENALEX_V3_UNIT='+unit+'.service',
            '--setenv=PYTHONNOUSERSITE=1', '--setenv=PYTHONPATH='+str(source),
            '--setenv=MPLCONFIGDIR='+str(out/'supervision/matplotlib'), *command]
    subprocess.run(argv, check=True)
    return {'status': 'START_SUBMITTED', 'unit': unit+'.service', 'workers': workers,
            'note': 'Service still enforces approval, frozen pilots and shared ledger before any request. No automatic restart.'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    for name in ('snapshot', 'verify-runtime', 'start', 'resume', 'supervise', 'pause', 'status'):
        a = sub.add_parser(name); a.add_argument('--campaign', type=Path, default=runner.DEFAULT)
        if name == 'snapshot': a.add_argument('--destination', type=Path, required=True)
        if name in ('start','resume','supervise'):
            a.add_argument('--approval', type=Path, required=True); a.add_argument('--phase', choices=('pilot','main'), required=True)
            a.add_argument('--study', choices=('S','L','M'), required=True); a.add_argument('--workers', type=int, choices=WORKERS, default=1)
            a.add_argument('--env-file', type=Path)
            a.add_argument('--pilot-selection',choices=('feasibility','full'),default='feasibility')
    args = p.parse_args(argv)
    try:
        if args.command == 'snapshot': result = snapshot(args.campaign, args.destination)
        elif args.command == 'verify-runtime': result = verify_runtime(args.campaign)
        elif args.command == 'pause': result = request_pause(args.campaign)
        elif args.command == 'status':
            from .multihorizon_status import status
            result = status(args.campaign)
        else:
            f = supervise if args.command == 'supervise' else start
            result = f(args.campaign,args.approval,args.phase,args.study,args.workers,args.env_file,selection=args.pilot_selection)
    except (ValueError, OSError, ExperimentStopped, subprocess.CalledProcessError) as exc:
        print(json.dumps({'status':'BLOCKED','reason':getattr(exc,'reason',str(exc))})); return 2
    print(json.dumps(result, indent=2)); return 0


if __name__ == '__main__': raise SystemExit(main())
