"""Read-only V3 supervision/accounting observer. No provider/executor imports."""
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
import json
import shutil
import sqlite3
import tempfile
import time
from .process_identity import alive


@contextmanager
def copied_database(path):
    """Copy DB+WAL consistently; SQLite writes only into a disposable directory."""
    path = Path(path)
    def stamps(paths):
        return [(p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in paths]
    with tempfile.TemporaryDirectory(prefix='v3-observer-') as temp:
        target = Path(temp)/path.name; paths = [path, Path(str(path)+'-wal')]
        for _ in range(10):
            before = stamps(paths)
            for p, mark in zip(paths, before):
                q = target if p == path else Path(str(target)+'-wal')
                if mark is not None: shutil.copyfile(p, q)
                elif q.exists(): q.unlink()
            if before == stamps(paths): break
        else: raise RuntimeError('Active database changing; retry observer')
        with sqlite3.connect(target) as db:
            db.row_factory = sqlite3.Row; db.execute('PRAGMA query_only=ON')
            if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok': raise RuntimeError('Observer copy failed integrity check')
            yield db


def status(out):
    out = Path(out); spec = json.loads((out/'protocol.json').read_text())
    saved = out/'supervision/status.json'
    state = json.loads(saved.read_text()) if saved.exists() else {'status':'NOT_STARTED','configured_workers':1}
    with copied_database(out/'queue.sqlite') as db:
        jobs = [dict(r) for r in db.execute('SELECT * FROM jobs')]
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        claims = [dict(r) for r in db.execute('SELECT * FROM trajectory_claims')] if 'trajectory_claims' in tables else []
    with copied_database(Path(spec['accounting']['shared_directory'])/'spending.sqlite') as db:
        attempts = [dict(r) for r in db.execute('SELECT * FROM attempts')]
        control = {r['key']:r['value'] for r in db.execute('SELECT * FROM control')}
    terminal = {'COMPLETED','FAILED','INTERRUPTED'}
    main = [j for j in jobs if j['stage'].startswith('main-')]
    stage = state.get('phase','')+'-'+state.get('study','')
    current = [j for j in jobs if j['stage']==stage]
    selection=state.get('pilot_selection')
    selected=[]
    if state.get('phase')=='pilot' and selection in spec.get('pilot_selections',{}):
        ids=set(json.loads((out/spec['pilot_selections'][selection]['path']).read_text())['trajectory_ids'])
        current=[j for j in current if j['id'] in ids];selected=[j for j in jobs if j['id'] in ids]
    done = [j for j in main if j['state'] in terminal]
    live = alive(state.get('owner'))
    if state['status'] in ('RUNNING','PAUSING') and not live: state = {**state,'status':'STOPPED','reason':'supervisor_identity_not_alive'}
    active = sum(alive(w.get('identity')) for w in state.get('workers', []))
    owners = {c['job_id']:json.loads(c['owner']) for c in claims}
    elapsed = max(0, state.get('finished_epoch',time.time())-state.get('started_epoch',time.time()))
    session_done = sum(c['session']==state.get('session') and c['ended'] is not None for c in claims)
    throughput = session_done/elapsed if elapsed>0 and session_done else None
    remaining_stage = sum(j['state'] not in terminal for j in current)
    return {'status':state['status'],'reason':state.get('reason'),'configured_workers':state['configured_workers'],
        'active_workers':active,'session':state.get('session'),'service_unit':state.get('service_unit'),'snapshot_sha256':state.get('snapshot_sha256'),
        'core_planned':len(main),'completed':sum(j['state']=='COMPLETED' for j in main),
        'failed':sum(j['state'] in ('FAILED','INTERRUPTED') for j in main),'terminal':len(done),
        'running_trajectories':[j['id'] for j in jobs if j['state']=='RUNNING' and alive(owners.get(j['id']))],
        'interrupted_awaiting_recovery':[j['id'] for j in jobs if j['state']=='RUNNING' and not alive(owners.get(j['id']))],
        'remaining_core':len(main)-len(done),'current_stage':stage,'remaining_current_stage':remaining_stage,
        'pilot_selection':selection,'pilot_selected_total':len(selected) if selected else None,'pilot_selected_terminal':sum(j['state'] in terminal for j in selected),
        'elapsed_session_seconds':elapsed,'observed_trajectories_per_hour':throughput*3600 if throughput else None,
        'estimated_remaining_stage_seconds':remaining_stage/throughput if throughput else None,
        'eta_interpretation':'Estimate from this session only; mixed trajectory lengths, pauses and provider contention prevent guaranteed completion; no cross-study ETA',
        'provider_attempts':len(attempts),'rate_limit_events':sum((json.loads(a['diagnostic']) if a['diagnostic'] else {}).get('http_status')==429 for a in attempts),
        'unknown_requests':sum(a['status']=='UNKNOWN' for a in attempts),'inflight_requests':sum(a['status']=='RESERVED' for a in attempts),
        'conservative_exposure_usd':str(sum((Decimal(a['retained']) for a in attempts),Decimal(0))),
        'confirmed_conservative_usd':str(sum((Decimal(a['charge']) for a in attempts if a['charge'] is not None),Decimal(0))),
        'ceiling_usd':'68.00','financial_stop_reason':control.get('stop_reason'),'accounting_source':'Read-only copy of authoritative cumulative SQLite ledger',
        'workers':state.get('workers',[])}
