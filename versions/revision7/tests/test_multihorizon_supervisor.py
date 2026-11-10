"""Offline multiprocess diagnostics. No real authorization or provider traffic."""
from decimal import Decimal
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
import pytest

from openalex_agent import multihorizon as runner
from openalex_agent import multihorizon_supervisor as supervision
from openalex_agent.multihorizon_status import status
from openalex_agent.process_identity import identity
from openalex_agent.terra_accounting import TerraLedger
from openalex_agent.terra_v2 import TerraPlanner
from openalex_agent.execution_control import ExperimentStopped
from graph_helpers import settings, envelope, OfflinePermit


def setup(out, count=10):
    out.mkdir()
    ledger=TerraLedger(out/'money',profile_hash='OFFLINE_CONCURRENCY_FIXTURE')
    ledger.activate('SYNTHETIC_TEST_NOT_REAL_AUTHORIZATION',max_runtime_seconds=3600)
    (out/'protocol.json').write_text(json.dumps({'accounting':{'shared_directory':str(out/'money'),'profile_sha256':ledger.profile_hash}}))
    with runner.queue(out) as db:
        db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,ordinal INTEGER,stage TEXT,payload TEXT,state TEXT,started REAL,ended REAL,error TEXT)')
        for n in range(count):
            method=('standard','llm','llm_graph','clm','clm_graph')[n%5]
            job={'id':f'OFFLINE-{n}','phase':'main','study':'S','method':method,'task_id':str(n),'repetition':0,'domain':'fixture'}
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,'PENDING',NULL,NULL,NULL)",(job['id'],n,'main-S',json.dumps(job)))
    supervision.initialize(out)
    return ledger


def synthetic_trajectory(out, job, ledger):
    """Local fixture state and real adapter with MockTransport, not research results."""
    folder=out/'cells'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    local=[]; wires=[]
    def handler(request):
        body=json.loads(request.content);value=json.loads(body['messages'][1]['content'])
        assert value['trajectory']==job['id'] and value['ordered_steps']==local
        wires.append(value);time.sleep(.05)
        return httpx.Response(200,json=envelope({'accepted_step':len(local)}))
    if job['method']!='standard':
        with TerraPlanner(settings(),ledger=ledger,cell_id=job['id'],architecture=job['method'],permit=OfflinePermit(),transport=httpx.MockTransport(handler),receipt_directory=out/'provider') as planner:
            for step in range(2):
                planner._call('Return JSON for an offline engineering diagnostic',{'trajectory':job['id'],'ordered_steps':list(local)})
                local.append(step)
    result={'status':'COMPLETED','mocked_generation':True,'local_steps':local,'method':job['method']}
    (folder/'summary.json').write_text(json.dumps(result))
    return result


def timeout_trajectory(out, job, ledger):
    folder=out/'cells'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    def handler(request): raise httpx.ReadTimeout('OFFLINE unknown-usage diagnostic')
    p=TerraPlanner(settings(),ledger=ledger,cell_id=job['id'],architecture='llm',permit=OfflinePermit(),transport=httpx.MockTransport(handler),receipt_directory=out/'provider')
    try:p._call('Return JSON',{})
    finally:p.close()


def die_after_reservation(out, session, worker, ready):
    l=TerraLedger(out/'money',profile_hash='OFFLINE_CONCURRENCY_FIXTURE',session_id=session,worker_id=worker);l.register_worker()
    job=supervision.claim(out,'main-S',session,worker,1,'fixture')
    l.reserve(job['id']+':call1',job['id'],'llm',b'{}');ready.set()
    time.sleep(30)


def reserve_together(out, session, worker, ready, release):
    l=TerraLedger(out/'money',profile_hash='OFFLINE_CONCURRENCY_FIXTURE',session_id=session,worker_id=worker);l.register_worker()
    try:
        index=l.reserve(worker,worker,'llm',b'{}');ready.put(('reserved',index));release.wait(10)
        l.settle(index,{'prompt_tokens':100,'completion_tokens':20})
    except ExperimentStopped as exc: ready.put(('blocked',exc.reason))


def launch_workers(out, ledger, n, executor=synthetic_trajectory):
    session='fixture-'+str(time.time_ns());ledger.session_id=session;ledger.begin_session(n)
    ctx=mp.get_context('spawn')
    children=[ctx.Process(target=supervision.worker_loop,args=(str(out),'NOT_AN_APPROVAL','main-S',session,f'w{i}',n,'OFFLINE_SNAPSHOT',None),kwargs={'diagnostic_executor':executor}) for i in range(n)]
    for p in children:p.start()
    for p in children:
        p.join(30)
        assert not p.is_alive()
        assert p.exitcode==0
    ledger.recover_interrupted();supervision.recover_claims(out);ledger.end_session()
    return session


@pytest.mark.parametrize('workers',[1,2,4])
def test_unique_claims_ordered_local_state_and_mocked_requests(tmp_path,workers):
    out=tmp_path/'campaign';ledger=setup(out)
    launch_workers(out,ledger,workers)
    with runner.queue(out) as db:
        jobs=list(db.execute('SELECT * FROM jobs'));claims=list(db.execute('SELECT * FROM trajectory_claims'))
    assert len(claims)==10 and all(j['state']=='COMPLETED' for j in jobs)
    assert all(c['concurrency']==workers and c['outcome']=='COMPLETED' for c in claims)
    data=json.loads((out/'money/spending-ledger.json').read_text())
    assert len(data['attempts'])==16 and len({a['request_key'] for a in data['attempts']})==16
    assert all(a['status']=='SETTLED' for a in data['attempts'])
    assert Decimal(data['retained_usd'])==sum(Decimal(a['charge']) for a in data['attempts'])
    assert all(a['diagnostic']['execution_session'] for a in data['attempts'])
    for p in (out/'cells').glob('*/summary.json'):
        value=json.loads(p.read_text());assert value['local_steps']==([] if value['method']=='standard' else [0,1])
    before={str(p):p.read_bytes() for p in (out/'cells').rglob('*') if p.is_file()}
    # A second session has no claimable work, never replays completed provider calls.
    launch_workers(out,ledger,workers)
    assert len(json.loads((out/'money/spending-ledger.json').read_text())['attempts'])==16
    assert before=={str(p):p.read_bytes() for p in (out/'cells').rglob('*') if p.is_file()}


def test_one_and_two_workers_produce_equivalent_fixture_outputs(tmp_path):
    outputs=[]
    for n in (1,2):
        out=tmp_path/str(n);ledger=setup(out);launch_workers(out,ledger,n)
        outputs.append({p.parent.name:json.loads(p.read_text()) for p in (out/'cells').glob('*/summary.json')})
    assert outputs[0]==outputs[1]


def test_concurrent_holds_are_atomic_and_cannot_cross_68(tmp_path):
    out=tmp_path/'campaign';ledger=setup(out,0);ledger.session_id='session';ledger.begin_session(2)
    with ledger.transaction() as db:
        db.execute("INSERT INTO attempts(request_key,status,retained,charge) VALUES('prior','SETTLED','67.95','67.95')")
    ctx=mp.get_context('spawn');ready=ctx.Queue();release=ctx.Event()
    children=[ctx.Process(target=reserve_together,args=(out,'session',str(i),ready,release)) for i in range(2)]
    for p in children:p.start()
    results=[ready.get(timeout=15) for _ in children]
    assert sorted(r[0] for r in results)==['blocked','reserved']
    data=json.loads((out/'money/spending-ledger.json').read_text())
    assert Decimal(data['retained_usd'])<=68 and sum(a['status']=='RESERVED' for a in data['attempts'])==1
    release.set()
    for p in children:p.join(15);assert p.exitcode==0
    ledger.end_session()


def test_two_reservations_really_overlap_and_owner_recovery_is_safe(tmp_path):
    out=tmp_path/'campaign';ledger=setup(out,0);ledger.session_id='session';ledger.begin_session(2)
    ctx=mp.get_context('spawn');ready=ctx.Queue();release=ctx.Event()
    children=[ctx.Process(target=reserve_together,args=(out,'session',str(i),ready,release)) for i in range(2)]
    for p in children:p.start()
    assert all(ready.get(timeout=15)[0]=='reserved' for _ in children)
    assert ledger.recover_interrupted()==0 # Live owners are never marked UNKNOWN.
    legacy=TerraLedger(out/'money',profile_hash=ledger.profile_hash)
    with pytest.raises(ExperimentStopped,match='live_supervisor'):legacy.recover_interrupted()
    with pytest.raises(ExperimentStopped):legacy.reserve('intruder','intruder','llm',b'{}')
    data=json.loads((out/'money/spending-ledger.json').read_text());assert sum(a['status']=='RESERVED' for a in data['attempts'])==2
    release.set()
    for p in children:p.join(15);assert p.exitcode==0
    ledger.end_session()


def test_forced_process_death_preserves_hold_and_never_requeues(tmp_path):
    out=tmp_path/'campaign';ledger=setup(out,1);ledger.session_id='session';ledger.begin_session(1)
    ctx=mp.get_context('spawn');ready=ctx.Event()
    p=ctx.Process(target=die_after_reservation,args=(out,'session','w',ready));p.start()
    assert ready.wait(15)
    os.kill(p.pid,signal.SIGKILL);p.join(15)
    assert ledger.recover_interrupted()==1
    supervision.recover_claims(out)
    with runner.queue(out) as db:assert db.execute('SELECT state FROM jobs').fetchone()[0]=='INTERRUPTED'
    data=json.loads((out/'money/spending-ledger.json').read_text());assert data['attempts'][0]['status']=='UNKNOWN'
    assert data['attempts'][0]['retained']==data['attempts'][0]['reservation']
    with pytest.raises(ExperimentStopped):ledger.reserve('replay','cell','llm',b'{}')
    ledger.end_session()


def test_pause_blocks_new_claims_and_resume_preserves_terminal_cells(tmp_path):
    out=tmp_path/'campaign';ledger=setup(out,2)
    job=supervision.claim(out,'main-S','session','w',1,'fixture')
    supervision.request_pause(out)
    assert supervision.claim(out,'main-S','session','w',1,'fixture') is None
    supervision.finish(out,job['id'],'session','w','COMPLETED')
    with runner.queue(out) as db:db.execute("DELETE FROM supervision_control WHERE key='pause'")
    next_job=supervision.claim(out,'main-S','resumed','w',2,'fixture')
    assert job['id']!=next_job['id']
    with pytest.raises(ExperimentStopped):supervision.finish(out,job['id'],'session','w','COMPLETED')


def test_unknown_usage_stops_other_workers_and_is_not_retried(tmp_path):
    out=tmp_path/'campaign';ledger=setup(out,8)
    launch_workers(out,ledger,2,timeout_trajectory)
    data=json.loads((out/'money/spending-ledger.json').read_text())
    assert data['stop_reason']=='campaign_unaccountable_usage'
    assert 1<=len(data['attempts'])<=2
    assert all(a['status']=='UNKNOWN' for a in data['attempts'])
    assert len({a['request_key'] for a in data['attempts']})==len(data['attempts'])
    with pytest.raises(ExperimentStopped):ledger.check()


def test_status_reads_authoritative_accounting_without_writes(tmp_path):
    out=tmp_path/'campaign';ledger=setup(out,2)
    n=ledger.reserve('fixture','fixture','llm',b'{}')
    (out/'supervision').mkdir();(out/'supervision/status.json').write_text(json.dumps({'status':'RUNNING','owner':identity(),'configured_workers':2,'started_epoch':time.time()-10,'workers':[{'identity':identity()}]}))
    before={str(p):(p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.rglob('*') if p.is_file()}
    view=status(out)
    assert view['inflight_requests']==1 and Decimal(view['conservative_exposure_usd'])>0 and view['active_workers']==1
    assert view['estimated_remaining_stage_seconds'] is None
    assert before=={str(p):(p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.rglob('*') if p.is_file()}
    ledger.settle(n,{'prompt_tokens':100,'completion_tokens':20})


def test_separate_runner_locks_reject_duplicate_supervisors(tmp_path):
    path=tmp_path/'execution.lock'
    with supervision.exclusive(path):
        with pytest.raises(ExperimentStopped,match='execution_lock'):
            with supervision.exclusive(path):pytest.fail('Duplicate lock acquired')


def test_orphan_output_and_wrong_owner_cannot_be_overwritten(tmp_path):
    out=tmp_path/'campaign';setup(out,2)
    job=supervision.claim(out,'main-S','session','owner',1,'fixture')
    with pytest.raises(ExperimentStopped,match='owner_mismatch'):
        supervision.finish(out,job['id'],'other-session','owner','COMPLETED')
    orphan=out/'cells/OFFLINE-1';orphan.mkdir(parents=True);(orphan/'checkpoint.json').write_text('PRESERVE')
    with pytest.raises(ExperimentStopped,match='orphan_output'):
        supervision.claim(out,'main-S','session','owner',1,'fixture')
    assert (orphan/'checkpoint.json').read_text()=='PRESERVE'


def test_start_builds_detached_systemd_command_without_codex(tmp_path,monkeypatch):
    out=tmp_path/'campaign';out.mkdir();snapshot=tmp_path/'snapshot'
    monkeypatch.setattr(supervision,'verify_runtime',lambda _: {'snapshot':str(snapshot),'snapshot_sha256':'fixture'})
    calls=[];monkeypatch.setattr(supervision.subprocess,'run',lambda argv,**kw:calls.append((argv,kw)))
    result=supervision.start(out,tmp_path/'NOT_A_REAL_APPROVAL','pilot','S',2,tmp_path/'credentials.env')
    command=calls[0][0]
    assert command[0]=='systemd-run' and '--user' in command and '--property=Restart=no' in command
    assert '--property=KillMode=mixed' in command and '--property=TimeoutStopSec=infinity' in command
    assert not any('codex' in part.lower() for part in command)
    assert '--workers' in command and command[command.index('--workers')+1]=='2'
    assert result['status']=='START_SUBMITTED'


def test_existing_unapproved_configuration_is_rejected_before_runtime_or_credentials(tmp_path,monkeypatch):
    out=tmp_path/'campaign';out.mkdir()
    monkeypatch.setattr(supervision.runner,'Permit',lambda *a:(_ for _ in ()).throw(ExperimentStopped('explicit_approval_required')))
    monkeypatch.setattr(supervision,'verify_runtime',lambda *a,**kw:pytest.fail('Runtime used before authorization'))
    monkeypatch.setattr(supervision,'_settings',lambda *a:pytest.fail('Credentials loaded before authorization'))
    with pytest.raises(ExperimentStopped,match='approval_required'):supervision.supervise(out,tmp_path/'missing','main','M')


def test_snapshot_copies_dependencies_and_survives_development_edits(tmp_path,monkeypatch):
    dev=tmp_path/'dev';dev.mkdir();(dev/'module.py').write_text('PINNED = True\n')
    out=tmp_path/'campaign';out.mkdir()
    spec={'source_pins':{'module.py':runner.sha(dev/'module.py')}}
    (out/'protocol.json').write_text(json.dumps(spec))
    prefix=tmp_path/'installed';site=prefix/'lib'/f'python{supervision.sys.version_info.major}.{supervision.sys.version_info.minor}'/'site-packages';site.mkdir(parents=True)
    (site/'dependency.py').write_text('IMMUTABLE = True\n');(site/'__editable__.example.pth').write_text('UNSAFE_EDITABLE_HOOK')
    class Builder:
        def __init__(self,**kw):pass
        def create(self,path):(path/'lib'/site.parent.name/'site-packages').mkdir(parents=True)
    monkeypatch.setattr(runner,'ROOT',dev);monkeypatch.setattr(runner,'check',lambda out:None)
    monkeypatch.setattr(supervision.sys,'prefix',str(prefix));monkeypatch.setattr(supervision.venv,'EnvBuilder',Builder)
    runtime=tmp_path/'runtime';supervision.snapshot(out,runtime)
    assert (runtime/'.venv/lib'/site.parent.name/'site-packages/dependency.py').read_text()=='IMMUTABLE = True\n'
    assert not list(runtime.rglob('*.pth'))
    (dev/'module.py').write_text('DEVELOPMENT_CHANGED = True\n');(site/'dependency.py').write_text('LOCAL_ENV_CHANGED = True\n')
    supervision.verify_runtime(out)
    assert (runtime/'module.py').read_text()=='PINNED = True\n'
    with pytest.raises(ValueError,match='isolated snapshot'):supervision.verify_runtime(out,require_current=True)
    (runtime/'module.py').chmod(0o644);(runtime/'module.py').write_text('MODIFIED\n')
    with pytest.raises(ValueError,match='Runtime source/dependency changed'):supervision.verify_runtime(out)


def supervisor_fixture_process(out, workers):
    """Patch gates only inside a synthetic no-network test process; no approval files."""
    class Permit:
        digest='SYNTHETIC_TEST_NOT_REAL_AUTHORIZATION'
        record={'max_runtime_seconds':3600}
        def __init__(self,*a):pass
        def check(self):pass
    supervision.runner.Permit=Permit
    supervision.verify_runtime=lambda *a,**kw:{'snapshot_sha256':'SYNTHETIC_FIXTURE'}
    supervision._settings=lambda *a:settings()
    supervision.runner.report=lambda *a:None
    supervision.supervise(out,'NO_APPROVAL_FILE','main','S',workers,diagnostic_executor=synthetic_trajectory)


def test_supervisor_sigterm_drains_and_resume_does_not_duplicate(tmp_path):
    out=tmp_path/'campaign';setup(out,30)
    ctx=mp.get_context('spawn');process=ctx.Process(target=supervisor_fixture_process,args=(out,2));process.start()
    deadline=time.time()+20
    while time.time()<deadline:
        with runner.queue(out) as db:running=db.execute("SELECT COUNT(*) FROM jobs WHERE state='RUNNING'").fetchone()[0]
        if running:break
        time.sleep(.03)
    assert running
    os.kill(process.pid,signal.SIGTERM);process.join(30);assert process.exitcode==0
    state=json.loads((out/'supervision/status.json').read_text());assert state['status']=='PAUSED'
    with runner.queue(out) as db:
        assert db.execute("SELECT COUNT(*) FROM jobs WHERE state='RUNNING'").fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM jobs WHERE state='PENDING'").fetchone()[0]>0
    process=ctx.Process(target=supervisor_fixture_process,args=(out,1));process.start();process.join(30);assert process.exitcode==0
    with runner.queue(out) as db:assert db.execute("SELECT COUNT(*) FROM jobs WHERE state='COMPLETED'").fetchone()[0]==30
    data=json.loads((out/'money/spending-ledger.json').read_text())
    assert len(data['attempts'])==48 and len({a['request_key'] for a in data['attempts']})==48
    assert json.loads((out/'supervision/status.json').read_text())['status']=='STUDY_COLLECTION_TERMINAL'


def test_pinned_control_entry_preserves_spawn_import_identity(tmp_path):
    """Exercise the actual generated entry, not only the module's direct CLI."""
    out=tmp_path/'campaign';setup(out,2)
    control=tmp_path/'v3ctl.py';control.write_text(supervision.CONTROL_ENTRY)
    driver=tmp_path/'driver.py';root=Path(__file__).resolve().parents[1]
    driver.write_text('import sys,runpy\n'
        +f'sys.path[:0] = {[str(root),str(root/"tests")]!r}\n'
        +'from openalex_agent import multihorizon_supervisor as supervisor\n'
        +'from test_multihorizon_supervisor import supervisor_fixture_process\n'
        +'if __name__ == "__main__":\n'
        +f'    supervisor.main = lambda: supervisor_fixture_process({str(out)!r}, 2) or 0\n'
        +f'    runpy.run_path({str(control)!r}, run_name="__main__")\n')
    result=subprocess.run([sys.executable,str(driver)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    with runner.queue(out) as db:assert db.execute("SELECT COUNT(*) FROM jobs WHERE state='COMPLETED'").fetchone()[0]==2
