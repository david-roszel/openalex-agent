"""Single cumulative $68 Terra ledger; receipts are primary, unknown usage stops.

No Cloud Monitoring dependency, retries, credit purchases or stage-local allowances.
"""
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
import fcntl,hashlib,json,os,re,sqlite3,time
from .corpus import atomic_json
from .execution_control import ExperimentStopped
from .process_identity import identity,alive

CEILING=Decimal('68.00')
INPUT=Decimal('1.00');OUTPUT=Decimal('5.00')
# Reserve/account all prompt tokens at the highest published short-context cache
# write rate, even though the profile does not request long cache retention.
INPUT_UPPER=Decimal('2.50')


class TerraLedger:
    def __init__(self,directory,*,profile_hash,clock=time.time,session_id=None,worker_id=None):
        self.out=Path(directory);self.out.mkdir(parents=True,exist_ok=True)
        self.profile_hash=profile_hash;self.clock=clock;self.session_id=session_id;self.worker_id=worker_id
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY,value TEXT)')
            db.execute('''CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,request_key TEXT UNIQUE,
                phase TEXT,method TEXT,status TEXT,retained TEXT,charge TEXT,reservation TEXT,
                started REAL,ended REAL,diagnostic TEXT,wire_sha256 TEXT,input_upper INTEGER,output_upper INTEGER)''')
            stored=db.execute("SELECT value FROM control WHERE key='profile_sha256'").fetchone()
            if stored and stored['value']!=profile_hash:raise ValueError('Ledger profile identity changed')
            db.execute("INSERT OR IGNORE INTO control VALUES('profile_sha256',?)",(profile_hash,))
            db.execute('CREATE TABLE IF NOT EXISTS execution_sessions(id TEXT PRIMARY KEY,owner TEXT,workers INTEGER,started REAL,ended REAL)')
            db.execute('CREATE TABLE IF NOT EXISTS execution_workers(session TEXT,worker TEXT,owner TEXT,PRIMARY KEY(session,worker))')
            db.execute('CREATE TABLE IF NOT EXISTS reservation_owners(attempt INTEGER PRIMARY KEY,session TEXT,worker TEXT,owner TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS spending_scopes(id TEXT PRIMARY KEY,ceiling TEXT,job_ids TEXT,attempt_limit INTEGER,policy_sha256 TEXT)')

    def register_scope(self,scope_id,*,ceiling_usd,job_ids,attempt_limit,policy_sha256):
        cap=Decimal(str(ceiling_usd));ids=sorted(set(job_ids))
        if not 0<cap<=CEILING or not ids or any(not isinstance(v,str) or ':' in v for v in ids) or type(attempt_limit) is not int or attempt_limit<1:raise ValueError('Invalid bounded spending scope')
        record=(scope_id,str(cap),json.dumps(ids),attempt_limit,policy_sha256)
        with self.transaction() as db:
            existing=db.execute('SELECT * FROM spending_scopes WHERE id=?',(scope_id,)).fetchone()
            if existing and tuple(existing)!=record:raise ExperimentStopped('immutable_spending_scope_changed')
            db.execute('INSERT OR IGNORE INTO spending_scopes VALUES(?,?,?,?,?)',record)

    def begin_session(self,workers):
        if not self.session_id or type(workers) is not int or workers not in (1,2,4):raise ValueError('Bounded supervisor session required')
        with self.transaction() as db:
            for row in db.execute('SELECT * FROM execution_sessions WHERE ended IS NULL'):
                if alive(json.loads(row['owner'])):raise ExperimentStopped('another_live_ledger_supervisor')
                db.execute('UPDATE execution_sessions SET ended=? WHERE id=?',(self.clock(),row['id']))
            if db.execute("SELECT 1 FROM attempts WHERE status IN ('RESERVED','UNKNOWN')").fetchone():raise ExperimentStopped('campaign_unaccountable_usage')
            db.execute('INSERT INTO execution_sessions VALUES(?,?,?,?,NULL)',(self.session_id,json.dumps(identity()),workers,self.clock()))

    def register_worker(self):
        with self.transaction() as db:
            session=db.execute('SELECT * FROM execution_sessions WHERE id=? AND ended IS NULL',(self.session_id,)).fetchone()
            if not session or not alive(json.loads(session['owner'])) or not self.worker_id:raise ExperimentStopped('supervisor_not_alive')
            workers=list(db.execute('SELECT * FROM execution_workers WHERE session=?',(self.session_id,)))
            if sum(alive(json.loads(w['owner'])) for w in workers)>=session['workers']:raise ExperimentStopped('worker_limit')
            db.execute('INSERT INTO execution_workers VALUES(?,?,?)',(self.session_id,self.worker_id,json.dumps(identity())))

    def end_session(self):
        with self.transaction() as db:
            row=db.execute('SELECT * FROM execution_sessions WHERE id=? AND ended IS NULL',(self.session_id,)).fetchone()
            if not row or json.loads(row['owner'])!=identity():raise ExperimentStopped('not_supervisor_owner')
            if any(alive(json.loads(w['owner'])) for w in db.execute('SELECT * FROM execution_workers WHERE session=?',(self.session_id,))):raise ExperimentStopped('workers_still_alive')
            db.execute('UPDATE execution_sessions SET ended=? WHERE id=?',(self.clock(),self.session_id))

    @contextmanager
    def transaction(self):
        with (self.out/'spending.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            db=sqlite3.connect(self.out/'spending.sqlite',timeout=30);db.row_factory=sqlite3.Row
            db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('BEGIN IMMEDIATE')
            try:
                yield db
                db.commit()
                self.publish(db)
            except BaseException:db.rollback();raise
            finally:db.close();fcntl.flock(lock,fcntl.LOCK_UN)

    def activate(self,authorization_sha256,*,max_runtime_seconds):
        if type(max_runtime_seconds) is not int or not 150<max_runtime_seconds<=86400:raise ValueError('Runtime must be150s–24h')
        with self.transaction() as db:
            old=db.execute("SELECT value FROM control WHERE key='authorization_sha256'").fetchone()
            # Later stage authorizations append identities, never reset money/time.
            db.execute('CREATE TABLE IF NOT EXISTS authorizations(sha256 TEXT PRIMARY KEY,recorded REAL)')
            db.execute('INSERT OR IGNORE INTO authorizations VALUES(?,?)',(authorization_sha256,self.clock()))
            runtime=db.execute("SELECT value FROM control WHERE key='max_runtime_seconds'").fetchone()
            if runtime and int(runtime['value'])!=max_runtime_seconds:raise ValueError('Stage authorization cannot reset or extend the original deadline')
            for key,value in [('authorization_sha256',authorization_sha256),('start_epoch',str(self.clock())),('max_runtime_seconds',str(max_runtime_seconds))]:
                db.execute('INSERT OR IGNORE INTO control VALUES(?,?)',(key,value))

    def reason(self,db,*,new_reservation=None):
        control={r['key']:r['value'] for r in db.execute('SELECT * FROM control')}
        if 'stop_reason' in control:return control['stop_reason']
        if 'authorization_sha256' not in control:return 'execution_not_authorized'
        if self.clock()+120>=float(control['start_epoch'])+int(control['max_runtime_seconds']):return 'campaign_runtime_limit'
        rows=list(db.execute('SELECT * FROM attempts'))
        if any(r['status']=='UNKNOWN' for r in rows):return 'campaign_unaccountable_usage'
        sessions=list(db.execute('SELECT * FROM execution_sessions WHERE ended IS NULL'))
        session=next((s for s in sessions if s['id']==self.session_id and alive(json.loads(s['owner']))),None)
        if (sessions or self.worker_id or (new_reservation is not None and self.session_id)) and session is None:return 'supervised_ledger_session_required'
        reserved=[r for r in rows if r['status']=='RESERVED']
        if reserved:
            if session is None:return 'campaign_inflight_or_interrupted_request'
            for row in reserved:
                owner=db.execute('SELECT * FROM reservation_owners WHERE attempt=?',(row['id'],)).fetchone()
                if not owner or owner['session']!=self.session_id or not alive(json.loads(owner['owner'])):return 'campaign_inflight_or_interrupted_request'
        if new_reservation is not None and session:
            worker=db.execute('SELECT * FROM execution_workers WHERE session=? AND worker=?',(self.session_id,self.worker_id)).fetchone()
            if not worker or json.loads(worker['owner'])!=identity():return 'registered_request_worker_required'
            if len(reserved)>=session['workers']:return 'concurrent_request_limit'
            if db.execute("SELECT 1 FROM reservation_owners o JOIN attempts a ON a.id=o.attempt WHERE o.session=? AND o.worker=? AND a.status='RESERVED'",(self.session_id,self.worker_id)).fetchone():return 'worker_request_already_inflight'
        if new_reservation is not None and sum((Decimal(r['retained']) for r in rows),Decimal(0))+new_reservation>CEILING:return 'campaign_68_dollar_limit'
        return None

    def check(self):
        with self.transaction() as db:
            reason=self.reason(db)
            if reason:raise ExperimentStopped(reason)

    def reserve(self,key,phase,method,raw,output_upper=4096):
        if type(output_upper) is not int or not 1<=output_upper<=4096:raise ValueError('Invalid output cap')
        # For two text-only messages and no tools: UTF8 serialized bytes plus
        #4096 framing tokens bound input more conservatively than BPE counts.
        upper=len(raw)+4096
        if upper>100000:raise ExperimentStopped('request_exceeds_pinned_short_context_byte_bound')
        amount=(Decimal(upper)*INPUT_UPPER+Decimal(output_upper)*OUTPUT)/1000000
        digest=hashlib.sha256(raw).hexdigest()
        with self.transaction() as db:
            if db.execute('SELECT id FROM attempts WHERE request_key=?',(key,)).fetchone():raise ExperimentStopped('duplicate_request_identity')
            reason=self.reason(db,new_reservation=amount)
            if reason:raise ExperimentStopped(reason)
            # Every matching immutable scope applies, even if a client omits a
            # stage label. Prior settled/active/UNKNOWN attempts all count.
            rows=list(db.execute('SELECT * FROM attempts'))
            root=phase.split(':',1)[0]
            for scope in db.execute('SELECT * FROM spending_scopes'):
                ids=set(json.loads(scope['job_ids']))
                if root not in ids:continue
                scoped=[r for r in rows if r['phase'] and r['phase'].split(':',1)[0] in ids]
                if len(scoped)>=scope['attempt_limit']:raise ExperimentStopped('pilot_provider_attempt_limit')
                if sum((Decimal(r['retained']) for r in scoped),Decimal(0))+amount>Decimal(scope['ceiling']):raise ExperimentStopped('pilot_spending_subceiling')
            cur=db.execute('INSERT INTO attempts(request_key,phase,method,status,retained,reservation,started,wire_sha256,input_upper,output_upper) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (key,phase,method,'RESERVED',str(amount),str(amount),self.clock(),digest,upper,output_upper))
            if self.session_id:db.execute('INSERT INTO reservation_owners VALUES(?,?,?,?)',(cur.lastrowid,self.session_id,self.worker_id,json.dumps(identity())))
            return cur.lastrowid

    def settle(self,index,diagnostic):
        d=dict(diagnostic);incoming=d.get('prompt_tokens');outgoing=d.get('completion_tokens')
        with self.transaction() as db:
            row=db.execute('SELECT * FROM attempts WHERE id=?',(index,)).fetchone()
            if row is None or row['status']!='RESERVED':raise ValueError('Settle exactly one active reservation')
            owner=db.execute('SELECT * FROM reservation_owners WHERE attempt=?',(index,)).fetchone()
            if owner:
                if owner['session']!=self.session_id or owner['worker']!=self.worker_id or json.loads(owner['owner'])!=identity():raise ExperimentStopped('reservation_owner_mismatch')
                d['execution_session']=self.session_id;d['execution_worker']=self.worker_id
            valid=(type(incoming) is int and 0<=incoming<=row['input_upper'] and type(outgoing) is int and 0<=outgoing<=row['output_upper'])
            valid=valid and d.get('returned_service_tier') in (None,'default')
            if 'provider_usage' in d:
                # Validate the original envelope too: a parser must not silently
                # discard malformed negative/boolean optional billing counts.
                usage=d['provider_usage']
                valid=valid and isinstance(usage,dict)
                if isinstance(usage,dict):
                    valid=valid and usage.get('prompt_tokens')==incoming and usage.get('completion_tokens')==outgoing
                    for detail,field,cap in [('prompt_tokens_details','cached_tokens',incoming),('completion_tokens_details','reasoning_tokens',outgoing)]:
                        if detail in usage:
                            data=usage[detail]
                            valid=valid and isinstance(data,dict)
                            if isinstance(data,dict) and field in data:
                                v=data[field];valid=valid and type(v) is int and type(cap) is int and 0<=v<=cap
                    if 'total_tokens' in usage:
                        total=usage['total_tokens'];valid=valid and type(total) is int and type(incoming) is int and type(outgoing) is int and total==incoming+outgoing
                model=d.get('returned_model')
                valid=valid and isinstance(model,str) and bool(re.fullmatch(r'gpt-6\.1-terra(?:-\d{4}-\d{2}-\d{2})?',model))

            for key,cap in [('cached_tokens',incoming),('cache_write_tokens',incoming),('reasoning_tokens',outgoing)]:
                if key in d:valid=valid and type(d[key]) is int and type(cap) is int and 0<=d[key]<=cap
            if 'total_tokens' in d:valid=valid and type(d['total_tokens']) is int and valid and d['total_tokens']==incoming+outgoing
            charge=(Decimal(incoming)*INPUT_UPPER+Decimal(outgoing)*OUTPUT)/1000000 if valid else None
            valid=valid and charge<=Decimal(row['reservation'])
            status='SETTLED' if valid else 'UNKNOWN'
            retained=str(charge) if valid else row['reservation']
            if valid:d['standard_uncached_token_estimate_usd']=str((Decimal(incoming)*INPUT+Decimal(outgoing)*OUTPUT)/1000000)
            d['accounted_cost_basis']='Conservative maximum published short-context cache-write input rate; output includes reasoning; not invoice'
            db.execute('UPDATE attempts SET status=?,retained=?,charge=?,ended=?,diagnostic=? WHERE id=?',
                (status,retained,str(charge) if valid else None,self.clock(),json.dumps(d),index))
            if not valid:db.execute("INSERT OR REPLACE INTO control VALUES('stop_reason','campaign_unaccountable_usage')")
            elif d.get('http_status') in (401,403,404,429):db.execute("INSERT OR REPLACE INTO control VALUES('stop_reason','provider_access_or_rate_limit_stop')")
        if not valid:raise ExperimentStopped('campaign_unaccountable_usage')

    def recover_interrupted(self):
        # Never infer that a committed reservation did not reach the provider.
        with self.transaction() as db:
            sessions=list(db.execute('SELECT * FROM execution_sessions WHERE ended IS NULL'))
            for s in sessions:
                if alive(json.loads(s['owner'])) and (s['id']!=self.session_id or json.loads(s['owner'])!=identity()):raise ExperimentStopped('live_supervisor_recovery_forbidden')
            changed=0
            for row in list(db.execute("SELECT * FROM attempts WHERE status='RESERVED'")):
                owner=db.execute('SELECT * FROM reservation_owners WHERE attempt=?',(row['id'],)).fetchone()
                if owner and alive(json.loads(owner['owner'])):continue
                changed+=db.execute("UPDATE attempts SET status='UNKNOWN',ended=?,diagnostic=? WHERE id=?",(self.clock(),json.dumps({'error_code':'interrupted_request_usage_unknown'}),row['id'])).rowcount
            if not any(alive(json.loads(s['owner'])) for s in sessions):db.execute('UPDATE execution_sessions SET ended=? WHERE ended IS NULL',(self.clock(),))
            if changed:db.execute("INSERT OR REPLACE INTO control VALUES('stop_reason','campaign_unaccountable_usage')")
            return changed

    def publish(self,db):
        rows=[dict(r) for r in db.execute('SELECT * FROM attempts ORDER BY id')]
        for r in rows:r['diagnostic']=json.loads(r['diagnostic']) if r['diagnostic'] else None
        control={r['key']:r['value'] for r in db.execute('SELECT * FROM control')}
        exposure=sum((Decimal(r['retained']) for r in rows),Decimal(0));confirmed=sum((Decimal(r['charge']) for r in rows if r['charge'] is not None),Decimal(0))
        scopes=[]
        for s in db.execute('SELECT * FROM spending_scopes'):
            ids=set(json.loads(s['job_ids']));selected=[r for r in rows if r['phase'] and r['phase'].split(':',1)[0] in ids]
            scopes.append({'id':s['id'],'ceiling_usd':s['ceiling'],'retained_exposure_usd':str(sum((Decimal(r['retained']) for r in selected),Decimal(0))),'attempts':len(selected),'attempt_limit':s['attempt_limit'],'policy_sha256':s['policy_sha256']})
        atomic_json(self.out/'spending-ledger.json',{'version':'terra-cumulative-68-v2','provider':'openai','model':'gpt-6.1-terra','ceiling_usd':str(CEILING),
            'cumulative_retained_exposure_usd':str(exposure),'retained_usd':str(exposure),'confirmed_upper_bound_gross_usd':str(confirmed),'remaining_authorization_usd':str(CEILING-exposure),
            'prior_retained_exposure_usd':'0','historical_google_costs_excluded':True,'attempts':rows,'stop_reason':control.get('stop_reason'),
            'profile_sha256':self.profile_hash,'start_epoch':control.get('start_epoch'),'max_runtime_seconds':control.get('max_runtime_seconds'),'spending_scopes':scopes,
            'execution_authorized':bool(control.get('authorization_sha256')),'credits_or_discounts_assumed':False,'automatic_retries':0,
            'pricing':{'input_usd_per_million':str(INPUT),'output_usd_per_million':str(OUTPUT),'input_conservative_accounting_usd_per_million':str(INPUT_UPPER)},
            'usage_source':'Actual returned provider usage; no dependency on external aggregate Monitoring counters'})
