"""Quality-first staged collection. Free allowance only; never calls an LLM."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import time

from openalex_agent.config import Settings
from openalex_agent.corpus import atomic_json
from openalex_agent.dataset_integration import load_package, PLAN_FILE
from openalex_agent.models import canonical_id, utcnow
from openalex_agent.quality_collection import (CollectionStop, CollectionStore, FIELD_GROUPS,
    FreeCollectionClient, base_filter, freeze_collection, quality_report)
from openalex_agent.storage import Database


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--phase',choices=['sample','collect'],required=True)
    p.add_argument('--max-requests',type=int,default=1500)
    p.add_argument('--max-free-spend',type=float,default=.80)
    p.add_argument('--sample-review',type=Path)
    p.add_argument('--wait-for-pace',action='store_true')
    args=p.parse_args()
    if not 1<=args.max_requests<=2500 or not 0<args.max_free_spend<=.90:raise ValueError('Invalid free-only ceiling')
    data=load_package(args.package);plan=data['tables'][PLAN_FILE]
    if args.phase=='collect':
        if not args.sample_review:raise ValueError('Collection needs a recorded sample strategy review')
        review=json.loads(args.sample_review.read_text())
        if review.get('package_sha256')!=data['sha256'] or review.get('strategy')!='abstract_available_source_balanced' or not review.get('sample_quality_checked'):
            raise ValueError('Sample strategy review does not match this package')
    args.output.mkdir(parents=True,exist_ok=True)
    sources=args.output/'source';sources.mkdir(exist_ok=True)
    for path in (Path(__file__),Path(__file__).resolve().parents[1]/'openalex_agent/quality_collection.py'):
        destination=sources/path.name
        if not destination.exists():shutil.copy2(path,destination)
    config=dict(package_sha256=data['sha256'],phase=args.phase,field_groups=FIELD_GROUPS,dates=['2016-01-01','2026-10-31'],
      require_abstract=args.phase=='collect',source_quotas={'keyword_seed':350,'citation_neighborhood':300,'field_background':250,'unjudged_challenge':100},
      citation_sampling='up to 150 outgoing plus incoming to 300; bounded shortfalls retained',
      target=30000 if args.phase=='collect' else 600,paid_allowed=False,fulltext_downloads_allowed=False,heldout_queries_used=False,
      limits={'max_data_requests':args.max_requests,'max_free_spend_usd':args.max_free_spend,'reserve_usd':.02,'prepaid_limit_usd':0},
      sample_review_sha256=hashlib.sha256(args.sample_review.read_bytes()).hexdigest() if args.sample_review else None)
    config_path=args.output/'collection-config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise ValueError('Resume configuration mismatch')
    if not config_path.exists():atomic_json(config_path,config)
    run_id='quality-'+args.phase
    settings=replace(Settings.from_env(),database=str(args.output/'collection.sqlite'),max_requests=args.max_requests,min_interval=1.0,timeout=45,max_retries=3)
    with Database(settings.database) as db:
      store=CollectionStore(db,args.output)
      previous=db.job_get('usage') or {}
      with FreeCollectionClient(db,settings,run_id=run_id,max_spend=args.max_free_spend) as client:
        client.estimated_spend=previous.get('estimated_spend',0)
        for key,value in previous.items():
            if hasattr(client.usage,key):setattr(client.usage,key,value)
        client.initial_quota=previous.get('initial_quota')
        client.errors.update(previous.get('error_counts',{}))
        client.quota_checks=previous.get('quota_checks',0)
        status='running';stopped=None;failures=[]
        def checkpoint():
            db.job_put('usage',{**asdict(client.usage),'estimated_spend':client.estimated_spend,'quota_checks':client.quota_checks,'initial_quota':client.initial_quota,'error_counts':dict(client.errors)})
            summary=dict(status=status,stop_category=stopped,updated_at=utcnow(),phase=args.phase,
               unique_works=db.connection.execute('SELECT count(*) FROM raw_works').fetchone()[0],
               usage=asdict(client.usage),estimated_spend_ceiling=client.estimated_spend,quota_checks=client.quota_checks,
               initial_quota=client.initial_quota,last_quota=client.last_quota,errors=dict(client.errors),failed_or_short_targets=failures)
            atomic_json(args.output/'collection-status.json',summary)
        def request(params):
            pauses=0
            while True:
                try:return client._request('/works',params)
                except CollectionStop as e:
                    checkpoint()
                    # Only short-term rate/pace waits; never wait through daily depletion.
                    if args.wait_for_pace and e.category in ('new_account_pace','rate_limit_unclassified','per_second_rate_limit') and 0<e.retry_after<=600 and pauses<8:
                        pauses+=1
                        db.event(run_id,'collection_pause',{'category':e.category,'seconds':e.retry_after})
                        remaining=e.retry_after+1
                        while remaining>0:
                            step=min(10,remaining);time.sleep(step);remaining-=step
                        continue
                    raise
        def fetch(row,role,target,params,max_pages=30):
            signature=hashlib.sha256(json.dumps([row['Stratum ID'],role,target,params],sort_keys=True).encode()).hexdigest()
            job=db.job_get(signature) or dict(cursor='*',pages=0,done=False)
            if job['done']:return
            while job['cursor'] and job['pages']<max_pages:
                current=store.role_ids(row['Stratum ID'],role)
                if len(current)>=target:break
                remaining=target-len(current)
                sampled='sample' in params
                call=dict(params,per_page=100,page=job['pages']+1) if sampled else dict(params,per_page=min(100,remaining),cursor=job['cursor'])
                payload=request(call)
                exclude=store.stratum_ids(row['Stratum ID'])-current
                if role=='dedup_background_refill':
                    exclude.update({r[0] for r in db.connection.execute('SELECT id FROM raw_works')}-current)
                store.ingest(payload,row,role,client.acquired_at,exclude_ids=exclude,require_abstract=args.phase=='collect',max_memberships=remaining)
                cursor=(str(job['pages']+2) if len(payload.get('results',[]))==100 and (job['pages']+1)*100<params['sample'] else None) if sampled else (payload.get('meta',{}).get('next_cursor') if payload.get('results') else None)
                if cursor==job['cursor']:raise CollectionStop('cursor_cycle')
                job.update(cursor=cursor,pages=job['pages']+1)
                db.job_put(signature,job);checkpoint()
            job['done']=True;db.job_put(signature,job)
            count=len(store.role_ids(row['Stratum ID'],role))
            if count<target:failures.append(dict(stratum=row['Stratum ID'],role=role,target=target,actual=count,reason='bounded_source_shortfall'))
        try:
            client.quota();checkpoint()
            if args.phase=='sample':
                for row in plan:
                    fetch(row,'representative_sample',20,{'search':row['OpenAlex search expression'],'filter':base_filter(row['Discipline'])},max_pages=3)
            else:
                for row in plan:
                    base=base_filter(row['Discipline'],True);sid=row['Stratum ID']
                    query=review.get('search_overrides',{}).get(sid,row['OpenAlex search expression'])
                    fetch(row,'keyword_seed',350,{'search':query,'filter':base},max_pages=12)
                    seed_ids=sorted(store.role_ids(sid,'keyword_seed'))
                    anchors=[];references=set()
                    for wid in seed_ids:
                        raw=json.loads(db.connection.execute('SELECT payload FROM raw_works WHERE id=?',(wid,)).fetchone()[0]);anchors.append(wid)
                        references.update(raw.get('referenced_works',[]))
                    # Interleave outgoing and incoming sources, both backed by actual metadata.
                    batches=[sorted(references)[i:i+100] for i in range(0,min(2000,len(references)),100)]
                    for batch in batches:
                        if len(store.role_ids(sid,'citation_neighborhood'))>=150:break
                        ids='|'.join(canonical_id(w).rsplit('/',1)[-1] for w in batch)
                        fetch(row,'citation_neighborhood',min(150,len(store.role_ids(sid,'citation_neighborhood'))+100),{'filter':'openalex:'+ids+','+base},max_pages=1)
                    if anchors and len(store.role_ids(sid,'citation_neighborhood'))<300:
                        ids='|'.join(w.rsplit('/',1)[-1] for w in anchors[:40])
                        fetch(row,'citation_neighborhood',300,{'filter':'cites:'+ids+','+base},max_pages=4)
                    seed=int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)%1000000
                    fetch(row,'field_background',250,{'filter':base,'sample':250,'seed':seed},max_pages=5)
                    fetch(row,'unjudged_challenge',100,{'filter':base,'sample':150,'seed':seed+1},max_pages=3)
                # Deduplication may reduce global totals; bounded background refill is explicit.
                for disc in FIELD_GROUPS:
                    for wave in range(5):
                        n=db.connection.execute('SELECT count(DISTINCT id) FROM memberships WHERE discipline=?',(disc,)).fetchone()[0]
                        if n>=5000:break
                        remaining=5000-n;row={'Stratum ID':'REFILL-'+str(list(FIELD_GROUPS).index(disc)),'Discipline':disc}
                        target=len(store.role_ids(row['Stratum ID'],'dedup_background_refill'))+remaining
                        fetch(row,'dedup_background_refill',target,{'filter':base_filter(disc,True),'sample':min(10000,remaining+500),'seed':11000+100*list(FIELD_GROUPS).index(disc)+wave},max_pages=80)
            status='acquired'
        except CollectionStop as e:
            status='incomplete';stopped=e.category
        except KeyboardInterrupt:
            status='interrupted';stopped='operator_interrupt'
        finally:
            checkpoint()
            try:client.quota()
            except CollectionStop:pass
            checkpoint()
        raws=store.rows();members=list(db.connection.execute('SELECT id,stratum,discipline,role FROM memberships'))
        if args.phase=='collect':
            targets=[]
            for row in plan:
                sid=row['Stratum ID']
                targets.append(dict(stratum=sid,discipline=row['Discipline'],unique=len(store.stratum_ids(sid)),
                    roles={role:{'target':target,'actual':len(store.role_ids(sid,role))} for role,target in config['source_quotas'].items()}))
            atomic_json(args.output/'stratum-targets.json',targets)
        if raws:
            report=quality_report(raws,members);atomic_json(args.output/'quality-progress.json',report)
            final=args.output/'frozen';version=1
            while final.exists():
                version+=1;final=args.output/('frozen-v'+str(version))
            count=report['unique_papers'];complete=args.phase=='collect' and count>=30000 and all(report['papers_per_discipline'].get(d,0)>=5000 for d in FIELD_GROUPS)
            report=freeze_collection(store,final,{**config,'requests':client.usage.requests,'quota_checks':client.quota_checks,
                    'reported_cost_usd':client.usage.reported_cost_usd,'acquisition_complete':complete,'collection_status':status,
                'note':'Hard-negative role is an unjudged candidate source, not a human relevance assertion; no full text downloaded'})
            print(json.dumps(dict(status=status,stop_category=stopped,unique_papers=report['unique_papers'],papers_per_discipline=report['papers_per_discipline'],abstracts=report['abstracts'],http_attempts=client.usage.requests,quota_checks=client.quota_checks,reported_cost_usd=client.usage.reported_cost_usd,frozen=str(final))))
        else:print(json.dumps(dict(status=status,stop_category=stopped,unique_papers=0)))


if __name__=='__main__':main()
