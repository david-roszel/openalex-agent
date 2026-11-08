"""Mocked free-budget and raw-evidence safeguards; no real API requests."""

import json
from dataclasses import replace

import httpx
import pytest

from openalex_agent.config import Settings
from openalex_agent.quality_collection import (FreeCollectionClient,CollectionStop,CollectionStore,classify_error,quality_report,freeze_collection)
from openalex_agent.storage import Database
from openalex_agent.corpus import LocalCorpus


def quota(remaining=1,prepaid=0):
 return {'api_key':'TEST PRIVATE KEY MUST NOT LOG','rate_limit':{'daily_budget_usd':1,'daily_used_usd':1-remaining,'daily_remaining_usd':remaining,'prepaid_remaining_usd':prepaid,
   'endpoint_costs_usd':{'singleton':0,'list':.0001,'search':.001},'resets_in_seconds':10}}


def make_client(db,handler):
 return FreeCollectionClient(db,Settings(api_key='TEST PRIVATE KEY MUST NOT LOG',min_interval=0,max_retries=1),transport=httpx.MockTransport(handler),sleep=lambda _:None)


@pytest.mark.parametrize('remaining,prepaid,category',[(0,0,'free_allowance_stop'),(1,1,'prepaid_authorization_required')])
def test_no_data_request_after_free_or_prepaid_guard(tmp_path,remaining,prepaid,category):
 calls=[]
 def handler(r):
  calls.append(r.url.path)
  assert r.url.path=='/rate-limit'
  return httpx.Response(200,json=quota(remaining,prepaid))
 with Database(':memory:') as db,make_client(db,handler) as c:
  with pytest.raises(CollectionStop) as e:c._request('/works',{'search':'test'})
  assert e.value.category==category and c.usage.requests==0 and calls==['/rate-limit']
  assert 'TEST PRIVATE KEY MUST NOT LOG' not in json.dumps(db.events(c.run_id))


def test_cached_metadata_needs_no_network_or_current_allowance():
 with Database(':memory:') as db:
  payload={'meta':{'cost_usd':.001},'results':[]};params={'search':'fixture'}
  import hashlib
  key=hashlib.sha256(json.dumps(['/works',params],sort_keys=True).encode()).hexdigest();db.cache_put(key,payload)
  with make_client(db,lambda _:pytest.fail('Cache cannot use network')) as c:
   assert c._request('/works',params)==payload and c.usage.requests==0


def test_pace_error_is_not_daily_exhaustion_and_does_not_leak_body():
 def handler(r):
  if r.url.path=='/rate-limit':return httpx.Response(200,json=quota())
  return httpx.Response(429,headers={'Retry-After':'280'},json={'message':'New account spending pace: wait 5 minutes. TEST PRIVATE KEY MUST NOT LOG'})
 with Database(':memory:') as db,make_client(db,handler) as c:
  with pytest.raises(CollectionStop) as e:c._request('/works',{'search':'test'})
  assert e.value.category=='new_account_pace' and e.value.retry_after==280 and c.usage.requests==1
  assert 'TEST PRIVATE KEY MUST NOT LOG' not in json.dumps(db.events(c.run_id))


@pytest.mark.parametrize('status,payload,remaining,delay,expected',[
 (429,{},0,30,'daily_budget_exhausted'),(429,{'message':'per second rate limit'},1,1,'per_second_rate_limit'),
 (429,{'message':'Plan upgrade required'},1,1,'paid_plan_required'),(400,{},1,0,'invalid_request'),
 (404,{},1,0,'missing_source'),(401,{},1,0,'authentication')])
def test_failure_categories(status,payload,remaining,delay,expected):
 assert classify_error(status,payload,remaining,delay)==expected


def test_request_receipts_and_free_preflight_happen_for_each_retry():
 calls=[]
 def handler(r):
  calls.append(r.url.path)
  if r.url.path=='/rate-limit':return httpx.Response(200,json=quota())
  if calls.count('/works')==1:return httpx.Response(503,json={})
  return httpx.Response(200,json={'meta':{'cost_usd':.001},'results':[]})
 with Database(':memory:') as db,make_client(db,handler) as c:
  c._request('/works',{'search':'fixture'})
  assert calls==['/rate-limit','/works','/rate-limit','/works']
  assert c.usage.requests==2 and c.usage.retries==1 and c.usage.reported_cost_usd==.001


def test_raw_preservation_dedup_membership_and_frozen_offline_compatibility(tmp_path):
 raw={'id':'https://openalex.org/W1','title':'Test raw record','publication_date':'2020-01-01','publication_year':2020,
   'abstract_inverted_index':{'Numerical':[0],'evidence':[1]},'referenced_works':['https://openalex.org/W2'],
   'primary_topic':{'field':{'id':'https://openalex.org/fields/17'}},'open_access':{'is_oa':True},'has_content':{'grobid_xml':True},'custom_unconsumed_field':'must survive'}
 with Database(':memory:') as db:
  store=CollectionStore(db,tmp_path/'working')
  for sid in ('a','b'):store.ingest({'results':[raw]}, {'Stratum ID':sid,'Discipline':'Computer Science'},'keyword_seed','2026-11-06T00:00:00+00:00')
  assert len(store.rows())==1 and store.rows()[0]['custom_unconsumed_field']=='must survive'
  report=freeze_collection(store,tmp_path/'frozen',{'acquisition_complete':False})
  assert report['unique_papers']==1 and report['membership_rows']==2 and report['unresolved_reference_targets']==1
  assert report['fulltext_downloaded']==0 and not report['ground_truth_verified']
  corpus=LocalCorpus(tmp_path/'frozen');assert len(corpus.papers)==1 and corpus.usage.requests==0
  with pytest.raises(ValueError):freeze_collection(store,tmp_path/'frozen',{})
