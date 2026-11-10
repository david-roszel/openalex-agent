"""Free-allowance-only acquisition, separate from agent/ranking algorithms."""

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

import httpx
import networkx as nx

from .corpus import atomic_json, freeze
from .git_provenance import git_provenance
from .models import Paper, canonical_id, utcnow
from .openalex import OpenAlexClient, RetrievalError


FIELD_GROUPS = {
    "Computer Science": [17],
    "Physics": [31],
    "Biomedicine": [13,24,27,28,30],
    "Environmental Science": [19,21,23],
    "Chemistry & Materials": [15,16,25],
    "Economics & Social Science": [14,18,20,32,33],
}


class CollectionStop(RetrievalError):
    def __init__(self, category, retry_after=0):
        self.category, self.retry_after = category, retry_after
        super().__init__(category)


def classify_error(status, payload, remaining=None, retry_after=0):
    # Interpret privately, but never persist provider error bodies or URLs.
    text=json.dumps(payload).lower()
    if status==404:return "missing_source"
    if status in (401,403):return "authentication"
    if status==429:
        if "plan upgrade" in text:return "paid_plan_required"
        if any(s in text for s in ("new account","spending pace","5 minutes","five minutes","spend pace")):
            return "new_account_pace"
        if remaining is not None and remaining<=0:return "daily_budget_exhausted"
        if any(s in text for s in ("per second","per-second","requests per second","rate limit exceeded")) and retry_after<=10:
            return "per_second_rate_limit"
        if any(s in text for s in ("daily budget","daily limit","daily quota")):return "daily_budget_exhausted"
        return "rate_limit_unclassified"
    if status in (408,500,502,503,504):return "transient_server_error"
    return "invalid_request"


class FreeCollectionClient(OpenAlexClient):
    """Check daily USD separately from credits/prepaid before every uncached call.

    There is deliberately no paid override. Positive prepaid balance stops this
    command: the server otherwise falls back to it automatically. Quota calls
    are counted separately; unknown quota/cost fields fail closed.
    """
    def __init__(self,*args,max_spend=.80,reserve=.02,clock=time.monotonic,**kwargs):
        super().__init__(*args,**kwargs)
        self.max_spend=max_spend;self.reserve=reserve;self.clock=clock
        self.estimated_spend=0.;self.quota_checks=0;self.initial_quota=None;self.last_quota=None
        self.errors=Counter();self.last_category=None;self.last_retry_after=0
        self.quota_http=httpx.Client(base_url="https://api.openalex.org",headers=dict(self.http.headers),
            timeout=self.settings.timeout,transport=kwargs.get("transport"))

    def close(self):
        self.quota_http.close();super().close()

    def quota(self):
        self.quota_checks+=1
        try:
            response=self.quota_http.get('/rate-limit')
            data=response.json()
        except (httpx.HTTPError,ValueError):raise CollectionStop('quota_unavailable') from None
        if response.status_code!=200 or not isinstance(data.get('rate_limit'),dict):
            raise CollectionStop('quota_unavailable')
        rate=data['rate_limit']
        needed=('daily_budget_usd','daily_used_usd','daily_remaining_usd','prepaid_remaining_usd')
        if any(self._number(rate.get(k)) is None for k in needed):raise CollectionStop('quota_schema_unknown')
        safe={k:rate[k] for k in needed}
        safe.update({k:rate.get(k) for k in ('resets_at','resets_in_seconds','endpoint_costs_usd')})
        if not self.initial_quota:self.initial_quota=safe
        self.last_quota=safe
        self.db.event(self.run_id,'collection_quota',{'quota':safe})
        return safe

    def _preflight(self,endpoint,params):
        quota=self.quota()
        if quota['prepaid_remaining_usd']>0:raise CollectionStop('prepaid_authorization_required')
        kind='singleton' if endpoint.startswith('/works/') else 'search' if params.get('search') else 'list'
        price=(quota.get('endpoint_costs_usd') or {}).get(kind)
        if self._number(price) is None:raise CollectionStop('price_unknown')
        if quota['daily_remaining_usd']<price+self.reserve:raise CollectionStop('free_allowance_stop')
        if self.estimated_spend+price>self.max_spend:raise CollectionStop('local_spending_cap')
        if self.usage.requests>=self.settings.max_requests:raise CollectionStop('request_cap')
        return price

    def _request(self,endpoint,params=None,*,refresh=False):
        params=dict(params or {})
        if endpoint not in ('/works','/fields') and not re.fullmatch(r'/works/W\d+',endpoint):
            raise CollectionStop('endpoint_not_authorized')
        if any(k in params for k in ('api_key','rerank','semantic')):raise CollectionStop('operation_not_authorized')
        key=hashlib.sha256(json.dumps([endpoint,params],sort_keys=True).encode()).hexdigest()
        if not refresh and (cached:=self.db.cache_get(key)) is not None:
            self._validate_payload(endpoint,cached,params.get('cursor'))
            self.acquired_at=self.db.cache_acquired(key);self.usage.cache_hits+=1
            return cached
        for attempt in range(self.settings.max_retries+1):
            price=self._preflight(endpoint,params)
            self.sleep(max(0,self.settings.min_interval-(self.clock()-self.last_request)))
            self.last_request=self.clock();started=self.clock();self.usage.requests+=1
            try:
                response=self.http.get(endpoint,params=params)
                try:payload=response.json()
                except ValueError:payload={}
            except httpx.HTTPError:
                response=None;payload={}
            self.usage.api_seconds+=self.clock()-started
            self.estimated_spend+=price
            if response is None:
                category='network_failure';delay=min(2**attempt,30)
            else:
                cost=self._number(payload.get('meta',{}).get('cost_usd')) if isinstance(payload,dict) and isinstance(payload.get('meta',{}),dict) else None
                if cost is not None:
                    self.usage.reported_cost_usd+=cost;self.usage.cost_reported_requests+=1
                else:self.usage.cost_unreported_requests+=1
                credits=self._number(response.headers.get('X-RateLimit-Credits-Used'))
                if credits is not None:self.usage.reported_credits+=credits
                try:delay=max(0,float(response.headers.get('Retry-After',2**attempt)))
                except ValueError:delay=min(2**attempt,30)
                if not math.isfinite(delay):delay=30
                if response.is_success:
                    try:self._validate_payload(endpoint,payload,params.get('cursor'))
                    except RetrievalError:raise CollectionStop('invalid_response_schema') from None
                    self.acquired_at=self.db.cache_put(key,payload)
                    self.db.event(self.run_id,'collection_request',{'endpoint':endpoint,'params':params,'status':response.status_code,
                        'cost_usd':cost,'predicted_cost_usd':price,'quota_before':self.last_quota,'seconds':self.clock()-started})
                    self.last_category=None
                    return payload
                category=classify_error(response.status_code,payload,self.last_quota['daily_remaining_usd'],delay)
            self.usage.errors+=1;self.errors[category]+=1;self.last_category=category;self.last_retry_after=delay
            self.db.event(self.run_id,'collection_error',{'endpoint':endpoint,'status':response.status_code if response is not None else None,
                'category':category,'retry_after':delay,'attempt':attempt+1,'predicted_cost_usd':price})
            if category not in ('network_failure','transient_server_error','per_second_rate_limit') or delay>60:
                raise CollectionStop(category,delay)
            if attempt<self.settings.max_retries:
                self.usage.retries+=1;self.sleep(delay)
        raise CollectionStop(self.last_category or 'retry_limit',self.last_retry_after)


def base_filter(discipline,abstracts=False):
    fields='|'.join(map(str,FIELD_GROUPS[discipline]))
    value=f'primary_topic.field.id:{fields},from_publication_date:2016-01-01,to_publication_date:2026-10-31,is_retracted:false'
    return value+(',has_abstract:true,type:article|review|conference-paper|preprint' if abstracts else '')


class CollectionStore:
    def __init__(self,db,directory):
        self.db=db;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        db.connection.executescript('''
          CREATE TABLE IF NOT EXISTS raw_works(id TEXT PRIMARY KEY,payload TEXT NOT NULL,acquired TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS memberships(id TEXT NOT NULL,stratum TEXT NOT NULL,discipline TEXT NOT NULL,role TEXT NOT NULL,PRIMARY KEY(id,stratum,role));
          CREATE TABLE IF NOT EXISTS rejected(id TEXT NOT NULL,reason TEXT NOT NULL,PRIMARY KEY(id,reason));
        ''')

    def ingest(self,payload,stratum,role,acquired,exclude_ids=(),require_abstract=False,max_memberships=None):
        added=0;normalized=[];assigned=0;existing_role=self.role_ids(stratum['Stratum ID'],role)
        for raw in payload.get('results',[]):
            if max_memberships is not None and assigned>=max_memberships:break
            try:paper=Paper.from_openalex(raw)
            except (ValueError,TypeError):continue
            if paper.id in exclude_ids or paper.id in existing_role:continue
            if not paper.title or not paper.publication_date or not '2016-01-01'<=paper.publication_date<='2026-10-31' or raw.get('is_retracted') is True:
                with self.db.connection:self.db.connection.execute('INSERT OR IGNORE INTO rejected VALUES (?,?)',(paper.id,'title_date_or_retraction'))
                continue
            if require_abstract and (not paper.abstract or len(paper.abstract)<200):
                with self.db.connection:self.db.connection.execute('INSERT OR IGNORE INTO rejected VALUES (?,?)',(paper.id,'missing_or_short_reconstructed_abstract'))
                continue
            field=(raw.get('primary_topic') or {}).get('field',{}).get('id','').rsplit('/',1)[-1]
            if not field.isdigit() or int(field) not in FIELD_GROUPS[stratum['Discipline']]:
                with self.db.connection:self.db.connection.execute('INSERT OR IGNORE INTO rejected VALUES (?,?)',(paper.id,'primary_field_mismatch'))
                continue
            # Keep data classification provenance separate from model-visible Paper.
            with self.db.connection:
                exists=self.db.connection.execute('SELECT 1 FROM raw_works WHERE id=?',(paper.id,)).fetchone()
                self.db.connection.execute('''INSERT INTO raw_works VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,acquired=excluded.acquired WHERE excluded.acquired>=raw_works.acquired''',(paper.id,json.dumps(raw,ensure_ascii=False),acquired))
                added+=not bool(exists)
                self.db.connection.execute('INSERT OR IGNORE INTO memberships VALUES (?,?,?,?)',(paper.id,stratum['Stratum ID'],stratum['Discipline'],role))
                assigned+=1
                existing_role.add(paper.id)
            paper.retrieved_at=acquired;normalized.append(paper)
        self.db.save_papers(normalized)
        return added

    def rows(self):
        return [json.loads(r[0]) for r in self.db.connection.execute('SELECT payload FROM raw_works ORDER BY id')]

    def role_ids(self,sid,role):
        return {r[0] for r in self.db.connection.execute('SELECT id FROM memberships WHERE stratum=? AND role=?',(sid,role))}

    def stratum_ids(self,sid):
        return {r[0] for r in self.db.connection.execute('SELECT DISTINCT id FROM memberships WHERE stratum=?',(sid,))}


def quality_report(raws,memberships):
    papers=[Paper.from_openalex(r) for r in raws];ids={p.id for p in papers}
    graph=nx.DiGraph();graph.add_nodes_from(ids)
    graph.add_edges_from((p.id,r) for p in papers for r in p.references if r in ids)
    counts=Counter();abstracts=Counter();topics=Counter();years=Counter();venues=Counter();types=Counter()
    roles=Counter();by_disc=defaultdict(set)
    for wid,sid,disc,role in memberships:by_disc[disc].add(wid);roles[role]+=1
    by_id={p.id:p for p in papers};raw_by_id={r['id']:r for r in raws}
    for disc,works in by_disc.items():
        counts[disc]=len(works);abstracts[disc]=sum(bool(by_id[w].abstract) for w in works)
    for raw,p in zip(raws,papers):
        years[p.year]+=1;venues[p.venue or 'UNKNOWN']+=1;types[p.work_type or 'UNKNOWN']+=1
        for topic in raw.get('topics',[]):topics[topic.get('display_name','UNKNOWN')]+=1
    components=sorted((len(c) for c in nx.weakly_connected_components(graph)),reverse=True)
    # Exact bounded reachability distributions: at most three citation hops.
    sampled=sorted(ids,key=lambda w:hashlib.sha256(w.encode()).digest())[:512]
    reach=[len(nx.single_source_shortest_path_length(graph,w,cutoff=3))-1 for w in sampled]
    dates=[p.publication_date for p in papers if p.publication_date];dois=Counter(p.doi for p in papers if p.doi)
    return dict(unique_papers=len(ids),papers_per_discipline=dict(counts),abstracts=sum(bool(p.abstract) for p in papers),abstracts_per_discipline=dict(abstracts),
        topics=dict(topics),publication_years=dict(sorted(years.items(),key=lambda x:str(x[0]))),publication_dates=dict(min=min(dates,default=None),max=max(dates,default=None)),
        venue_count=len(venues),largest_venues=venues.most_common(15),work_types=dict(types),internal_citation_edges=graph.number_of_edges(),
        all_reference_entries=sum(len(p.references) for p in papers),unresolved_reference_targets=len({r for p in papers for r in p.references}-ids),
        isolated_papers=sum(graph.degree(w)==0 for w in graph),weak_components=len(components),largest_weak_components=components[:10],
        sampled_three_hop_reachable_pairs=sum(reach),sampled_papers_with_citation_paths=sum(n>0 for n in reach),
        reachability_sample_size=len(sampled),reachability_scope='Deterministic hash sample, maximum 512 source papers; counts are sample-only',
        missing_titles=sum(not p.title for p in papers),missing_doi=sum(not p.doi for p in papers),missing_abstracts=sum(not p.abstract for p in papers),
        duplicate_doi_groups=sum(n>1 for n in dois.values()),duplicate_ids=len(papers)-len(ids),membership_rows=len(memberships),collection_roles=dict(roles),
        open_access=sum((r.get('open_access') or {}).get('is_oa') is True for r in raws),
        searchable_fulltext=sum(r.get('has_fulltext') is True for r in raws),
        downloadable_pdf=sum((r.get('has_content') or {}).get('pdf') is True for r in raws),
        machine_readable_xml=sum((r.get('has_content') or {}).get('grobid_xml') is True for r in raws),
        fulltext_downloaded=0,independently_verified_labels=0,scientifically_validated=False,ground_truth_verified=False)


def freeze_collection(store,output,provenance):
    output=Path(output)
    if output.exists():raise ValueError('Choose a new immutable corpus version')
    raws=store.rows();papers=[]
    acquired={r[0]:r[1] for r in store.db.connection.execute('SELECT id,acquired FROM raw_works')}
    for raw in raws:
        p=Paper.from_openalex(raw);p.retrieved_at=acquired[p.id];papers.append(p)
    manifest=freeze(papers,output,acquisition=[provenance])
    raw_path=output/'raw-works.jsonl.gz'
    with gzip.open(raw_path,'wt',encoding='utf-8') as f:
        for raw in raws:f.write(json.dumps(raw,ensure_ascii=False)+'\n')
    memberships=list(store.db.connection.execute('SELECT id,stratum,discipline,role FROM memberships ORDER BY id,stratum,role'))
    atomic_json(output/'memberships.json',memberships)
    report=quality_report(raws,memberships);atomic_json(output/'quality.json',report)
    # Availability shortlist, never synthetic answers or downloaded evidence.
    shortlist=[];paper_by_id={p.id:p for p in papers}
    for disc in FIELD_GROUPS:
        candidates=[]
        eligible={wid for wid,_,d,_ in memberships if d==disc}
        for raw in raws:
            if raw['id'] not in eligible:continue
            p=paper_by_id.get(raw['id'])
            if p and p.abstract and (raw.get('has_content') or {}).get('grobid_xml'):
                # Predeclared corpus-only proxy; no system outcomes or held-out text.
                quantitative=bool(re.search(r'\b\d+(?:\.\d+)?\s*(?:%|percent|patients|participants|samples)',p.abstract,re.I))
                candidates.append((not quantitative,raw['id'],raw.get('content_urls'),(raw.get('best_oa_location') or {}).get('license')))
        for _,wid,urls,license_name in sorted(candidates)[:10]:
            shortlist.append(dict(id=wid,discipline=disc,content_urls=urls,license=license_name,human_evidence_review='PENDING',downloaded=False))
    atomic_json(output/'fulltext-review-shortlist.json',shortlist)
    receipt=dict(corpus_sha256=manifest['sha256'],files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        git=git_provenance(),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        acquisition_complete=provenance.get('acquisition_complete',False),scientifically_validated=False,ground_truth_verified=False)
    atomic_json(output/'freeze-receipt.json',receipt)
    return report
