"""Outcome-blind frozen-corpus construction for Multi-Horizon V3.

Oracle information is confined to the offline builder/validator and private keys.
No provider calls, hidden-question reads, or scientific outcome selection.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,re
from .corpus import LocalCorpus,atomic_json
from .models import canonical_id,OPENALEX_SOURCES
from .token_budget import count
from .context_memory import serialized

SEED=20261109
VERSION='openalex-multihorizon-v3-engineering-1'
V2=Path('/home/bs/Documents/OpenAlex-Agent-Graph-Visible-V2-2026-11-09/data/terra-graph-visible-v2-2026-11-09-v3')
CORPUS=Path('/home/bs/Documents/NewProject/data/corpus/multidisciplinary-2026-11-06-v1')


def digest(value):return hashlib.sha256(serialized(value).encode()).hexdigest()
def short(value):return canonical_id(value).rsplit('/',1)[1]
def order(*values):return hashlib.sha256('|'.join(map(str,(SEED,*values))).encode()).hexdigest()


def passage(p,maximum=250):
    text=p.abstract or '';words=list(re.finditer(r'\S+',text));end=0
    for w in words:
        if count(text[:w.end()])>maximum:break
        end=w.end()
    return text[:end]


def clue(text,exclude_year=None):
    for i in range(0,min(300,len(text)-72)+1,16):
        c=text[i:i+72]
        if len(c.split())>=6 and (exclude_year is None or str(exclude_year) not in c) and not re.search(r'\bW\d+\b',c):return c
    raise ValueError('No unique descriptive clue candidate')


def keywords(corpus,p):
    row=corpus.matrix[corpus.papers.index(p)]
    names=corpus.vectorizer.get_feature_names_out()
    positions=sorted(zip(row.indices,row.data),key=lambda z:(-z[1],names[z[0]]))
    terms=[]
    for i,_ in positions:
        t=str(names[i])
        if t not in terms and len(t)>4 and not re.search(r'\d{5,}|openalex|https',t):terms.append(t)
        if len(terms)==3:break
    return ' '.join(terms)


def construct(output,*,corpus=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    corpus=corpus or LocalCorpus(CORPUS);by=corpus.by_id
    memberships=json.loads((CORPUS/'memberships.json').read_text());domains=sorted({r[2] for r in memberships})
    pools={d:{r[0] for r in memberships if r[2]==d} for d in domains}
    oldkeys=json.loads((V2/'proposal/keys/answer-keys.json').read_text())
    forbidden={x for k in oldkeys for x in [*k['target_ids'],k['source_id'],*(p for edge in k['citation_edges'] for p in edge)]}
    original_main=json.loads((V2/'proposal/main-agent.json').read_text());original_pilot=json.loads((V2/'proposal/pilot-agent.json').read_text())
    s_meta={k['id']:k for k in oldkeys}
    short_tasks=[{**q,'study':'S','domain':s_meta[q['id']]['domain'],'objective_keys_withheld':True} for q in original_main]
    short_pilot=[{**q,'study':'S','domain':s_meta[q['id']]['domain'],'objective_keys_withheld':True} for q in original_pilot]
    audit=[];eligible={};excerpts={}
    for d in domains:
        ids=[];abstracts=set()
        for pid in sorted(pools[d],key=lambda p:order('eligibility',d,p)):
            p=by[pid];reason=None
            if pid in forbidden:reason='original_S_or_pilot_target_overlap'
            elif p.metadata_source not in OPENALEX_SOURCES or 'references' not in (p.metadata_fields or []):reason='citation_provenance_unavailable'
            elif not p.title or type(p.year) is not int:reason='missing_title_or_year'
            elif count(p.abstract or '')<220:reason='abstract_less_than_220_study_tokens'
            elif digest(p.abstract) in abstracts:reason='duplicate_abstract'
            if reason:audit.append({'domain':d,'candidate':pid,'stage':'eligibility','accepted':False,'reason':reason});continue
            text=passage(p);excerpts[pid]=text;abstracts.add(digest(p.abstract));ids.append(pid)
        eligible[d]=set(ids)
    used=set(forbidden);long=[];long_pilot=[];memory=[];memory_pilot=[];keys={'S':{k['id']:k for k in oldkeys},'L':{},'M':{}}
    positions={p.id:i for i,p in enumerate(corpus.papers)}
    feature_names=corpus.vectorizer.get_feature_names_out()
    def terms(p):
        # Avoid repeated linear .index lookup during deterministic construction.
        row=corpus.matrix[positions[p.id]];names=feature_names
        choices=sorted(zip(row.indices,row.data),key=lambda z:(-z[1],names[z[0]]));out=[]
        for i,_ in choices:
            t=str(names[i])
            if len(t)>4 and not re.search(r'\d{5,}|openalex|https',t) and t not in out:out.append(t)
            if len(out)==3:break
        return ' '.join(out)
    def candidate_paths(d,stratum='dense'):
        ids=eligible[d]-used
        sources=sorted(ids,key=lambda p:((-1 if stratum=='dense' else 1)*len(set(by[p].references)&ids),order('chain',d,p)))
        for cid in sources:
            for bid in [r for r in by[cid].references if r in by][:10]:
                if bid not in ids or bid==cid:continue
                targets=[a for a in [r for r in by[bid].references if r in by][:10] if a in ids and a not in (bid,cid) and bid not in by[a].references and cid not in by[a].references and cid not in by[bid].references]
                for aid in targets:
                    extra=[x for x in targets if x!=aid]
                    if extra:yield aid,bid,cid,extra[0]
    def unique_clues(ids,anchor_ids):
        cues={pid:clue(excerpts[pid],by[pid].year) for pid in anchor_ids}
        for pid,c in cues.items():
            if sum(c in excerpts[x] for x in ids)!=1:raise ValueError('nonunique_excerpt_clue')
        return cues
    def add_long(d,index,pilot=False):
        for a,b,c,other in candidate_paths(d,'dense' if index%2==0 else 'sparse'):
            ps=[by[x] for x in (a,b,c,other)]
            topicsA={t.id:t.name for t in ps[0].topics if t.name};topicsB={t.id:t.name for t in ps[3].topics if t.name}
            distinct=[(x,y) for x in sorted(topicsA) for y in sorted(topicsB) if x!=y and topicsA[x]!=topicsB[y]]
            if not distinct:
                audit.append({'domain':d,'candidate':[a,b,c,other],'stage':'L','accepted':False,'reason':'no_distinct_observed_topic_branches'});continue
            ta,tb=distinct[0];qs=[terms(ps[i]) for i in (2,0,3)]
            if any(not q or pid not in {p.id for p in corpus.search(q,10)} for q,pid in zip(qs,(c,a,other))):
                audit.append({'domain':d,'candidate':[a,b,c,other],'stage':'L','accepted':False,'reason':'public_query_top10_oracle_infeasible'});continue
            prefix='LP' if pilot else 'L';case_id=f'{prefix}-{domains.index(d)+1:02d}-{index+1:02d}'
            project=f'Investigate research on {topicsA[ta]} and {topicsB[tb]}, including {qs[0]}. Compare sources, methods and verifiable citation provenance without inferring unsupported findings.'
            descriptions=[f'Discover the initial literature cluster using these descriptive terms: {qs[0]}.',f'Investigate branch A, {topicsA[ta]}, using {qs[1]}.',f'Investigate branch B, {topicsB[tb]}, using {qs[2]}.','Explore directed citation paths connecting the acquired branches. Follow references of legitimately observed sources; do not infer missing links.',
                'Compare the collected study methods and evidence. Revisit earlier sources and distinguish observable metadata from claims that need full-text verification.',
                'Audit source identities, publication years and directed links. Synthesize only source-backed conclusions and explicitly identify remaining evidence gaps.']
            queries=[qs[0],qs[1],qs[2],qs[0]+' citation evidence',qs[1]+' '+qs[2],qs[0]+' '+qs[1]]
            phases=[{'index':i+1,'objective':desc,'baseline_query':queries[i]} for i,desc in enumerate(descriptions)]
            case={'id':case_id,'study':'L','domain':d,'question':project,'split':'pilot' if pilot else 'development','phases':phases}
            gold={'source_ids':[a,b,c,other],'citation_edges':[[c,b],[b,a]],'item_denominator':6,'phase_objectives':[[c],[a],[other],[[c,b],[b,a]],[a,other],[a,b,c,other]],'oracle':{'queries':dict(zip((c,a,other),qs)),'permitted_actions':['SEARCH_PAPERS','EXPAND_REFERENCES'],'hops':2,'references_within_first_page':True,'topic_branches':[ta,tb]}}
            keys['L'][case_id]=gold;(long_pilot if pilot else long).append(case);used.update((a,b,c,other))
            audit.append({'domain':d,'candidate':case_id,'stage':'L','accepted':True,'source_ids':[a,b,c,other],'citation_edges':gold['citation_edges']});return
        atomic_json(output/'selection-audit.json',audit)
        atomic_json(output/'construction-failure.json',{'domain':d,'study':'L','index':index,'pilot':pilot,'eligible':{k:len(v) for k,v in eligible.items()},'rejections':dict(Counter(a.get('reason') for a in audit if not a.get('accepted')))})
        raise ValueError(f'No oracle-feasible distinct-branch L case: {d}/{index}/{pilot}')
    for i in range(4):
        for d in domains:add_long(d,i)
    # One fully disjoint staged pilot per domain; never used in main analysis.
    for d in domains:add_long(d,0,True)
    def add_memory(d,index,pilot=False):
        stratum='dense' if index==0 else 'sparse';status='SUPPORTED' if index==0 else 'NOT_ESTABLISHED'
        for a,b,c,yearid in candidate_paths(d,stratum):
            anchors={a,b,c,yearid};pair=None
            for src in sorted(eligible[d]-used-anchors,key=lambda x:order('calibration',d,index,pilot,x)):
                dst=next((x for x in by[src].references if x in eligible[d]-used-anchors and x!=src and src not in by[x].references),None)
                if dst:pair=(src,dst);break
            if pair is None:continue
            f,e=pair;anchors.update((e,f))
            distractors=[x for x in eligible[d]-used-anchors if not(set(by[x].references)&anchors)]
            distractors.sort(key=lambda x:order('distractor',d,index,pilot,x))
            if len(distractors)<90:continue
            positions_main={0:a,3:yearid,6:e,9:b,15:c,18:f}
            selected=[];iterator=iter(distractors[:90])
            for pos in range(96):selected.append(positions_main[pos] if pos in positions_main else next(iterator))
            try:cues=unique_clues(selected,anchors)
            except ValueError as exc:
                audit.append({'domain':d,'candidate':[a,b,c,yearid,e,f],'stage':'M','accepted':False,'reason':str(exc)});continue
            prefix='MP' if pilot else 'M';case_id=f'{prefix}-{domains.index(d)+1:02d}-{index+1:02d}'
            q={'q1':f'Give the work ID of the earlier source whose passage contains: "{cues[a]}"',
               'q2':f'Give the publication year of the study whose passage contains: "{cues[yearid]}"',
               'q3':f'Give the work ID of the source whose passage contains: "{cues[b]}"',
               'q4':f'Give the citing and cited IDs for the directed connection between the sources with passages "{cues[b]}" and "{cues[yearid]}".',
               'q5':'Give the three-work citing-to-cited path connecting these passage clues (clues are unordered): '+serialized(sorted([cues[a],cues[b],cues[c]],key=lambda x:order('clue-order',x))),
               'q6':f'Is a citation from the source with passage "{cues[f if status=="SUPPORTED" else e]}" to the source with passage "{cues[e if status=="SUPPORTED" else f]}" established by the exposed records? Answer SUPPORTED or NOT_ESTABLISHED.'}
            batches=[];seen=set();sourcehash=[]
            for batch in range(32):
                batchids=selected[batch*3:batch*3+3];seen.update(batchids);records=[]
                for pid in batchids:
                    p=by[pid];h=digest(p.to_dict());sourcehash.append(h)
                    records.append({'work_id':short(pid),'title':p.title,'year':p.year,'excerpt':excerpts[pid],'references_to_exposed_works':[short(r) for r in p.references if r in seen],'source_record_sha256':h})
                batches.append({'batch_number':batch+1,'papers':records})
            case={'case_id':case_id,'domain':d,'density_stratum':stratum,'batches':batches,'questions':q,'case_source_sha256':digest(sourcehash)}
            gold={'q1':{'work_id':short(a)},'q2':{'year':by[yearid].year},'q3':{'work_id':short(b)},'q4':{'citing_work_id':short(b),'cited_work_id':short(yearid)},'q5':{'path':[short(c),short(b),short(a)]},'q6':{'status':status}}
            refs={(short(p),short(r)) for p in selected[:24] for r in by[p].references if r in selected[:24]}
            if (short(c),short(b)) not in refs or (short(b),short(a)) not in refs or (short(b),short(yearid)) not in refs:raise ValueError('Missing verified memory gold path')
            keys['M'][case_id]={'answers':gold,'source_ids':selected,'gold_edges':[[c,b],[b,a],[b,yearid],[f,e]],'gold_anchors':sorted(anchors),'probe_clues':cues,
                'pressure':{'aggregate_excerpt_tokens':sum(count(excerpts[x]) for x in selected),'active_tokens':4096,'ratio':sum(count(excerpts[x]) for x in selected)/4096,'max_batch_record_tokens':max(count(serialized(batch)) for batch in batches)},
                'known_source_work_overlaps':[],'calibration_edge':[f,e] if status=='SUPPORTED' else [e,f]}
            (memory_pilot if pilot else memory).append(case);used.update(selected)
            audit.append({'domain':d,'candidate':case_id,'stage':'M','accepted':True,'paper_ids':selected,'gold_edges':keys['M'][case_id]['gold_edges']});return
        atomic_json(output/'selection-audit.json',audit)
        raise ValueError(f'No eligible disjoint 96-paper M case: {d}/{index}/{pilot}')
    for i in range(2):
        for d in domains:add_memory(d,i)
    # Two disjoint pilots, spanning different density/calibration strata.
    for i,d in enumerate(domains[:2]):add_memory(d,i,True)
    atomic_json(output/'S.json',short_tasks);atomic_json(output/'S-pilot.json',short_pilot)
    atomic_json(output/'L.json',long);atomic_json(output/'L-pilot.json',long_pilot)
    atomic_json(output/'M.json',memory);atomic_json(output/'M-pilot.json',memory_pilot)
    atomic_json(output/'keys.private.json',keys);atomic_json(output/'selection-audit.json',audit)
    report={'version':VERSION,'selection_seed':SEED,'corpus_sha256':corpus.fingerprint,'main_counts':{'S':len(short_tasks),'L':len(long),'M':len(memory)},'pilot_counts':{'S':len(short_pilot),'L':len(long_pilot),'M':len(memory_pilot)},
        'domains':domains,'eligible_records_per_domain':{d:len(ids) for d,ids in eligible.items()},'selection_uses_model_outcomes':False,'hidden_keys_passed_to_models':False,
        'L_source_target_count':sum(len(v['source_ids']) for v in keys['L'].values()),'M_total_scheduled_papers':96*len(memory),'M_min_pressure_ratio':min(keys['M'][c['case_id']]['pressure']['ratio'] for c in memory),
        'main_and_pilot_anchor_sets_disjoint':True,'M_complete_streams_disjoint':True,'source_overlap_policy':'All M physical papers exclude original S/pilot gold and L/pilot source targets; M streams disjoint globally. Non-target S/L retrieval overlap remains possible and is reported, not hidden.',
        'requires_disjoint_live_pilots_before_scored_freeze':True,'status':'offline_constructed_not_scored_frozen'}
    atomic_json(output/'construction-report.json',report)
    return report
