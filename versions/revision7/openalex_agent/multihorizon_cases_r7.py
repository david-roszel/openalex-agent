"""Outcome-blind phased workloads from genuine frozen records; no inference."""
from pathlib import Path
from collections import Counter
import json,math,re,shutil
from .corpus import LocalCorpus,atomic_json
from .multihorizon_cases import CORPUS,order,digest,clue
from .models import OPENALEX_SOURCES

BASE=Path('/home/bs/Documents/OpenAlex-Agent-Multihorizon-V3-2026-11-09/data/openalex-multihorizon-v3-2026-11-09-v6/proposal')
REVISION='multihorizon-scientific-r7'
PAGE_SIZE=5
PHASES=('Initial literature discovery','Investigate branch A','Investigate branch B',
        'Citation-path exploration','Cross-study verification','Evidence audit and synthesis limitations')


def construct(output,corpus=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(BASE,output,dirs_exist_ok=True)
    for file in output.rglob("*"):
        if file.is_file():file.chmod(0o644)
    corpus=corpus or LocalCorpus(CORPUS);by=corpus.by_id
    membership=json.loads((CORPUS/'memberships.json').read_text());domains=sorted({r[2] for r in membership})
    pools={d:{r[0] for r in membership if r[2]==d} for d in domains}
    keys=json.loads((BASE/'keys.private.json').read_text());old_long=keys['L'];keys['L']={}
    forbidden={pid for k in keys['S'].values() for pid in [*k['target_ids'],k['source_id']]}
    forbidden.update(pid for k in keys['M'].values() for pid in k['source_ids'])
    forbidden.update(pid for k in old_long.values() for pid in k['source_ids'])
    used=set(forbidden);positions={p.id:i for i,p in enumerate(corpus.papers)}
    names=corpus.vectorizer.get_feature_names_out();queries={};pages={};audit=[]
    def terms(pid):
        if pid not in queries:
            row=corpus.matrix[positions[pid]];candidates=sorted(zip(row.indices,row.data),key=lambda z:(-z[1],names[z[0]]));out=[]
            for i,_ in candidates:
                word=str(names[i])
                if len(word)>4 and not re.search(r'\d{5,}|https|openalex',word) and word not in out:out.append(word)
                if len(out)==3:break
            queries[pid]=' '.join(out)
        return queries[pid]
    def page(pid):
        if pid not in pages:pages[pid]=[p.id for p in corpus.search(terms(pid),PAGE_SIZE)] if terms(pid) else []
        return pages[pid]
    incoming=corpus.citing
    def eligible(pid,d):
        p=by[pid]
        return pid in pools[d] and pid not in used and p.metadata_source in OPENALEX_SOURCES and 'references' in (p.metadata_fields or []) and p.title and p.year and len(p.abstract or '')>=180
    def connected(pid):return {p.id for p in incoming.get(pid,[])}|set(by[pid].references)
    def build(d,index,pilot=False):
        pool={pid for pid in pools[d] if eligible(pid,d)}
        roots=sorted(pool,key=lambda p:order('r7-root',d,index,pilot,p))
        for root in roots[:300]:
            selected=set();acquired={};phases=[];groups=[];links=[];oracle=[];anchor=None;ok=True;local_queries=set()
            for phase in range(6):
                prior=set(selected);routes=[];group=set();phase_seeds=[]
                if phase in (1,2):
                    # A genuine oriented two-hop chain spans the first3 phases.
                    first_candidates={p.id for p in incoming.get(anchor,[])}&pool-selected
                elif prior:first_candidates=set().union(*(connected(p) for p in prior))&pool-selected
                else:first_candidates={root}
                candidates=sorted(first_candidates,key=lambda p:order('r7-route',d,index,pilot,phase,p))
                first_seed=None
                for route in range(3):
                    if route:
                        neighbors=set().union(*(connected(p) for p in (prior|group)))&pool-selected-group
                        # Neighbor discovery rather than unrelated corpus-wide targets.
                        candidates=sorted(neighbors,key=lambda p:order('r7-route-extra',d,index,pilot,phase,route,p))
                    chosen=None
                    for pid in candidates[:600]:
                        q=terms(pid)
                        if not q or q in local_queries:continue
                        result=page(pid)
                        novel={x for x in result if x in pool and x not in selected and x not in group}
                        if pid not in result or len(novel)<(4 if route<2 else 3):continue
                        chosen=(pid,q,result,novel);break
                    if chosen is None:ok=False;break
                    pid,q,result,novel=chosen
                    if first_seed is None:first_seed=pid
                    local_queries.add(q);routes.append(q);group.update(novel);phase_seeds.append(pid)
                    acquired.update({x:by[x] for x in result});oracle.append({'phase':phase+1,'action':'SEARCH_PAPERS','arguments':{'query':q},'expected_returned_ids':result})
                if not ok or len(group)<11:ok=False;break
                cross=[]
                if prior:
                    cross=sorted({(a,b) for a in group|prior for b in by[a].references if b in group|prior and ((a in group and b in prior) or (a in prior and b in group))})
                    if not cross:ok=False;break
                    links.extend([list(e) for e in cross[:2]])
                group=sorted(group);groups.append(group);selected.update(group)
                if phase in (1,2):links.append([first_seed,anchor])
                earlier=groups[0][0] if phase else None
                if earlier:
                    try:cue=clue((by[earlier].abstract or '')[:350],by[earlier].year)
                    except ValueError:ok=False;break
                    reuse=f'Revisit the earlier source whose abstract contains "{cue}". Give its publication year with a verbatim supporting excerpt, or explicitly abstain if unavailable.'
                else:reuse='Give one available publication year and source quotation, or explicitly abstain.'
                phases.append({'index':phase+1,'objective':PHASES[phase]+'. Investigate the literature described by these three evidence routes: '+ '; '.join(routes)+'. Check directed citation connections with earlier sources; distinguish metadata from scientific claims.',
                    'evidence_routes':routes,'baseline_query':routes[0],'reuse_question':reuse})
                anchor=first_seed
            if not ok:
                audit.append({'domain':d,'root':root,'index':index,'pilot':pilot,'accepted':False,'reason':'insufficient_novel_connected_query_pages_or_reuse_clue','completed_phases':len(phases)});continue
            all_edges=sorted({tuple(e) for e in links});source_ids=sorted(selected)
            # Every required source is acquired by the oracle; no invented relation.
            if any(b not in by[a].references for a,b in all_edges):raise ValueError('Unverified phase dependency')
            topic_names=[]
            for pid in [root,*groups[0]]:
                for t in by[pid].topics:
                    if t.name and t.name not in topic_names:topic_names.append(t.name)
                if len(topic_names)>=2:break
            if len(topic_names)<2:continue
            cid=f'{"LP7" if pilot else "L7"}-{domains.index(d)+1:02d}-{index+1:02d}'
            question=('Investigate '+d+' research on '+topic_names[0]+' and '+topic_names[1]+'. Follow the evolving evidence routes, compare source methods and citation provenance, and preserve earlier identities and publication years for later checks. Do not infer scientific truth from metadata.')[:1000]
            from .ranking import relevance
            first_ids=oracle[0]['expected_returned_ids'];rank=relevance(question,[by[x] for x in first_ids])
            early_candidates=[x for x in sorted(first_ids,key=lambda x:(-rank[x],x))[:2] if x in groups[0]]
            earlier=None;cue=None
            for candidate in early_candidates:
                text=(by[candidate].abstract or '')[:350]
                cues=[text[i:i+100] for i in range(0,max(1,len(text)-100),16) if str(by[candidate].year) not in text[i:i+100] and len(text[i:i+100].split())>=8]
                cue=next((q for q in cues if sum(q in (p.abstract or '') for p in acquired.values())==1),None)
                if cue:earlier=candidate;break
            if earlier is None:
                audit.append({'domain':d,'root':root,'accepted':False,'reason':'no_unique_initially_selected_reuse_source'});continue
            for phase in phases[1:]:phase['reuse_question']=f'Revisit the earlier source whose abstract contains "{cue}". Give its publication year with a verbatim supporting excerpt, or explicitly abstain if unavailable.'
            case={'id':cid,'study':'L','domain':d,'question':question,'split':'pilot' if pilot else 'development','protocol_revision':REVISION,'phases':phases}
            phase_checks=[]
            for i,group in enumerate(groups):
                members=set(group);edgeitems=[list(e) for e in all_edges if any(x in members for x in e) and set(e)<=set().union(*(set(g) for g in groups[:i+1]))]
                phase_checks.append([*group,*edgeitems])
            reuse={str(i):{'source_id':earlier,'year':by[earlier].year} for i in range(2,7)}
            gold={'source_ids':source_ids,'citation_edges':[list(e) for e in all_edges],
                'item_denominator':len(source_ids)+len(all_edges),'phase_objectives':phase_checks,'phase_source_ids':groups,'reuse':reuse,
                'oracle':{'steps':oracle,'executable_acquisition_operations':len(oracle),'cardinality_lower_bound':math.ceil(len(source_ids)/PAGE_SIZE),
                    'lower_bound_scope':'Any acquisition returns at most5 records; not an exact optimum over unrestricted queries, nor a claim about autonomous behavior',
                    'phase_minimum_new_record_operations':[math.ceil(len(g)/PAGE_SIZE) for g in groups],
                    'phase_seed_ids':[s['expected_returned_ids'][0] for s in oracle[::3]],'dependencies':'Real citation links cross successive disclosed research stages; phase2/3 introduce a verified oriented chain',
                    'reuse_requires_earlier_information':True,'reuse_scope':'Secondary fixed response probes; primary remains objective acquisition'},
                'source_hashes':{pid:digest(by[pid].to_dict()) for pid in source_ids}}
            used.update(selected);keys['L'][cid]=gold
            audit.append({'case_id':cid,'domain':d,'accepted':True,'source_count':len(source_ids),'edge_count':len(all_edges),'oracle_operations':18,'cardinality_lower_bound':gold['oracle']['cardinality_lower_bound']})
            return case
        atomic_json(output/'r7-selection-audit.json',audit)
        raise ValueError('No feasible revised L project: '+str((d,index,pilot,len(pool))))
    main=[];pilot=[]
    for index in range(4):
        for d in domains:
            main.append(build(d,index));atomic_json(output/'r7-selection-audit.json',audit)
    for d in domains:pilot.append(build(d,0,True));atomic_json(output/'r7-selection-audit.json',audit)
    atomic_json(output/'L.json',main);atomic_json(output/'L-pilot.json',pilot);atomic_json(output/'keys.private.json',keys)
    for name in ('M.json','M-pilot.json'):
        cases=json.loads((BASE/name).read_text())
        for c in cases:c['protocol_revision']=REVISION
        atomic_json(output/name,cases)
    report={'revision':REVISION,'selection_uses_architecture_outcomes':False,'source_corpus_sha256':corpus.fingerprint,'counts':{'S':30,'L':24,'M':12},
        'pilot_counts':{'S':6,'L':6,'M':2},'page_size':PAGE_SIZE,'acquisition_record_cap':200,'maximum_records_in36_acquisitions':180,
        'L_cases':[{ 'id':c['id'],'domain':c['domain'],**{k:keys['L'][c['id']]['oracle'][k] for k in ('executable_acquisition_operations','cardinality_lower_bound','phase_minimum_new_record_operations')}} for c in main+pilot],
        'S_unchanged':True,'M_physical_streams_and_gold_unchanged':True,'live_sustained_work':'NOT VERIFIED; genuine pilot gate remains mandatory'}
    shutil.copyfile(output/'construction-report.json',output/'base-revision6-construction-report.json')
    shutil.copyfile(output/'selection-audit.json',output/'base-revision6-selection-audit.json')
    atomic_json(output/'construction-report.json',report);atomic_json(output/'selection-audit.json',audit)
    atomic_json(output/'r7-construction-report.json',report);return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(construct(a.output),indent=2))
