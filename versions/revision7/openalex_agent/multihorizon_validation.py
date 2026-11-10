"""Independent offline validation of V3 construction; no provider access."""
from pathlib import Path
from collections import Counter
from itertools import permutations
import json
import jsonschema
from .multihorizon_cases import short,digest,CORPUS
from .models import canonical_id
from .token_budget import count


def validate(directory,corpus):
    root=Path(directory);keys=json.loads((root/'keys.private.json').read_text())
    candidate=json.loads((root/'L.json').read_text())
    if candidate and candidate[0].get('protocol_revision')=='multihorizon-scientific-r7':
        from .multihorizon_validation_r7 import validate as revised
        return revised(root,corpus)
    short_tasks=json.loads((root/'S.json').read_text());long=json.loads((root/'L.json').read_text());memory=json.loads((root/'M.json').read_text())
    report={'short_tasks':len(short_tasks),'long_tasks':len(long),'memory_cases':len(memory),'errors':[],'main_trajectories':len(short_tasks)*5*2+len(long)*5*2+len(memory)*4*2,'oracle_only':True,'provider_calls':0}
    def require(ok,message):
        if not ok:report['errors'].append(message)
    require((len(short_tasks),len(long),len(memory))==(30,24,12),'Wrong primary case counts')
    schema=json.loads((Path(__file__).resolve().parents[1]/'protocols/OpenAlex_Memory_Case_Manifest_Schema_V3.json').read_text())
    from .multihorizon_cases import VERSION,SEED
    try:jsonschema.Draft202012Validator(schema).validate({'protocol_version':VERSION,'corpus_sha256':corpus.fingerprint,'sampling_seed':SEED,'cases':memory})
    except jsonschema.ValidationError as exc:report['errors'].append('V3 case schema: '+exc.message)
    require(len(json.loads((root/'S-pilot.json').read_text()))==6 and len(json.loads((root/'L-pilot.json').read_text()))==6 and len(json.loads((root/'M-pilot.json').read_text()))==2,'Incorrect pinned pilot case counts')
    for cases,expected,field in [(short_tasks,5,'id'),(long,4,'id'),(memory,2,'case_id')]:
        require(set(Counter(c['domain'] for c in cases).values())=={expected},'Unbalanced domain case counts')
        require(len({c[field] for c in cases})==len(cases),'Duplicate case identity')
    forbidden={x for k in keys['S'].values() for x in [*k['target_ids'],k['source_id'],*(p for edge in k['citation_edges'] for p in edge)]}
    used=set(forbidden);long_ids=set()
    for case in [*long,*json.loads((root/'L-pilot.json').read_text())]:
        gold=keys['L'][case['id']];ids=set(gold['source_ids']);require(not used&ids,'L target overlap '+case['id']);used.update(ids);long_ids.update(ids)
        require(len(case['phases'])==6,'Missing research phases')
        for a,b in gold['citation_edges']:require(b in corpus.get(a).references,'Missing/reversed L citation '+case['id'])
        for pid,q in gold['oracle']['queries'].items():require(pid in {p.id for p in corpus.search(q,10)},'L public-query discovery infeasible')
        require(all(pid not in json.dumps(case) and short(pid) not in json.dumps(case) for pid in ids),'L hidden ID leaked')
    memory_all=[*memory,*json.loads((root/'M-pilot.json').read_text())];per_case=[]
    for case in memory_all:
        ident=case['case_id'];gold=keys['M'][ident];records=[r for b in case['batches'] for r in b['papers']];ids=[canonical_id(r['work_id']) for r in records]
        require(len(case['batches'])==32 and all(len(b['papers'])==3 for b in case['batches']),'Wrong memory stream geometry')
        require(len(set(ids))==96 and not used&set(ids),'M duplicate or cross-case/target overlap '+ident);used.update(ids)
        seen=set();source_hashes=[]
        for batch in case['batches']:
            seen.update(r['work_id'] for r in batch['papers'])
            for r in batch['papers']:
                p=corpus.get(r['work_id']);source_hashes.append(digest(p.to_dict()))
                require(r['source_record_sha256']==digest(p.to_dict()),'Wrong source hash '+ident)
                require(r['excerpt'] in p.abstract and r['year']==p.year and r['title']==p.title,'Fabricated source content '+ident)
                require(set(r['references_to_exposed_works'])<=seen,'Future reference disclosed '+ident)
                require(all(canonical_id(t) in p.references for t in r['references_to_exposed_works']),'Fabricated direction '+ident)
            require(count(json.dumps(batch,ensure_ascii=False))<4096,'Batch exceeds working-token ceiling '+ident)
        require(case['case_source_sha256']==digest(source_hashes),'Changed case source list '+ident)
        for a,b in gold['gold_edges']:require(b in corpus.get(a).references,'Missing memory gold edge '+ident)
        first=set(ids[:24]);require(set(gold['gold_anchors'])<=first,'Quiz target unavailable at checkpoint24 '+ident)
        answers=gold['answers']
        require(answers['q1']=={'work_id':short(ids[0])} and answers['q2']=={'year':records[3]['year']} and answers['q3']=={'work_id':short(ids[9])},'Source/year answer key conflicts with fixed source roles '+ident)
        require(answers['q4']=={'citing_work_id':short(ids[9]),'cited_work_id':short(ids[3])} and answers['q5']=={'path':[short(ids[15]),short(ids[9]),short(ids[0])]},'Directed answer key conflicts with source roles '+ident)
        for pid,cue in gold['probe_clues'].items():require(sum(cue in r['excerpt'] for r in records)==1,'Nonunique question clue '+ident)
        chain=gold['answers']['q5']['path'];edges={(r['work_id'],t) for r in records[:24] for t in r['references_to_exposed_works']}
        alternatives=[list(x) for x in permutations(chain) if (x[0],x[1]) in edges and (x[1],x[2]) in edges]
        require(alternatives==[chain],'Ambiguous oriented path '+ident)
        src,target=map(short,gold['calibration_edge']);status='SUPPORTED' if (src,target) in edges else 'NOT_ESTABLISHED'
        require(status==gold['answers']['q6']['status'],'Calibration key not supported by shown references '+ident)
        require(gold['pressure']['ratio']>=1.5,'Insufficient actual context pressure '+ident)
        per_case.append({'case_id':ident,'papers':96,'pressure_ratio':gold['pressure']['ratio'],'max_batch_tokens':gold['pressure']['max_batch_record_tokens'],'all_targets_queryable_by_24':True})
    require(Counter(keys['M'][c['case_id']]['answers']['q6']['status'] for c in memory)=={'SUPPORTED':6,'NOT_ESTABLISHED':6},'Calibration imbalance')
    report.update(memory_details=per_case,main_and_pilot_disjoint=not report['errors'],status='PASS' if not report['errors'] else 'BLOCKED')
    return report
