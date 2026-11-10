"""Outcome-blind R7 workload checks, exact-source witnesses and replayable oracles."""
from pathlib import Path
from collections import Counter
import json,math
from .models import canonical_id,OPENALEX_SOURCES
from .multihorizon_cases import digest
from .multihorizon_cases_r7 import BASE,REVISION,PAGE_SIZE
from .ranking import relevance


def validate(root,corpus):
    root=Path(root);keys=json.loads((root/'keys.private.json').read_text());oldkeys=json.loads((BASE/'keys.private.json').read_text());errors=[]
    def require(condition,msg):
        if not condition:errors.append(msg)
    from .multihorizon_validation import validate as legacy
    base_validation=legacy(BASE,corpus)
    require(not base_validation['errors'],'Preserved S/M source validation failed')
    import jsonschema
    schema=json.loads((Path(__file__).resolve().parents[1]/'protocols/OpenAlex_Memory_Case_Manifest_Schema_V3_R7.json').read_text())
    from .multihorizon_cases import VERSION,SEED
    try:jsonschema.validate({'protocol_version':'openalex-multihorizon-v3-scientific-r7','corpus_sha256':corpus.fingerprint,'sampling_seed':SEED,'cases':json.loads((root/'M.json').read_text())},schema)
    except jsonschema.ValidationError as exc:errors.append('R7 memory case schema: '+exc.message)
    for name in ('S.json','S-pilot.json'):
        require(json.loads((root/name).read_text())==json.loads((BASE/name).read_text()),'Changed S questions '+name)
    forbidden={pid for k in keys['S'].values() for pid in [*k['target_ids'],k['source_id']]}
    for name in ('M.json','M-pilot.json'):
        values=json.loads((root/name).read_text());original=json.loads((BASE/name).read_text())
        require([{k:v for k,v in c.items() if k!='protocol_revision'} for c in values]==original,'Changed M physical stream or questions')
        for c in values:
            ident=c['case_id'];require(keys['M'][ident]==oldkeys['M'][ident],'Changed M gold key');forbidden.update(keys['M'][ident]['source_ids'])
    main=json.loads((root/'L.json').read_text());pilot=json.loads((root/'L-pilot.json').read_text());require(len(main)==24 and len(pilot)==6,'Wrong L counts')
    require(set(Counter(c['domain'] for c in main).values())=={4},'L domain imbalance')
    used=set(forbidden);details=[]
    for c in main+pilot:
        gold=keys['L'][c['id']];ids=set(gold['source_ids']);require(not used&ids,'Cross-study/target overlap '+c['id']);used.update(ids)
        require(c.get('protocol_revision')==REVISION and len(c['phases'])==6,'Wrong R7 phase identity')
        require(len(ids)>=66 and all(len(g)>=11 for g in gold['phase_source_ids']),'Insufficient distinct staged evidence '+c['id'])
        require(len(set().union(*map(set,gold['phase_source_ids'])))==sum(map(len,gold['phase_source_ids'])),'Phase target overlap')
        require(gold['item_denominator']==len(ids)+len(gold['citation_edges']),'Changed denominator')
        observed={};phase_records=[];minimum_promising=5
        for phase in c['phases']:
            steps=[x for x in gold['oracle']['steps'] if x['phase']==phase['index']];new=set()
            require(len(steps)==3 and [s['arguments']['query'] for s in steps]==phase['evidence_routes'],'Oracle/public route mismatch')
            for step in steps:
                papers=corpus.search(step['arguments']['query'],PAGE_SIZE);actual=[p.id for p in papers]
                require(actual==step['expected_returned_ids'],'Non-reproducible oracle query '+c['id'])
                require(len(actual)<=5,'Oracle violates page size');new.update(actual);observed.update({p.id:p for p in papers})
                scores=relevance(c['question'],papers);minimum_promising=min(minimum_promising,sum(v>=.03 for v in scores.values()))
            required=set(gold['phase_source_ids'][phase['index']-1]);require(required<=new,'Phase target not acquired by public oracle')
            phase_records.append({'phase':phase['index'],'required_new_sources':len(required),'oracle_acquisitions':len(steps),'conditional_lower_bound':math.ceil(len(required)/5),'cumulative_oracle_records':len(observed)})
        require(ids<=set(observed),'Unacquired gold target')
        for a,b in gold['citation_edges']:
            p=corpus.get(a);require(b in p.references and p.metadata_source in OPENALEX_SOURCES and 'references' in (p.metadata_fields or []),'Unverified/reversed dependency')
        for pid,h in gold['source_hashes'].items():require(h==digest(corpus.get(pid).to_dict()),'Source identity changed')
        for phase in c['phases'][1:]:
            q=phase['reuse_question'];reuse=gold['reuse'][str(phase['index'])]
            cue=q.split(chr(34))[1]
            require(cue in (corpus.get(reuse['source_id']).abstract or '')[:350] and sum(cue in (p.abstract or '') for p in observed.values())==1,'Reuse cue inaccessible or ambiguous')
            initial=[corpus.get(pid) for pid in gold['oracle']['steps'][0]['expected_returned_ids']]
            rank=relevance(c['question'],initial)
            require(reuse['source_id'] in sorted(rank,key=lambda x:(-rank[x],x))[:2],'Reuse source was not initially selectable')
            require(reuse['source_id'] in gold['phase_source_ids'][0] and corpus.get(reuse['source_id']).year==reuse['year'],'Invalid early reuse target')
            require('W'+reuse['source_id'].split('W')[-1] not in q and str(reuse['year']) not in q,'Reuse question leaks answer')
        require(all(pid not in json.dumps(c) and pid.rsplit('/',1)[1] not in json.dumps(c) for pid in ids),'Hidden target leaked')
        lower=math.ceil(len(ids)/5);require(lower>4 and len(gold['oracle']['steps'])==18,'Four-operation shortcut remains')
        details.append({'id':c['id'],'domain':c['domain'],'source_checks':len(ids),'directed_checks':len(gold['citation_edges']),
            'denominator':gold['item_denominator'],'cardinality_lower_bound':lower,'oracle_operations':18,
            'minimum_promising_results_per_oracle_query':minimum_promising,'phase_details':phase_records,
            'late_reuse_probes':5,'exact_global_minimum_claimed':False})
    return {'version':REVISION,'status':'PASS' if not errors else 'BLOCKED','errors':errors,'main_trajectories':636,
        'short_tasks':30,'long_tasks':24,'memory_cases':12,'L_difficulty':details,'source_corpus_sha256':corpus.fingerprint,
        'lower_bound_limitation':'Phase bounds assume no prior incidental acquisition; global cardinality bound holds for any query under page5. Oracle bounds are not autonomous outcomes.',
        'live_sustained_work':'NOT VERIFIED; live pilot gate retained','oracle_only':True,'provider_calls':0}
