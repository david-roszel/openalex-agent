"""Prospective R7 outcome/visibility diagnostics, separate from legacy scores."""
from pathlib import Path
import json
from .models import canonical_id,OPENALEX_SOURCES,Paper


def evidence_objects(value):
    if isinstance(value,dict):
        yield value
        for k,v in value.items():
            if k not in ('protected','proposals','memory_probe','factual_task'):yield from evidence_objects(v)
    elif isinstance(value,list):
        for v in value:yield from evidence_objects(v)
    elif isinstance(value,str):
        decoder=json.JSONDecoder();i=0
        while i<len(value):
            pos=value.find('{',i)
            if pos<0:break
            try:obj,n=decoder.raw_decode(value[pos:])
            except ValueError:i=pos+1;continue
            yield from evidence_objects(obj);i=pos+n


def visible_records(payload,papers):
    visible=set();edges=set()
    for obj in evidence_objects(payload):
        try:pid=canonical_id(obj.get('id',obj.get('work_id',obj.get('source'))))
        except (ValueError,TypeError):continue
        if pid not in papers:continue
        p=papers[pid]
        if (obj.get('title')==p.title and isinstance(obj.get('abstract_excerpt'),str) and obj['abstract_excerpt'] in (p.abstract or '')):visible.add(pid)
        if p.metadata_source not in OPENALEX_SOURCES or 'references' not in (p.metadata_fields or []):continue
        if obj.get('relation')=='PAPER_CITES_PAPER' and obj.get('direction')=='source_cites_target':
            if (obj.get('provenance') or {}).get('metadata_source') not in OPENALEX_SOURCES:continue
            refs=[obj.get('target')]
        else:refs=obj.get('references',[])
        if not isinstance(refs,list):continue
        for value in refs:
            try:target=canonical_id(value)
            except (ValueError,TypeError):continue
            if target in p.references:edges.add((pid,target))
    return visible,edges


def directed_diagnostics(directory,key,result):
    directory=Path(directory);state=json.loads((directory/'state.json').read_text())
    values=state.get('papers',[]);values=values.values() if isinstance(values,dict) else values
    papers={p.id:p for p in map(Paper.from_dict,values)}
    gold={tuple(map(canonical_id,e)) for e in key['citation_edges']};acquired=set();delivered=set();calls=[]
    for a,b in gold:
        p=papers.get(a)
        if p and b in p.references and p.metadata_source in OPENALEX_SOURCES and 'references' in (p.metadata_fields or []):acquired.add((a,b))
    for path in sorted((directory/'context').glob('request-*.json')):
        r=json.loads(path.read_text());d=r.get('diagnostics',{})
        if not d.get('wire_sha256') or d.get('request_not_sent',True):continue
        _,edges=visible_records(r.get('input',{}),papers);hits=gold&acquired&edges
        delivered.update(hits);calls.append({'call':r['call'],'directed_gold_edges':[list(e) for e in sorted(hits)],'wire_sha256':d['wire_sha256']})
    n=len(gold)
    return {'edge_diagnostic_version':'observed-directed-links-r7','legacy_edge_metric_meaning':'Gold-link endpoint co-discovery; not model traversal',
        'acquired_directed_edge_recall':len(acquired)/n if n else None,'delivered_directed_edge_recall':len(delivered)/n if n else None,
        'directed_edge_delivery_calls':calls,'acquired_directed_gold_edges':[list(e) for e in sorted(acquired)],
        'delivered_directed_gold_edges':[list(e) for e in sorted(delivered)],'directed_edge_gold_denominator':n,
        'delivery_is_not_causal_use':True,'diagnostic_scope':'Complete structured records/reference lists; unparsed prose is not inferred',
        'target_resolution':'Relation visibility does not imply retrieval of the cited publication'}


def final_outcome(method,raw_score,responses,*,collection_ok=True):
    required=method!='standard'
    valid=not required or bool(responses) and all(r and r.get('status')=='valid' for r in responses)
    success=bool(collection_ok and valid)
    return {'collection_status':'COMPLETED' if collection_ok else 'FAILED','required_final_response':required,
        'factual_response_success':valid if required else None,'overall_failure_reason':None if success else 'required_final_response_missing_or_malformed' if not valid else 'collection_failure',
        'status':'COMPLETED' if success else 'FAILED','raw_primary_score':raw_score,
        'operational_itt_score':raw_score if success else 0.0,'complete_case_eligible':success,
        'final_response_status':'not_applicable' if not required else 'valid' if valid else 'missing_or_malformed'}
