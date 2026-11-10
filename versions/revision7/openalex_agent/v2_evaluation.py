"""Exact V1 objective score functions, reused without new quality labels."""
from pathlib import Path
import json,csv,re

def selected_text(payload):
    from openalex_agent.evaluation_artifacts import _views
    views,_=_views(payload)
    def strings(value):
        if isinstance(value,str):yield value
        elif isinstance(value,dict):
            for item in value.values():yield from strings(item)
        elif isinstance(value,list):
            for item in value:yield from strings(item)
    return '\n'.join(strings([*views,*[payload.get(k) for k in ('observation','active_context','recovered_observation')]]))

def objective_score(job,key,result):
    directory=Path(result['directory'])
    ordered=[r['id'] for r in result.get('ranked',[]) if isinstance(r,dict) and 'id' in r]
    if not ordered and (directory/'ranked.csv').exists():
        ordered=[r['id'] for r in csv.DictReader((directory/'ranked.csv').open())]
    observed=set(result.get('inspected_ids',[]))
    targets=set(key['target_ids']);top=set(ordered[:10])
    ranks=[i+1 for i,pid in enumerate(ordered[:10]) if pid in targets]
    requests=[json.loads(p.read_text()) for p in sorted((directory/'context').glob('request-*.json'))]
    passage=key['exact_passage']
    delivered=[r['call'] for r in requests if passage in selected_text(r['input'])]
    edges=key['citation_edges'];found=[edge for edge in edges if set(edge)<=observed]
    from openalex_agent.evaluation_artifacts import graph_delivery
    graph=graph_delivery(requests)
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    edits=[e for e in events if e.get('kind')=='context_edit' and e.get('source')=='model']
    recovered=[e for e in events if e.get('kind')=='context_recovery']
    evidence=[e for e in events if e.get('kind')=='evidence_record']
    factual=(result.get('factual_responses') or [{}])[0];claim=factual.get('evidence_record') or {}
    source=(claim.get('provenance') or {}).get('source_record') or {}
    years=set(map(int,re.findall(r'\b(?:18|19|20)\d{2}\b',claim.get('statement',''))))
    year_match=(next(iter(years))==source.get('year')) if len(years)==1 and source.get('year') else None
    last_has=bool(requests) and passage in selected_text(requests[-1]['input'])
    delivered_edits=0
    for edit in edits:
        call=(edit.get('usage') or {}).get('call_index');after=edit.get('after','')
        if edit.get('committed') and edit.get('applied') and isinstance(call,int) and after and any(r['call']>call and after in r['input'].get('active_context','') for r in requests):delivered_edits+=1
    state=json.loads((directory/'state.json').read_text());actions=state.get('history',[])
    stats={};citation_edges=0
    if (directory/'graph.json').exists():
        from openalex_agent.graph import ScholarlyGraph
        stored=ScholarlyGraph.load(directory/'graph.json');stats=stored.stats();citation_edges=stored.citation_graph().number_of_edges()
    return {'target_recall_at_10':len(targets&top)/len(targets),'target_mrr_at_10':1/min(ranks) if ranks else 0.,'target_discovery_rate':len(targets&observed)/len(targets),'all_targets_discovered':int(targets<=observed),'verified_edge_recall':len(found)/len(edges) if edges else None,'chain_complete':int(len(found)==len(edges)) if edges else None,'evidence_source_discovered':int(key['source_id'] in observed),'evidence_delivered':int(bool(delivered)) if job['method']!='standard' else None,'evidence_final_input':int(last_has) if job['method']!='standard' else None,'retention_given_exposure':int(last_has) if delivered else None,'evidence_delivery_calls':delivered,'graph_payload_calls':graph['structured_graph_calls'],'candidate_payload_calls':graph['candidate_calls'],'citation_payload_calls':graph['citation_edge_calls'],'graph_target_overlap_calls':graph['action_target_overlap_calls'],'stored_graph_statistics':stats,'stored_citation_edges':citation_edges,'reference_expansions':sum(a['action']=='EXPAND_REFERENCES' for a in actions),'citation_expansions':sum(a['action']=='EXPAND_CITATIONS' for a in actions),'search_actions':sum(a['action']=='SEARCH_PAPERS' for a in actions),'context_edits_attempted':len(edits),'context_edits_accepted':sum(e.get('accepted') is True for e in edits),'context_edits_committed':sum(e.get('committed') is True and e.get('applied') is True for e in edits),'modified_context_delivery_count':delivered_edits,'recoveries':len(recovered),'evidence_records':len(evidence),'nonempty_evidence_quotes':sum(bool(e.get('evidence_quote')) for e in evidence),'quote_source_exact_matches':sum(e.get('quote_matches_observed_metadata') is True for e in evidence),'factual_response_status':factual.get('status'),'metadata_year_consistency':year_match,'factual_source_identifier_valid':claim.get('identifier_valid'),'factual_quote_exact':claim.get('quote_matches_observed_metadata'),'independent_grounding':'NOT EVALUATED','ndcg_general_relevance':None}
