"""OFFLINE reviewer support: no model decisions, grades, approvals or source writes."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
from openalex_agent.corpus import LocalCorpus,atomic_json
from openalex_agent.pooled_review import _development_questions
from openalex_agent.evidence_visibility import audit_passage
from openalex_agent.models import Paper


def investigate(corpus_path,questions_path,audit_path,historical_path,output):
    output=Path(output)
    if output.exists():raise ValueError('Choose fresh immutable audit output')
    questions=_development_questions(questions_path)
    audit_path=Path(audit_path);selected=json.loads((audit_path/'selected-papers.json').read_text())['papers']
    chains=json.loads((audit_path/'citation-chains.json').read_text())['chains']
    searches=json.loads((audit_path/'retrieval.json').read_text())
    corpus=LocalCorpus(corpus_path);questions={q['id']:q for q in questions}
    assert len(selected)==24 and all(p['question_id'] in questions for p in selected)
    # Diagnostic queries are reviewer-discovered witnesses, not model-visible hints.
    hits={qid:[dict(query=s['query'],ids=[p.id for p in corpus.search(s['query'],limit=10)]) for s in rows] for qid,rows in searches.items()}
    historic=Path(historical_path);manifest=json.loads((historic/'experiment.json').read_text())
    import csv
    native=list(csv.DictReader((historic/'metrics.csv').open()))
    methods=('standard','llm','llm_graph','clm','clm_graph')
    records=[]
    for row in selected:
        paper=corpus.get(row['id']);quote=row['exact_abstract_quote'];assert quote in paper.abstract
        qid=row['question_id'];question_search=[p.id for p in corpus.search(questions[qid]['question'],limit=10)]
        evidence=dict(question_id=qid,paper_id=paper.id,title=paper.title,year=paper.year,
            corpus_abstract_contains_quote=True,quote_sha256=hashlib.sha256(quote.encode()).hexdigest(),
            quote_offset=paper.abstract.index(quote),quote_in_350_character_prefix=quote in paper.abstract[:350],
            initial_question_search_rank_1_based=question_search.index(paper.id)+1 if paper.id in question_search else None,
            reviewer_query_witnesses=[dict(query=s['query'],rank=s['ids'].index(paper.id)+1) for s in hits[qid] if paper.id in s['ids']],
            allowed_search_can_return_paper=any(paper.id in s['ids'] for s in hits[qid]),
            witness_is_model_decision=False,current_three_action_trajectory_delivery='NOT_OBSERVED: proposal has no collected outputs',
            full_abstract_tool_available=False,archive_scope='selected observations only, <=512 characters per read',
            full_text='NOT_PRESENT; external access not checked',historical_nov6=[])
        for cell in native:
            if cell['method'] not in methods or cell['query_id']!=qid:continue
            run=historic/'runs'/Path(cell['directory']).name
            observed=audit_passage(run,paper,quote)
            graph=json.loads((run/'graph.json').read_text()) if (run/'graph.json').exists() else {}
            node=next((n for n in graph.get('nodes',[]) if n.get('id')==paper.id),None)
            active=run/'context/active_context.txt'
            observed.update(method=cell['method'],trial=int(cell['trial']),action_budget=int(cell['action_budget']),
                graph_node_present=bool(node),graph_node_resolved=node.get('resolved') if node else None,
                final_context_file_contains_quote=quote in active.read_text() if active.exists() else None,
                graph_presence_is_not_model_delivery=True,context_presence_is_not_semantic_retention=True)
            evidence['historical_nov6'].append(observed)
        records.append(evidence)
    edges=[]
    for chain in chains:
        for e in chain['edges']:
            citing=corpus.get(e['citing_id']);cited=corpus.get(e['cited_id'])
            assert cited.id in citing.references
            internal=[wid for wid in citing.references if wid in corpus.by_id]
            from openalex_agent.ranking import relevance
            from openalex_agent.selection import Scoring
            from openalex_agent.evaluation_artifacts import _views
            source_score=relevance(questions['Q-PHY-03-2']['question'],[citing])[citing.id]
            edge_history=[]
            for cell in native:
                if cell['method'] not in methods or cell['query_id']!='Q-PHY-03-2':continue
                run=historic/'runs'/Path(cell['directory']).name
                result=json.loads((run/'result.json').read_text())
                graph=json.loads((run/'graph.json').read_text()) if (run/'graph.json').exists() else {}
                delivered=[]
                for path in sorted((run/'context').glob('request-*.json')):
                    req=json.loads(path.read_text())
                    for view in _views(req.get('input',{}))[0]:
                        if any(edge.get('source')==citing.id and edge.get('target')==cited.id
                               for edge in view.get('subgraph',[]) if isinstance(edge,dict)):
                            delivered.append(req['call']);break
                edge_history.append(dict(method=cell['method'],trial=int(cell['trial']),
                    citing_metadata_retrieved=citing.id in result['inspected_ids'],
                    cited_metadata_retrieved=cited.id in result['inspected_ids'],
                    stored_in_graph=any(edge.get('source')==citing.id and edge.get('target')==cited.id
                        and edge.get('relation')=='PAPER_CITES_PAPER' for edge in graph.get('edges',[])),
                    explicit_structured_edge_delivered_to_model_calls=delivered))
            edges.append(dict(chain_id=chain['chain_id'],source=citing.id,target=cited.id,
                corpus_reference_index=citing.references.index(cited.id),
                source_eligible_as_anchor_at_depth_zero=source_score>=Scoring().min_relevance,
                inspect_unseen_target_valid_if_source_eligible=True,
                first_eight_is_display_limit_not_full_validator_target_set=True,
                historical_nov6=edge_history,
                in_corpus_reference_position=internal.index(cited.id),
                within_first_eight_display=cited.id in citing.references[:8],
                returned_by_first_expansion_page=cited.id in [p.id for p in corpus.references(citing.id,limit=10)],
                reverse_citation_query_returns_source=citing.id in [p.id for p in corpus.citations(cited.id,limit=10)],
                graph_insertion_of_source_preserves_edge=True,
                graph_payload_delivery='depends on retrieved seed, top-three seeds, 12-node/12-edge context cap and packing',
                all_three_action_paths_observed=False))
    summary=dict(classification='OFFLINE ACCESSIBILITY AUDIT; not hosted inference or human evaluation',
        selected_papers=len(records),quotes_beyond_prefix=sum(not r['quote_in_350_character_prefix'] for r in records),
        reachable_via_recorded_reviewer_query_at_page10=sum(r['allowed_search_can_return_paper'] for r in records),
        initial_question_search_hits=dict(Counter(r['question_id'] for r in records if r['initial_question_search_rank_1_based'])),
        corpus_edges_verified=len(edges),edges_visible_in_eight_display=sum(e['within_first_eight_display'] for e in edges),
        edges_returned_by_expansion_page10=sum(e['returned_by_first_expansion_page'] for e in edges),
        historical_input_policy=sorted({(lambda r:r.get('context_policy_version',r.get('provider_configuration',{}).get('context_policy','historical/unrecorded')))(json.loads((historic/'runs'/Path(c['directory']).name/'result.json').read_text())) for c in native if c['method'].startswith('clm')}),
        current_three_action_live_delivery='NOT EVALUATED; no current experiment executed',
        limitations=['Search witnesses are evaluator observations; they were not supplied to a planner.',
            'Independent INSPECT of an already loaded paper is invalid and cannot reveal a longer abstract.',
            'Graph storage retains full source records and reference edges, but graph observations do not expose complete abstracts.',
            'Archive reads cannot recover source characters never included in an archived selected observation.',
            'Three actions include the mandatory initial search: only two further acquisition actions remain.',
            'STOP is checked after action/token/round guards; no extra free planning call is guaranteed at the action ceiling.',
            'For one chain, initial retrieval of its root plus two expansions can use all three actions; returned metadata is not guaranteed delivery into a subsequent model call.',
            'Results from historical ten-action runs do not establish delivery in the corrected three-action protocol.'])
    for m in methods:
        observations=[o for r in records for o in r['historical_nov6'] if o['method']==m]
        summary.setdefault('historical_by_method',{})[m]=dict(paper_trial_pairs=len(observations),
            retrieved=sum(o['source_retrieved'] for o in observations),
            exact_quote_delivered=sum(o['exact_passage_delivered_to_model'] for o in observations),
            retained_in_last_input=sum(o['retained_in_final_selected_input'] for o in observations),
            graph_nodes=sum(o['graph_node_present'] for o in observations))
    output.mkdir(parents=True)
    atomic_json(output/'paper-accessibility.json',records);atomic_json(output/'citation-edge-accessibility.json',edges)
    atomic_json(output/'summary.json',summary)
    print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['corpus','questions','audit','historical','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();investigate(a.corpus,a.questions,a.audit,a.historical,a.output)
