"""Exact-passage delivery audit, deliberately separate from truth/support judgments."""
import hashlib
import json
from pathlib import Path

from .models import canonical_id


def text_values(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for child in value.values():yield from text_values(child)
    elif isinstance(value,list):
        for child in value:yield from text_values(child)


def audit_passage(run_directory,paper,passage,*,human_judgment=None):
    if not isinstance(passage,str) or not passage.strip():raise ValueError('An exact independently selected passage is required')
    run=Path(run_directory);result=json.loads((run/'result.json').read_text())
    wid=canonical_id(paper.id)
    corpus_contains=bool(paper.abstract and passage in paper.abstract)
    def contains(value):
        # CLM text can contain a serialized observation, while conventional
        # payloads contain decoded source strings. Compare the same passage.
        return passage in value or json.dumps(passage,ensure_ascii=False)[1:-1] in value
    requests=[json.loads(p.read_text()) for p in sorted((run/'context').glob('request-*.json'))]
    delivered=[];recovered=[]
    for r in requests:
        body=r.get('input',{})
        # Protected questions/proposals and arbitrary output text are not evidence.
        evidence={k:body[k] for k in ('active_context','observation','recovered_observation') if k in body}
        if body.get('protected',{}).get('context_management')=='meta_clm':
            end=1+body.get('measurement_suffix_messages',0)
            evidence['meta_conversation']=body.get('messages',[])[2:-end]
        values=list(text_values(evidence))
        if any(contains(value) for value in values):delivered.append(r['call'])
        recovery=body.get('recovered_observation')
        if recovery and contains(recovery.get('text','')):recovered.append(r['call'])
    last=requests[-1].get('input',{}) if requests else {}
    current={k:last[k] for k in ('active_context','observation') if k in last}
    if last.get('protected',{}).get('context_management')=='meta_clm':
        end=1+last.get('measurement_suffix_messages',0)
        current['meta_conversation']=last.get('messages',[])[2:-end]
    visible_at_measurement=any(contains(value) for value in text_values(current))
    independent=human_judgment if human_judgment and human_judgment.get('independent_human_review') is True else {}
    return dict(paper_id=wid,passage_sha256=hashlib.sha256(passage.encode()).hexdigest(),
        evidence_exists_in_stored_abstract=corpus_contains,source_retrieved=wid in result['inspected_ids'],
        exact_passage_delivered_to_model=bool(delivered),delivery_calls=delivered,
        retained_in_final_selected_input=visible_at_measurement,recovered_into_model_input=bool(recovered),recovery_calls=recovered,
        memory_failure_eligible=corpus_contains and wid in result['inspected_ids'] and bool(delivered),
        factual_answer_correct=independent.get('factually_correct'),cited_source_supports_answer=independent.get('citation_support_correct'),
        input_policy='Stored abstracts and actual selected/recovered request text only; matching alone does not certify semantic support',
        standard_nonagentic=not bool(requests))
