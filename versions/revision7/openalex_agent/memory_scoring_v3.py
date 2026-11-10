"""Revision7 itemwise memory measurement. Original reference V2 is unchanged."""
import json
from collections.abc import Mapping

VERSION='memory-itemwise-abstention-v3-r7'
QUESTIONS=tuple(f'q{i}' for i in range(1,7))
FIELDS={'q1':{'work_id'},'q2':{'year'},'q3':{'work_id'},
        'q4':{'citing_work_id','cited_work_id'},'q5':{'path'},'q6':{'status'}}


def work(value):return isinstance(value,str) and value.startswith('W') and len(value)>1 and value[1:].isdigit()


def classify(q,value):
    if value is None:return 'abstained'
    if not isinstance(value,Mapping) or set(value)!=FIELDS[q]:return 'malformed'
    if all(v is None for v in value.values()):return 'abstained'
    if q in ('q1','q3'):valid=work(value['work_id'])
    elif q=='q2':valid=type(value['year']) is int and 1<=value['year']<=2100
    elif q=='q4':valid=all(work(v) for v in value.values())
    elif q=='q5':valid=isinstance(value['path'],list) and len(value['path'])==3 and all(work(v) for v in value['path'])
    else:valid=value['status'] in ('SUPPORTED','NOT_ESTABLISHED')
    return 'answered' if valid else 'malformed'


def score_response(response,gold,*,terminal_failure=False):
    from .memory_scoring_v2 import _valid_answer
    if not _valid_answer(gold):raise ValueError('Invalid frozen gold key')
    parsed=None;error=None
    if not terminal_failure:
        try:
            def unique(pairs):
                out={}
                for k,v in pairs:
                    if k in out:raise ValueError('duplicate_json_key')
                    out[k]=v
                return out
            parsed=json.loads(response,object_pairs_hook=unique) if isinstance(response,str) else response
        except (ValueError,TypeError):error='unparseable_or_duplicate_json'
    scoreable=isinstance(parsed,Mapping) and not terminal_failure
    statuses={};items={}
    for q in QUESTIONS:
        status='missing' if not scoreable or q not in parsed else classify(q,parsed[q])
        if status=='answered':status='correct' if parsed[q]==gold[q] else 'incorrect'
        statuses[q]=status;items[q]=int(status=='correct')
    valid=scoreable and set(parsed)==set(QUESTIONS) and all(v not in ('missing','malformed') for v in statuses.values())
    counts={s:sum(v==s for v in statuses.values()) for s in ('correct','incorrect','abstained','malformed','missing')}
    return {'scoring_policy':VERSION,'valid_final_response':bool(valid),'scoreable_response':bool(scoreable),
        'terminal_failure':bool(terminal_failure or not scoreable),'response_status':'valid' if valid else 'partial' if scoreable else 'missing' if response is None else 'malformed',
        'parse_error':error,'unexpected_keys':sorted(str(k) for k in parsed if k not in QUESTIONS) if scoreable else [],
        'item_status':statuses,'item_counts':counts,'items':items,
        'primary_score':sum(items[q] for q in QUESTIONS[:5])/5,
        'source_identity_year_score':sum(items[q] for q in QUESTIONS[:3])/3,
        'citation_relation_score':(items['q4']+items['q5'])/2,'evidence_calibration_score':items['q6']}
