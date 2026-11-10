"""Shared terminal-only capacity projection of already selected evidence.

Never mutates CLM text, reads archives, inspects authoritative papers, selects a
benchmark answer, or adds evidence. Complete provenance-bearing records remain
atomic. All displacement is durably auditable by the caller.
"""
from copy import deepcopy
import hashlib,json,re
from .context_memory import serialized
from .graph_evidence import extract_edges
from .openalex import BudgetExceeded

VERSION='capacity-aware-final-response-v1'


def _sha(value):return hashlib.sha256(serialized(value).encode()).hexdigest()


def units(value):
    result=[]
    def add(path,data):result.append({'path':path,'value':deepcopy(data),'sha256':_sha(data)})
    def walk(x,path):
        if isinstance(x,dict):
            if x.get('relation')=='PAPER_CITES_PAPER' or any(k in x for k in ('id','work_id','source_identifier')):
                add(path,x)
            else:
                for k,v in x.items():walk(v,path+'/'+k)
        elif isinstance(x,list):
            for i,v in enumerate(x):walk(v,path+'/'+str(i))
        elif isinstance(x,str):
            # CLM observation JSON lines and model notes. No arbitrary byte
            # slicing: provenance JSON objects and verbatim text lines are whole.
            for i,line in enumerate(x.splitlines()):
                if not line.strip():continue
                try:obj=json.loads(line)
                except ValueError:add(path+f'/line/{i}',line)
                else:
                    if isinstance(obj,(dict,list)):walk(obj,path+f'/line/{i}')
                    else:add(path+f'/line/{i}',line)
        elif x is not None:add(path,x)
    walk(value,'')
    return result


def prepare(planner,payload,system):
    original=deepcopy(payload);candidate=deepcopy(payload)
    report={'policy':VERSION,'scope':'terminal request projection only; active context and archives unchanged',
        'before_sha256':_sha(original),'before_capacity':planner.capacity_measurements(original,system=system),
        'controls_removed':[],'evidence_displaced':[],'selected_units':[],
        'new_evidence_loaded':False,'context_edit_committed':False,'recovery_calls':0}
    def finish(status):
        report.update(status=status,after_sha256=_sha(candidate),after_capacity=planner.capacity_measurements(candidate,system=system),
            graph_edges_before=len({(e['source'],e['target']) for e in extract_edges(original)}),
            graph_edges_after=len({(e['source'],e['target']) for e in extract_edges(candidate)}))
        return candidate,report
    if planner.payload_fits(candidate,system=system):return finish('unchanged_fits')
    # No acquisition/edit/recovery is permitted in the final response. These
    # operating controls carry no source evidence and are removed before it.
    for key in ('operation_feedback','context_file','context_budget_chars','pending_observation_chars'):
        if key in candidate:report['controls_removed'].append({'path':'/'+key,'value':candidate.pop(key)})
    protected=candidate.get('protected',{})
    for key in list(protected):
        if key not in ('question','context_management','phase','phase_objective'):
            report['controls_removed'].append({'path':'/protected/'+key,'value':protected.pop(key)})
    if planner.payload_fits(candidate,system=system):return finish('controls_only_compaction')
    evidence={k:candidate[k] for k in ('active_context','observation','recovered_observation') if k in candidate}
    records=units(evidence)
    words=set(re.findall(r'[a-z0-9]+',str(protected.get('question',''))+' '+str(candidate.get('factual_task',{}).get('question','')),flags=re.I))
    words={w.lower() for w in words if len(w)>2}
    def rank(record):
        data=record['value'];tokens=set(re.findall(r'[a-z0-9]+',serialized(data).lower()))
        # Public research/factual question only. No target IDs or answer key.
        kind=0 if isinstance(data,dict) and any(k in data for k in ('abstract_excerpt','excerpt','evidence_quote','year')) else 1 if isinstance(data,dict) and data.get('relation')=='PAPER_CITES_PAPER' else 2
        return (kind,-len(tokens&words),str(data.get('id',data.get('source',''))) if isinstance(data,dict) else '',record['path'])
    for k in evidence:candidate.pop(k,None)
    selected=[]
    field='active_context' if planner.mode=='clm' else 'observation'
    def put():
        body={'selected_evidence':[r['value'] for r in selected]}
        candidate[field]=serialized(body) if field=='active_context' else body
    put()
    if not planner.payload_fits(candidate,system=system):
        report['evidence_displaced']=records
        finish('required_envelope_does_not_fit')
        return None,report
    for record in sorted(records,key=rank):
        selected.append(record);put()
        if not planner.payload_fits(candidate,system=system):
            selected.pop();put();report['evidence_displaced'].append(record)
    report['selected_units']=[{'path':r['path'],'sha256':r['sha256']} for r in selected]
    if records and not selected:
        finish('no_complete_evidence_unit_fits')
        return None,report
    return finish('evidence_projection_compaction')
