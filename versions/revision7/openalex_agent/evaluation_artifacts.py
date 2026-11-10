"""Read-only presentation adapter over existing benchmark/context/trace readers.

No model, collection, scoring, approval or mutation entry point is called here.
Counters describe observable records; they are not new scientific estimators.
"""
import json
import math
from pathlib import Path

from .controlled import ARCHITECTURES, GRAPH_FIELDS
from .experiment_dashboard import (allowed_path, read_comparison, context_records,
                                   trace_records, safe_json, saved_experiments)


def number(value):
    if isinstance(value,bool):return None
    try:value=float(value)
    except (ValueError,TypeError):return None
    return value if math.isfinite(value) else None


def _count(value):
    n=number(value)
    return int(n) if n is not None and n>=0 and n.is_integer() else None


def _mapping(value):
    return value if isinstance(value,dict) else {}


def _records(value):
    return [r for r in value if isinstance(r,dict)] if isinstance(value,list) else []


def _views(payload):
    """Only actual evidence payloads, excluding protected proposals/instructions."""
    views=[];unstructured=False
    payload=_mapping(payload)
    for key in ('observation','active_context','recovered_observation'):
        value=payload.get(key)
        if key=='recovered_observation' and isinstance(value,dict):value=value.get('text')
        if isinstance(value,dict):views.append(value)
        elif isinstance(value,str) and value.strip():
            parsed=False
            for index,text in enumerate([value,*value.splitlines()]):
                try:obj=json.loads(text)
                except (ValueError,TypeError):continue
                if isinstance(obj,dict):
                    view=obj.get('changes',obj)
                    if isinstance(view,dict):views.append(view);parsed=True
                    if index==0:break
            unstructured|=not parsed
    return views,unstructured


def graph_delivery(requests):
    counters=dict(structured_graph_calls=0,candidate_calls=0,citation_edge_calls=0,
                  action_target_overlap_calls=0,unstructured_context_calls=0)
    examples=[]
    for item in requests:
        record=item.get('request',item);payload=record.get('input',{})
        views,unstructured=_views(payload);counters['unstructured_context_calls']+=unstructured
        graph=[{k:v[k] for k in GRAPH_FIELDS if k in v} for v in views if isinstance(v,dict) and any(v.get(k) for k in GRAPH_FIELDS)]
        if not graph:continue
        counters['structured_graph_calls']+=1
        candidates=[c for v in graph for c in _records(v.get('candidates'))]
        edges=[e for v in graph for e in _records(v.get('subgraph'))]
        counters['candidate_calls']+=bool(candidates);counters['citation_edge_calls']+=bool(edges)
        output=record.get('output') or {};decision=output.get('decision',{}) if isinstance(output,dict) else {}
        decision=_mapping(decision)
        target=_mapping(decision.get('arguments')).get('paper_id')
        overlap=bool(target and any(target==c.get('id',c.get('paper_id')) for c in candidates))
        counters['action_target_overlap_calls']+=overlap
        examples.append(dict(call=record.get('call'),delivered=graph,decision=decision,
                             target_in_delivered_candidates=overlap))
    return {**counters,'examples':examples}


def _run(directory,root,row):
    warnings=[]
    if not directory:
        return dict(result={},events=[],requests=[],transitions=[],warnings=['Run artifacts unavailable'],
                    graph_delivery=graph_delivery([]),graph_snapshot={},progression=[],rankings=[],directory=None)
    try:trace=trace_records(directory,root)
    except (OSError,ValueError,KeyError,TypeError):trace=dict(events=[],result={},state={});warnings.append('Trace unavailable or incomplete')
    try:context=context_records(directory,root)
    except (OSError,ValueError,KeyError,TypeError):context=dict(requests=[],transitions=[]);warnings.append('Context snapshots unavailable or incomplete')
    result=trace.get('result') or {}
    if not isinstance(result,dict):result={}
    if not result:warnings.append('No completed result; recorded partial evidence only')
    events=[e for e in trace.get('events',[]) if isinstance(e,dict)]
    snapshot=safe_json(Path(directory)/'graph.json',root,{})
    snapshot=snapshot if isinstance(snapshot,dict) else {}
    nodes=snapshot.get('nodes',[]);edges=snapshot.get('edges',snapshot.get('links',[]))
    if not isinstance(nodes,list) or not isinstance(edges,list):
        snapshot={};nodes=[];edges=[];warnings.append('Graph snapshot is malformed')
    nodes=[n for n in nodes if isinstance(n,dict)]
    stats=dict(nodes=len(nodes),edges=len(edges),
               retrieved_papers=sum(n.get('type')=='paper' and n.get('resolved') is True for n in nodes),
               unresolved_papers=sum(n.get('type')=='paper' and n.get('resolved') is False for n in nodes)) if snapshot else {}
    progression=[dict(event=e.get('event_seq',i),actions=e.get('actions'),inspected=e.get('inspected'),
        new_papers=e.get('new_papers'),retrieved=e.get('retrieved')) for i,e in enumerate(events)
        if e.get('kind')=='observation']
    if not progression:
        progression=[dict(event=p.get('actions',i),actions=p.get('actions'),inspected=p.get('inspected'),
            new_papers=p.get('new_papers'),retrieved=p.get('retrieved')) for i,p in enumerate(result.get('trace',[])) if isinstance(p,dict)]
    return dict(result=result,events=events,requests=context.get('requests',[]),transitions=context.get('transitions',[]),
        warnings=warnings,graph_delivery=graph_delivery(context.get('requests',[])),graph_snapshot=stats,
        progression=progression,rankings=result.get('ranked',[]),directory=str(directory))


def discover(root):
    """Existing catalog discovers paths only; selected contents load separately."""
    paths=saved_experiments(root)
    frozen=[]
    for p in (Path(root)/'data').rglob('unscored-freeze.json'):
        try:frozen.append(str(allowed_path(p.parent,root).relative_to(Path(root).resolve())))
        except ValueError:continue
    return list(dict.fromkeys([*frozen,*paths]))


def _cell(row):
    return (str(row.get('query_id')),row.get('method'),row.get('variant','full'),
            _count(row.get('trial')),_count(row.get('action_budget')))


def _pooled_scores(selected,assessment,root):
    """Read recorded scores only; match existing policy/approval/freeze pins."""
    from .pooled_review import VERSION, POLICY, digest, sha, load_freeze
    freeze,_=load_freeze(selected)
    score=safe_json(allowed_path(assessment,root)/'scores.json',root,{})
    if not isinstance(score,dict):raise ValueError('Malformed recorded assessment')
    pins=score.get('immutable_inputs',{});approval=score.get('scientific_approval',{})
    if not isinstance(pins,dict) or not isinstance(approval,dict):raise ValueError('Malformed assessment provenance')
    if (score.get('protocol')!=VERSION or approval.get('protocol')!=VERSION
        or approval.get('stage')!='score-preserved' or approval.get('approved') is not True
        or not approval.get('approver') or not approval.get('approved_at') or approval.get('pins')!=pins
        or pins.get('frozen_outputs_sha256')!=sha(selected/'unscored-freeze.json')
        or pins.get('policy_sha256')!=digest(POLICY) or freeze.get('policy')!=POLICY):
        raise ValueError('Recorded assessment does not match this approved freeze and pooled policy')
    rows=score.get('rows',[])
    if not isinstance(rows,list) or not rows or len({_cell(r) for r in rows})!=len(rows):raise ValueError('Missing or duplicate scored cells')
    return {_cell(r):r for r in rows}


def load_evaluation(directory,root,assessment=None):
    root=Path(root).resolve();selected=allowed_path(directory,root)
    native=selected/'raw-output' if (selected/'unscored-freeze.json').is_file() else selected
    try:loaded=read_comparison(native,root)
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        loaded=dict(manifest={},rows=[],partial_runs=[],relevance_evaluated=False,factual_evaluated=False,category='UNVERIFIED SAVED EXPERIMENT')
    manifest=loaded['manifest'];rows=[];excluded=set();warnings=[]
    if not isinstance(manifest,dict):manifest={};warnings.append('Manifest is incomplete or malformed')
    for row in loaded['rows']:
        if row.get('method') not in ARCHITECTURES:
            excluded.add(str(row.get('method')));continue
        if not row.get('query_id') or any(_count(row.get(k)) is None for k in ('trial','action_budget')):
            warnings.append('Malformed cell identity omitted; no trial or budget was imputed');continue
        try:run=_run(row.get('artifact_directory'),root,row)
        except (OSError,ValueError,KeyError,TypeError,AttributeError):
            run=_run(None,root,row)
        rows.append({**row,'detail':run})
    if excluded:warnings.append('Exploratory/nonprimary cells excluded from this view: '+', '.join(sorted(excluded)))
    if manifest.get('status')!='complete':warnings.append('Incomplete experiment: show available cells without imputing missing outcomes')
    partial=[]
    for path in loaded.get('partial_runs',[]):
        try:partial.append(_run(path,root,{}))
        except (OSError,ValueError,TypeError):continue
    pooled=False
    if assessment:
        try:
            scores=_pooled_scores(selected,assessment,root)
            if set(scores)!={_cell(r) for r in rows}:raise ValueError('Scored and native cell identities differ')
            for row in rows:
                score=scores.get(_cell(row),{})
                row.update(ndcg_at_10=score.get('ndcg_at_10_pooled'),precision_at_10=score.get('precision_at_10'),
                           pool_coverage=score.get('pool_coverage'),judgment_coverage_at_10=score.get('judgment_coverage_at_10'))
            loaded['relevance_evaluated']=pooled=True
        except (OSError,ValueError,KeyError,TypeError):
            warnings.append('Pooled assessment was not accepted: matching frozen outputs, policy and recorded approval are required')
    synthetic=manifest.get('deterministic_smoke') is True
    questions=manifest.get('questions',[])
    questions=[q for q in questions if isinstance(q,dict)] if isinstance(questions,list) else []
    classification=('Synthetic diagnostics' if synthetic else 'Recorded independently scored findings'
        if loaded.get('relevance_evaluated') or loaded.get('factual_evaluated') else
        'Unscored development results' if questions and all(q.get('split')=='development' for q in questions) else
        'Observed behavior · independent evaluation unavailable')
    policies=set()
    for row in rows:
        result=row['detail']['result'];provider=result.get('provider_configuration') or {}
        provider=provider if isinstance(provider,dict) else {}
        policy=result.get('context_policy_version',provider.get('context_policy'))
        if row['method'].startswith('clm') and isinstance(policy,str) and policy:policies.add(policy)
    policies=sorted(policies)
    if classification=='Unscored development results' and policies and policies!=['controlled-context-v2-capacity-stop']:
        classification='Historical unscored development results'
    # Do not promote raw claims or unapproved files into scientific metrics.
    return dict(directory=str(selected),native_directory=str(native),manifest=manifest,rows=rows,
        questions=questions,classification=classification,
        relevance_evaluated=bool(loaded.get('relevance_evaluated')),
        factual_evaluated=bool(loaded.get('factual_evaluated')),synthetic=synthetic,
        warnings=warnings,partial_runs=partial,excluded_methods=sorted(excluded),
        source_status=manifest.get('status','not recorded'),pooled_assessment=pooled,context_policies=policies)


def summary_row(row):
    result=row['detail']['result'];usage=_mapping(result.get('llm_usage'));events=row['detail']['events']
    baseline=row['method']=='standard'
    transitions=_records(row['detail']['transitions']);delivery=row['detail']['graph_delivery']
    committed=sum(t.get('source')=='model' and (t.get('committed') is True or t.get('status')=='committed') for t in transitions)
    recoveries=[e for e in events if e.get('kind')=='context_recovery']
    sources=set()
    for r in row['detail']['requests']:
        for view in _views(r['request'].get('input',{}))[0]:
            sources.update(p['id'] for p in _records(view.get('top_papers')) if p.get('id') and p.get('abstract_excerpt'))
    unreported=_count(usage.get('usage_unreported_calls'))
    return dict(method=row['method'],actions=_count(row.get('actions',result.get('actions'))),
        inspected=_count(row.get('inspected')),model_calls=_count(row.get('llm_calls',usage.get('calls'))),
        input_tokens=None if unreported else number(row.get('input_tokens',usage.get('input_tokens'))),
        output_tokens=None if unreported else number(row.get('output_tokens',usage.get('output_tokens'))),
        reported_tokens=number(row.get('total_reported_tokens')),
        estimated_tokens=number(row.get('estimated_unreported_tokens')),
        mean_latency=number(row.get('mean_response_seconds')),runtime=number(row.get('end_to_end_seconds')),
        cost=number(row.get('total_api_cost_usd')) if row.get('cost_known') in (True,'True') else None,
        fallbacks=_count(row.get('fallbacks')),failures=_count(row.get('llm_failures',usage.get('failures'))),
        committed_edits=committed if transitions else (0 if not row['method'].startswith('clm') and result else None),
        recovery_attempts=len(recoveries) if events else None,
        accepted_recoveries=sum(e.get('accepted') is True for e in recoveries) if events else None,
        graph_payload_calls=delivery['structured_graph_calls'] if row['detail']['requests'] else None,
        candidate_payload_calls=delivery['candidate_calls'] if row['detail']['requests'] else None,
        citation_payload_calls=delivery['citation_edge_calls'] if row['detail']['requests'] else None,
        ndcg=number(row.get('ndcg_at_10')),precision=number(row.get('precision_at_10')),
        factual_accuracy=None if baseline else number(row.get('task_overall_answer_accuracy')),
        supported_claims=None if baseline else number(row.get('supported_claim_rate')),
        unsupported_claims=None if baseline else number(row.get('unsupported_claim_rate')),
        retention=None,stop_reason=result.get('stop_reason',row.get('stop_reason','not recorded')),
        excerpt_sources=len(sources) if row['detail']['requests'] else None,
        pending_chars=_count(result.get('pending_observation_chars')),
        graph_enabled=result.get('graph_enabled',row.get('graph_enabled')),
        context_policy=result.get('context_policy_version',_mapping(result.get('provider_configuration')).get('context_policy','not recorded')))


def timeline(rows):
    result=[]
    for row in rows:
        method=row['method']
        for index,event in enumerate(row['detail']['events']):
            kind=event.get('kind','unknown')
            family=('Failure' if kind in ('invalid_decision','controller_error','request_error','context_capacity_failure') else
                    'Context' if kind in ('context_edit','context_recovery','context_admission','context_management_required','context_termination','context_budget_stop') else
                    'Planning' if kind in ('context_model_call','planner_call') else
                    'Retrieval' if kind in ('decision','observation','request','cache_hit') else 'Control')
            result.append(dict(id=f"{row.get('run_id',method)}:{index}",method=method,
                sequence=event.get('event_seq',index+1),kind=kind,family=family,
                action=event.get('action',''),timestamp=event.get('timestamp'),record=event,
                summary=event.get('reason',event.get('error_code',kind))))
    return result
