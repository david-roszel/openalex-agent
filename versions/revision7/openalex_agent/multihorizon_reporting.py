"""Saved-output V3 exports and mechanical freeze gates, never hosted inference."""
from pathlib import Path
from collections import Counter
import csv,json,hashlib,math
from decimal import Decimal
from .corpus import atomic_json
from .multihorizon_metrics import comparisons,architecture_performance


def _csv(path,rows,fields):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    tmp.replace(path)


def saved_rows(out):
    from .multihorizon import queue
    out=Path(out);rows=[]
    spec=json.loads((out/'protocol.json').read_text())
    ledger=json.loads((Path(spec['accounting']['shared_directory'])/'spending-ledger.json').read_text());attempts={}
    for a in ledger['attempts']:attempts.setdefault(a['phase'].split(':',1)[0],[]).append(a)
    with queue(out) as db:jobs=[dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY ordinal')]
    for j in jobs:
        job=json.loads(j['payload']);p=out/'cells'/job['id']/'summary.json'
        row=json.loads(p.read_text()) if p.exists() else {**job,'primary_score':None,'attempted':False}
        if not p.exists():
            cell=out/'cells'/job['id']
            row['phases']=[json.loads(f.read_text()) for f in sorted(cell.glob('phase-*.json'))]
            row['checkpoints']=[json.loads(f.read_text()) for f in sorted(cell.glob('checkpoint-*.json'))]
        row.update(queue_state=j['state'],status=j['state'],error=j['error'],started=j['started'],ended=j['ended'])
        execution=out/'cells'/job['id']/'execution.json'
        if execution.exists():
            receipt=json.loads(execution.read_text())
            row.update(execution_session=receipt['session'],execution_worker=receipt['worker'],execution_workers=receipt['configured_workers'],runtime_snapshot_sha256=receipt['snapshot_sha256'],trajectory_wall_seconds=receipt['trajectory_wall_seconds'])
        if j['state'] in ('FAILED','INTERRUPTED') and not p.exists():
            # A prepared JSON input is not evidence that inference was sent.
            # Financial pre-send refusal must stay missing, not become zero.
            if job['method']!='standard':row['attempted']=any((a.get('diagnostic') or {}).get('request_not_sent') is not True for a in attempts.get(job['id'],[]))
            else:
                states=[json.loads(f.read_text()) for f in (out/'cells'/job['id']).rglob('state.json') if 'snapshot' not in str(f)]
                row['attempted']=any(s.get('actions',0)>0 for s in states)
        accounts=attempts.get(job['id'],[])
        row['conservative_retained_exposure_usd']=str(sum((Decimal(a['retained']) for a in accounts),Decimal(0)))
        row['cost_accounted']=all(a.get('charge') is not None for a in accounts)
        row['unknown_usage']=sum(a['status']=='UNKNOWN' for a in accounts)
        rows.append(row)
    return rows


def export(out):
    out=Path(out);rows=saved_rows(out);main=[r for r in rows if r['phase']=='main'];spec=json.loads((out/'protocol.json').read_text())
    fields=['id','phase','study','task_id','domain','method','repetition','trial_identifier','status','attempted','primary_score','exposure_complete','actual_exposures','actual_substantive_actions','acquisition_actions','early_stops','provider_attempts','successful_http_requests','known_input_tokens','known_output_tokens','reported_reasoning_tokens','conservative_cost_usd','conservative_retained_exposure_usd','cost_accounted','unknown_usage','per_request_latency_seconds','graph_delivery_calls','verified_graph_edges_transmitted','graph_tokens_transmitted','graph_capacity_exclusions','context_edits_committed','modified_context_delivery_calls','accepted_recoveries','context_limit_events','context_pressure_events','evidence_displacement_events','fallback_events','final_context_compactions','final_capacity_failures','latency_mean_seconds','latency_median_seconds','latency_p95_seconds','latency_max_seconds','seconds','mocked_generation','error','protocol_sha256']
    fields += ['execution_session','execution_worker','execution_workers','runtime_snapshot_sha256','trajectory_wall_seconds']
    fields += ['collection_status','raw_primary_score','operational_itt_score','required_final_response','final_response_status','factual_response_success','overall_failure_reason','complete_case_eligible','factual_response_status','valid_llm_run','response_contract_valid','memory_item_status','memory_item_counts',
        'verified_edge_recall','chain_complete','legacy_edge_metric_meaning','acquired_directed_edge_recall','delivered_directed_edge_recall','directed_edge_gold_denominator','reuse_probes_correct','reuse_probes_expected','acquisition_record_limit','records_per_acquisition']
    fields += ['returned_records','unique_acquired_metadata_records','explicit_inspection_actions','publications_delivered_to_model','publications_in_CLM_inputs','publications_in_recovery_inputs']
    _csv(out/'RESULTS.csv',rows,fields)
    effects=comparisons(main);atomic_json(out/'TASK_EFFECTS.json',effects)
    performance=architecture_performance(main);atomic_json(out/'ARCHITECTURE_PERFORMANCE.json',performance)
    _csv(out/'ARCHITECTURE_PERFORMANCE.csv',performance,['study','policy','method','tasks','mean','ci95','independent_unit','scope'])
    _csv(out/'TASK_EFFECTS.csv',effects,['study','policy','contrast','tasks','mean','ci95','sign_flip_p','holm_within_study','holm_across_studies','independent_unit','caveat'])
    phases=[];checkpoints=[]
    for r in rows:
        for p in r.get('phases',[]):phases.append({**{k:r[k] for k in ('id','task_id','domain','method','repetition')},'collection_phase':r['phase'],**p})
        for c in r.get('checkpoints',[]):
            for branch,b in c['branches'].items():checkpoints.append({**{k:r[k] for k in ('id','phase','task_id','domain','method','repetition')},'papers':c['papers_scheduled'],'actual_exposures':c['actual_exposures'],'branch':branch,'primary_score':b['scoring']['primary_score'],'operational_itt':b['scoring']['operational_itt'],'recovery_operations':b['recovery_operations'],'model_calls':b['model_calls'],'snapshot_unchanged':b['snapshot_unchanged'],'error':b['error'],
                'item_status':b['scoring'].get('item_status'),'item_counts':b['scoring'].get('item_counts'),'calibration_score':b['scoring'].get('evidence_calibration_score'),'response_status':b['scoring'].get('response_status'),'response_contract_valid':b['scoring'].get('valid_final_response')})
    _csv(out/'PHASES.csv',phases,['id','collection_phase','task_id','domain','method','repetition','phase','actual_substantive_actions','acquisition_actions','accepted_distinct_recoveries','new_evidence_sources','evidence_carryover_sources','model_calls','stop','premature_stop','objective_complete','reason','seconds','fallbacks','returned_records','acquired_metadata_records','newly_admitted_records','explicit_inspection_actions','acquisition_record_limit','records_per_acquisition','early_evidence_reuse_correct','reuse_source_previously_delivered','publications_delivered_in_requests','publications_in_CLM_inputs','publications_in_recovery_inputs','actual_provider_request_fixtures','context_framing_displaced_records'])
    _csv(out/'MEMORY_CHECKPOINTS.csv',checkpoints,['id','phase','task_id','domain','method','repetition','papers','actual_exposures','branch','primary_score','operational_itt','recovery_operations','model_calls','snapshot_unchanged','error','item_status','item_counts','calibration_score','response_status','response_contract_valid'])
    curve_effects=[]
    for cp in (24,48,72,96):
        for branch in ('A','B'):
            observation_rows=[]
            for row in main:
                if row['study']!='M':continue
                checkpoint=next((c for c in row.get('checkpoints',[]) if c['papers_scheduled']==cp),None)
                if checkpoint is None or branch not in checkpoint['branches']:continue
                b=checkpoint['branches'][branch];scoring=b['scoring']
                observation_rows.append({**{k:row[k] for k in ('task_id','method','domain','repetition')},'study':'M','status':'COMPLETED' if scoring.get('scoreable_response',scoring.get('valid_final_response')) and checkpoint['actual_exposures']==cp else 'FAILED','attempted':scoring.get('status')!='NOT RUN','primary_score':scoring['primary_score']})
            for effect in comparisons(observation_rows):
                if effect['study']=='M':curve_effects.append({**effect,'checkpoint_papers':cp,'branch':branch,'secondary_correlated_endpoint':not(cp==96 and branch=='A')})
    atomic_json(out/'MEMORY_CURVE_EFFECTS.json',curve_effects)
    primary=[e for e in effects if e['policy']=='operational_itt']
    (out/'PAPER_TABLES.tex').write_text('%% Development-only V3; task-clustered effects, no human relevance/semantic truth judgments.\n\\begin{tabular}{llrr}\nStudy & Contrast & N tasks & Effect \\\\\n'+''.join(f"{e['study']} & {e['contrast']} & {e['tasks']} & "+(f"{e['mean']:.4f}" if e['mean'] is not None else 'Not evaluated')+' \\\\\n' for e in primary)+'\\end{tabular}\n')
    counts={s:dict(Counter(r['status'] for r in main if r['study']==s)) for s in ['S','L','M']}
    audit={'version':spec['version'],'core_planned':636,'study_counts':counts,'pilot_counts':dict(Counter(r['status'] for r in rows if r['phase']=='pilot')),'source_protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),
        'human_relevance':'NOT EVALUATED','semantic_factual_accuracy':'NOT EVALUATED','scored_frozen':(out/'SCORED-FREEZE.json').exists(),'independent_task_counts':{'S':30,'L':24,'M':12},'read_only_dashboard_inputs':['RESULTS.csv','TASK_EFFECTS.json','PHASES.csv','MEMORY_CHECKPOINTS.csv']}
    atomic_json(out/'progress.json',audit)
    paragraphs=[]
    for study in ['S','L','M']:
        observed=[r for r in performance if r['study']==study and r['policy']=='operational_itt' and r['mean'] is not None]
        paragraphs.append('Study '+study+': '+('; '.join(r['method']+' '+f"{r['mean']:.4f}"+' (N='+str(r['tasks'])+' independent tasks)' for r in observed) if observed else 'NOT EVALUATED: no complete matched task blocks.') )
    (out/'RESULTS_SECTION.md').write_text('# Separate development-study results data\n\n'+'\n\n'.join(paragraphs)+'\n\nNo independent human relevance, semantic factual accuracy or hallucination-reduction claim is established. See task-level uncertainty, failures, action counts, exposure and cost tables; V1/V2 are separate.\n')
    (out/'BENCHMARK_REPORT.md').write_text('# Multi-Horizon V3 saved-output status\n\n'+json.dumps(audit,indent=2)+'\n\nS: target-only Recall@10. L: acquired source/directed-edge checks. M: delayed exact answer accuracy, checkpoint96 branch A. These endpoints are not pooled. Pending cells are missing; attempted failed finals use operational ITT zero with complete-case sensitivity. Repetitions, phases, papers and checkpoint branches are not independent tasks. V1/V2 are separate. No semantic hallucination or independent human-relevance claim is inferred.\n')
    # Figures require actual complete paired task data; never plot pending as0.
    available=[e for e in primary if e['mean'] is not None]
    if available:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figures=out/'FIGURES';figures.mkdir(exist_ok=True)
        for study in ['S','L','M']:
            es=[e for e in available if e['study']==study]
            if not es:continue
            fig,ax=plt.subplots(figsize=(5.5,3.2));ax.bar([e['contrast'] for e in es],[e['mean'] for e in es],color=['#244B3B','#A7794C','#657A3F'][:len(es)])
            for i,e in enumerate(es):
                if e['ci95'] is not None:ax.plot([i,i],e['ci95'],color='#29332D')
            ax.axhline(0,color='#786B57',lw=.8);ax.set_title(f'Study {study}: matched task effects');ax.set_ylabel('Paired primary endpoint difference');fig.tight_layout()
            for ext in ['png','svg','pdf']:fig.savefig(figures/f'{study}-factorial.{ext}',dpi=200)
            plt.close(fig)
    return audit


def pilot_gates(out):
    out=Path(out);rows=[r for r in saved_rows(out) if r['phase']=='pilot'];required=len(rows)
    complete=[r for r in rows if r['status']=='COMPLETED' and not r.get('mocked_generation',True)]
    live=[r for r in complete if r['method']!='standard'];reasons=[]
    if len(complete)!=required:reasons.append('All disjoint pilot cells must complete with unmocked receipts; mocked engineering outputs never satisfy live validation')
    long=[r for r in live if r['study']=='L']
    sustained=[r for r in long if r.get('phase_count')==6 and sum(p['actual_substantive_actions']>=3 for p in r.get('phases',[]))>=3]
    if not long or len(sustained)<math.ceil(len(long)/2):reasons.append('L pilots have not established sustained multi-phase substantive research')
    pressured=[r for r in long if r.get('context_pressure_events',0)>0 or r.get('evidence_displacement_events',0)>0]
    if not long or len(pressured)<math.ceil(len(long)/2):reasons.append('L pilots have not established measured context competition or evidence displacement')
    mem=[r for r in live if r['study']=='M']
    for r in mem:
        if not r.get('exposure_complete') or r.get('actual_exposures')!=96 or len(r.get('checkpoints',[]))!=4:reasons.append('M pilot lacks96 actual exposures/four isolated checkpoints')
        if not any(b.get('context_evictions') or b.get('pending_observation_chars') or any(e.get('kind')=='context_admission' and e.get('status')=='pending' for e in b.get('context_events',[])) for b in r.get('batches',[])):reasons.append('M pilot lacks demonstrated context competition/displacement')
        if not all(c.get('snapshot_unchanged') and all(b['recovery_operations']<=2 for b in c['branches'].values()) for c in r.get('checkpoints',[])):reasons.append('M fork/recovery safeguards not established')
        if not all(set(c['branches'])=={'A','B'} and all(b.get('scoring',{}).get('valid_final_response') is True and b.get('error') is None for b in c['branches'].values()) for c in r.get('checkpoints',[])):reasons.append('M pilot final-answer schema/capacity did not pass both branches at every checkpoint')
    if not mem:reasons.append('No eligible unmocked M pilot')
    for method in ('llm_graph','clm_graph'):
        arm=[r for r in live if r['method']==method]
        if not any(r.get('graph_delivery_calls',0)>0 for r in arm):reasons.append('No actual verified graph-edge input delivery for '+method)
    for method in ('clm','clm_graph'):
        arm=[r for r in live if r['method']==method]
        if not any(r.get('context_edits_committed',0)>0 and r.get('modified_context_delivery_calls',0)>0 for r in arm):reasons.append('No committed model-initiated context editing for '+method)
    if any(r.get('final_capacity_failures',0)>0 or r.get('study')=='S' and r.get('method')!='standard' and r.get('factual_response_status')!='valid' for r in complete):reasons.append('Final-response construction/format did not pass all required pilots')
    if any(r.get('unknown_usage') for r in rows):reasons.append('Unaccountable provider usage')
    result={'status':'PASS' if not reasons else 'BLOCKED','expected_pilot_cells':required,'completed_unmocked_cells':len(complete),'mechanical_reasons':sorted(set(reasons)),
        'relative_main_scores_inspected':False,'pilot_scores_used_for_task_tuning':False,'semantic_truth':'NOT EVALUATED'}
    atomic_json(out/'PILOT_GATES.json',result);return result


def freeze(out,review):
    from .multihorizon import check,sha
    out=Path(out);check(out);gates=pilot_gates(out)
    if gates['status']!='PASS':raise ValueError('Scored freeze blocked: '+str(gates['mechanical_reasons']))
    r=json.loads(Path(review).read_text())
    if r.get('protocol_sha256')!=sha(out/'protocol.json') or r.get('verified') is not True or not r.get('reviewer') or not r.get('reviewed_at') or not r.get('source'):raise ValueError('Genuine external objective-source audit required; no review is fabricated')
    target=out/'SCORED-FREEZE.json'
    if target.exists():raise ValueError('Scored freeze is immutable')
    result={'protocol_sha256':sha(out/'protocol.json'),'external_source_review_sha256':sha(review),'pilot_gates_sha256':sha(out/'PILOT_GATES.json'),'scored_source_and_task_freeze':True,'no_heldout_or_human_relevance_approval_implied':True}
    atomic_json(target,result);target.chmod(0o444);return result
