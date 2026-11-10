"""Six-phase externally staged research using existing scholarly execution.

STOP closes a phase; evidence, completed operations and context survive. This
is staged autonomous research, not an unconstrained natural-stop trajectory.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json,time
from .controlled import ControlledController
from .controller import Controller,Budget
from .context_planner import ContextPlanner,InferenceLimits
from .llm import DeterministicPlanner,PlannerError
from .actions import Action,make_action
from .openalex import BudgetExceeded
from .storage import Database
from .corpus import atomic_json
from .multihorizon_metrics import source_edge_checks
from .multihorizon_state import snapshot

PHASE_ACTIONS=6
PHASE_GENERATIONS=13
TOTAL_GENERATIONS=78
TOTAL_TOKENS=600000


class PhaseController(ControlledController):
    def _hard_stop(self):
        reason=super()._hard_stop()
        return 'max_acquired_records' if reason=='max_inspected' and getattr(self,'revision7_resources',False) else reason
    def _common_context(self):
        result=super()._common_context()
        if getattr(self,'revision7_resources',False):
            result['remaining']['acquisition_records']=result['remaining'].pop('inspections')
            result['counts']['acquired_metadata_records']=result['counts'].pop('inspected')
            result['counts']['explicit_inspection_actions']=sum(a['action']=='INSPECT_PAPER' for a in self.history)
        return result
    def context(self):
        context=super().context();objective=self.phase['objective']
        if getattr(self,'revision7_resources',False):objective+=' '+self.phase.get('reuse_question','')
        context.update(phase=self.phase['index'],phase_objective=objective)
        if getattr(self,'revision7_resources',False):
            from .context_memory import serialized
            from .token_budget import count
            protected,body=self.planner._parts(context);protected['context_management']='clm'
            index=len(self.planner.store.observations);protected['archive_count']=index+1
            dropped={}
            def fits():
                _,selected=self.planner._parts(context)
                delta='\n'+serialized({'observation':index,'changes':selected})
                envelope={'protected':protected,'active_context':delta,'context_file':'active_context.txt','pending_observation_chars':len(delta),'context_budget_chars':16000}
                return len(serialized(envelope))+128<=16000 and count(delta)+32<=4096
            while not fits():
                for name,floor in [('candidates',4),('recent_actions',1),('observed_metadata',2),('top_papers',2),('candidates',0),('recent_actions',0),('proposals',1),('observed_metadata',0),('top_papers',1)]:
                    if len(context.get(name,[]))>floor:
                        context[name].pop();dropped[name]=dropped.get(name,0)+1
                        if name=='proposals':protected['proposals']=context[name]
                        break
                else:raise ValueError('Required phase envelope cannot admit one complete observation')
            if dropped:
                record={'phase':self.phase['index'],'policy':'shared-empty-CLM-frame-r7','displaced_records':dropped,'limits_unchanged':True}
                self.framing_events=getattr(self,'framing_events',[])+[record];self._log('phase_context_framing',record)
        return context


def _phase_complete(gold,index,papers):
    items=gold['phase_objectives'][index-1];checks=[]
    for item in items:
        if isinstance(item,list):checks.append(item[0] in papers and item[1] in papers and item[1] in papers[item[0]].references)
        else:checks.append(item in papers)
    return all(checks)


def execute(case,method,directory,corpus,factory,gold,*,cancel_check=lambda:None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);started=time.perf_counter();phases=[];events=[];controller=None
    budget=Budget(max_actions=36,max_rounds=40,max_inspected=200,max_api_requests=50,max_llm_calls=TOTAL_GENERATIONS,page_size=10,output_k=20,patience=3,max_context_chars=16000)
    revised=case.get('protocol_revision')=='multihorizon-scientific-r7'
    if revised:budget=replace(budget,page_size=5)
    with Database(str(directory/'metadata.sqlite')) as db:
        if method=='standard':
            controller=Controller(case['question'],corpus,db,method='standard',planner=DeterministicPlanner(),budget=budget,output=str(directory/'runs'))
            planner=None
        else:
            planner=ContextPlanner(factory('main'),mode='clm' if method.startswith('clm') else 'conventional',limits=InferenceLimits(max_total_tokens=TOTAL_TOKENS,max_active_tokens=4096,max_input_tokens=8192),completion_guidance=True)
            controller=PhaseController(case['question'],corpus,db,architecture=method,graph_policy='graph-visible-v2',planner=planner,budget=budget,output=str(directory/'runs'))
            controller.revision7_resources=revised
        original_sink=controller._log
        phase_state={'start_actions':0,'recoveries':set()}
        def sink(kind,value):
            events.append({'kind':kind,**value})
            if kind=='context_recovery' and value.get('accepted'):
                r=value['request'];phase_state['recoveries'].add((r['index'],r['start'],r['chars']))
            original_sink(kind,value)
        if planner:
            planner.sink=sink;planner.store.sink=sink
            def recovery_guard():
                if controller.actions-phase_state['start_actions']+len(phase_state['recoveries'])>=PHASE_ACTIONS:raise BudgetExceeded('phase_substantive_action_limit')
            planner.recovery_action_guard=recovery_guard
        controller.started=time.perf_counter()
        try:
            for phase in case['phases']:
                cancel_check();phase_started=time.perf_counter();before_actions=controller.actions;before_papers=set(controller.papers);before_calls=controller.planner.usage.calls
                phase_state.update(start_actions=before_actions,recoveries=set())
                controller._memory=replace(controller._memory,stop_reason='',stagnant=0)
                controller.budget=replace(budget,max_actions=before_actions+6,max_rounds=controller.rounds+6,max_llm_calls=min(TOTAL_GENERATIONS,before_calls+12))
                controller.phase=phase
                if planner:planner.budget=controller.budget
                stop=False;reason='phase_action_limit';safety=False;fallbacks_before=sum(e['kind']=='invalid_decision' for e in events)
                if method=='standard':
                    action=make_action('SEARCH_PAPERS',phase['baseline_query'],'Pinned phase-specific non-LLM query')
                    try:controller._execute(action)
                    except (ValueError,BudgetExceeded) as exc:reason=type(exc).__name__
                    else:reason='pinned_nonagentic_phase_query'
                else:
                    while controller.actions-before_actions+len(phase_state['recoveries'])<6:
                        cancel_check()
                        if controller._hard_stop():reason=controller._hard_stop();break
                        if planner.usage.calls>=controller.budget.max_llm_calls:reason='phase_model_call_limit';break
                        context=controller.context()
                        try:
                            action=Action.parse(planner.decide(deepcopy(context)));controller.validate(action)
                        except BudgetExceeded as exc:reason=str(exc);break
                        except ValueError as exc:
                            if isinstance(exc,PlannerError) and exc.code=='refused':reason='safety_refusal';safety=True;break
                            action=controller._fallback(context)
                            sink('invalid_decision',{'error_code':getattr(exc,'code','invalid_action'),'validation_rule':str(exc),'fallback_action':action.to_dict()})
                        if controller.actions-before_actions+len(phase_state['recoveries'])>=6 and action.action!='STOP_RETRIEVAL':reason='phase_substantive_action_limit';break
                        controller._execute(action)
                        if action.action=='STOP_RETRIEVAL':stop=True;reason='model_phase_stop';break
                response=None
                if planner and not safety:
                    planner.budget=replace(controller.budget,max_llm_calls=min(TOTAL_GENERATIONS,before_calls+13));planner._factual_response_used=False
                    response=planner.factual_response(case['id']+f'-phase{phase["index"]}',phase.get('reuse_question',phase['objective']+' State one available publication-year/source quotation or abstain. Do not infer semantic scientific truth.'))
                complete=_phase_complete(gold,phase['index'],controller.papers)
                record={'phase':phase['index'],'objective':phase['objective'],'allowed_substantive_actions':6,'acquisition_actions':controller.actions-before_actions,'accepted_distinct_recoveries':len(phase_state['recoveries']),
                    'actual_substantive_actions':controller.actions-before_actions+len(phase_state['recoveries']),'new_evidence_sources':sorted(set(controller.papers)-before_papers),'evidence_carryover_sources':len(before_papers),
                    'model_calls':controller.planner.usage.calls-before_calls,'stop':stop,'premature_stop':stop and not complete,'objective_complete':complete,'reason':reason,'safety_refusal':safety,
                    'factual_response':response,'seconds':time.perf_counter()-phase_started,'fallbacks':sum(e['kind']=='invalid_decision' for e in events)-fallbacks_before}
                record.update(acquired_metadata_records=len(controller.papers),newly_admitted_records=len(set(controller.papers)-before_papers),
                    explicit_inspection_actions=sum(a['action']=='INSPECT_PAPER' for a in controller.history[before_actions:]),
                    returned_records=sum(t.get('returned_papers',0) for t in controller.trace if t.get('actions',0)>before_actions),
                    acquisition_record_limit=budget.max_inspected,records_per_acquisition=budget.page_size)
                record['context_framing_displaced_records']=sum(sum(e['displaced_records'].values()) for e in getattr(controller,'framing_events',[]) if e['phase']==phase['index'])
                if planner:
                    from .revision7_measurements import visible_records
                    delivered=set();recovered=set();retained=set();request_count=0
                    for path in sorted(planner.store.directory.glob('request-*.json')):
                        r=json.loads(path.read_text());d=r.get('diagnostics',{})
                        if r['call']<=before_calls or not d.get('wire_sha256') or d.get('request_not_sent',True):continue
                        request_count+=1;payload=r.get('input',{})
                        delivered.update(visible_records(payload,controller.papers)[0])
                        if 'recovered_observation' in payload:recovered.update(visible_records(payload['recovered_observation'],controller.papers)[0])
                        if 'active_context' in payload:retained.update(visible_records(payload['active_context'],controller.papers)[0])
                    record.update(publications_delivered_in_requests=len(delivered),delivered_publication_ids=sorted(delivered),
                        publications_in_CLM_inputs=len(retained),publications_in_recovery_inputs=len(recovered),actual_provider_request_fixtures=request_count)
                if revised:
                    import re
                    from .models import canonical_id
                    expected=gold.get('reuse',{}).get(str(phase['index']));answer=(response or {}).get('response') or {}
                    try:identity=canonical_id(answer.get('work_id'))
                    except (ValueError,TypeError):identity=None
                    record['early_evidence_reuse_correct']=bool(expected and (response or {}).get('status')=='valid' and identity==expected['source_id'] and re.search(r'(?<![0-9])'+str(expected['year'])+r'(?![0-9])',answer.get('answer','')) is not None and answer.get('evidence') and answer['evidence'] in (((response or {}).get('evidence_record') or {}).get('provenance') or {}).get('assessment_context','')) if expected else None
                    record['reuse_source_previously_delivered']=bool(expected and expected['source_id'] in {pid for prior in phases for pid in prior.get('delivered_publication_ids',[])}) if expected else None
                phases.append(record);atomic_json(directory/f'phase-{phase["index"]:02d}.json',record)
                controller._checkpoint(time.perf_counter()-started)
                if planner:snapshot(planner,directory/f'snapshot-phase-{phase["index"]:02d}',extra={'phase':phase['index'],'papers':{k:v.to_dict() for k,v in controller.papers.items()},'graph_registry':controller._graph_registry,'graph_allocation':controller._graph_allocation})
                if safety:break
            scores=source_edge_checks(controller.papers,gold)
            result=controller._export_result(time.perf_counter()-started)
            report={'study':'L','task_id':case['id'],'method':method,'status':'COMPLETED' if len(phases)==6 else 'FAILED','attempted':True,**scores,'phases':phases,'phase_count':len(phases),
                'actual_substantive_actions':sum(p['actual_substantive_actions'] for p in phases),'acquisition_actions':controller.actions,'provider_calls':controller.planner.usage.calls,'retrieved_sources':len(controller.papers),
                'objectives_completed':sum(p['objective_complete'] for p in phases),'meaningful_phases':sum(p['actual_substantive_actions']>0 for p in phases),'sustained_work_demonstrated':sum(p['actual_substantive_actions']>=3 for p in phases)>=3,'early_stops':sum(p['premature_stop'] for p in phases),'factual_opportunities':len(phases) if planner else 0,'events':events,'directory':str(controller.directory),'seconds':time.perf_counter()-started,
                'interpretation':'Externally staged autonomous research; STOP ends one phase; semantic factual accuracy NOT EVALUATED'}
            if revised:
                from .revision7_measurements import final_outcome
                report.update(final_outcome(method,scores['primary_score'],[p['factual_response'] for p in phases],collection_ok=len(phases)==6))
                report.update(reuse_probes_correct=sum(p.get('early_evidence_reuse_correct') is True for p in phases),reuse_probes_expected=5 if planner else 0,
                    acquisition_record_limit=200,records_per_acquisition=5,resource_policy='bounded-acquisition-records-r7',
                    context_framing_displaced_records=sum(p['context_framing_displaced_records'] for p in phases))
            atomic_json(directory/'trajectory.json',report);return report
        finally:
            if planner:planner.close()
