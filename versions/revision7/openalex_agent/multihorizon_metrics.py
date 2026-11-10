"""V3 objective scoring and task-clustered analyses; never human judgments."""
from itertools import permutations
import math
import numpy as np
from .models import canonical_id,OPENALEX_SOURCES
from .memory_scoring_v2 import score_response,factorial_effects

HOSTED=('llm','llm_graph','clm','clm_graph')


def source_edge_checks(papers,gold):
    observed=set(papers)
    source_checks={pid:int(pid in observed) for pid in gold['source_ids']}
    edge_checks={a+'->'+b:int(a in observed and b in observed and b in papers[a].references and papers[a].metadata_source in OPENALEX_SOURCES and 'references' in (papers[a].metadata_fields or [])) for a,b in gold['citation_edges']}
    values=[*source_checks.values(),*edge_checks.values()]
    if len(values)!=gold['item_denominator']:raise ValueError('Changed frozen objective denominator')
    return {'primary_score':sum(values)/len(values),'source_checks':source_checks,'directed_edge_checks':edge_checks,'objective_denominator':len(values),'semantic_truth':'NOT EVALUATED'}


def score_memory(answer,gold,*,attempted=True,failed=False,exposure_complete=True,policy='v2'):
    if not attempted:return {'primary_score':None,'operational_itt':None,'status':'NOT RUN'}
    scorer=score_response
    if policy=='r7':
        from .memory_scoring_v3 import score_response as scorer
    result=scorer(answer,gold,terminal_failure=failed)
    result.update(operational_itt=0.0 if failed or result.get('terminal_failure') or not exposure_complete else result['primary_score'],exposure_complete=exposure_complete,
        general_relevance='NOT EVALUATED',semantic_truth='NOT EVALUATED')
    return result


def holm(values):
    ordered=sorted((float(v),k) for k,v in values.items() if v is not None)
    out={k:None for k in values};previous=0.0;n=len(ordered)
    for i,(value,key) in enumerate(ordered):previous=max(previous,min(1,(n-i)*value));out[key]=previous
    return out


def comparisons(rows,*,samples=10000,seed=73019):
    """Exact two repetitions per arm, average within task, then domain bootstrap."""
    if type(samples) is not int or samples<1:raise ValueError('Invalid bootstrap size')
    outputs=[]
    for study in ['S','L','M']:
        groups={};domains={}
        for r in rows:
            if r.get('study')!=study or r.get('method') not in HOSTED:continue
            key=(r['task_id'],r['method'],r['repetition'])
            if key in groups:raise ValueError('Duplicate task/method/repetition')
            groups[key]=r;domains[r['task_id']]=r['domain']
        for policy in ['operational_itt','complete_case']:
            per={};excluded={}
            for task in sorted(domains):
                arm={};bad=None
                for method in HOSTED:
                    scores=[]
                    for rep in [0,1]:
                        r=groups.get((task,method,rep))
                        if r is None or r.get('status') in ('PENDING','RUNNING') or not r.get('attempted',False):bad='missing_or_unattempted';break
                        failed=r.get('status') not in ('COMPLETED',) or r.get('exposure_complete') is False
                        value=0. if failed and policy=='operational_itt' else None if failed else r.get('primary_score')
                        if value is None:bad='failed_or_unscored';break
                        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid primary endpoint')
                        scores.append(float(value))
                    if bad:break
                    arm[method]=float(np.mean(scores))
                if bad:excluded[task]=bad
                else:per[task]=factorial_effects(arm)
            rng=np.random.default_rng(seed);stats=[]
            for contrast in ['graph','clm','interaction']:
                tasks=sorted(per);x=np.array([per[t][contrast] for t in tasks],dtype=float);ci=None;p=None;mean=float(x.mean()) if len(x) else None
                if len(x)>=2:
                    draws=np.zeros(samples)
                    for d in sorted({domains[t] for t in tasks}):
                        part=x[[domains[t]==d for t in tasks]];draws+=part[rng.integers(0,len(part),size=(samples,len(part)))].sum(axis=1)/len(x)
                    ci=[float(z) for z in np.quantile(draws,[.025,.975])]
                    flips=rng.choice([-1,1],size=(samples,len(x)))
                    p=float((np.sum(np.abs((flips*x).mean(axis=1))>=abs(mean)-1e-12)+1)/(samples+1))
                stats.append({'study':study,'policy':policy,'contrast':contrast,'tasks':len(tasks),'mean':mean,'ci95':ci,'sign_flip_p':p,
                    'per_task':{t:per[t][contrast] for t in tasks},'excluded_tasks':excluded,'independent_unit':'task/case','replications_per_method':2,
                    'caveat':'Exploratory fixed-domain task bootstrap; sign-flip sensitivity assumes symmetry. Zero-width intervals do not establish equivalence.'})
            adjusted=holm({s['contrast']:s['sign_flip_p'] for s in stats})
            for s in stats:s['holm_within_study']=adjusted[s['contrast']]
            outputs.extend(stats)
    adjusted=holm({f"{s['study']}:{s['contrast']}":s['sign_flip_p'] for s in outputs if s['policy']=='operational_itt'})
    for s in outputs:s['holm_across_studies']=adjusted.get(s['study']+':'+s['contrast']) if s['policy']=='operational_itt' else None
    return outputs


def architecture_performance(rows,*,samples=10000,seed=73019):
    """Matched task means for each study; standard is a non-LLM reference."""
    output=[]
    for study in ['S','L','M']:
        methods=HOSTED if study=='M' else ('standard',*HOSTED)
        subset=[r for r in rows if r['study']==study];domains={r['task_id']:r['domain'] for r in subset}
        cells={(r['task_id'],r['method'],r['repetition']):r for r in subset}
        for policy in ['operational_itt','complete_case']:
            scores={m:{} for m in methods}
            for task in sorted(domains):
                record={};valid=True
                for method in methods:
                    values=[]
                    for rep in (0,1):
                        r=cells.get((task,method,rep))
                        if r is None or not r.get('attempted') or r['status'] in ('PENDING','RUNNING'):valid=False;break
                        failed=r['status']!='COMPLETED' or r.get('exposure_complete') is False
                        value=0.0 if failed and policy=='operational_itt' else None if failed else r.get('primary_score')
                        if value is None:valid=False;break
                        values.append(float(value))
                    if not valid:break
                    record[method]=float(np.mean(values))
                if valid:
                    for method,value in record.items():scores[method][task]=value
            for method in methods:
                tasks=sorted(scores[method]);x=np.asarray([scores[method][t] for t in tasks]);ci=None;rng=np.random.default_rng(seed)
                if len(x)>=2:
                    draws=np.zeros(samples)
                    for domain in sorted({domains[t] for t in tasks}):
                        part=x[[domains[t]==domain for t in tasks]];draws+=part[rng.integers(0,len(part),(samples,len(part)))].sum(axis=1)/len(x)
                    ci=[float(v) for v in np.quantile(draws,[.025,.975])]
                output.append({'study':study,'policy':policy,'method':method,'tasks':len(tasks),'mean':float(x.mean()) if len(x) else None,'ci95':ci,'per_task':scores[method],
                    'scope':'Matched five-method task blocks; standard external non-LLM reference' if study!='M' else 'Matched four-hosted-method case blocks','independent_unit':'task/case'})
    return output
