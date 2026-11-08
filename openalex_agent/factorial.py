"""Descriptive, predeclared 2 x 3 contrasts; the old 2 x 2 scorer is unchanged."""
import math

import numpy as np

STRATEGIES={'conventional':('llm','llm_graph'),'inspired':('clm','clm_graph'),
            'meta':('meta_clm','meta_clm_graph')}


def contrasts_2x3():
    rows=[]
    def add(name,weights,primary=False):
        rows.append(dict(name=name,weights=weights,primary=primary))
    add('graph_main_effect',{m:(1/3 if graph else -1/3) for pair in STRATEGIES.values() for graph,m in enumerate(pair)},True)
    for left,right in [('inspired','conventional'),('meta','conventional'),('meta','inspired')]:
        name=left+'_minus_'+right
        add(name,{**{m:.5 for m in STRATEGIES[left]},**{m:-.5 for m in STRATEGIES[right]}},right=='conventional')
        add('graph_interaction_'+name,{STRATEGIES[left][1]:1,STRATEGIES[left][0]:-1,
            STRATEGIES[right][1]:-1,STRATEGIES[right][0]:1},right=='conventional')
        for graph in (0,1):
            add(name+('_graph' if graph else '_no_graph'),{STRATEGIES[left][graph]:1,STRATEGIES[right][graph]:-1})
    for strategy,pair in STRATEGIES.items():add('graph_within_'+strategy,{pair[1]:1,pair[0]:-1})
    return rows


def paired_comparisons_2x3(rows,metric,*,samples=2000,seed=17):
    if type(samples) is not int or samples<1:raise ValueError('Bootstrap samples must be a positive integer')
    methods={m for pair in STRATEGIES.values() for m in pair}
    groups={}
    for row in rows:
        if row['method'] not in methods or row.get('variant','full')!='full':continue
        trial=row.get('trial')
        if type(trial) is not int or trial<0:raise ValueError('Paired comparisons require nonnegative integer trial IDs')
        key=(row['query_id'],row['method'],row['action_budget'])
        cell=groups.setdefault(key,{})
        if trial in cell:raise ValueError('Duplicate experimental trial/cell identity')
        cell[trial]=row.get(metric)
    rng=np.random.default_rng(seed);result=[]
    for budget in sorted({key[2] for key in groups}):
        for contrast in contrasts_2x3():
            weights=contrast['weights'];differences={};excluded={};counts={}
            for question in sorted({key[0] for key in groups if key[2]==budget}):
                cells=[groups.get((question,m,budget)) for m in weights]
                if any(c is None for c in cells):excluded[question]='missing_method_cell';continue
                if any(set(c)!=set(cells[0]) for c in cells[1:]):raise ValueError('Unbalanced trial identities in paired comparison')
                counts[question]=len(cells[0])
                if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for c in cells for v in c.values()):
                    excluded[question]='missing_or_nonfinite_metric_in_trial';continue
                differences[question]=sum(w*float(np.mean(list(groups[question,m,budget].values()))) for m,w in weights.items())
            values=list(differences.values());interval=None
            if len(values)>=2:
                bootstrap=rng.choice(values,size=(samples,len(values)),replace=True).mean(axis=1)
                interval=[float(v) for v in np.quantile(bootstrap,[.025,.975])]
            result.append(dict(contrast=contrast['name'],primary=contrast['primary'],weights=weights,
                action_budget=budget,metric=metric,variant='full',questions=len(values),trial_counts=counts,
                excluded_questions=excluded,per_question_difference=differences,
                mean_difference=float(np.mean(values)) if values else None,bootstrap_95_ci=interval,
                analysis_version='factorial-2x3-descriptive-v1',
                interpretation='Descriptive question-paired contrasts; intervals are not multiplicity-adjusted significance tests'))
    return result
