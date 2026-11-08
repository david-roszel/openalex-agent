"""Offline fixed-evidence readiness matrix; never a hosted comparative study."""

import argparse
from pathlib import Path
import json

from openalex_agent.context_pressure import run_pressure
from openalex_agent.corpus import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a fresh pressure validation directory')
    args.output.mkdir(parents=True)
    results=[]
    architectures=('llm','llm_graph','clm','clm_graph')
    for seed in (11,23,37):
        for condition in ('low','medium','high'):
            directory=args.output/(condition+'-seed'+str(seed))
            report=run_pressure(condition,directory,seed=seed,architectures=architectures,return_invalid=True)
            results.append(dict(seed=seed,condition=condition,source=str(directory/'pressure.json'),valid=report['pressure_valid'],
               pressure_ratio=report['pressure_ratio'],distinct_source_ratio=report['unique_source_pressure_ratio'],
               shared_archives=report['shared_observation_archives'],arms=len(report['rows']),
               limits_equal=len({json.dumps(r['limits'],sort_keys=True) for r in report['rows']})==1))
            atomic_json(args.output/'matrix.json',dict(kind='SCRIPTED FIXED-EVIDENCE READINESS, NOT EMPIRICAL RETRIEVAL',rows=results))
            print(json.dumps(results[-1]),flush=True)
    correction=run_pressure('medium',args.output/'correction-seed11',seed=11,architectures=architectures,correction_at=16,return_invalid=True)
    atomic_json(args.output/'summary.json',dict(matrix_conditions=len(results),valid_conditions=sum(r['valid'] for r in results),
      architectures=architectures,seeds=[11,23,37],rows=results,correction_pressure_valid=correction['pressure_valid'],
      correction_factual_errors={r['architecture']:r['factual_errors'] for r in correction['rows']},
      independent_conflicting_source_scenario='Not independently validated; correction does not establish unresolved-conflict handling',
      real_openalex_requests=0,hosted_calls=0,package_native_probes_executed=0))
    low=run_pressure('low',args.output/'correction-low-seed11',seed=11,architectures=architectures,correction_at=2,return_invalid=True)
    atomic_json(args.output/'correction-low-summary.json',dict(valid=low['pressure_valid'],source=str(args.output/'correction-low-seed11/pressure.json'),
        factual_errors={r['architecture']:r['factual_errors'] for r in low['rows']},
        note='Small functional correction check, not evidence of long-horizon correction readiness'))


if __name__=='__main__':main()
