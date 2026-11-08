"""Versioned medium-pressure correction repair; no retrieval-policy changes or LLMs."""
import argparse
from pathlib import Path
from openalex_agent.context_pressure import run_pressure
from openalex_agent.corpus import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a fresh correction-validation output')
    args.output.mkdir(parents=True)
    reports=[]
    for seed in (11,23,37):
        report=run_pressure('medium',args.output/f'seed-{seed}',seed=seed,
            architectures=('llm','llm_graph','clm','clm_graph'),correction_at=4,return_invalid=True)
        reports.append(report)
        print(f"seed={seed} pressure_and_delivery_valid={report['pressure_valid']}",flush=True)
    summary=dict(version='medium-correction-delivery-v2',passed=all(r['pressure_valid'] for r in reports),
        conditions=3,arms=sum(len(r['rows']) for r in reports),hosted_calls=0,openalex_requests=0,
        change='Only synthetic correction placement changes from observation 16 to 4; original top-eight lexical selection, model script and scoring unchanged.',
        reason='The high-ID original source is still in the selected top-eight when corrected at 4, then can leave the selected view during the remaining horizon.',
        initial_evidence_position=0,correction_at=4,horizon=32,archive_access_equal=True,
        rows=[dict(seed=r['seed'],valid=r['pressure_valid'],pressure_ratio=r['pressure_ratio'],
              distinct_source_ratio=r['unique_source_pressure_ratio'],shared_archives=r['shared_observation_archives'],
              equal_limits=len({str(a['limits']) for a in r['rows']})==1,
              factual_errors={a['architecture']:a['factual_errors'] for a in r['rows']},
              correction_delivered={a['architecture']:a['correction_delivered_to_model'] for a in r['rows']}) for r in reports],
        limitation='Scripted engineering validation only. Retention, correction errors and attribution must remain separate; stale answers are not edited away.')
    atomic_json(args.output/'summary.json',summary)
    if not summary['passed']:raise ValueError('A correction arm failed delivery or pressure validation')


if __name__=='__main__':main()
