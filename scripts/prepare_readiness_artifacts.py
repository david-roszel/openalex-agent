"""Offline workload estimates and exact-input audits; no answers, labels, or API calls."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from openalex_agent.controller import Budget
from openalex_agent.context_planner import InferenceLimits
from openalex_agent.corpus import atomic_json
from openalex_agent.evidence_visibility import audit_passage
from openalex_agent.models import Paper
from openalex_agent.pooled_review import VERSION,POLICY,digest,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--corpus',required=True,type=Path)
    p.add_argument('--existing-offline-experiment',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh preparation directory')
    a.output.mkdir(parents=True)
    rows=[]
    for methods in (5,7):
        for budgets in (1,3):
            agents=methods-1;agent_runs=18*agents*budgets*3;calls=agent_runs*20
            # Expected use is an explicit scenario, not an empirical prediction.
            tin=agent_runs*8*3000;tout=agent_runs*8*500
            rows.append(dict(methods=methods,action_budgets=budgets,questions=18,trials=3,
                total_runs=18*methods*budgets*3,agent_runs=agent_runs,max_model_calls=calls,
                max_accounted_total_tokens=agent_runs*100000,
                request_box_input_tokens=calls*8192,request_box_output_tokens=calls*4096,
                expected_scenario_input_tokens=tin,expected_scenario_output_tokens=tout,
                planning_list_price_scenario_usd=(tin+5*tout)/1e6,
                planning_accounted_token_ceiling_usd=agent_runs*100000*5/1e6,
                conservative_request_box_usd=calls*(8192+4096*5)/1e6,
                expected_scenario_serial_model_hours=agent_runs*8*2.041/3600,
                capped_call_count_at_pilot_latency_hours=calls*2.041/3600,
                capped_call_count_at_configured_timeout_hours=calls*120/3600,
                annotation_rank_slots_before_dedup=18*methods*budgets*3*20,
                existing_configured_invoice_cost='UNKNOWN: input/output price fields are zero',
                meta_transport_retries_included=False))
    atomic_json(a.output/'workload.json',dict(scope='Track A retrieval; Track B/C require separate approval and resource reservations',
        assumptions=dict(model='gpt-6.1-terra',planning_input_usd_per_million=1,planning_output_usd_per_million=5,
            prices_source='https://developers.openai.com/api/docs/models/gpt-6.1-terra',verified_on='2026-11-06',
            price_scope='Published Standard direct OpenAI prices; account billing/tier is not verified, configuration unchanged',
            expected_calls_per_agent=8,expected_input_tokens_per_call=3000,expected_output_tokens_per_call=500,
            pilot_calls=12,pilot_input_tokens=29276,pilot_output_tokens=705,pilot_mean_seconds=2.041,
            pilot_scope='Historical short 39-work graph/no-graph pilot, not the new corpus, context editing, Meta or long-horizon latency',
            retries='Existing five-method client has one attempt per invocation; Meta default retries require a separate bound'),options=rows))
    spec=dict(corpus=str(a.corpus),questions='COORDINATOR_RELEASED_FINAL_QUESTIONS.json',
        output='data/APPROVED_TRACK_A_OUTPUT',action_budgets=[10],trials=3,trial_seeds=[11,23,37],
        budget=asdict(Budget(max_actions=10,max_rounds=20,max_inspected=200,max_llm_calls=20,page_size=10)),
        inference_limits=asdict(InferenceLimits(max_total_tokens=100000,max_active_tokens=4096,max_input_tokens=8192)),
        max_total_cost_usd=None)
    atomic_json(a.output/'collection-spec.UNAPPROVED.json',spec)
    for stage,keys in [('collect-unscored',('spec_sha256','question_review_sha256','corpus_sha256','questions_sha256','source_sha256','provider_profile_sha256','policy_sha256')),
                       ('freeze-unscored',('experiment_sha256','metrics_sha256','raw_output_tree_sha256','policy_sha256')),
                       ('score-preserved',('frozen_outputs_sha256','pool_sha256','annotation_sha256','policy_sha256')),
                       ('release-heldout-review',('corpus_sha256','questions_sha256'))]:
        atomic_json(a.output/f'{stage}.UNAPPROVED.json',dict(protocol=VERSION,stage=stage,approved=False,approver='',approved_at='',
            heldout_independence_attested=False,paid_inference_authorized=False,pins={k:digest(POLICY) if k=='policy_sha256' else '' for k in keys}))
    manifest=json.loads((a.existing_offline_experiment/'experiment.json').read_text())
    assert manifest['deterministic_smoke'] and not manifest['has_relevance_judgments']
    import csv
    with (a.existing_offline_experiment/'metrics.csv').open(newline='') as f:runs=list(csv.DictReader(f))
    wanted=set();source_runs=[]
    for row in runs:
        result=json.loads((Path(row['directory'])/'result.json').read_text())
        for wid in result['inspected_ids']:
            wanted.add(wid);source_runs.append((row,wid))
    papers={}
    with (a.corpus/'papers.jsonl').open() as f:
        for line in f:
            raw=json.loads(line)
            if raw['id'] in wanted:papers[raw['id']]=Paper.from_dict(raw)
    audits=[]
    for row,wid in source_runs:
        paper=papers[wid]
        for location,passage in [('abstract_start',(paper.abstract or '')[:100]),('abstract_tail',(paper.abstract or '')[-100:])]:
            if passage:audits.append(dict(method=row['method'],probe_location=location,
                **audit_passage(row['directory'],paper,passage)))
    atomic_json(a.output/'real-corpus-evidence-visibility.json',dict(kind='Engineering input audit; no independent factual ground truth',
        source_experiment_sha256=sha(a.existing_offline_experiment/'experiment.json'),corpus_sha256=manifest['corpus'],
        deterministic_only=True,hosted_calls=0,rows=audits))
    sealed=Path('data/final-dataset-readiness-2026-11-06/integrated/sealed/heldout-index.json')
    shutil.copyfile(sealed,a.output/'heldout-index-only.json')
    print('Prepared workload options:',len(rows),'exact-input diagnostic checks:',len(audits))


if __name__=='__main__':main()
