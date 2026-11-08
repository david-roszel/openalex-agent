"""Run a small no-network engineering check over a real frozen collection."""

import argparse
from pathlib import Path
from unittest.mock import patch
import json

import httpx

from openalex_agent.controller import Budget
from openalex_agent.context_planner import InferenceLimits
from openalex_agent.corpus import LocalCorpus, atomic_json
from openalex_agent.experiments import run_experiments
from openalex_agent.llm import DeterministicPlanner
from openalex_agent.storage import Database


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh offline-validation directory')
    a.output.mkdir(parents=True)
    questions=a.output/'questions.json'
    atomic_json(questions,[dict(id='offline-interface-check',question='scientific literature methods evidence',split='diagnostic')])
    def forbidden(*args,**kwargs):raise AssertionError('Offline corpus attempted an HTTP request')
    with patch.object(httpx.Client,'send',forbidden),patch.object(httpx.AsyncClient,'send',forbidden):
        c=LocalCorpus(a.corpus)
        c.search('scientific literature',limit=2)
        first=c.papers[0]
        c.get(first.id);c.references(first.id,limit=2);c.citations(first.id,limit=2)
        try:c.get('W999999999999999999999')
        except ValueError:pass
        with Database(str(a.output/'metadata.sqlite')) as db:
            rows=run_experiments(str(a.corpus),str(questions),db,DeterministicPlanner,output=str(a.output/'benchmark'),
                suite='controlled-five',methods=('standard','llm','llm_graph','clm','clm_graph'),trials=1,action_budgets=(2,),trial_seeds=(11,),
                budget=Budget(max_actions=2,max_rounds=2,max_inspected=4,page_size=2,max_llm_calls=5),
                inference_limits=InferenceLimits(max_total_tokens=70000,max_active_tokens=4096,max_input_tokens=8192),allow_deterministic=True)
    report=dict(kind='NO-NETWORK INTERFACE VALIDATION, NOT EMPIRICAL QUALITY',passed=True,methods=5,rows=len(rows),
        api_requests=sum(r['api_requests'] for r in rows),real_hosted_calls=0,independent_labels_present=False,
        corpus_sha256=c.fingerprint,http_transport_forbidden=True)
    atomic_json(a.output/'offline-validation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
