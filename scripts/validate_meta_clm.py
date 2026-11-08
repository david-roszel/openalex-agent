"""Offline seven-configuration integration check; never instantiates a hosted client."""
import argparse
import hashlib
import json
from pathlib import Path

from openalex_agent.context_planner import InferenceLimits
from openalex_agent.controlled import ALL_ARCHITECTURES
from openalex_agent.controller import Budget
from openalex_agent.corpus import atomic_json
from openalex_agent.experiments import run_experiments
from openalex_agent.llm import DeterministicPlanner
from openalex_agent.meta_context import reference_provenance
from openalex_agent.storage import Database


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--corpus',type=Path,default=Path('data/corpus/multidisciplinary-2026-11-06-v1'))
    parser.add_argument('--questions',type=Path,default=Path('data/hosted-development-preflight-2026-11-06-v1/questions.json'))
    parser.add_argument('--provider-mock',choices=['script','gemini'],default='script')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a fresh offline output directory')
    questions=json.loads(args.questions.read_text())
    if len(questions)!=3 or any(q.get('split')!='development' for q in questions):
        raise ValueError('This check uses the three frozen development questions only')
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    receipt=json.loads((args.corpus/'freeze-receipt.json').read_text())
    before={name:digest(args.corpus/name) for name in receipt['files']}
    if before!=receipt['files']:raise ValueError('Frozen corpus receipt mismatch')
    args.output.mkdir(parents=True)
    budget=Budget(max_actions=10,max_rounds=20,max_inspected=200,max_llm_calls=20,page_size=10)
    limits=InferenceLimits(max_total_tokens=100000,max_active_tokens=4096,max_input_tokens=8192)
    factory=DeterministicPlanner
    if args.provider_mock=='gemini':
        from openalex_agent.gemini import GeminiADCPlanner,GeminiSettings
        def factory():
            count=0
            def scripted(body):
                nonlocal count
                count+=1
                payload=json.loads(body['contents'][-1]['parts'][-1]['text'])
                protected=payload['protected'];mode=protected['context_management']
                if 'tools' in body['config']:
                    if count==1:
                        name,arguments='context_file',dict(path='LIVE_CTX_MAIN.txt',op='append',text='\nMOCKED GEMINI CONTEXT MARKER')
                    else:
                        proposal=protected['proposals'][0];name,arguments=proposal['action'],proposal['arguments']
                    part=dict(function_call=dict(id='fixture-id',name=name,args=arguments),thought_signature='U1lOVEhFVElDLU5PVC1SRUFM')
                else:
                    value=({'context_edit':dict(path='active_context.txt',op='append',text='\nSCRIPTED ENGINEERING MEMO')}
                        if mode=='clm' and count==1 else {'decision':protected['proposals'][0],'evidence_records':[]})
                    part={'text':json.dumps(value)}
                return dict(http_status=200,response=dict(candidates=[dict(content=dict(role='model',parts=[part]),finish_reason='STOP')],
                    usage_metadata=dict(prompt_token_count=8,candidates_token_count=5,thoughts_token_count=95,total_token_count=108)))
            return GeminiADCPlanner(GeminiSettings(project='SYNTHETIC-OFFLINE-NO-ADC'),executor=scripted)
    with Database(str(args.output/'metadata.sqlite')) as db:
        rows=run_experiments(str(args.corpus),str(args.questions),db,factory,
            output=str(args.output/'benchmark'),suite='controlled-seven',methods=ALL_ARCHITECTURES,
            action_budgets=(10,),trials=1,trial_seeds=(11,),budget=budget,inference_limits=limits,allow_deterministic=True)
    after={name:digest(args.corpus/name) for name in receipt['files']}
    assert before==after
    native=[]
    for row in rows:
        if not row['method'].startswith('meta_clm'):continue
        run=Path(row['directory'])
        requests=[json.loads(p.read_text()) for p in sorted((run/'context').glob('request-*.json'))]
        marker='MOCKED GEMINI CONTEXT MARKER' if args.provider_mock=='gemini' else 'SCRIPTED META SYNCHRONIZATION MARKER'
        delivered=[r['call'] for r in requests if marker in json.dumps(r['input'])]
        native.append(dict(query_id=row['query_id'],method=row['method'],calls=row['llm_calls'],fallbacks=row['fallbacks'],
            accepted_edits=row['context_edits'],reference_syncs=row['meta_reference_syncs'],marker_delivery_calls=delivered,
            stop_reason=row['stop_reason'],max_active_tokens=row['max_active_context_tokens'],max_input_tokens=row['max_full_input_tokens']))
    result=dict(classification='OFFLINE SCRIPTED ENGINEERING; not hosted results or scientific effectiveness',
        provider_fixture=args.provider_mock,token_fields_are_synthetic=args.provider_mock=='gemini',
        cells=len(rows),methods=list(ALL_ARCHITECTURES),native_cells=native,reference=reference_provenance(),
        corpus_unchanged=before==after,corpus_sha256=before['papers.jsonl'],
        hosted_calls=0,live_openalex_calls=sum(r['api_requests'] for r in rows),
        labels_used=False,heldout_questions_used=False,budget=budget.__dict__,limits=limits.__dict__,
        fallback_count=sum(r['fallbacks'] for r in rows),
        all_meta_cells_edit_and_deliver=all(r['accepted_edits'] and r['marker_delivery_calls'] for r in native))
    atomic_json(args.output/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
