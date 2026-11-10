"""SYNTHETIC offline integration diagnostics, never model-quality evidence."""
import json
from pathlib import Path
import pytest
from openalex_agent import evaluation_artifacts as artifacts
from openalex_agent.corpus import atomic_json
from test_scientific_evaluation_ui import saved


@pytest.mark.parametrize('malformed',[
    {'active_context':'{"changes":null}'},
    {'observation':{'top_papers':None}},
    {'observation':{'graph':{'nodes':1},'candidates':None,'subgraph':None}},
])
def test_malformed_saved_context_cannot_crash_comparison_or_invent_quality(tmp_path,malformed):
    directory=saved(tmp_path)
    path=directory/'runs/clm_graph/context/request-00001.json'
    record=json.loads(path.read_text());record['input']=malformed;record['output']={'decision':None};atomic_json(path,record)
    rows=artifacts.load_evaluation(directory,tmp_path)['rows']
    summaries=[artifacts.summary_row(r) for r in rows]
    assert len(summaries)==5
    assert all(s['ndcg'] is None and s['retention'] is None for s in summaries)
    row=next(r for r in rows if r['method']=='clm_graph')
    assert len(row['detail']['requests'])==1 # Preserve the valid trace around malformed evidence.


def test_null_saved_usage_and_provider_are_unknown_instead_of_crashing(tmp_path):
    directory=saved(tmp_path);path=directory/'runs/clm/result.json'
    record=json.loads(path.read_text());record.update(llm_usage=None,provider_configuration=None);atomic_json(path,record)
    row=next(r for r in artifacts.load_evaluation(directory,tmp_path)['rows'] if r['method']=='clm')
    assert artifacts.summary_row(row)['retention'] is None


@pytest.mark.parametrize('stop_early',[False,True])
def test_mocked_native_five_method_retrieval_edit_recovery_graph_and_stop(tmp_path,db,monkeypatch,stop_early):
    """Real orchestration, HTTP MockTransport only; no scientific grades."""
    import httpx
    from openalex_agent.models import Paper
    from openalex_agent.config import Settings
    from openalex_agent.llm import HostedJSONPlanner
    from openalex_agent.corpus import freeze
    from openalex_agent.controller import Budget
    from openalex_agent.experiments import run_experiments
    from openalex_agent.controlled import ARCHITECTURES
    from openalex_agent.experiment_integrity import validate_schedule
    from openalex_agent import pooled_review as pooled
    question='synthetic astronomy signal retrieval root'
    papers=[Paper.from_openalex(dict(id='W1',title=question,
        abstract_inverted_index={'SYNTHETIC':[0],'astronomy':[1],'signal':[2],'retrieval':[3],'root':[4]},referenced_works=['W2'])),
        Paper.from_openalex(dict(id='W2',title='synthetic astronomy signal intermediate',referenced_works=['W3'])),
        Paper.from_openalex(dict(id='W3',title='synthetic astronomy signal foundation'))]
    freeze(papers,tmp_path/'corpus',acquisition=[{'source':'SYNTHETIC engineering diagnostic'}])
    atomic_json(tmp_path/'questions.json',[dict(id='SYNTHETIC-DEV',question=question,split='development')])
    wires=[]
    def factory():
        state={'edit':False,'read':False,'expansions':0}
        def handler(request):
            wire=json.loads(request.content);payload=json.loads(wire['messages'][1]['content']);wires.append(payload)
            clm=payload['protected']['context_management']=='clm'
            if clm and not state['edit']:
                state['edit']=True
                output={'context_edit':{'path':'active_context.txt','op':'append','text':'\nSYNTHETIC retained memo'}}
            elif not state['read']:
                state['read']=True;output={'read_observation':{'index':0,'start':0,'chars':64}}
            elif stop_early and state['expansions']==1:
                output={'decision':{'action':'STOP_RETRIEVAL','arguments':{}}}
            else:
                wid='W1' if state['expansions']==0 else 'W2';state['expansions']+=1
                output={'decision':{'action':'EXPAND_REFERENCES','arguments':{'paper_id':wid}}}
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(output)}}],
                'usage':{'prompt_tokens':100,'completion_tokens':20}})
        return HostedJSONPlanner(Settings(llm_base_url='https://mock.invalid/v1',llm_api_key='SYNTHETIC-NOT-A-KEY',
            llm_model='SYNTHETIC-OFFLINE-MOCK',llm_input_price=1,llm_output_price=2),transport=httpx.MockTransport(handler))
    rows=run_experiments(str(tmp_path/'corpus'),str(tmp_path/'questions.json'),db,factory,
        output=str(tmp_path/'data/benchmark'),suite='controlled-five',trials=1,trial_seeds=(11,),action_budgets=(3,),
        budget=Budget(max_actions=3,max_llm_calls=6,page_size=1,max_rounds=20),completion_guidance=True)
    root=tmp_path/'data/benchmark';manifest=json.loads((root/'experiment.json').read_text())
    validate_schedule(manifest,rows);assert {r['method'] for r in rows}==set(ARCHITECTURES)
    for row in rows:
        run=Path(row['directory']);result=json.loads((run/'result.json').read_text())
        assert row['actions']<=3 and row['llm_calls']<=6 and row['api_requests']==0
        if row['method']=='standard':
            assert row['actions']==1 and row['llm_calls']==0 and result['stop_reason']=='single_pass';continue
        assert result['context_policy_version']=='controlled-context-v2-capacity-stop'
        assert result['stop_reason']==('planner_stop' if stop_early else 'max_actions')
        assert not row['fallbacks'] and result['llm_usage']['input_tokens']==100*row['llm_calls']
        assert result['llm_usage']['output_tokens']==20*row['llm_calls']
        snapshots=[json.loads(p.read_text()) for p in sorted((run/'context').glob('request-*.json'))]
        assert any(p['input'].get('recovered_observation') for p in snapshots)
        assert any('abstract_excerpt' in json.dumps(p['input']) for p in snapshots)
        if row['method'].startswith('clm'):
            assert result['context_edits']==1
            assert any('SYNTHETIC retained memo' in p['input']['active_context'] for p in snapshots[1:])
        else:assert result['context_edits']==0
        if row['method'].endswith('_graph'):
            graph=json.loads((run/'graph.json').read_text())
            assert any(e['source'].endswith('/W1') and e['target'].endswith('/W2') and e['relation']=='PAPER_CITES_PAPER' for e in graph['edges'])
            assert any(v.get('graph') for p in snapshots for v in artifacts._views(p['input'])[0])
        else:
            assert not (run/'graph.json').exists()
            assert not any(set(v)&{'graph','candidates','subgraph','topic_coverage'} for p in snapshots for v in artifacts._views(p['input'])[0])
    # Native saved output -> existing immutable freeze and blank blinded pool.
    # Approval validation is unit-stubbed in memory; no approval file is made.
    with pytest.raises((OSError,ValueError)):
        pooled.freeze_outputs(root,tmp_path/'without-approval',tmp_path/'no-approval.json')
    assert not (tmp_path/'without-approval').exists()
    monkeypatch.setattr(pooled,'approval',lambda *_:{'engineering_diagnostic_only':True})
    original_atomic=pooled.atomic_json
    monkeypatch.setattr(pooled,'atomic_json',lambda p,v:None if Path(p).name=='approval.json' else original_atomic(p,v))
    frozen=tmp_path/'frozen';before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    pooled.freeze_outputs(root,frozen,tmp_path/'in-memory-only.json')
    fm,cells=pooled.load_freeze(frozen);assert len(cells)==5
    pooled.prepare_pool(tmp_path/'corpus',tmp_path/'questions.json',tmp_path/'pool',frozen_outputs=frozen)
    for role in ('A','B'):
        assert all(r['grade']=='UNKNOWN' for r in pooled.csv_rows(tmp_path/f'pool/reviewer-{role}/relevance.csv'))
    assert all(p.read_bytes()==data for path,data in before.items() for p in [Path(path)])
    assert not list(tmp_path.glob('*approval*.json')) and not (frozen/'approval.json').exists()
    # A missing cell cannot become a complete scientific evaluation.
    atomic_json(root/'experiment.json',{**manifest,'status':'incomplete'})
    with pytest.raises(ValueError,match='incomplete'):
        pooled.freeze_outputs(root,tmp_path/'incomplete-freeze',tmp_path/'in-memory-only.json')
    assert not (tmp_path/'incomplete-freeze').exists()


def test_malformed_saved_artifacts_render_readonly_without_hosted_or_history_mutation(tmp_path,monkeypatch):
    import httpx
    from streamlit.testing.v1 import AppTest
    from openalex_agent import web_ui, experiment_dashboard, research
    from openalex_agent.config import Settings
    from test_scientific_evaluation_ui import APP,hashes
    directory=saved(tmp_path)
    path=directory/'runs/clm_graph/context/request-00001.json'
    record=json.loads(path.read_text());record['input']={'active_context':'{"changes":null}'};atomic_json(path,record)
    path=directory/'runs/llm/result.json';record=json.loads(path.read_text());record['llm_usage']=None;atomic_json(path,record)
    def prohibited(*args,**kwargs):pytest.fail('Read-only UI must not start work or make requests')
    monkeypatch.setattr(httpx.Client,'send',prohibited)
    monkeypatch.setattr(experiment_dashboard,'start_dashboard',prohibited)
    monkeypatch.setattr(research,'start_research',prohibited)
    monkeypatch.setattr(web_ui,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(web_ui.Settings,'from_env',lambda:Settings())
    before=hashes(tmp_path);app=AppTest.from_file(APP,default_timeout=20).run()
    assert not app.exception
    assert len([m for m in app.markdown if '<div class="oa-card-title">' in m.value])==5
    assert hashes(tmp_path)==before
