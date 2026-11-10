"""Read-only presentation tests. All authored records are synthetic, unjudged.

No real approval or human judgment is created; no provider is contacted.
"""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import re
import tomllib

import pandas as pd
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

from openalex_agent import web_ui, evaluation_artifacts as artifacts
from openalex_agent.config import Settings
from openalex_agent.controlled import ARCHITECTURES
from openalex_agent.corpus import atomic_json
from openalex_agent.graph import ScholarlyGraph
from openalex_agent.models import Paper
from openalex_agent.ui_theme import METHOD_COLORS, PINE, CREAM, MISSING

APP=str(Path(__file__).resolve().parents[1]/'legacy_app.py')


def widget(app,kind,label):return next(e for e in app.get(kind) if e.label==label)
def hashes(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def saved(root,*,synthetic=False,empty_graph=False):
    directory=root/'data/recorded/benchmark';directory.mkdir(parents=True)
    question=dict(id='SYNTHETIC-DEV',question='SYNTHETIC retrieval question',split='development')
    p=Paper.from_openalex(dict(id='W1',title='SYNTHETIC paper',abstract_inverted_index={'SYNTHETIC':[0],'EVIDENCE':[1]},referenced_works=['W2']))
    q=Paper.from_openalex(dict(id='W2',title='SYNTHETIC cited paper'))
    rows=[]
    for method in ARCHITECTURES:
        run=directory/'runs'/method;run.mkdir(parents=True)
        baseline=method=='standard';graph=method.endswith('_graph')
        usage=dict(calls=0 if baseline else 1,input_tokens=0 if baseline else 100,output_tokens=0 if baseline else 20,
            usage_unreported_calls=0,failures=0,estimated_cost_usd=0 if baseline else .001,seconds=0 if baseline else 2)
        result=dict(run_id=method,architecture=method,question=question['question'],graph_enabled=graph,
            actions=1,inspected_ids=[p.id],stop_reason='single_pass' if baseline else 'planner_stop',
            llm_usage=usage,evidence_records=[],context_policy_version='controlled-context-v2-capacity-stop',
            provider_configuration={'context_policy':'controlled-context-v2-capacity-stop'},
            ranked=[dict(paper=p.to_dict(),score=.5,discovery_path=[p.id])])
        atomic_json(run/'result.json',result)
        atomic_json(run/'state.json',{'question':question['question'],'actions':1,'synthetic_fixture':True})
        events=[dict(kind='start',event_seq=1,architecture=method),
            dict(kind='decision',event_seq=2,action='SEARCH_PAPERS',arguments={'query':'SYNTHETIC'}),
            dict(kind='observation',event_seq=3,actions=1,new_papers=1,inspected=1,retrieved=1)]
        if graph:
            g=ScholarlyGraph();g.add(p);g.add(q);atomic_json(run/'graph.json',g.to_dict())
            # Assembled context is deliberately not proof of model delivery.
            events.append(dict(kind='context',event_seq=4,context={'graph':{'nodes':2},'subgraph':[{'source':p.id,'target':q.id}]}))
        if not baseline:
            (run/'context').mkdir()
            observation={'top_papers':[dict(id=p.id,abstract_excerpt=p.abstract)]}
            if graph and not empty_graph:observation.update(graph={'nodes':2},candidates=[dict(id=p.id,score=.5)],subgraph=[])
            payload={'protected':{'question':question['question']},'active_context':json.dumps({'changes':observation})} if method.startswith('clm') else {'protected':{},'observation':observation}
            atomic_json(run/'context/request-00001.json',dict(call=1,system='SYNTHETIC protected instructions',input=payload,
                output={'decision':{'action':'EXPAND_REFERENCES','arguments':{'paper_id':p.id}}},
                study_tokens={'active_tokens':10,'full_input_tokens':100}))
            events.append(dict(kind='context_model_call',event_seq=5,call=1,seconds=2))
            if method.startswith('clm'):
                atomic_json(run/'context/transition-00001.json',dict(index=1,source='model',committed=True,accepted=True,status='committed',before='SYNTHETIC',after='SYNTHETIC retained',before_chars=9,after_chars=18))
                events.append(dict(kind='context_edit',event_seq=6,committed=True))
                events.append(dict(kind='context_recovery',event_seq=7,accepted=False))
        (run/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        rows.append(dict(query_id=question['id'],method=method,variant='full',trial=0,seed=11,action_budget=3,
            run_id=method,directory='/not/an/allowed/original/path',actions=1,inspected=1,llm_calls=usage['calls'],
            input_tokens=usage['input_tokens'],output_tokens=usage['output_tokens'],estimated_input_tokens=0,estimated_output_tokens=0,
            cost_known=True,total_api_cost_usd=usage['estimated_cost_usd'],end_to_end_seconds=3,
            mean_response_seconds=2 if not baseline else None,fallbacks=0,llm_failures=0,graph_enabled=graph,
            precision_at_10=.99,ndcg_at_10=.99,task_overall_answer_accuracy=.99)) # Untrusted, unapproved fixture values MUST be masked.
    pd.DataFrame(rows).to_csv(directory/'metrics.csv',index=False)
    atomic_json(directory/'experiment.json',dict(suite='controlled-five',questions=[question],methods=list(ARCHITECTURES),
        status='complete',planned_runs=5,deterministic_smoke=synthetic,has_relevance_judgments=False,judgments=None,
        run_order=[{k:r[k] for k in ('query_id','method','variant','trial','action_budget')} for r in rows],
        legacy_context_policy_version='controlled-context-v2-capacity-stop'))
    return directory


def test_actual_readers_preserve_five_methods_and_mask_unapproved_quality(tmp_path):
    directory=saved(tmp_path);before=hashes(tmp_path)
    data=artifacts.load_evaluation(directory,tmp_path)
    assert [r['method'] for r in data['rows']]==list(ARCHITECTURES)
    assert data['classification']=='Unscored development results'
    for row in data['rows']:
        summary=artifacts.summary_row(row)
        assert summary['ndcg'] is summary['factual_accuracy'] is summary['retention'] is None
        if row['method']=='standard':assert summary['model_calls']==0 and summary['cost']==0
        elif row['method'].startswith('clm'):assert summary['committed_edits']==1 and summary['accepted_recoveries']==0
    assert hashes(tmp_path)==before


def test_graph_availability_is_not_actual_delivery_or_causal_utilization(tmp_path):
    directory=saved(tmp_path,empty_graph=True);data=artifacts.load_evaluation(directory,tmp_path)
    row=next(r for r in data['rows'] if r['method']=='llm_graph')
    assert row['detail']['graph_snapshot']['nodes']>0
    assert row['detail']['graph_delivery']['structured_graph_calls']==0
    assert any(e['kind']=='context' for e in row['detail']['events'])
    assert artifacts.graph_delivery([{'request':{'input':{'protected':{'graph':{'nodes':10}}}}}])['structured_graph_calls']==0


def test_graph_payload_candidate_overlap_and_absent_citation_edges_stay_distinct(tmp_path):
    data=artifacts.load_evaluation(saved(tmp_path),tmp_path)
    row=next(r for r in data['rows'] if r['method']=='clm_graph')
    detail=row['detail']['graph_delivery']
    assert detail['structured_graph_calls']==detail['candidate_calls']==detail['action_target_overlap_calls']==1
    assert detail['citation_edge_calls']==0
    assert len(detail['examples'][0]['delivered'])==1
    free=artifacts.graph_delivery([{'request':{'input':{'active_context':'Unstructured research note'}}}])
    assert free['unstructured_context_calls']==1


def test_partial_corrupt_or_missing_artifacts_do_not_fabricate_outcomes(tmp_path):
    directory=saved(tmp_path)
    (directory/'runs/clm/result.json').unlink()
    (directory/'runs/clm/context/request-00001.json').write_text('{unfinished')
    with (directory/'runs/llm/events.jsonl').open('a') as f:f.write('{incomplete tail')
    data=artifacts.load_evaluation(directory,tmp_path)
    row=next(r for r in data['rows'] if r['method']=='clm')
    assert not row['detail']['result'] and row['detail']['warnings']
    assert artifacts.summary_row(row)['retention'] is None
    assert artifacts.number('not recorded') is None and artifacts.number(float('nan')) is None


def test_only_primary_cells_are_compared_and_diagnostics_remain_labelled(tmp_path):
    directory=saved(tmp_path,synthetic=True)
    frame=pd.read_csv(directory/'metrics.csv');extra=frame.iloc[0].copy();extra['method']='meta_clm'
    pd.concat([frame,pd.DataFrame([extra])]).to_csv(directory/'metrics.csv',index=False)
    data=artifacts.load_evaluation(directory,tmp_path)
    assert len(data['rows'])==5 and data['excluded_methods']==['meta_clm']
    assert data['classification']=='Synthetic diagnostics'


def test_historical_clm_policy_is_not_presented_as_current_corrected_results(tmp_path):
    directory=saved(tmp_path)
    for method in ('clm','clm_graph'):
        path=directory/'runs'/method/'result.json';record=json.loads(path.read_text())
        record['context_policy_version']='controlled-context-v1';atomic_json(path,record)
    data=artifacts.load_evaluation(directory,tmp_path)
    assert data['classification']=='Historical unscored development results'
    assert data['context_policies']==['controlled-context-v1']


def test_unknown_provider_usage_does_not_become_zero_cost_or_known_total(tmp_path):
    directory=saved(tmp_path);p=directory/'runs/llm/result.json';r=json.loads(p.read_text())
    r['llm_usage'].update(usage_unreported_calls=1,estimated_cost_usd=None);atomic_json(p,r)
    frame=pd.read_csv(directory/'metrics.csv');frame.loc[frame.method=='llm','cost_known']=False
    frame.to_csv(directory/'metrics.csv',index=False)
    row=next(r for r in artifacts.load_evaluation(directory,tmp_path)['rows'] if r['method']=='llm')
    s=artifacts.summary_row(row)
    assert s['cost'] is s['reported_tokens'] is s['input_tokens'] is s['output_tokens'] is None


def test_unmatched_assessment_and_path_escape_are_rejected(tmp_path):
    directory=saved(tmp_path);data=artifacts.load_evaluation(directory,tmp_path,assessment=directory)
    assert not data['pooled_assessment'] and any('not accepted' in w for w in data['warnings'])
    with pytest.raises(ValueError):artifacts.load_evaluation(tmp_path.parent,tmp_path)


def test_pooled_export_requires_recorded_approval_and_exact_freeze_policy_pins(tmp_path,monkeypatch):
    from openalex_agent import pooled_review
    freeze=tmp_path/'data/frozen';assessment=tmp_path/'data/assessed'
    freeze.mkdir(parents=True);assessment.mkdir()
    pins={'frozen_outputs_sha256':'SYNTHETIC-FREEZE','policy_sha256':pooled_review.digest(pooled_review.POLICY)}
    # Entire approval/score payload is in memory: no approval or human labels
    # are written. The fixture tests validation, not scientific outcomes.
    score={'protocol':pooled_review.VERSION,'immutable_inputs':pins,'rows':[dict(query_id='SYNTHETIC',
        method='llm',variant='full',trial=0,action_budget=3,ndcg_at_10_pooled=.5)],
        'scientific_approval':{'protocol':pooled_review.VERSION,'stage':'score-preserved','approved':True,
            'approver':'SYNTHETIC MOCK','approved_at':'SYNTHETIC MOCK','pins':pins}}
    monkeypatch.setattr(pooled_review,'load_freeze',lambda _:({'policy':pooled_review.POLICY},[]))
    monkeypatch.setattr(pooled_review,'sha',lambda _:'SYNTHETIC-FREEZE')
    monkeypatch.setattr(artifacts,'safe_json',lambda *a,**k:score)
    assert len(artifacts._pooled_scores(freeze,assessment,tmp_path))==1
    score['scientific_approval']['stage']='collect-development'
    with pytest.raises(ValueError):artifacts._pooled_scores(freeze,assessment,tmp_path)
    assert not list(tmp_path.rglob('*.json'))


def test_empty_metrics_partial_trajectories_and_malformed_identity_are_explicit(tmp_path):
    directory=saved(tmp_path)
    frame=pd.read_csv(directory/'metrics.csv');frame['trial']=frame['trial'].astype(float)
    frame.loc[frame.method=='standard','trial']=.5;frame.to_csv(directory/'metrics.csv',index=False)
    data=artifacts.load_evaluation(directory,tmp_path)
    assert len(data['rows'])==4 and any('identity omitted' in w for w in data['warnings'])
    (directory/'metrics.csv').write_text('')
    data=artifacts.load_evaluation(directory,tmp_path)
    assert not data['rows'] and len(data['partial_runs'])==5


def test_readonly_scientific_workspace_renders_all_methods_and_controls_without_api_or_mutation(tmp_path,monkeypatch):
    import httpx
    from openalex_agent import experiment_dashboard, research, pooled_review
    directory=saved(tmp_path);before=hashes(tmp_path)
    def forbidden(*a,**k):pytest.fail('Saved inspection must not execute retrieval, scoring or HTTP')
    monkeypatch.setattr(httpx.Client,'send',forbidden)
    monkeypatch.setattr(experiment_dashboard,'start_dashboard',forbidden)
    monkeypatch.setattr(research,'start_research',forbidden)
    monkeypatch.setattr(pooled_review,'score_pool',forbidden)
    monkeypatch.setattr(web_ui,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(web_ui.Settings,'from_env',lambda:Settings())
    ui=AppTest.from_file(APP,default_timeout=20).run()
    assert not ui.exception
    assert widget(ui,'radio','Workspace').value=='Scientific Evaluation'
    assert len([m for m in ui.markdown if '<div class="oa-card-title">' in m.value])==5
    assert any(f.value.astype(str).eq(MISSING).any().any() for f in ui.dataframe)
    widget(ui,'selectbox','Architecture to inspect').select_index(2).run()
    widget(ui,'selectbox','Ranked architecture').select_index(3).run()
    widget(ui,'multiselect','Event types').set_value(['Context']).run()
    assert not ui.exception
    assert any('not causal' in c.value for c in ui.caption)
    assert hashes(tmp_path)==before


def test_no_emoji_ui_chrome_and_nature_theme_is_valid_and_consistent():
    root=Path(APP).parent
    emoji=re.compile('[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]')
    for name in ['web_ui.py','dashboard_ui.py','evaluation_ui.py','ui_theme.py']:
        tree=ast.parse((root/'openalex_agent'/name).read_text())
        assert not any(emoji.search(n.value) for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str))
    cfg=tomllib.loads((root/'.streamlit/config.toml').read_text())['theme']
    assert cfg['light']['primaryColor']==PINE and cfg['light']['backgroundColor']==CREAM
    assert cfg['light']['chartCategoricalColors']==list(METHOD_COLORS.values())
    # Forest text on cream comfortably exceeds WCAG normal-text contrast.
    def luminance(color):
        channels=[int(color[i:i+2],16)/255 for i in (1,3,5)]
        channels=[c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in channels]
        return sum(a*b for a,b in zip(channels,[.2126,.7152,.0722]))
    assert (luminance(CREAM)+.05)/(luminance(PINE)+.05)>4.5


def test_dark_palette_and_aligned_card_layout():
    from openalex_agent.ui_theme import palette, method_colors
    root=Path(APP).parent
    cfg=tomllib.loads((root/'.streamlit/config.toml').read_text())['theme']
    assert cfg['base']=='dark'
    assert cfg['dark']['backgroundColor']==palette('dark')['background']
    assert cfg['dark']['textColor']==palette('dark')['text']
    assert cfg['dark']['chartCategoricalColors']==list(method_colors('dark').values())
    assert set(method_colors('dark'))==set(METHOD_COLORS)
    assert palette('light')['background']==CREAM
    def luminance(color):
        c=[int(color[i:i+2],16)/255 for i in (1,3,5)]
        c=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in c]
        return sum(v*w for v,w in zip(c,[.2126,.7152,.0722]))
    for mode in ('light','dark'):
        p=palette(mode)
        for fg in ('text','muted','accent'):
            a,b=sorted([luminance(p[fg]),luminance(p['surface'])])
            assert (b+.05)/(a+.05)>4.5
    source=(root/'openalex_agent/ui_theme.py').read_text()
    assert 'grid-template-rows:32px 48px 42px 45px auto' in source
    assert "vertical_alignment='bottom'" in (root/'openalex_agent/evaluation_ui.py').read_text()


def test_native_appearance_switch_does_not_keep_server_side_text_colors():
    from openalex_agent.ui_theme import chart_style
    import altair as alt
    source=(Path(APP).parent/'openalex_agent/ui_theme.py').read_text()
    assert '--oa-scheme' not in source
    assert 'color-scheme:' not in source
    chart=chart_style(alt.Chart().mark_point()).to_dict()
    assert 'background' not in chart['config']
    assert not {'labelColor','titleColor'} & chart['config']['axis'].keys()
    assert 'legend' not in chart['config']
