"""Synthetic preparation/guard checks; never hosted stopping-behavior results."""
from copy import deepcopy
from dataclasses import asdict
import csv
import json

import pytest

from openalex_agent.context_memory import serialized
from openalex_agent.context_planner import ContextPlanner, InferenceLimits
from openalex_agent.controlled import ControlledController
from openalex_agent.controlled_diagnostics import SeededCorpus
from openalex_agent.controller import Budget
from openalex_agent.gemini import GeminiSettings
from openalex_agent.meta_planner import MetaContextPlanner
from openalex_agent.research_completion import VERSION, SUFFIX
from openalex_agent import stopping_probe as probe
from test_gemini_adc import corpus, model, response


@pytest.mark.parametrize('architecture', ['llm', 'llm_graph', 'clm', 'clm_graph', 'meta_clm', 'meta_clm_graph'])
def test_same_opt_in_guidance_reaches_all_six_real_wrappers(tmp_path, db, architecture):
    def script(body, n):
        if 'tools' in body['config']: return response(name='STOP_RETRIEVAL')
        return response(text=serialized(dict(decision=dict(action='STOP_RETRIEVAL', arguments={}))))
    base, wire = model(script)
    planner = (MetaContextPlanner(base, completion_guidance=True) if architecture.startswith('meta_') else
        ContextPlanner(base, mode='clm' if architecture.startswith('clm') else 'conventional', completion_guidance=True))
    budget = Budget(max_actions=10, max_llm_calls=20, page_size=1)
    controller = ControlledController('SYNTHETIC research', SeededCorpus(corpus(tmp_path), [1]), db,
        architecture=architecture, planner=planner, budget=budget, output=str(tmp_path/'runs'))
    try:
        result = controller.run()
        assert len(wire) == 1 and wire[0]['config']['system_instruction'].endswith(SUFFIX)
        assert wire[0]['config']['system_instruction'].count(SUFFIX) == 1
        assert planner.configuration['research_completion_guidance'] == VERSION
        assert result['budget'] == asdict(budget) and result['stop_reason'] == 'planner_stop'
        assert ('tools' in wire[0]['config']) == architecture.startswith('meta_')
    finally: planner.close()


def synthetic_proposal(tmp_path, monkeypatch):
    pilot = tmp_path/'pilot'; (pilot/'proposal/execution/benchmark').mkdir(parents=True)
    monkeypatch.setattr(probe, 'PILOT', pilot)
    provider = {**asdict(GeminiSettings(project='SYNTHETIC-PROJECT')), 'auth':'ADC'}
    questions = [dict(id=q, question='SYNTHETIC '+q, split='development') for q in probe.QUESTIONS]
    qpath = pilot/'questions.json'; qpath.write_text(json.dumps(questions))
    old = dict(provider=provider, budget=asdict(Budget(max_actions=10, max_llm_calls=20)),
        limits=asdict(InferenceLimits(max_active_tokens=4096, max_input_tokens=8192)),
        corpus_sha256='SYNTHETIC-NO-CORPUS', questions=str(qpath))
    (pilot/'proposal/spec.json').write_text(json.dumps(old))
    rows = []
    for q in questions:
        for method in probe.METHODS:
            run = pilot/f"{q['id']}-{method}"; (run/'context').mkdir(parents=True)
            protected = dict(question=q['question'], remaining={'actions':9})
            payload = dict(protected=protected, observation={'top_papers':[]})
            if method == 'clm': payload = dict(protected=protected, active_context='SYNTHETIC OBSERVATION')
            if method == 'meta_clm':
                payload = dict(protected=protected, tools=[], messages=[dict(role='system', content='SYNTHETIC SYSTEM'),
                    dict(role='user', content=q['question']), dict(role='user', content='SYNTHETIC OBSERVATION'),
                    dict(role='user', content='SYNTHETIC FROZEN BUDGET NUDGE'), dict(role='user', content=serialized(protected))])
            record = dict(system='SYNTHETIC SYSTEM', input=payload,
                diagnostics=dict(prompt_tokens=8, completion_tokens=100))
            (run/'context/request-00001.json').write_text(json.dumps(record))
            rows.append(dict(method=method, query_id=q['id'], trial='0', directory=str(run)))
    with (pilot/'proposal/execution/benchmark/metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    output = tmp_path/'proposal'
    assert probe.prepare(output)['verified_cells'] == 12
    return output


def test_matched_matrix_has_only_system_suffix_and_preserves_nudges_tools_and_limits(tmp_path, monkeypatch):
    out = synthetic_proposal(tmp_path, monkeypatch)
    spec = json.loads((out/'spec.json').read_text())
    for q in probe.QUESTIONS:
        for method in probe.METHODS:
            cells = {c['condition']:json.loads(open(c['request']).read()) for c in spec['cases'] if c['query_id']==q and c['method']==method}
            original, clarified = cells['original'], deepcopy(cells['clarified'])
            assert clarified['system'] == original['system']+SUFFIX
            clarified['system'] = original['system']
            clarified['body']['config']['system_instruction'] = original['body']['config']['system_instruction']
            if method == 'meta_clm': clarified['payload']['messages'][0]['content'] = original['system']
            clarified['study_tokens'] = original['study_tokens']
            assert clarified == original
    assert probe.verify(out/'spec.json')['execution_authorized'] is False


def test_unapproved_probe_cannot_create_execution_or_send_requests(tmp_path, monkeypatch):
    out = synthetic_proposal(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='explicit approval'):
        probe.execute(out/'spec.json', out/'approval-template.json', transport=lambda _:pytest.fail('No inference'))
    assert not (out/'execution').exists()


@pytest.mark.parametrize('change', ['system', 'provider', 'duplicate'])
def test_changed_request_provider_or_matrix_is_rejected_offline(tmp_path, monkeypatch, change):
    from pathlib import Path
    out = synthetic_proposal(tmp_path, monkeypatch)
    spec = json.loads((out/'spec.json').read_text())
    if change == 'provider': spec['provider']['thinking_level'] = 'HIGH'
    elif change == 'duplicate': spec['cases'][1] = deepcopy(spec['cases'][0])
    else:
        file = Path(spec['cases'][0]['request'])
        value = json.loads(file.read_text()); value['body']['config']['system_instruction'] += 'UNAPPROVED EXTRA'
        file.write_text(json.dumps(value)); spec['cases'][0]['sha256'] = probe.sha(file)
    (out/'spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError): probe.verify(out/'spec.json')


def test_mocked_runner_executes_at_most_twelve_single_responses_without_tools_or_retries(tmp_path, monkeypatch):
    out = synthetic_proposal(tmp_path, monkeypatch)
    approval = json.loads((out/'approval-template.json').read_text())
    approval.update(approved=True, paid_inference_authorized=True, approver='SYNTHETIC TEST', approved_at='SYNTHETIC TEST')
    (out/'synthetic-approval.json').write_text(json.dumps(approval))
    calls = []
    def script(body):
        calls.append(body)
        return response(name='STOP_RETRIEVAL') if 'tools' in body['config'] else response(text='{"decision":{"action":"STOP_RETRIEVAL","arguments":{}}}')
    result = probe.execute(out/'spec.json', out/'synthetic-approval.json', transport=script)
    assert len(calls) == result['provider_attempts'] == result['completed_cells'] == 12
    assert result['classification'] == 'MOCKED OFFLINE TEST' and result['action_execution'] is False


def test_unknown_usage_stops_after_first_mocked_response(tmp_path, monkeypatch):
    out = synthetic_proposal(tmp_path, monkeypatch)
    approval = json.loads((out/'approval-template.json').read_text())
    approval.update(approved=True, paid_inference_authorized=True, approver='SYNTHETIC TEST', approved_at='SYNTHETIC TEST')
    (out/'synthetic-approval.json').write_text(json.dumps(approval))
    calls = []
    def script(body):
        calls.append(body)
        return response(text='{}', usage={'prompt_token_count':8, 'candidates_token_count':2})
    result = probe.execute(out/'spec.json', out/'synthetic-approval.json', transport=script)
    assert len(calls) == 1 and result['stop_reason'] == 'probe_unaccountable_usage'
    assert result['retained_usd'] == '2.13072'
