"""Transport mocks and synthetic arithmetic, not hosted or scientific results."""
import json

import httpx
import pytest

from openalex_agent.config import Settings
from openalex_agent.factorial import contrasts_2x3, paired_comparisons_2x3
from openalex_agent.llm import PlannerError
from openalex_agent.responses import HostedResponsesPlanner, response_input


def config():
    return Settings(llm_base_url='https://provider.example/v1',llm_api_key='SECRET-SYNTHETIC',
        llm_model='gpt-6.1-terra',llm_reasoning_effort='low',llm_temperature=None,llm_max_tokens=4096)


@pytest.mark.parametrize('status,code',[(401,'authentication_error'),(429,'rate_limited'),(503,'http_error')])
def test_responses_errors_are_classified_with_no_retries(status,code):
    calls=[]
    def handler(request):calls.append(request);return httpx.Response(status,json={'error':'SECRET-SYNTHETIC'})
    with HostedResponsesPlanner(config(),transport=httpx.MockTransport(handler)) as p:
        with pytest.raises(PlannerError,match=code):p._call('Return JSON',{})
        assert len(calls)==p.usage.calls==p.usage.failures==1
        assert p.usage.usage_unreported_calls==1 and p.usage.estimated_cost_usd is None
        assert 'SECRET-SYNTHETIC' not in json.dumps(p.last_call)


@pytest.mark.parametrize('error,code',[(httpx.ReadTimeout,'timeout'),(httpx.ConnectError,'transport_error')])
def test_responses_transport_failures_remain_distinct(error,code):
    calls=[]
    def handler(request):calls.append(1);raise error('PRIVATE provider failure',request=request)
    with HostedResponsesPlanner(config(),transport=httpx.MockTransport(handler)) as p:
        with pytest.raises(PlannerError,match=code):p._call('Return JSON',{})
        assert len(calls)==1 and p.last_call['error_code']==code


def test_responses_json_mode_preserves_profile_and_unknown_usage():
    bodies=[]
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200,json={'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"decision": {}}'}]}]})
    with HostedResponsesPlanner(config(),transport=httpx.MockTransport(handler)) as p:
        assert p._call('Return JSON',{'question':'SYNTHETIC'})=={'decision':{}}
        assert p.usage.estimated_cost_usd is None and p.usage.usage_unreported_calls==1
        assert bodies[0]['model']=='gpt-6.1-terra' and bodies[0]['reasoning']=={'effort':'low'}
        assert bodies[0]['max_output_tokens']==4096 and bodies[0]['store'] is False
        assert 'previous_response_id' not in bodies[0] and 'temperature' not in bodies[0]


def test_native_history_conversion_keeps_function_identity_and_outputs():
    messages=[{'role':'assistant','content':'','tool_calls':[{'id':'c1','function':{'name':'INSPECT_PAPER','arguments':'{"paper_id":"W2"}'}}]},
        {'role':'tool','tool_call_id':'c1','content':'SYNTHETIC observation'}]
    assert response_input(messages)==[dict(type='function_call',call_id='c1',name='INSPECT_PAPER',arguments='{"paper_id":"W2"}'),
                                      dict(type='function_call_output',call_id='c1',output='SYNTHETIC observation')]


@pytest.mark.parametrize('output,code',[
    ([{'type':'message','content':None}],'invalid_envelope'),
    ([{'type':'message','content':[{'type':'refusal','refusal':'no'}]}],'refused'),
    ([{'type':'message','content':[{'type':'output_text','text':'not JSON'}]}],'invalid_json')])
def test_malformed_native_envelopes_fail_after_preserving_reported_usage(output,code):
    with HostedResponsesPlanner(config(),transport=httpx.MockTransport(lambda _:httpx.Response(200,json={
        'status':'completed','output':output,'usage':{'input_tokens':20,'output_tokens':5}}))) as p:
        with pytest.raises(PlannerError,match=code):p._call('Return JSON',{})
        assert p.usage.input_tokens==20 and p.usage.output_tokens==5 and p.usage.usage_reported_calls==1


def factorial_rows():
    values={'llm':0,'llm_graph':1,'clm':2,'clm_graph':3,'meta_clm':4,'meta_clm_graph':6,'standard':999}
    return [dict(query_id=q,method=m,trial=t,action_budget=10,score=v) for q in ['SYNTHETIC-Q1','SYNTHETIC-Q2']
            for m,v in values.items() for t in (0,1)]


def test_2x3_contrasts_include_meta_and_interactions_without_standard_or_fake_significance():
    contrasts=contrasts_2x3()
    assert len(contrasts)==16 and sum(c['primary'] for c in contrasts)==5
    rows=factorial_rows();result=paired_comparisons_2x3(rows,'score')
    by_name={r['contrast']:r for r in result}
    assert by_name['graph_main_effect']['mean_difference']==pytest.approx(4/3)
    assert by_name['inspired_minus_conventional']['mean_difference']==2
    assert by_name['meta_minus_conventional']['mean_difference']==4.5
    assert by_name['meta_minus_inspired']['mean_difference']==2.5
    assert by_name['graph_interaction_meta_minus_conventional']['mean_difference']==1
    assert all(r['questions']==2 and 'standard' not in r['weights'] and 'p_value' not in r for r in result)
    assert result==paired_comparisons_2x3(rows,'score')


def test_2x3_missing_grades_exclude_whole_question_only_for_affected_contrasts():
    rows=factorial_rows()
    next(r for r in rows if r['query_id']=='SYNTHETIC-Q2' and r['method']=='meta_clm_graph')['score']=None
    result={r['contrast']:r for r in paired_comparisons_2x3(rows,'score')}
    assert result['graph_main_effect']['questions']==1 and result['meta_minus_conventional']['questions']==1
    assert result['inspired_minus_conventional']['questions']==2
    assert result['graph_main_effect']['excluded_questions']=={'SYNTHETIC-Q2':'missing_or_nonfinite_metric_in_trial'}
    with pytest.raises(ValueError,match='Duplicate'):paired_comparisons_2x3(rows+[rows[0]],'score')
    with pytest.raises(ValueError,match='Unbalanced'):paired_comparisons_2x3(rows[1:],'score')
