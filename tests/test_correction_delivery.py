"""Delivery validity is independent of whether the fixed script answers correctly."""
from openalex_agent.context_pressure import run_pressure


def test_medium_correction_is_delivered_before_selected_source_ages_out(tmp_path):
    architectures=('llm','llm_graph','clm','clm_graph')
    repaired=run_pressure('medium',tmp_path/'repaired',seed=11,correction_at=4,architectures=architectures,return_invalid=True)
    assert repaired['pressure_valid'] and repaired['shared_observation_archives']
    assert repaired['pressure_ratio']>1 and repaired['unique_source_pressure_ratio']>1
    assert len(repaired['rows'])==4
    for arm in repaired['rows']:
        assert arm['initial_evidence_delivered_to_model'] and arm['correction_delivered_to_model']
        assert arm['all_stream_source_excerpts_delivered_to_model'] and arm['requests_all_within_limits']
        assert arm['factual_errors']==1  # Do not make a validity check depend on CLM winning.
    invalid=run_pressure('medium',tmp_path/'old-placement',seed=11,correction_at=16,architectures=architectures,return_invalid=True)
    assert not invalid['pressure_valid']
    assert all(not arm['correction_delivered_to_model'] for arm in invalid['rows'])
