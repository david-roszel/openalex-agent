"""Approval-gated, single-response replay of preserved development first requests.

No controller actions execute. Preparation and verification never authenticate.
"""
from copy import deepcopy
from dataclasses import fields
from decimal import Decimal
import json
from pathlib import Path
import time

from .context_memory import serialized
from .context_planner import ContextPlanner, InferenceLimits
from .controller import Budget
from .corpus import atomic_json
from .execution_control import ExperimentStopped
from .gemini import GeminiADCPlanner, GeminiSettings
from .gemini_validation import GeminiSpendLedger, GeminiValidationBackend
from .meta_planner import MetaContextPlanner
from .meta_validation import code_pins
from .pooled_review import sha
from .research_completion import VERSION as GUIDANCE_VERSION, SUFFIX

VERSION = "meta-stopping-first-request-v1"
PILOT = Path("data/gemini-seven-development-pilot-2026-11-06-v1")
METHODS = ("llm", "clm", "meta_clm")
QUESTIONS = ("Q-CS-02-3", "Q-PHY-03-2")


def settings_for(provider):
    return GeminiSettings(**{f.name: provider[f.name] for f in fields(GeminiSettings)})


def request_case(record, method, clarified, settings, budget, limits):
    payload = deepcopy(record["input"])
    system = record["system"] + (SUFFIX if clarified else "")
    base = GeminiADCPlanner(settings)
    if method == "meta_clm":
        assert payload["messages"][0] == {"role": "system", "content": record["system"]}
        payload["messages"][0]["content"] = system
        planner = MetaContextPlanner(base)
        body = base.body(payload["messages"], payload["tools"])
    else:
        planner = ContextPlanner(base, mode="clm" if method == "clm" else "conventional")
        body = base.body([dict(role="system", content=system),
                          dict(role="user", content=serialized(payload))])
    planner.budget, planner.limits = Budget(**budget), InferenceLimits(**limits)
    try:
        if not planner.payload_fits(payload, system=system):
            raise ValueError("Frozen input exceeds unchanged method capacity")
        measurements = planner.token_measurements(system, payload)
    finally:
        planner.close()
    return dict(system=system, payload=payload, body=body, study_tokens=measurements)


def prepare(output):
    import csv
    out = Path(output)
    if out.exists():
        raise ValueError("Use a fresh proposal directory")
    parent = PILOT / "proposal/spec.json"
    old = json.loads(parent.read_text())
    settings = settings_for(old["provider"])
    rows = list(csv.DictReader((PILOT / "proposal/execution/benchmark/metrics.csv").open()))
    selected = [r for r in rows if r["method"] in METHODS and r["query_id"] in QUESTIONS and r["trial"] == "0"]
    if len(selected) != 6 or len({(r['method'], r['query_id']) for r in selected}) != 6:
        raise ValueError("Expected six unique development snapshots")
    out.mkdir(parents=True)
    sources, cases, historical_cost = {old['questions']: sha(old['questions'])}, [], Decimal(0)
    # Counterbalance condition order without changing the preserved trial input.
    for n, (query, method) in enumerate((q, m) for q in QUESTIONS for m in METHODS):
        row = next(r for r in selected if r["query_id"] == query and r["method"] == method)
        source = Path(row["directory"]) / "context/request-00001.json"
        record = json.loads(source.read_text())
        if record['input']['protected']['question'] != next(q['question'] for q in json.loads(Path(old['questions']).read_text()) if q['id'] == query and q['split'] == 'development'):
            raise ValueError("Selected input is not the original DEV question")
        sources[str(source)] = sha(source)
        historical_cost += (Decimal(record['diagnostics']['prompt_tokens'])*Decimal('1.5') +
                            Decimal(record['diagnostics']['completion_tokens'])*Decimal('7.5'))/1000000
        for clarified in ((False, True) if n % 2 == 0 else (True, False)):
            condition = "clarified" if clarified else "original"
            name = f"{query}-{method}-{condition}"
            value = request_case(record, method, clarified, settings, old['budget'], old['limits'])
            path = out / "requests" / f"{name}.json"
            atomic_json(path, value)
            cases.append(dict(id=name, query_id=query, method=method, condition=condition,
                              request=str(path), sha256=sha(path), source=str(source)))
    spec = dict(protocol=VERSION, guidance_version=GUIDANCE_VERSION, guidance_suffix=SUFFIX,
        purpose="UNSCORED DEVELOPMENT FIRST-DECISION PROMPT PROBE ONLY", provider=old['provider'],
        budget=old['budget'], limits=old['limits'], corpus_sha256=old['corpus_sha256'],
        methods=list(METHODS), questions=list(QUESTIONS), repetitions=1, source_trial=0, source_seed=11,
        conditions=['original', 'clarified'], cases=cases, source_files=sources,
        parent_spec=str(parent), parent_spec_sha256=sha(parent), source_pins=code_pins(),
        max_provider_calls=12, spending_ceiling_usd=6, checkpoint_usd=3, wall_seconds=7200,
        automatic_retries=0, actions_executed=False, final_evaluation_authorized=False,
        output=str(out/'execution'),
        estimate=dict(historical_six_first_request_cost_usd=str(historical_cost),
            twelve_calls_at_historical_usage_usd=str(historical_cost*2),
            caveat="Projection only; clarification adds input tokens and can change output length. No credit discounts.",
            study_token_upper_envelope_usd=str(12*(Decimal(8192)*Decimal('1.5')+Decimal(4096)*Decimal('7.5'))/1000000),
            envelope_is_billing_guarantee=False))
    atomic_json(out/'spec.json', spec)
    atomic_json(out/'approval-template.json', dict(protocol=VERSION, approved=False,
        paid_inference_authorized=False, final_evaluation_authorized=False,
        spec_sha256=sha(out/'spec.json'), max_provider_calls=12, spending_ceiling_usd=6,
        checkpoint_usd=3, approver='', approved_at=''))
    return verify(out/'spec.json')


def verify(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if (spec['protocol'] != VERSION or spec['guidance_version'] != GUIDANCE_VERSION or
        spec['guidance_suffix'] != SUFFIX or spec['source_pins'] != code_pins() or
        spec['max_provider_calls'] != 12 or spec['spending_ceiling_usd'] != 6 or
        spec['checkpoint_usd'] != 3 or spec['wall_seconds'] != 7200 or spec['automatic_retries'] != 0 or
        spec['methods'] != list(METHODS) or spec['questions'] != list(QUESTIONS) or
        spec['repetitions'] != 1 or spec['source_trial'] != 0 or
        spec['actions_executed'] is not False or spec['final_evaluation_authorized'] is not False):
        raise ValueError("Probe protocol, source pins or safeguards changed")
    if sha(spec['parent_spec']) != spec['parent_spec_sha256']:
        raise ValueError("Historical specification changed")
    old = json.loads(Path(spec['parent_spec']).read_text())
    for key in ('provider', 'budget', 'limits', 'corpus_sha256'):
        if spec[key] != old[key]:
            raise ValueError("Historical provider/resource profile changed")
    if any(sha(p) != checksum for p, checksum in spec['source_files'].items()):
        raise ValueError("Historical input changed")
    if len(spec['cases']) != 12:
        raise ValueError("Exactly twelve cells required")
    seen = set()
    settings = settings_for(spec['provider'])
    development = {q['id']: q['question'] for q in json.loads(Path(old['questions']).read_text())
                   if q.get('split') == 'development' and q['id'] in QUESTIONS}
    for case in spec['cases']:
        key = (case['query_id'], case['method'], case['condition'])
        if key in seen or key[0] not in QUESTIONS or key[1] not in METHODS or key[2] not in ('original', 'clarified'):
            raise ValueError("Invalid or duplicate matrix cell")
        seen.add(key)
        if case['source'] not in spec['source_files'] or sha(case['request']) != case['sha256']:
            raise ValueError("Unpinned request")
        record = json.loads(Path(case['source']).read_text())
        if record['input']['protected']['question'] != development.get(case['query_id']):
            raise ValueError("Request is not matched to its original development question")
        expected = request_case(record, case['method'], case['condition']=='clarified', settings, spec['budget'], spec['limits'])
        if json.loads(Path(case['request']).read_text()) != expected:
            raise ValueError("Request differs from exact frozen input plus optional shared suffix")
    return dict(protocol=VERSION, verified_cells=12, original_inputs=6,
        sole_intervention='shared system-instruction suffix', provider_and_tools_unchanged=True,
        budget_nudges_and_evidence_unchanged=True, capacity_checks_passed=True,
        hosted_requests=0, execution_authorized=False)


def execute(spec_path, approval_path, *, transport=None):
    verify(spec_path)
    spec = json.loads(Path(spec_path).read_text())
    approval = json.loads(Path(approval_path).read_text())
    expected = dict(protocol=VERSION, approved=True, paid_inference_authorized=True,
        final_evaluation_authorized=False, spec_sha256=sha(spec_path), max_provider_calls=12,
        spending_ceiling_usd=6, checkpoint_usd=3)
    if any(approval.get(k) != v for k, v in expected.items()) or not approval.get('approver') or not approval.get('approved_at'):
        raise ValueError("Separate explicit approval of this exact twelve-call probe is required")
    out = Path(spec['output'])
    if out.exists():
        raise ValueError("Never resume, overwrite or automatically retry a probe")
    out.mkdir(parents=True)
    atomic_json(out/'spec.json', spec)
    atomic_json(out/'approval.json', approval)
    ledger = GeminiSpendLedger(out/'spending-ledger.json', spec['spending_ceiling_usd'],
        checkpoint=spec['checkpoint_usd'], max_calls=12, max_seconds=spec['wall_seconds'])
    factory = GeminiValidationBackend.guarded_factory(settings_for(spec['provider']), ledger, transport=transport)
    results, stop = [], 'completed'
    try:
        for case in spec['cases']:
            ledger.check()
            planner = factory()
            planner.request_input_limit = spec['limits']['max_input_tokens']
            value = json.loads(Path(case['request']).read_text())
            result = dict(id=case['id'], method=case['method'], condition=case['condition'],
                          query_id=case['query_id'], status='running', body=value['body'])
            try:
                result['response'] = planner._request(value['body'])
                result['status'] = 'returned'  # Parsed response is not semantic action validity.
            except Exception as exc:
                result.update(status='failed', error_code=getattr(exc, 'code', getattr(exc, 'reason', type(exc).__name__)))
            finally:
                result.update(diagnostics=planner.last_call, provider_response=planner.last_provider_response,
                              wire_requests=planner.last_wire_requests)
                planner.close()
                atomic_json(out/f"{case['id']}.json", result)
                results.append(result)
            if planner.usage.usage_unreported_calls:
                raise ExperimentStopped('probe_unaccountable_usage')
            if planner.usage.input_tokens+planner.usage.output_tokens > spec['limits']['max_total_tokens']:
                raise ExperimentStopped('probe_cell_token_ceiling')
            if ledger.stop_reason:
                raise ExperimentStopped(ledger.stop_reason)
            if ledger.reported_spend >= Decimal(str(spec['checkpoint_usd'])):
                raise ExperimentStopped('pilot_three_dollar_checkpoint')
    except ExperimentStopped as exc:
        stop = exc.reason
    finally:
        summary = dict(protocol=VERSION, classification='MOCKED OFFLINE TEST' if transport else 'HOSTED DEVELOPMENT PROBE',
            completed_cells=len(results), provider_attempts=len(ledger.attempts), stop_reason=stop,
            gross_reported_cost_usd=str(ledger.reported_spend), retained_usd=str(ledger.charged),
            runtime_seconds=time.monotonic()-ledger.started, action_execution=False)
        atomic_json(out/'summary.json', summary)
    return summary
