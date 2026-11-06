"""C1–C7 mechanism tests with scripted HTTP responses, not model-quality claims."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from openalex_agent.config import Settings
from openalex_agent.context_memory import ContextStore, serialized
from openalex_agent.context_planner import ContextPlanner, InferenceLimits
from openalex_agent.controller import Budget
from openalex_agent.llm import HostedJSONPlanner, PlannerError
from openalex_agent.models import Paper
from openalex_agent.openalex import BudgetExceeded
from openalex_agent.research_metrics import context_metrics, factual_metrics, validate_evidence


def context(round=1, fact="FACT[early-evidence-73]"):
    return {"question": "retrieval", "round": round, "remaining": {"actions": 20-round},
        "retrieval": {"mode": "frozen_corpus", "live_openalex_available": False},
        "permitted_actions": ["SEARCH_PAPERS", "STOP_RETRIEVAL"],
        "proposals": [{"action": "SEARCH_PAPERS", "arguments": {"query": "retrieval"}}],
        "top_papers": [{"id": "https://openalex.org/W1", "abstract_excerpt": fact}], "recent_actions": []}


STOP = {"decision": {"action": "STOP_RETRIEVAL", "arguments": {}}}


def test_context_serialization_is_reproducible_across_python_hash_seeds():
    code = '''
from types import SimpleNamespace
from openalex_agent.context_planner import ContextPlanner
from openalex_agent.context_memory import serialized
p=SimpleNamespace(question="retrieval",mode="clm",store=SimpleNamespace(observations=[]))
c={"question":"retrieval","round":1,"remaining":{},"permitted_actions":[],"retrieval":{},"proposals":[]}
print(serialized(ContextPlanner._parts(p,c)[0]))
'''
    outputs = [subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed})
               for seed in ("1", "2")]
    assert outputs[0] == outputs[1]


def planner(tmp_path, mode, script, *, max_chars=8000, max_calls=20, limits=None, prices=False, require_evidence=False):
    inputs, events = [], []
    def handler(request):
        payload = json.loads(request.content)
        data = json.loads(payload["messages"][1]["content"])
        inputs.append(data)
        output = script(data, len(inputs)) if callable(script) else script[len(inputs)-1]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(output)}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 10}})
    base = HostedJSONPlanner(Settings(llm_base_url="https://test.example/v1", llm_api_key="PRIVATE-CLM-KEY",
        llm_model="SYNTHETIC-MOCK-MODEL", llm_max_tokens=64,
        llm_input_price=1 if prices else 0, llm_output_price=2 if prices else 0), transport=httpx.MockTransport(handler))
    p = ContextPlanner(base, mode=mode, limits=limits, require_evidence=require_evidence)
    p.bind(tmp_path / mode, "retrieval", Budget(max_context_chars=max_chars, max_llm_calls=max_calls),
           lambda kind, value: events.append({"kind": kind, **value}), lambda: {})
    return p, inputs, events


def test_c5_real_edit_synchronizes_to_next_wire_request_and_preserves_protected_question(tmp_path):
    p, inputs, events = planner(tmp_path, "clm", [
        {"context_edit": {"path": "active_context.txt", "op": "replace", "text": "KEEP evidence 73"}}, STOP, STOP])
    with p.base:
        assert p.decide(context())["action"] == "STOP_RETRIEVAL"
        assert inputs[1]["active_context"] == "KEEP evidence 73"
        assert "FACT[early-evidence-73]" not in inputs[1]["active_context"]
        assert inputs[0]["protected"]["question"] == inputs[1]["protected"]["question"] == "retrieval"
        # Identical old observation does not resurrect model-deleted facts.
        p.decide(context(2))
        assert "FACT[early-evidence-73]" not in inputs[-1]["active_context"]
    edits = [e for e in events if e["kind"] == "context_edit"]
    assert len(edits) == 1 and edits[0]["accepted"]
    assert "FACT[early-evidence-73]" in edits[0]["before"] and edits[0]["after"] == "KEEP evidence 73"
    snapshots = list(p.store.directory.glob("request-*.json"))
    assert len(snapshots) == 3 and p.usage.calls == 3
    assert p.usage.input_tokens == 120 and p.usage.output_tokens == 30
    assert edits[0]["usage"]["input_tokens"] == 40


@pytest.mark.parametrize("edit_request", [
    {"path": "../credentials", "op": "replace", "text": "bad"},
    {"path": "/tmp/other-run", "op": "replace", "text": "bad"},
    {"path": "active_context.txt", "op": "replace", "text": "bad", "question": "overwrite protected"},
    {"path": "active_context.txt", "op": "splice", "old": "missing", "new": "bad"},
    {"path": "active_context.txt", "op": "execute", "text": "rm -rf /"},
])
def test_c5_invalid_edits_are_audited_and_never_modify_other_files(tmp_path, edit_request):
    sentinel = tmp_path / "credentials"
    sentinel.write_text("IMMUTABLE")
    store = ContextStore(tmp_path / "run-context", 1000)
    store.transition("original", source="environment", operation={})
    result = store.edit(edit_request)
    assert not result["accepted"] and result["before"] == result["after"] == store.read() == "original"
    assert sentinel.read_text() == "IMMUTABLE"
    assert json.loads((store.directory / "transition-00001.json").read_text()) == result


def test_c5_symlink_escape_is_rejected(tmp_path):
    store = ContextStore(tmp_path / "workspace", 1000)
    victim = tmp_path / "source.py"
    victim.write_text("untouched")
    store.path.unlink()
    store.path.symlink_to(victim)
    with pytest.raises(ValueError, match="redirection"):
        store.edit({"path": "active_context.txt", "op": "replace", "text": "bad"})
    assert victim.read_text() == "untouched"


def test_c2_authoritative_correction_replaces_outdated_active_statement(tmp_path):
    scripts = [STOP, {"context_edit": {"path": "active_context.txt", "op": "replace", "text": "FACT[corrected-value-9]"}}, STOP]
    p, inputs, _ = planner(tmp_path, "clm", scripts)
    try:
        p.decide(context(1, "FACT[outdated-value-3]"))
        p.decide(context(2, "FACT[corrected-value-9]"))
        assert "FACT[corrected-value-9]" in inputs[1]["active_context"]
        assert inputs[2]["active_context"] == "FACT[corrected-value-9]"
        assert "outdated" not in p.store.read()
    finally:
        p.close()


@pytest.mark.parametrize("mode", ["conventional", "clm"])
def test_c6_both_modes_can_recover_identical_previously_delivered_evidence(tmp_path, mode):
    output = [STOP, {"read_observation": {"index": 0, "start": 0, "chars": 512}}, STOP]
    p, inputs, events = planner(tmp_path, mode, output)
    try:
        p.decide(context(1))
        p.decide(context(2, "later observation"))
        recovered = inputs[-1]["recovered_observation"]
        assert "FACT[early-evidence-73]" in recovered["text"] and recovered["historical"]
        assert p.usage.calls == 3
        assert sum(e["kind"] == "context_recovery" and e["accepted"] for e in events) == 1
    finally:
        p.close()


def test_invalid_edit_is_feedback_not_a_paid_retry_and_calls_are_bounded(tmp_path):
    p, inputs, events = planner(tmp_path, "clm", lambda *_: {"context_edit": {
        "path": "../../.env", "op": "replace", "text": "bad"}}, max_calls=2)
    try:
        with pytest.raises(BudgetExceeded, match="llm_call_limit"):
            p.decide(context())
        assert p.usage.calls == len(inputs) == 2
        assert inputs[1]["operation_feedback"]["edit_accepted"] is False
        assert sum(e["kind"] == "context_edit" for e in events) == 2
    finally:
        p.close()


def test_token_ceiling_prevents_model_request_and_unknown_prices_cannot_enforce_dollar_cap(tmp_path):
    p, inputs, _ = planner(tmp_path, "clm", [STOP], limits=InferenceLimits(max_total_tokens=1))
    try:
        with pytest.raises(BudgetExceeded, match="token_budget_exhausted"):
            p.decide(context())
        assert not inputs and p.usage.calls == 0
    finally:
        p.close()
    with pytest.raises(ValueError, match="prices"):
        planner(tmp_path / "priced", "conventional", [STOP], limits=InferenceLimits(max_llm_cost_usd=.001))


def test_known_cost_ceiling_stops_next_call_and_keeps_observed_usage(tmp_path):
    p, inputs, _ = planner(tmp_path, "conventional", [STOP], prices=True,
                           limits=InferenceLimits(max_llm_cost_usd=.00001))
    try:
        p.decide(context())
        with pytest.raises(BudgetExceeded, match="llm_cost_budget_exhausted"):
            p.decide(context(2))
        assert len(inputs) == 1 and p.usage.estimated_cost_usd == pytest.approx(.00006)
    finally:
        p.close()


def test_capacity_failure_feedback_directs_model_to_edit_instead_of_repeating_oversized_reads(tmp_path):
    def script(payload, number):
        if number == 1:
            assert payload["pending_observation_chars"] > 0
            return {"read_observation": {"index": 0, "start": 0, "chars": 512}}
        if number == 2:
            assert payload["operation_feedback"]["detail"] == "archive_read_does_not_fit"
            assert "context_edit" in payload["operation_feedback"]["next_step"]
            return {"context_edit": {"path": "active_context.txt", "op": "replace", "text": "Retained working note"}}
        assert payload["active_context"].startswith("Retained working note")
        return STOP
    p, inputs, events = planner(tmp_path, "clm", script, max_chars=4000, max_calls=3)
    try:
        protected, body = p._parts(context())
        overhead = len(serialized(p._payload(protected, body, active="")))
        p.store.transition("Z"*(4000-overhead-100), source="environment", operation={"pressure_fixture": True})
        assert p.decide(context())["action"] == "STOP_RETRIEVAL"
        assert len(inputs) == 3
        assert sum(e["kind"] == "context_edit" and e["accepted"] for e in events) == 1
    finally:
        p.close()


def test_c4_c7_valid_identifiers_and_matching_quotes_do_not_establish_support():
    paper = Paper.from_openalex({"id": "W1", "title": "A trial did not improve outcomes"}, provenance="SYNTHETIC TEST ONLY")
    records = [validate_evidence({"statement": "The trial improved outcomes", "work_id": "W1",
               "evidence_quote": paper.title, "insufficient": False}, {paper.id: paper})]
    assert records[0]["identifier_valid"] and records[0]["quote_matches_observed_metadata"]
    assert records[0]["support_judgment"] == "not_evaluated"
    evaluated = [{**records[0], "judgment": "contradicted", "factually_correct": False, "citation_support_correct": False},
        {"judgment": "unsupported", "identifier_valid": False},
        {"judgment": "abstain", "appropriate": True},
        {"judgment": "supported", "factually_correct": True, "citation_support_correct": True, "identifier_valid": True}]
    values = factual_metrics(evaluated)
    assert values["claim_factual_accuracy"] == .5
    assert values["contradicted_claim_rate"] == values["unsupported_claim_rate"] == values["supported_claim_rate"] == pytest.approx(1/3)
    assert values["appropriate_abstention_rate"] == 1
    assert factual_metrics([])["claim_factual_accuracy"] is None
    with pytest.raises(ValueError):
        factual_metrics([records[0]])


def test_c4_required_structured_evidence_cannot_be_replaced_by_silent_empty_output(tmp_path):
    p, inputs, _ = planner(tmp_path, "conventional", [STOP], require_evidence=True)
    try:
        with pytest.raises(PlannerError, match="evidence_record_required"):
            p.decide(context())
        assert len(inputs) == 1
    finally:
        p.close()


def test_c1_context_metrics_measure_verbatim_retention_and_loss_not_summary_similarity():
    edits = [{"source": "model", "accepted": True, "before": "exact evidence A", "after": "paraphrase"}]
    result = context_metrics(["exact evidence A", "exact evidence B"], "exact evidence A", edits=edits,
                             recovered={"exact evidence A": "paraphrase"})
    assert result["exact_evidence_retention"] == .5 and result["exact_recovery_accuracy"] == 0
    assert result["evidence_lost_through_editing"] == 1


def test_c7_model_assertions_cannot_merge_similar_papers_or_create_citation_edges(tmp_path, db):
    from openalex_agent.controlled import ControlledController
    from openalex_agent.corpus import LocalCorpus, freeze
    from openalex_agent.research_metrics import validate_evidence
    raw = [{"id": "W1", "title": "SYNTHETIC shared study", "referenced_works": [],
            "authorships": [{"author": {"id": "A1", "display_name": "Alex Taylor"}}]},
           {"id": "W2", "title": "SYNTHETIC shared study revised", "referenced_works": [],
            "authorships": [{"author": {"id": "A2", "display_name": "Alex Taylor"}}]}]
    papers = [Paper.from_openalex(r, provenance="SYNTHETIC ADVERSARIAL FIXTURE ONLY") for r in raw]
    papers[0].abstract, papers[1].abstract = "The trial did not improve outcomes.", "No outcome measurement was made."
    corpus = tmp_path / "corpus"
    freeze(papers, corpus, acquisition=[{"source": "SYNTHETIC ADVERSARIAL TEST"}])
    replies = [
        {"context_edit": {"path": "active_context.txt", "op": "replace", "text": "Unverified assertion: W1 cites W2 and the trial improved outcomes."}},
        {"decision": {"action": "STOP_RETRIEVAL", "arguments": {}}, "evidence_records": [
            {"statement": "The trial improved outcomes.", "work_id": "W1", "evidence_quote": papers[0].abstract, "insufficient": False},
            {"statement": "The revised trial improved outcomes.", "work_id": "W2", "evidence_quote": papers[1].abstract, "insufficient": False},
            {"statement": "Unseen result", "work_id": "W999", "evidence_quote": "invented", "insufficient": False},
            {"statement": "Insufficient evidence for revised outcomes", "work_id": "W2", "evidence_quote": "", "insufficient": True}]}]
    count = []
    def handler(request):
        count.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(replies[len(count)-1])}}]})
    base = HostedJSONPlanner(Settings(llm_base_url="https://test.example/v1", llm_api_key="PRIVATE", llm_model="MOCK"),
                             transport=httpx.MockTransport(handler))
    p = ContextPlanner(base, mode="clm")
    try:
        c = ControlledController("shared study", LocalCorpus(corpus), db, architecture="clm_graph", planner=p,
            budget=Budget(max_actions=2, max_llm_calls=2, page_size=2), output=str(tmp_path / "runs"))
        result = c.run()
        assert len(c.papers) == 2 and c.graph.citation_graph().number_of_edges() == 0
        assert c.papers[papers[0].id].authorships[0].author.id != c.papers[papers[1].id].authorships[0].author.id
        assert c.papers[papers[0].id].abstract == "The trial did not improve outcomes."
        records = result["evidence_records"]
        assert records[0]["identifier_valid"] and records[0]["support_judgment"] == "not_evaluated"
        assert not records[2]["identifier_valid"]
        oracle = [{**r, "judgment": judgment} for r, judgment in zip(records, ("contradicted", "unsupported", "unsupported", "abstain"))]
        assert factual_metrics(oracle)["contradicted_claim_rate"] == pytest.approx(1/3)
        assert all(r["support_judgment"] == "not_evaluated" for r in records)
    finally:
        p.close()
