"""Offline scientific audit diagnostics; never runs a benchmark or a real API call.

Run from the project root with .venv/bin/python scripts/experiment_validity_audit.py
--output /tmp/openalex-scientific-validity-audit (a fresh directory).
Findings deliberately preserve and demonstrate production defects rather than fix them.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd

from openalex_agent.actions import action_schema
from openalex_agent.config import Settings
from openalex_agent.context_memory import ContextStore, serialized
from openalex_agent.context_planner import ContextPlanner, InferenceLimits
from openalex_agent.controlled import ControlledController, GRAPH_FIELDS
from openalex_agent.controlled_diagnostics import SeededCorpus, fixture
from openalex_agent.controller import Budget
from openalex_agent.corpus import freeze
from openalex_agent.evaluation import metrics
from openalex_agent.llm import HostedJSONPlanner
from openalex_agent.models import Paper
from openalex_agent.research_assessment import claim_id, score_research
from openalex_agent.research_metrics import factual_metrics, paired_comparisons, validate_evidence
from openalex_agent.selection import Scoring
from openalex_agent.storage import Database


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def estimate(text):
    """Documented bytes/4 approximation, NOT a native tokenizer."""
    return (len(text.encode()) + 3) // 4


def statistical_checks():
    def row(method, value, variant="full", trial=0, question="synthetic-q"):
        return dict(query_id=question, method=method, variant=variant, trial=trial,
                    action_budget=5, score=value)
    clean = [row(m, v) for m, v in zip(("llm", "llm_graph", "clm", "clm_graph"), (1, 3, 5, 9))]
    contrasts = paired_comparisons(clean, "score", samples=100)
    interaction = next(r for r in contrasts if r["left"] == "interaction")
    assert interaction["mean_difference"] == 2
    contaminated = [row("llm", 1), row("llm_graph", 1)] + [
        row("llm_graph", 0, v) for v in ("no_novelty", "no_importance", "no_coverage")]
    bad = paired_comparisons(contaminated, "score", samples=100)[0]
    assert bad["mean_difference"] == -.75
    duplicate = paired_comparisons([row("llm", 0), row("llm", 1, trial=1),
        row("llm_graph", 1), row("llm_graph", 1)], "score", samples=100)
    second = [dict(r, query_id="synthetic-q2", score=r["score"] - (1 if r["method"] == "llm_graph" else 0)) for r in clean]
    paired = paired_comparisons(clean + second, "score", samples=100)
    assert paired[0]["bootstrap_95_ci"] is not None
    null = paired_comparisons([row("llm", 1), row("llm_graph", 1)], "score")[0]
    regression = paired_comparisons([row("llm", 1), row("llm_graph", 0)], "score")[0]
    assert null["mean_difference"] == 0 and regression["mean_difference"] == -1
    return dict(interaction=dict(expected=2, actual=interaction),
        marginal_main_effects=dict(expected_graph=3, expected_clm=5,
            exported=False, observed_contrasts=contrasts),
        variant_contamination=dict(correct_full_contrast=0, actual=bad),
        duplicate_unbalanced_rows_accepted=duplicate, two_question_paired_intervals=paired,
        null_result=null, regression=regression)


def scoring_checks(root):
    w1, w2 = "https://openalex.org/W1", "https://openalex.org/W2"
    paper = Paper.from_openalex({"id": "W1", "title": "SYNTHETIC: treatment reduced symptoms"})
    claim = dict(statement="Treatment increased symptoms", work_id=w1,
                 evidence_quote=paper.title, insufficient=False)
    structural = validate_evidence(claim, {w1: paper})
    assert structural["identifier_valid"] and structural["quote_matches_observed_metadata"]
    assert structural["support_judgment"] == "not_evaluated"
    records = [dict(judgment="supported", factually_correct=True, citation_support_correct=True, identifier_valid=True),
        dict(judgment="unsupported", citation_support_correct=False, identifier_valid=True),
        dict(judgment="contradicted", factually_correct=False, citation_support_correct=False, identifier_valid=True),
        dict(judgment="abstain", appropriate=True)]
    facts = factual_metrics(records)
    assert facts["claims"] == 3 and facts["abstentions"] == 1
    assert facts["claim_factual_accuracy"] == .5 and facts["citation_to_claim_support_accuracy"] == 1/3
    assert all(facts[k+"_claim_rate"] == 1/3 for k in ("supported", "unsupported", "contradicted"))
    empty = factual_metrics([])
    assert empty["supported_claim_rate"] is None and empty["claim_factual_accuracy"] is None
    invalid = validate_evidence(dict(claim, work_id="W999", evidence_quote="invented"), {w1: paper})
    assert not invalid["identifier_valid"] and not invalid["quote_matches_observed_metadata"]
    grades = {w1: 3, w2: 1, "https://openalex.org/W3": 0}
    ranking = metrics([w2, w1], grades, k=10, scope="exhaustive")
    expected_ndcg = (1 + 7/math.log2(3))/(7 + 1/math.log2(3))
    assert ranking["precision_at_k"] == .2 and math.isclose(ranking["ndcg_at_k"], expected_ndcg)
    unjudged = metrics([w1, "https://openalex.org/W999"], grades, k=10)
    assert unjudged["precision_at_k"] is None and unjudged["ndcg_at_k"] is None
    empty_ranking = metrics([], grades, k=10)
    assert empty_ranking["precision_at_k"] == 0 and empty_ranking["ndcg_at_k"] == 0
    try:
        factual_metrics([dict(judgment="unjudged")])
    except ValueError:
        rejects_unjudged = True
    else:
        rejects_unjudged = False
    assert rejects_unjudged

    # Minimal deliberately incomplete assessment fixture, not a benchmark execution.
    directory = root / "scorer-fixture"
    run = directory / "synthetic-run"
    run.mkdir(parents=True)
    wrong_source = validate_evidence(dict(claim, work_id=w2), {w1: paper})
    save(run / "result.json", dict(evidence_records=[wrong_source]))
    save(directory / "experiment.json", dict(suite="controlled-five", status="running",
        corpus_manifest=dict(ids=[w1, w2]), planned_runs=4))
    pd.DataFrame([dict(query_id="audit-q", method="llm", variant="full", trial=0,
        action_budget=5, run_id="synthetic-run", directory=str(run), total_api_cost_usd=0)]).to_csv(directory / "metrics.csv", index=False)
    judgments = directory / "independent-synthetic-judgments.csv"
    pd.DataFrame([dict(record_id=claim_id("audit-q", wrong_source), judgment="unsupported",
        factually_correct="", citation_support_correct="false", appropriate="",
        assessor="SYNTHETIC AUDIT FIXTURE; NOT REAL LABELS")]).to_csv(judgments, index=False)
    scored = score_research(directory, claim_judgments_path=judgments)
    output = pd.read_csv(directory / "research-metrics.csv")
    assert not wrong_source["identifier_valid"] and output.citation_identifier_validity.iloc[0] == 1
    assert "variant" not in output.columns and scored == 1
    return dict(known_outcome_factual=facts, empty_claims=empty, valid_id_does_not_prove_support=structural,
        nonexistent_source=invalid, wrong_source_semantic_judgment_required=True,
        independent_wrong_source_support_rate=facts["citation_to_claim_support_accuracy"],
        precision_ndcg=dict(actual=ranking, expected_ndcg=expected_ndcg), unjudged_ranking=unjudged,
        empty_ranking=empty_ranking, rejects_unjudged_claims=rejects_unjudged,
        observed_source_scope_drift=dict(online_identifier_valid=False, offline_corpus_identifier_validity=1,
            wrong_source_support_accuracy=float(output.citation_to_claim_support_accuracy.iloc[0])),
        scorer_drops_variant=True, scorer_accepts_incomplete_experiment=True)


def write_failure_check(root):
    store = ContextStore(root / "fault-injected-context", 4000)
    store.transition("original", source="environment", operation={"fixture": True})
    try:
        with patch.object(store, "_write", side_effect=OSError("SYNTHETIC DISK FAILURE")):
            store.transition("replacement", source="model", operation={"fixture": True})
    except OSError:
        pass
    record = json.loads((store.directory / "transition-00001.json").read_text())
    assert record["accepted"] and store.read() == "original" and record["after"] == "replacement"
    return dict(persisted_audit=record, actual_active_text=store.read(),
                accepted_record_matches_active_file=False, injected_failure_only=True)


def wire_checks(root, settings):
    question, papers, seeds, _, _ = fixture("G1")
    corpus = root / "mock-corpus"
    freeze(papers, corpus, acquisition=[{"source": "SYNTHETIC OFFLINE AUDIT ONLY"}])
    results, inputs = {}, {}
    for method in ("llm", "llm_graph", "clm", "clm_graph"):
        wire = []
        def handler(request):
            wire.append(json.loads(request.content))
            value = ({"context_edit": {"path": "active_context.txt", "op": "replace", "text": "AUDIT RETAINED NOTE"}}
                     if method.startswith("clm") and len(wire) == 1 else
                     {"decision": {"action": "STOP_RETRIEVAL", "arguments": {}}})
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}})
        base = HostedJSONPlanner(settings, transport=httpx.MockTransport(handler))
        planner = ContextPlanner(base, mode="clm" if method.startswith("clm") else "conventional")
        def forbidden(*args, **kwargs):
            raise AssertionError("Graph constructed or scored in a no-graph arm")
        with Database(":memory:") as db:
            if not method.endswith("_graph"):
                with patch("openalex_agent.controller.ScholarlyGraph", forbidden), patch("openalex_agent.controlled.score_candidates", forbidden):
                    controller = ControlledController(question, SeededCorpus(corpus, seeds), db, architecture=method,
                        planner=planner, budget=Budget(max_actions=2, page_size=2, output_k=10),
                        scoring=Scoring(min_relevance=0), output=str(root / "mock-runs"))
                    result = controller.run()
            else:
                controller = ControlledController(question, SeededCorpus(corpus, seeds), db, architecture=method,
                    planner=planner, budget=Budget(max_actions=2, page_size=2, output_k=10),
                    scoring=Scoring(min_relevance=0), output=str(root / "mock-runs"))
                result = controller.run()
            requests = [json.loads(p.read_text()) for p in sorted(planner.store.directory.glob("request-*.json"))]
            first = requests[0]["input"]
            inputs[method] = first
            save(root / f"wire-{method}.json", dict(kind="HTTP MOCK; NO REAL MODEL INFERENCE", wire=wire))
            if method.startswith("clm"):
                assert requests[1]["input"]["active_context"] == "AUDIT RETAINED NOTE"
            assert controller.papers[papers[0].id].title == papers[0].title
            saved = json.loads((controller.directory / "state.json").read_text())
            assert next(p for p in saved["papers"] if p["id"] == papers[0].id)["title"] == papers[0].title
            results[method] = dict(directory=result["directory"], calls=len(wire),
                graph=result["graph_enabled"], graph_construction_and_scoring_forbidden=not method.endswith("_graph"),
                wire_parameters={k: v for k, v in wire[0].items() if k != "messages"},
                system_chars=len(requests[0]["system"]), input_chars=len(serialized(first)),
                protected=first["protected"], ranked=[dict(id=r["paper"]["id"], score=r["algorithmic"]["relevance"]) for r in result["ranked"]],
                original_metadata_unchanged=True, edit_visible_on_next_call=method.startswith("clm"))
        planner.close()
    for a, b in (("llm", "llm_graph"), ("clm", "clm_graph")):
        assert inputs[a]["protected"] == inputs[b]["protected"]
        bodies = [json.loads(inputs[m]["active_context"])["changes"] if m.startswith("clm") else inputs[m]["observation"] for m in (a, b)]
        assert not GRAPH_FIELDS.intersection(bodies[0]) and GRAPH_FIELDS <= bodies[1].keys()
        assert bodies[0] == {k: v for k, v in bodies[1].items() if k not in GRAPH_FIELDS}
        assert results[a]["wire_parameters"] == results[b]["wire_parameters"]
        assert results[a]["ranked"] == results[b]["ranked"]
        assert results[a]["system_chars"] == results[b]["system_chars"]
    assert len({serialized(r["ranked"]) for r in results.values()}) == 1
    return dict(kind="FOUR TINY MOCK MECHANISM PROBES, NOT A BENCHMARK", rows=results,
        equal_raw_information_and_protected_proposals_within_graph_pairs=True,
        equal_final_ranking_for_identical_observed_papers=True, api_calls=0)


def pressure_measurements():
    root = Path("data/clm-development-validation/diagnostics-v1")
    result = []
    needle = "early-evidence-73"
    for context in sorted(root.glob("C*/*/*/context")):
        requests = [json.loads(p.read_text()) for p in sorted(context.glob("request-*.json"))]
        if not requests:
            continue
        observations = [json.loads(p.read_text()) for p in sorted(context.glob("observation-*.json"))]
        state = json.loads((context.parent / "state.json").read_text())
        source_papers = [Paper.from_dict(p) for p in state["papers"]]
        active = [r["input"].get("active_context", serialized(r["input"].get("observation", {}))) for r in requests]
        transitions = [json.loads(p.read_text()) for p in sorted(context.glob("transition-*.json"))]
        first_missing = next((i + 1 for i, text in enumerate(active) if needle not in text), None)
        result.append(dict(directory=str(context), case=context.parts[-4], mode=context.parts[-3],
            observations=len(observations), calls=len(requests),
            unique_authoritative_papers=len(source_papers),
            total_unique_source_text_estimated_tokens=sum(estimate(p.text) for p in source_papers),
            total_unique_source_metadata_estimated_tokens=estimate(serialized(state["papers"])),
            early_evidence_in_authoritative_source=any(needle in p.text for p in source_papers),
            total_selected_observation_estimated_tokens=sum(estimate(serialized(o)) for o in observations),
            unique_selected_observation_estimated_tokens=estimate(serialized(list({serialized(o): o for o in observations}.values()))),
            observation_sum_repeats_selected_metadata=True,
            max_active_chars=max(map(len, active)), final_active_chars=len(active[-1]),
            max_active_estimated_tokens=max(map(estimate, active)),
            max_full_input_estimated_tokens=max(estimate(r["system"] + serialized(r["input"])) for r in requests),
            max_user_payload_chars=max(len(serialized(r["input"])) for r in requests),
            context_limit_chars=json.loads((context.parent / "result.json").read_text())["budget"]["max_context_chars"],
            first_request_without_early_evidence=first_missing,
            first_missing_active_chars=len(active[first_missing-1]) if first_missing else None,
            early_evidence_in_final_active=needle in active[-1],
            early_evidence_in_external_archive=any(needle in serialized(o) for o in observations),
            model_edits=sum(r["source"] == "model" for r in transitions),
            source_semantics="Selected observation bodies, not full raw acquisition volume; byte/4 estimates only"))
    return dict(native_token_counts=None, estimation="ceil(UTF-8 bytes / 4); not a bound", rows=result)


def hosted_snapshot_checks():
    results = []
    for validation in sorted(Path("data/clm-development-validation/hosted").glob("*/validation.json")):
        meta = json.loads(validation.read_text())
        context = Path(meta["run_directory"]) / "context"
        transitions = [json.loads(p.read_text()) for p in sorted(context.glob("transition-*.json"))]
        requests = [json.loads(p.read_text()) for p in sorted(context.glob("request-*.json"))]
        hashes = all(hashlib.sha256(r[k].encode()).hexdigest() == r[k+"_sha256"] for r in transitions for k in ("before", "after"))
        chain = all(a["after"] == b["before"] for a, b in zip(transitions, transitions[1:]))
        final_matches = not transitions or transitions[-1]["after"] == (context / "active_context.txt").read_text()
        edits = [r for r in transitions if r["source"] == "model" and r["accepted"]]
        deletion = []
        for r in edits:
            marker = r["before"][:64]
            later = [q["input"].get("active_context", "") for q in requests if q.get("usage_delta", {}).get("call_index", 0) > r.get("usage", {}).get("call_index", 0)]
            deletion.append(dict(before_chars=len(r["before"]), after_chars=len(r["after"]),
                nonempty_old_prefix_removed=bool(marker and later and all(marker not in s for s in later)),
                empty_after_makes_startswith_check_vacuous=r["after"] == ""))
        assert hashes and chain and final_matches
        results.append(dict(validation=str(validation), model=meta["model"], original_validation=meta,
            snapshot_hashes_correct=hashes, transitions_contiguous=chain, final_file_matches=final_matches,
            independent_deletion_checks=deletion))
    return results


def matrix(settings, wire):
    shared = dict(model=settings.llm_model, temperature=settings.llm_temperature,
        reasoning_effort=settings.llm_reasoning_effort, max_output_tokens=settings.llm_max_tokens,
        timeout_seconds=settings.llm_timeout, input_price_per_million=settings.llm_input_price,
        output_price_per_million=settings.llm_output_price, sampling_seed=None,
        price_status="unknown when prices are zero", provider_release_pinned=False,
        response_format={"type": "json_object"}, token_limit_field="max_completion_tokens")
    rows = []
    for method in ("standard", "llm", "llm_graph", "clm", "clm_graph"):
        standard = method == "standard"
        clm = method.startswith("clm")
        rows.append(dict(method=method, graph=method.endswith("_graph"), clm=clm,
            model=None if standard else shared, tool_schema=None if standard else action_schema(),
            auxiliary_operations=None if standard else dict(
                read_observation=dict(required_fields=["index", "start", "chars"], additional_fields=False,
                    index="integer in this run's archived observations", start="integer >=0", chars="integer 1..512"),
                context_edit=None if not clm else dict(path="active_context.txt", additional_fields=False,
                    replace_or_append=dict(required_fields=["path", "op", "text"], text="string"),
                    splice=dict(required_fields=["path", "op", "old", "new"], old="nonempty unique string", new="string"))),
            evidence_required_default=False,
            evidence_record_fields=["statement", "work_id", "evidence_quote", "insufficient"] if not standard else [],
            corpus_access="same fixed LocalCorpus snapshot; no live OpenAlex or hidden source tools",
            retrieval_budgets=asdict(Budget()),
            shared_acquisition_guards=asdict(Scoring()),
            protocol_overrides=dict(action_budgets=[5, 10, 20], max_rounds=20, output_k=10,
                require_evidence=True, trials=3, status="documented eventual protocol; NOT EXECUTED"),
            initial_search_limit="remaining inspection allowance" if standard else "page_size",
            context_token_ceiling=None, context_user_payload_character_ceiling=None if standard else 16000,
            system_in_character_ceiling=False, native_token_preflight=False,
            cumulative_inference_limits=None if standard else asdict(InferenceLimits()),
            planner_invocation_budget=0 if standard else 20,
            context_editing="none" if standard else "model-selected replace/append/splice" if clm else "top-eight/recent-four harness selection",
            context_edit_cost="one shared model call plus usage/time; no acquisition action" if clm else None,
            archive_access="none" if standard else "own selected observation archive, index/start/chars<=512; shared call budget",
            raw_source_access="authoritative observed metadata; proposals regenerated from all observed papers" if not standard else "frozen search result metadata",
            graph_overlay="candidate scores, verified citation edges, statistics, topic coverage" if method.endswith("_graph") else None,
            fallback="no hosted planner" if standard else "visible deterministic action on provider/invalid decision; inference budget exhaustion stops",
            ranking="shared HashingVectorizer cosine(original question, paper text), descending score then canonical ID",
            termination="one original-query search" if standard else "STOP, action/round/inspection/API/model-call ceilings, stagnation, inference token/cost stop, retrieval failure",
            diagnostic_wire=None if standard else wire["rows"][method]))
    differences = [
        ("graph derived information", "intended experimental intervention", "Only graph arms add the existing derived view; planning/discovery/stop decisions may change."),
        ("model control over persistent context", "intended experimental intervention", "CLM edit API and persistent delta history versus conventional selected current view."),
        ("edit/recovery invocations reduce remaining planning calls", "necessary architectural consequence", "Management charged to shared cap; report opportunity count and costs."),
        ("graph payload size and JSON-encoded active text overhead", "necessary architectural consequence", "Same character ceiling does not imply equal tokens or source exposure; report measured amounts."),
        ("CLM-specific retention/correction advice", "potential confound", "Separate necessary API explanation from memory-strategy coaching; review matched legitimate guidance."),
        ("CLM delays pending input until space is freed", "potential confound", "Temporal source exposure differs from immediate conventional selected view and paper's operation delivery."),
        ("context ceiling measured in characters; system excluded", "implementation defect", "Does not enforce requested common token ceiling; cumulative estimate is a different resource."),
        ("graph ablation variants pooled in paired inference", "implementation defect", "Variant absent from grouping and discarded by research scorer."),
        ("retention diagnostic uses different scripted behaviors", "potential confound", "CLM regex-retains the target; conventional script never requests supported recovery."),
        ("metadata-driven protected proposals use all observed papers", "necessary architectural consequence", "Shared external harness memory; cannot attribute all-agent forgetting to active-file loss."),
        ("standard baseline one-pass search/inspection and no evidence generation", "necessary architectural consequence", "Additional retrieval baseline, outside factorial main effects; factual rates unavailable."),
        ("model prices unknown and release alias unpinned", "potential confound", "No cost-benefit or exact provider-version reproducibility claim yet."),
        ("method order rotated, not randomized/full counterbalance", "potential confound", "Three trials cover only three of five order positions per question-budget."),
        ("claim output opportunities vary and optional by default", "potential confound", "Require-evidence does not establish fixed answer opportunities across methods."),
    ]
    return dict(schema_version=1, audited_date="2026-11-02", status="descriptive audit, not a frozen experimental protocol",
        production_files_modified=False, live_api_calls=0,
        profile_origin="local non-secret settings; all diagnostic HTTP replaced by MockTransport",
        default_vs_protocol="retrieval_budgets are Python defaults; protocol_overrides are proposed documented commands, not actual final runs",
        configurations=rows, differences=[dict(difference=a, classification=b, interpretation=c) for a,b,c in differences])


def pressure_plan(root):
    conditions = []
    for name, blocks in (("low", 4), ("medium", 32), ("high", 256)):
        records = []
        for i in range(blocks):
            needle = "SYNTHETIC SOURCE W1: verified audit datum = 73." if i == 0 else ""
            text = (needle + "\n" + (f"FILLER {i:04d} audit observation "+"x"*37+"\n")*20)[:1024].ljust(1024)
            records.append(dict(index=i, source="SYNTHETIC AUDIT PRESSURE ONLY", text=text))
        file = root / f"pressure-{name}.jsonl"
        file.write_text("".join(serialized(r)+"\n" for r in records))
        conditions.append(dict(name=name, observation_blocks=blocks, observation_estimated_tokens=blocks*256,
            active_working_target_tokens=4096, full_request_input_ceiling_tokens=8192,
            response_reserve_tokens=512, cumulative_token_ceiling=10000000,
            max_model_calls=blocks*4+16, acquisition_actions=blocks+1,
            max_rounds=blocks+1, max_inspected=blocks+1, patience=blocks+1, page_size=1,
            estimated_observation_to_working_ratio=blocks*256/4096,
            stimulus=str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    return dict(status="PREPARED, NOT EXECUTED; NATIVE TOKEN ENFORCEMENT NOT IMPLEMENTED",
        tokenizer=None, estimates="UTF-8 text bytes / 4; JSON/system/tool overhead not included; counts are not native tokens",
        required_before_execution=["Pin an available provider-compatible tokenizer or clearly named study tokenizer; recount all source text and full messages.",
            "Enforce 8192 full input tokens and 512 output reserve equally; reserve 4096 for working evidence; log protected/system/recovery overhead.",
            "Preserve existing conventional top-eight/recent-four/trim/archive strategy; no removal of its recovery tools.",
            "Verify actual delivered incremental volume, not repeated top-eight snapshots; require low <1x, medium >1x, high >8x measured working capacity.",
            "Keep required evidence under 1024 counted tokens and each operation under 512; reject infeasible instances before running.",
            "Use same probes, source attribution, model, advice quality, retrieval/management caps and at least three seeds per condition.",
            "Report selected-input retention separately from real retrieval/source-exposure effects; archives remain available in all agent cells.",
            "Freeze after development calibration; judge active retention, late answers, corrections and archival recovery separately."],
        conditions=conditions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Fresh output directory; never a historical experiment")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "diagnostic-results.json").exists():
        raise SystemExit("Audit diagnostics already exist; select a fresh --output directory")
    settings = replace(Settings.from_env(), llm_api_key="SYNTHETIC-AUDIT-NO-SECRET", api_key="", email="")
    wire = wire_checks(root, settings)
    diagnostics = dict(kind="OFFLINE SCIENTIFIC VALIDITY AUDIT; NOT PERFORMANCE EVIDENCE",
        real_hosted_calls=0, real_openalex_calls=0, statistical=statistical_checks(),
        scoring=scoring_checks(root), write_failure=write_failure_check(root), graph_and_edit_wire=wire,
        prior_hosted_snapshot_review=hosted_snapshot_checks())
    save(root / "diagnostic-results.json", diagnostics)
    save(root / "context-pressure-measurements.json", pressure_measurements())
    save(root / "configuration-matrix.json", matrix(settings, wire))
    save(root / "context-pressure-plan.json", pressure_plan(root))
    print(json.dumps(dict(output=str(root), real_api_calls=0,
        diagnostic_groups=list(diagnostics), recommendation="See docs/EXPERIMENT_READINESS_AUDIT.md")))


if __name__ == "__main__":
    main()
