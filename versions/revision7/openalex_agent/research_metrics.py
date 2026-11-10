"""Transparent research diagnostics. Labels are evaluator inputs, never planner state."""

import math
import numpy as np

from .models import canonical_id


def validate_evidence(record: dict, observed: dict) -> dict:
    fields = {"statement", "work_id", "evidence_quote", "insufficient"}
    if not isinstance(record, dict) or set(record) != fields or type(record["insufficient"]) is not bool:
        raise ValueError("Invalid structured evidence record")
    if not all(isinstance(record[k], str) for k in ("statement", "evidence_quote")) or not record["statement"].strip():
        raise ValueError("Evidence statements and excerpts must be text")
    try:
        identifier = canonical_id(record["work_id"]) if record["work_id"] is not None else None
    except ValueError:
        identifier = None
    paper = observed.get(identifier)
    return {**record, "work_id": identifier, "citation_id_format_valid": identifier is not None,
            "source_observed_at_claim": paper is not None, "source_accessible_at_claim": None,
            "identifier_valid": paper is not None,
            "quote_matches_observed_metadata": bool(paper and record["evidence_quote"] and
                                                     record["evidence_quote"] in paper.text),
            "support_judgment": "not_evaluated"}


def factual_metrics(judged_records: list[dict]) -> dict:
    """Requires separate support/correctness judgments, not identifier matching."""
    permitted = {"supported", "unsupported", "contradicted", "abstain"}
    if any(r.get("judgment") not in permitted for r in judged_records):
        raise ValueError("Independent claim judgments are required")
    claims = [r for r in judged_records if r["judgment"] != "abstain"]
    def fraction(n, d):
        return n/d if d else None
    judged_truth = [r for r in claims if type(r.get("factually_correct")) is bool]
    abstentions = [r for r in judged_records if r["judgment"] == "abstain" and type(r.get("appropriate")) is bool]
    citation_judged = [r for r in claims if type(r.get("citation_support_correct")) is bool]
    identifiers = [r for r in claims if type(r.get("identifier_valid")) is bool]
    result = {"claims": len(claims), "abstentions": len(judged_records)-len(claims),
        "claim_factual_accuracy": fraction(sum(r["factually_correct"] for r in judged_truth), len(judged_truth)),
        **{status+"_claim_rate": fraction(sum(r["judgment"] == status for r in claims), len(claims))
           for status in ("supported", "unsupported", "contradicted")},
        "citation_identifier_validity": fraction(sum(r["identifier_valid"] for r in identifiers), len(identifiers)),
        "citation_to_claim_support_accuracy": fraction(sum(r["citation_support_correct"] for r in citation_judged), len(citation_judged)),
        "appropriate_abstention_rate": fraction(sum(r["appropriate"] for r in abstentions), len(abstentions)),
        "truth_judgment_coverage": fraction(len(judged_truth), len(claims)), "judged_abstentions": len(abstentions),
        "support_judgment_source": "independently supplied; not inferred from identifier validity"}
    for field, name in (("citation_exists_in_corpus", "citation_corpus_membership"),
                        ("source_accessible_at_claim", "citation_source_accessibility"),
                        ("source_observed_at_claim", "citation_observation_rate")):
        known = [r[field] for r in claims if type(r.get(field)) is bool]
        result[name] = fraction(sum(known), len(known))
        result[name+"_coverage"] = fraction(len(known), len(claims))
        result[name+"_unknown"] = len(claims)-len(known)
    return result


def retrieval_diagnostics(inspected, judgments, actions, *, subtopics=None, expected_subtopics=None):
    relevant = {i for i in set(inspected) if judgments.get(i, 0) > 0}
    result = {"unique_relevant_papers": len(relevant), "useful_papers_per_action": len(relevant)/actions if actions else None,
              "actions_per_relevant_paper": actions/len(relevant) if relevant else None}
    if subtopics is not None and expected_subtopics is not None:
        expected = set(expected_subtopics)
        found = {topic for i in relevant for topic in subtopics.get(i, [])} & expected
        result.update(subtopic_coverage=len(found)/len(expected) if expected else None,
                      subtopic_label_source="independent evaluator labels")
    return result


def context_metrics(expected: list[str], active: str, *, recovered=None, edits=None):
    retained = [text for text in expected if text in active]
    result = {"exact_evidence_retention": len(retained)/len(expected) if expected else None,
              "retained_evidence": len(retained), "expected_evidence": len(expected), "active_context_chars": len(active)}
    if recovered is not None:
        result["exact_recovery_accuracy"] = sum(recovered.get(text) == text for text in expected)/len(expected) if expected else None
    if edits is not None:
        models = [r for r in edits if r["source"] == "model"]
        lost = {text for r in models if r["accepted"] for text in expected if text in r["before"] and text not in r["after"]}
        result.update(context_edit_frequency=len(models), accepted_context_edits=sum(r["accepted"] for r in models),
                      evidence_lost_through_editing=len(lost))
    return result


def efficiency_metrics(result, calls):
    latency = [r["seconds"] for r in calls if isinstance(r.get("seconds"), (int, float))]
    llm = result.get("llm_usage", {})
    cost = llm.get("estimated_cost_usd") if llm.get("calls", 0) else 0.0
    api = result.get("api_usage", {})
    if api.get("cost_unreported_requests", 0):
        total = None
    else:
        total = cost + api.get("reported_cost_usd", 0) if cost is not None else None
    return {"mean_response_seconds": float(np.mean(latency)) if latency else None,
            "median_response_seconds": float(np.median(latency)) if latency else None,
            "total_api_cost_usd": total, "cost_known": total is not None}


def paired_comparisons(rows: list[dict], metric: str, *, samples=2000, seed=17) -> list[dict]:
    """Full-variant factorial contrasts; repeats are not independent questions.

    Missing metric values exclude the whole question for that contrast, with
    explicit reasons. Unequal trial identities and duplicates are errors.
    Historical callers without a variant field are treated as full-only;
    explicitly named ablations and legacy method names never enter contrasts.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("Bootstrap samples must be a positive integer")
    methods = ("llm", "llm_graph", "clm", "clm_graph")
    groups = {}
    for row in rows:
        if row["method"] not in methods or row.get("variant", "full") != "full":
            continue
        key = (row["query_id"], row["method"], row["action_budget"])
        trial = row.get("trial")
        if type(trial) is not int or trial < 0:
            raise ValueError("Paired comparisons require nonnegative integer trial IDs")
        cell = groups.setdefault(key, {})
        if trial in cell:
            raise ValueError("Duplicate experimental trial/cell identity")
        cell[trial] = row.get(metric)
    pairs = [("llm_graph", "llm"), ("clm", "llm"), ("clm_graph", "llm_graph"), ("clm_graph", "clm")]
    coefficients = {(left, right): {left: 1, right: -1} for left, right in pairs}
    coefficients.update({("graph_main_effect", "factorial"): dict(llm=-.5, llm_graph=.5, clm=-.5, clm_graph=.5),
        ("clm_main_effect", "factorial"): dict(llm=-.5, llm_graph=-.5, clm=.5, clm_graph=.5),
        ("interaction", "factorial"): dict(llm=1, llm_graph=-1, clm=-1, clm_graph=1)})
    result = []
    rng = np.random.default_rng(seed)
    for budget in sorted({key[2] for key in groups}):
        for (left, right), weights in coefficients.items():
            if not set(weights) <= {key[1] for key in groups if key[2] == budget}:
                continue
            differences, excluded, trial_counts = {}, {}, {}
            for question in sorted({key[0] for key in groups if key[2] == budget}):
                cells = [groups.get((question, m, budget)) for m in weights]
                if any(c is None for c in cells):
                    excluded[question] = "missing_method_cell"
                    continue
                if any(set(c) != set(cells[0]) for c in cells[1:]):
                    raise ValueError("Unbalanced trial identities in paired comparison")
                trial_counts[question] = len(cells[0])
                values = [v for c in cells for v in c.values()]
                if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in values):
                    excluded[question] = "missing_or_nonfinite_metric_in_trial"
                    continue
                differences[question] = sum(weight * float(np.mean(list(groups[question, method, budget].values())))
                                            for method, weight in weights.items())
            values = list(differences.values())
            interval = None
            if len(values) >= 2:
                boot = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
                interval = [float(v) for v in np.quantile(boot, [.025, .975])]
            result.append({"left": left, "right": right, "action_budget": budget, "metric": metric,
                           "variant": "full", "trial_counts": trial_counts, "excluded_questions": excluded,
                           "questions": len(values), "per_question_difference": differences,
                           "mean_difference": float(np.mean(values)) if values else None, "bootstrap_95_ci": interval,
                           "interpretation": "Descriptive paired differences; a higher mean alone is not significance"})
    return result
