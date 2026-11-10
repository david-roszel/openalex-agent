"""Blinded claim assessment and independent subtopic/paired research scoring."""

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from .context_memory import serialized
from .corpus import atomic_json
from .evaluation import load_judgments, metrics
from .experiment_integrity import validate_schedule
from .evidence_provenance import claim_access
from .models import canonical_id
from .research_metrics import factual_metrics, paired_comparisons, retrieval_diagnostics


def assessment_claims(result, manifest, query_id):
    """Use only fixed opportunities when configured; planning claims stay traces."""
    tasks=manifest.get("factual_questions",[])
    if not tasks:
        question=next((q["question"] for q in manifest.get("questions",[]) if q["id"]==query_id),query_id)
        return [(query_id,question,c) for c in result.get("evidence_records",[])]
    from .factual_protocol import validate_record
    task=next(q for q in tasks if q["query_id"]==query_id)
    result_claims=[]
    for record in result.get("factual_responses",[]):
        if record.get("status")!="valid":continue
        try:validate_record(record,task["question_id"])
        except ValueError:continue
        result_claims.append((task["question_id"],task["question"],record["evidence_record"]))
    return result_claims


def _experiment(directory):
    root = Path(directory)
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("suite") not in ("controlled-five", "controlled-seven"):
        raise ValueError("Research assessment requires a controlled experiment; legacy outputs are preserved")
    rows = pd.read_csv(root / "metrics.csv", dtype={"query_id": str}).to_dict("records")
    validate_schedule(manifest, rows)
    return root, manifest, rows


def claim_id(query, record):
    identity = {k: record.get(k) for k in ("statement", "work_id", "evidence_quote", "insufficient")}
    source = record.get("provenance", {}).get("source_record")
    if source:
        identity["source_version"] = {k: source.get(k) for k in ("id", "title", "abstract", "retrieved_at")}
    if record.get("insufficient") and record.get("provenance",{}).get("assessment_context_sha256"):
        identity["abstention_context"]=record["provenance"]["assessment_context_sha256"]
    return hashlib.sha256(serialized([query, identity]).encode()).hexdigest()


def build_claim_pool(directory, output):
    root, manifest, rows = _experiment(directory)
    questions = {q["id"]: q["question"] for q in manifest["questions"]}
    records = {}
    for row in rows:
        result = json.loads((Path(row["directory"]) / "result.json").read_text(encoding="utf-8"))
        selected=assessment_claims(result,manifest,row["query_id"])
        for (scope_id,question,claim), access in zip(selected, claim_access(row["directory"], [c for _,_,c in selected])):
            identifier = claim_id(scope_id, claim)
            source = access.get("source_record") or {}
            records[identifier] = {"record_id": identifier, "query_id": row["query_id"], "question_id":scope_id,"question": question,
                **{k: claim[k] for k in ("statement", "work_id", "evidence_quote", "insufficient")},
                "source_title": source.get("title"), "source_abstract": source.get("abstract"),
                "source_evidence_basis": access["basis"],
                "available_evidence_context":claim.get("provenance",{}).get("assessment_context",""),
                "judgment": "", "factually_correct": "", "citation_support_correct": "", "appropriate": "", "assessor": ""}
    columns = ["record_id", "query_id", "question_id", "question", "statement", "work_id", "evidence_quote", "insufficient", "source_title", "source_abstract", "source_evidence_basis", "available_evidence_context",
               "judgment", "factually_correct", "citation_support_correct", "appropriate", "assessor"]
    frame = pd.DataFrame(records.values(), columns=columns)
    if len(frame):
        frame = frame.sample(frac=1, random_state=73)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return len(frame)


def _claim_judgments(path):
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("judgment") not in ("supported", "unsupported", "contradicted", "abstain") or not row.get("assessor", "").strip():
                raise ValueError("Claim judgments require independent status and assessor fields")
            record = {"judgment": row["judgment"]}
            for key in ("factually_correct", "citation_support_correct", "appropriate"):
                value = row.get(key, "").strip().lower()
                if value not in ("", "true", "false"):
                    raise ValueError("Claim judgment booleans must be true, false or unknown/blank")
                if value:
                    record[key] = value == "true"
            key = row["record_id"]
            if key in result and result[key] != record:
                raise ValueError("Conflicting claim judgments require adjudication")
            result[key] = record
    return result


def score_research(directory, *, labels_path=None, subtopics_path=None, claim_judgments_path=None):
    root, manifest, rows = _experiment(directory)
    if not labels_path and not claim_judgments_path:
        raise ValueError("Supply independent relevance or claim judgments")
    judgments, scope = load_judgments(labels_path) if labels_path else ({}, "pooled")
    corpus_ids = set(manifest["corpus_manifest"]["ids"])
    if any(set(labels)-corpus_ids for labels in judgments.values()):
        raise ValueError("Judged papers must belong to the experiment corpus")
    if scope == "exhaustive" and any(set(labels) != corpus_ids for labels in judgments.values()):
        raise ValueError("Exhaustive judgments must cover the whole corpus")
    subtopics = {}
    if subtopics_path:
        data = json.loads(Path(subtopics_path).read_text(encoding="utf-8"))
        if data.get("source") not in ("human_import", "curated"):
            raise ValueError("Subtopic labels require independent human/curated provenance")
        subtopics = data["queries"]
        for value in subtopics.values():
            value["papers"] = {canonical_id(k): v for k, v in value["papers"].items()}
            if set(value["papers"])-corpus_ids or any(set(v)-set(value["expected"]) for v in value["papers"].values()):
                raise ValueError("Subtopic labels must match corpus IDs and defined categories")
    claim_labels = _claim_judgments(claim_judgments_path) if claim_judgments_path else None
    used_claims = set()
    records = []
    access_records = []
    opportunity_groups={}
    for row in rows:
        result = json.loads((Path(row["directory"]) / "result.json").read_text(encoding="utf-8"))
        record = {k: row[k] for k in ("query_id", "method", "variant", "trial", "action_budget", "run_id")}
        from copy import deepcopy
        responses=deepcopy(result.get("factual_responses",[]))
        labels = judgments.get(row["query_id"])
        if labels_path:
            if not labels:
                raise ValueError("Every query requires independent relevance judgments")
            values = metrics([p["paper"]["id"] for p in result["ranked"]], labels, k=10, scope=scope)
            record.update(precision_at_10=values["precision_at_k"], ndcg_at_10=values["ndcg_at_k"],
                          recall_at_10=values["recall_at_k"], judgment_coverage_at_10=values["judged_fraction_at_k"])
            topics = subtopics.get(row["query_id"], {})
            record.update(retrieval_diagnostics(result["inspected_ids"], labels, result["actions"],
                subtopics=topics.get("papers"), expected_subtopics=topics.get("expected")))
        if claim_labels is not None:
            claims = []
            selected=assessment_claims(result,manifest,row["query_id"])
            for (scope_id,question,claim), access in zip(selected, claim_access(row["directory"], [c for _,_,c in selected])):
                identifier = claim_id(scope_id, claim)
                if identifier not in claim_labels:
                    raise ValueError("Claim pool is incompletely judged; unjudged claims are not scored as false")
                used_claims.add(identifier)
                observation = access["source_observed_at_claim"]
                scored = {**claim, **claim_labels[identifier], "citation_exists_in_corpus": claim.get("work_id") in corpus_ids,
                          "identifier_valid": observation, "source_observed_at_claim": observation,
                          "source_accessible_at_claim": access["source_accessible_at_claim"]}
                claims.append(scored)
                for response in responses:
                    if response.get("question_id")==scope_id:response["evidence_record"]=scored
                access_records.append({**record, "record_id": identifier, **scored, "access_basis": access["basis"]})
            record.update(factual_metrics(claims))
            cost = row.get("total_api_cost_usd")
            record["supported_claims_per_usd"] = (sum(c["judgment"] == "supported" for c in claims)/cost
                                                  if isinstance(cost, (int, float)) and cost > 0 else None)
        if manifest.get("factual_questions") and row["method"]!="standard":
            from .factual_protocol import metrics as opportunity_metrics
            expected=[t["question_id"] for t in manifest["factual_questions"] if t["query_id"]==row["query_id"]]
            if claim_labels is None:
                # Structural access is still reconstructed without semantic labels.
                for response in responses:
                    if "evidence_record" in response:
                        c=response["evidence_record"]
                        access=claim_access(row["directory"],[c])[0]
                        c.update({k:access[k] for k in ("source_observed_at_claim","source_accessible_at_claim")})
                        c["citation_exists_in_corpus"]=c.get("work_id") in corpus_ids
            values=opportunity_metrics(expected,responses,claim_labels)
            record.update({"task_"+k:v for k,v in values.items() if k not in ("legacy_claim_metrics","protocol")})
            key=(row["method"],row["variant"],row["action_budget"],row["trial"])
            opportunity_groups.setdefault(key,[]).extend(responses)
        records.append(record)
    if claim_labels is not None and set(claim_labels)-used_claims:
        raise ValueError("Claim judgment file contains records outside this experiment")
    pd.DataFrame(records).to_csv(root / "research-metrics.csv", index=False)
    comparison = paired_comparisons
    if manifest.get("suite") == "controlled-seven":
        from .factorial import paired_comparisons_2x3
        comparison = paired_comparisons_2x3
    contrasts = {metric: comparison(records, metric) for metric in
                 ("precision_at_10", "ndcg_at_10", "recall_at_10", "supported_claim_rate", "claim_factual_accuracy", "unique_relevant_papers",
                  "task_overall_answer_accuracy","task_overall_task_completion","task_answer_coverage")}
    atomic_json(root / "research-paired-comparisons.json", contrasts)
    atomic_json(root / "research-claim-access.json", access_records)
    if manifest.get("factual_questions"):
        from .factual_protocol import metrics as opportunity_metrics
        summaries=[dict(method=k[0],variant=k[1],action_budget=k[2],trial=k[3],
            **opportunity_metrics([t["question_id"] for t in manifest["factual_questions"]],v,claim_labels))
            for k,v in opportunity_groups.items()]
        atomic_json(root/"factual-opportunity-summary.json",summaries)
    atomic_json(root / "assessment-provenance.json", {"relevance": json.loads(Path(labels_path).read_text()) if labels_path else None,
        "subtopics": json.loads(Path(subtopics_path).read_text()) if subtopics_path else None,
        "claim_judgments_sha256": hashlib.sha256(Path(claim_judgments_path).read_bytes()).hexdigest() if claim_judgments_path else None,
        "note": "Independence is a study-design requirement; provenance fields cannot certify human behavior."})
    return len(records)
