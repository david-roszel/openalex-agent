"""Human relevance judgments and metrics that expose incomplete assessment."""

import csv
import json
import math
from pathlib import Path
from .corpus import atomic_json
from .models import canonical_id, utcnow


def import_judgments(csv_path: str, output: str, *, scope: str = "pooled") -> dict:
    if scope not in ("pooled", "exhaustive"):
        raise ValueError("Judgment scope must be pooled or exhaustive")
    judgments: dict[tuple[str, str], dict] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"query_id", "paper_id", "grade", "assessor"} <= set(reader.fieldnames or []):
            raise ValueError("Judgment CSV needs query_id,paper_id,grade,assessor")
        for row in reader:
            grade = int(row["grade"])
            if grade not in range(4) or not row["assessor"].strip() or not row["query_id"].strip():
                raise ValueError("Grades must be 0–3 and query/assessor IDs nonempty")
            paper_id = canonical_id(row["paper_id"])
            key = (row["query_id"], paper_id)
            if key in judgments and judgments[key]["grade"] != grade:
                raise ValueError("Conflicting judgments: adjudicate before importing")
            judgments[key] = {"query_id": row["query_id"], "paper_id": paper_id,
                              "grade": grade, "assessor": row["assessor"]}
    result = {"scope": scope, "source": "human_import", "created_at": utcnow(), "judgments": list(judgments.values())}
    atomic_json(Path(output), result)
    return result


def load_judgments(path: str) -> tuple[dict[str, dict[str, int]], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("source") not in ("human_import", "curated") or data.get("scope") not in ("pooled", "exhaustive"):
        raise ValueError("Judgments require human/curated provenance and explicit scope")
    result: dict[str, dict[str, int]] = {}
    for row in data["judgments"]:
        paper_id = canonical_id(row["paper_id"])
        grade = row["grade"]
        if not isinstance(grade, int) or isinstance(grade, bool) or grade not in range(4):
            raise ValueError("Judgment grades must be integers 0–3")
        query = result.setdefault(row["query_id"], {})
        if paper_id in query and query[paper_id] != grade:
            raise ValueError("Conflicting judgment grades")
        query[paper_id] = grade
    return result, data["scope"]


def metrics(ranking: list[str], judgments: dict[str, int], *, k: int = 20, scope: str = "pooled",
            inspected: list[str] | None = None, actions: int = 0) -> dict:
    if k < 1 or len(ranking) != len(set(ranking)) or scope not in ("pooled", "exhaustive"):
        raise ValueError("Require positive k, unique ranking IDs, and valid judgment scope")
    top = ranking[:k]
    unknown = sum(p not in judgments for p in top)
    known_relevant = sum(grade > 0 for grade in judgments.values())
    relevant = sum(judgments.get(p, 0) > 0 for p in top)
    judged_count = len(top) - unknown
    dcg = sum((2**judgments.get(p, 0)-1)/math.log2(i+2) for i, p in enumerate(top))
    ideal = sum((2**g-1)/math.log2(i+2) for i, g in enumerate(sorted(judgments.values(), reverse=True)[:k]))
    ndcg = dcg/ideal if ideal else None
    inspected = list(dict.fromkeys(inspected or []))
    if scope == "exhaustive" and any(p not in judgments for p in top + inspected):
        raise ValueError("Exhaustive scope cannot contain unjudged output papers")
    inspected_judged = [p for p in inspected if p in judgments]
    discovered_relevant = sum(judgments.get(p, 0) > 0 for p in inspected)
    return {"k": k, "returned": len(top), "judgment_scope": scope, "known_relevant": known_relevant,
            "judged_at_k": judged_count, "unjudged_at_k": unknown,
            "judged_fraction_at_k": judged_count/len(top) if top else 0.0,
            "precision_at_k": relevant/k if not unknown else None,
            "precision_lower_bound": relevant/k, "precision_upper_bound": (relevant+unknown)/k,
            "judged_precision": relevant/judged_count if judged_count else None,
            "recall_at_k": relevant/known_relevant if known_relevant else None,
            "recall_scope": "judged_pool" if scope == "pooled" else "candidate_corpus",
            "ndcg_at_k": ndcg if not unknown else None,
            "ndcg_scope": "judged_pool" if scope == "pooled" else "candidate_corpus",
            "ndcg_zero_unjudged_convention": ndcg,
            "known_relevant_per_action": discovered_relevant/actions if actions else None,
            "inspected_known_recall": discovered_relevant/known_relevant if known_relevant else None,
            "judged_irrelevant_inspection_fraction": sum(judgments[p] == 0 for p in inspected_judged)/len(inspected_judged) if inspected_judged else None,
            "inspected_judged_fraction": len(inspected_judged)/len(inspected) if inspected else 0.0}
