"""Independent exploratory system evaluation. No V3 imports, data or spending.

prepare/report never call providers. run requires explicit bounded authorization
for hosted arms. All cases, including failed and never-run cases, stay visible.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import platform
from importlib.metadata import version

from .corpus import atomic_json
from .live_profiles import load_live_settings, thorough_config, PROFILE_VERSION
from .live_safety import SpendingLedger, diagnostic
from .research import PROJECT_ROOT, start_research

QUESTIONS = PROJECT_ROOT / "benchmarks/live_submission_v1/questions.json"
VARIANTS = ("standard", "conventional", "combined")
SOURCES = ("live_benchmark.py", "live_research.py", "live_profiles.py", "live_quality.py", "live_safety.py",
    "research.py", "controller.py", "controlled.py", "context_planner.py", "context_memory.py", "llm.py",
    "openalex.py", "config.py", "models.py", "ranking.py", "selection.py", "storage.py", "corpus.py",
    "agent_state.py", "actions.py", "live_cli.py", "token_budget.py", "tokenizer_data/o200k_base.tiktoken")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def profile(settings):
    return {"model": settings.llm_model, "endpoint_sha256": hashlib.sha256(settings.llm_base_url.encode()).hexdigest(),
        "input_price_upper_bound": settings.llm_input_price, "output_price": settings.llm_output_price,
        "max_output_tokens": settings.llm_max_tokens, "temperature": settings.llm_temperature,
        "reasoning_effort": settings.llm_reasoning_effort, "timeout": settings.llm_timeout,
        "service_tier": "default for OpenAI direct", "discount_assumed": False}


def environment():
    return {"python": platform.python_version(), "packages": {name: version(name) for name in
        ("httpx", "networkx", "numpy", "scikit-learn", "tiktoken")}}


def prepare(output, settings, *, questions_path=QUESTIONS, subset=None, variants=VARIANTS):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Preparation is immutable; choose a new output directory")
    data = json.loads(Path(questions_path).read_text())
    questions = data["questions"]
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate fixed question ID")
    if subset:
        wanted = set(subset)
        if not wanted <= {q["id"] for q in questions}:
            raise ValueError("Unknown fixed question in subset")
        questions = [q for q in questions if q["id"] in wanted]
    if not questions or any(v not in ("standard", "conventional", "combined", "graph", "clm") for v in variants):
        raise ValueError("Empty questions or unsupported comparison")
    config = thorough_config("Frozen evaluation profile")
    cells = [{"id": q["id"] + "-" + v, "question_id": q["id"], "variant": v} for q in questions for v in variants]
    hosted = sum(c["variant"] != "standard" for c in cells)
    plan = {"schema": "live-system-evaluation-v1", "created_at": now(), "profile_version": PROFILE_VERSION,
        "questions": questions, "question_source_sha256": sha(questions_path), "cells": cells,
        "configuration": asdict(config), "provider": profile(settings),
        "environment": environment(),
        "source_sha256": {n: sha(PROJECT_ROOT / "openalex_agent" / n) for n in SOURCES},
        "proposed_hosted_trajectories": hosted, "proposed_total_trajectories": len(cells),
        "hosted_upper_bound_usd": str(Decimal(str(config.inference_limits.max_llm_cost_usd)) * hosted),
        "independent_of_V3": True, "paid_authorized": False,
        "target_resolution_status": data.get("openalex_target_resolution", "Fixture identities only"),
        "protocol": {"primary_retrieval": "Fixed target Recall@10 among the final lexical top ten; exact DOI/title/previously resolved ID only",
            "standard": "One nonagentic pinned keyword query, at most 30 returned records. Answer metrics are not applicable. No hosted synthesis.",
            "agents": "Same Thorough profile and final validation/revision opportunity for B/C; all acquisition, edit, recovery, analysis and synthesis calls count",
            "general_relevance": "NOT EVALUATED", "semantic_accuracy": "UNVERIFIED until independent claim review",
            "failure_policy": "Failed attempts retained; missing metrics remain null. Completion rate includes all attempted cells. Never-run cases remain missing.",
            "repetitions": 1, "independent_unit": "question", "inference": "Exploratory descriptive comparison; no superiority or statistical significance claim",
            "source_regime": "Live OpenAlex with timestamped metadata; changing index is a limitation. Saved trajectories support artifact reproduction.",
            "resumption": "Skip every terminal case. Interrupted or uncertain provider requests are never replayed automatically."}}
    output.mkdir(parents=True)
    atomic_json(output / "plan.json", plan)
    (output / "plan.json").chmod(0o444)
    for cell in cells:
        directory = output / "trajectories" / cell["id"]
        directory.mkdir(parents=True)
        atomic_json(directory / "trajectory.json", {**cell, "status": "NOT_RUN", "attempted": False, "result": None})
    report(output)
    return plan


def authorize(output, *, ceiling, confirmation, source):
    """Only call after a real user's explicit command; not part of prepare/run."""
    output = Path(output)
    if confirmation != "AUTHORIZE_LIVEDEMO_HOSTED" or not source.strip():
        raise ValueError("Explicit LiveDemo hosted authorization and its source are required")
    maximum = Decimal(str(ceiling))
    if not maximum.is_finite() or maximum <= 0:
        raise ValueError("Use a positive finite authorized ceiling")
    target = output / "authorization.json"
    if target.exists():
        raise ValueError("An authorization already exists; do not replace it or reset spending")
    authorization = {"version": "independent-live-demo-authorization-v1", "approved": True,
        "plan_sha256": sha(output / "plan.json"), "ceiling_usd": str(maximum),
        "authorization_source": source, "authorized_at": now(), "V3_funds": False}
    atomic_json(target, authorization)
    return authorization


def check_plan(output):
    plan = json.loads((output / "plan.json").read_text())
    if any(sha(PROJECT_ROOT / "openalex_agent" / n) != value for n, value in plan["source_sha256"].items()):
        raise ValueError("Research implementation changed since preparation; preserve this plan and prepare a new version")
    if plan.get("environment") != environment():
        raise ValueError("Runtime dependencies changed since preparation; preserve the plan and prepare a new version")
    return plan


def validate_targets(output, settings):
    """Identity lookup only, never acquisition evidence supplied to an agent."""
    from .storage import Database
    from .openalex import OpenAlexClient, RetrievalError
    from .models import Paper
    output = Path(output)
    plan = check_plan(output)
    receipt = {"plan_sha256": sha(output / "plan.json"), "checked_at": now(), "paid_model_calls": 0,
        "targets": [{"question_id": q["id"], "doi": t["doi"], "status": "NOT_CHECKED"} for q in plan["questions"] for t in q["targets"]],
        "scope": "Gold publication identity verification; no lookup records are supplied to research trajectories"}
    with Database(str(output / "target-identities.sqlite")) as db, OpenAlexClient(db,
            replace(settings, max_retries=0, timeout=min(15, settings.timeout), max_requests=min(40, settings.max_requests))) as client:
        for record in receipt["targets"]:
            target = next(t for q in plan["questions"] if q["id"] == record["question_id"] for t in q["targets"] if t["doi"] == record["doi"])
            try:
                try:
                    raw = client._request("/works/https://doi.org/" + target["doi"])
                    paper = Paper.from_openalex(raw)
                except RetrievalError:
                    # Only a successful HTTP response establishes that the API
                    # is reachable; do not repeat searches after transport loss.
                    errors = [e for e in db.events(client.run_id) if e["kind"] == "request_error"]
                    if not errors or errors[-1].get("status") != 404:
                        raise
                    candidates = client.search(target["title"], limit=3)
                    paper = next((p for p in candidates if matches(p.to_dict(), target)), None)
                if paper is None or not matches(paper.to_dict(), target):
                    record.update(status="UNRESOLVED")
                else:
                    record.update(status="VERIFIED", openalex_id=paper.id, title=paper.title, year=paper.year,
                        observed_doi=paper.doi, metadata_source=paper.metadata_source)
            except (RetrievalError, ValueError) as exc:
                record.update(status="FAILED", error_type=type(exc).__name__)
                break
        receipt["api_usage"] = asdict(client.usage)
    receipt["all_verified"] = all(r["status"] == "VERIFIED" for r in receipt["targets"])
    atomic_json(output / "target-verification.json", receipt)
    return receipt


@contextmanager
def runner_lock(output):
    with (output / "runner.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This independent benchmark already has an active runner") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run(output, settings, *, only=None, transport_factory=None):
    output = Path(output).resolve()
    plan = check_plan(output)
    selected = [c for c in plan["cells"] if not only or c["variant"] in only]
    ledger = None
    if any(c["variant"] != "standard" for c in selected):
        auth_path = output / "authorization.json"
        if not auth_path.exists():
            raise ValueError("No independent LiveDemo authorization. Prepare and review the cost proposal, then explicitly authorize; V3's $68 is not available here.")
        auth = json.loads(auth_path.read_text())
        if auth.get("approved") is not True or auth.get("plan_sha256") != sha(output / "plan.json") or auth.get("V3_funds") is not False:
            raise ValueError("Authorization does not match this independent frozen plan")
        if profile(settings) != plan["provider"]:
            raise ValueError("Hosted provider settings changed since the frozen plan")
        ledger = SpendingLedger(output / "live-spending.sqlite", ceiling=auth["ceiling_usd"], identity=auth["plan_sha256"])
    with runner_lock(output):
        mode_path = output / "execution-mode.json"
        mode = {"mocked_generation_and_retrieval": transport_factory is not None}
        if mode_path.exists() and json.loads(mode_path.read_text()) != mode:
            raise ValueError("Do not mix mocked diagnostics and real evaluation in one campaign; prepare a separate directory")
        if not mode_path.exists():
            atomic_json(mode_path, mode)
        if ledger is not None and transport_factory is None:
            from .live_cli import check_openalex
            if check_openalex(settings, root=output) != 0:
                report(output)
                raise ValueError("Live OpenAlex preflight failed. No hosted trajectory was started; run from a network-enabled environment.")
            verification_path = output / "target-verification.json"
            verification = json.loads(verification_path.read_text()) if verification_path.exists() else validate_targets(output, settings)
            if not verification.get("all_verified") or verification.get("plan_sha256") != sha(output / "plan.json"):
                raise ValueError("Predefined target identities are not all verified in OpenAlex. No hosted trajectory started; inspect target-verification.json.")
        for cell in selected:
            path = output / "trajectories" / cell["id"] / "trajectory.json"
            old = json.loads(path.read_text())
            if old["status"] == "RUNNING":
                old.update(status="INTERRUPTED", error="Previous in-flight trajectory preserved; no automatic replay")
                atomic_json(path, old)
                continue
            if old["status"] != "NOT_RUN":
                continue
            if ledger and ledger.summary()["uncertain_attempts"]:
                break
            question = next(q for q in plan["questions"] if q["id"] == cell["question_id"])
            record = {**cell, "status": "RUNNING", "attempted": True, "started_at": now(), "result": None,
                "evidence_mode": "MOCKED_INTEGRATION" if transport_factory else "REAL_LIVE_OPENALEX_AND_HOSTED" if cell["variant"] != "standard" else "REAL_LIVE_OPENALEX_NO_LLM"}
            atomic_json(path, record)
            try:
                config = thorough_config(question["question"] if cell["variant"] != "standard" else question["baseline_query"], variant=cell["variant"])
                effective = replace(settings, max_retries=0)
                transports = transport_factory(cell) if transport_factory else {}
                job = start_research(config, effective, root=path.parent, spending_ledger=ledger if cell["variant"] != "standard" else None,
                    trajectory_id=cell["id"], **transports)
                record["research_directory"] = str(job.directory)
                atomic_json(path, record)
                while job.snapshot()["status"] in ("starting", "running"):
                    job.join(0.25)
                job.join(5)
                state = job.snapshot()
                record.update(status="COMPLETED" if state["status"] == "completed" else "FAILED",
                    result=state.get("result"), error=state.get("error"), events=state["events"], ended_at=now())
                if cell["variant"] != "standard" and ((record.get("result") or {}).get("synthesis") or {}).get("status") != "qualified":
                    record["status"] = "PARTIAL" if record.get("result") else "FAILED"
            except KeyboardInterrupt:
                record.update(status="INTERRUPTED", error="Interrupted; paid holds and artifacts retained; no replay", ended_at=now())
                atomic_json(path, record)
                report(output)
                raise
            except Exception as exc:
                record.update(status="FAILED", error=diagnostic(exc, (settings.api_key, settings.llm_api_key, settings.email)), ended_at=now())
            atomic_json(path, record)
            report(output)
    return report(output)


def normalized(value):
    return " ".join(re.sub(r"[^a-z0-9]", " ", (value or "").lower()).split())


def doi(value):
    return (value or "").lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")


def matches(paper, target):
    return (bool(target.get("openalex_id") and target["openalex_id"] == paper.get("id")) or
        bool(paper.get("doi") and doi(paper["doi"]) == doi(target["doi"])) or
        (normalized(paper.get("title")) == normalized(target["title"]) and
         (not target.get("year") or paper.get("year") == target["year"])))


def score(record, question, provider):
    result = record.get("result") or {}
    ranked = [r["paper"] for r in result.get("ranked", [])]
    top = ranked[:10]
    targets = question["targets"]
    valid_retrieval = bool(result) and record["status"] in ("COMPLETED", "PARTIAL")
    found = [t["doi"] for t in targets if any(matches(p, t) for p in top)] if valid_retrieval else None
    synthesis = result.get("synthesis") or {}
    answer = synthesis.get("answer") or {}
    claims = [c for k in ("answer", "foundations", "comparison", "conclusion") for c in answer.get(k, [])]
    cited = {c["work_id"] for claim in claims for c in claim["citations"]}
    source_ids = {p["id"] for p in synthesis.get("sources", [])}
    usage = result.get("llm_usage") or {}
    events = record.get("events", [])
    coverage = result.get("research_coverage") or {}
    standard = record["variant"] == "standard"
    status = record["status"]
    performed = record.get("attempted", False)
    measured_cost = None
    if usage.get("calls", 0) and usage.get("usage_unreported_calls", 0) == 0 and usage.get("usage_reported_calls", 0) == usage["calls"]:
        measured_cost = (usage.get("input_tokens", 0) * provider["input_price_upper_bound"] +
                         usage.get("output_tokens", 0) * provider["output_price"]) / 1e6
    return {"id": record["id"], "question_id": question["id"], "domain": question["domain"], "variant": record["variant"],
        "status": status, "attempted": performed, "evidence_mode": record.get("evidence_mode", "NOT_RUN"),
        "target_recall_at_10": len(found) / len(targets) if found is not None else None, "targets_found_at_10": found,
        "unique_publications": len(result["retrieved_ids"]) if result else None,
        "sources_sent_to_model": (result.get("live_accounting") or {}).get("sources_sent_to_model") if result else None,
        "final_answer_produced": None if standard or not performed else bool(answer),
        "validated_synthesis_completed": None if standard or not performed else synthesis.get("status") == "qualified",
        "traceable_citations": len(cited & source_ids) if answer else None,
        "untraceable_citations": len(cited - source_ids) if answer else None,
        "scientifically_unsupported_claims": None, "independent_claim_review": "UNVERIFIED",
        "required_components": question["required_components"], "component_completeness_review": "UNVERIFIED",
        "model_component_coverage": result.get("research_coverage"), "limitations_present": bool(answer.get("limitations")) if answer else None,
        "actions": result.get("actions"), "http_requests": (result.get("api_usage") or {}).get("requests") if result else None,
        "model_calls": usage.get("calls") if result else None, "input_tokens": usage.get("input_tokens") if result else None,
        "output_tokens": usage.get("output_tokens") if result else None, "seconds": result.get("seconds"),
        "usage_based_cost_upper_bound_usd": measured_cost if not standard else 0 if result else None,
        "cost_basis": "Reported provider tokens at conservative configured prices; not invoice cost or discounts",
        "cost_unreported_calls": usage.get("usage_unreported_calls") if result else None,
        "openalex_reported_cost_usd": result["api_usage"]["reported_cost_usd"] if result and result["api_usage"].get("cost_reported_requests", 0) else None,
        "openalex_cost_unreported_requests": result["api_usage"].get("cost_unreported_requests") if result else None,
        "retrieval_failed": status == "FAILED" and (record.get("error") or {}).get("type") == "retrieval_error" if isinstance(record.get("error"), dict) else status == "FAILED",
        "planning_failures": sum(e["kind"] == "live_model_call" and e.get("stage") == "planning" and bool(e.get("error_code")) for e in events) if performed else None,
        "invalid_actions": sum(e["kind"] == "invalid_decision" for e in events) if performed else None,
        "repeated_action_rejections": sum(e["kind"] == "invalid_decision" and "Repeated" in e.get("validation_rule", "") for e in events) if performed else None,
        "unresolved_components_at_stop": coverage.get("unresolved_components"),
        "premature_stop_proxy": bool(result.get("stop_reason") == "planner_stop" and coverage.get("unresolved_components")) if result else None,
        "premature_stop_definition": "Unresolved model-assessed components at planner STOP; not independent scientific completeness",
        "synthesis_failed": None if standard or not performed else not bool(answer),
        "stop_reason": result.get("stop_reason"), "resource_limits_reached": result.get("binding_limits"),
        "resource_limit_failure": bool(not answer and not standard and (synthesis.get("resource_limit") or
            any(word in result.get("stop_reason", "") for word in ("budget", "max_", "time_limit", "token_")))) if result else None,
        "mechanisms": result.get("mechanisms"), "error": record.get("error"),
        "research_directory": record.get("research_directory")}


def mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (float, int)) and not isinstance(r[key], bool)]
    return sum(values) / len(values) if values else None


def table(output, name, headers, rows):
    def display(v):
        return "—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    with (output / (name + ".csv")).open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(headers); writer.writerows(rows)
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    md += ["| " + " | ".join(display(v) for v in row) + " |" for row in rows]
    (output / (name + ".md")).write_text("\n".join(md) + "\n")
    def tex(v):
        return display(v).replace("—", "--").replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{tabular}{" + "l" * len(headers) + "}", r"\hline",
        " & ".join(tex(v) for v in headers) + r" \\", r"\hline"]
    lines += [" & ".join(tex(v) for v in row) + r" \\" for row in rows]
    (output / (name + ".tex")).write_text("\n".join(lines + [r"\hline", r"\end{tabular}"]) + "\n")


def report(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    records = [json.loads((output / "trajectories" / cell["id"] / "trajectory.json").read_text()) for cell in plan["cells"]]
    rows = [score(r, next(q for q in plan["questions"] if q["id"] == r["question_id"]), plan["provider"]) for r in records]
    tables = output / "tables"; tables.mkdir(exist_ok=True)
    variants = list(dict.fromkeys(c["variant"] for c in plan["cells"]))
    retrieval, reliability, efficiency, mechanism = [], [], [], []
    for variant in variants:
        arm = [r for r in rows if r["variant"] == variant]
        attempted = sum(r["attempted"] for r in arm)
        completed = sum(r["status"] == "COMPLETED" for r in arm)
        failed = sum(r["status"] in ("FAILED", "PARTIAL", "INTERRUPTED") for r in arm)
        missing = sum(r["status"] == "NOT_RUN" for r in arm)
        retrieval.append([variant, len(arm), completed, failed, missing, mean(arm, "target_recall_at_10"), mean(arm, "unique_publications"), mean(arm, "sources_sent_to_model")])
        answers = sum(r["validated_synthesis_completed"] is True for r in arm) if variant != "standard" and attempted else None
        reliability.append([variant, attempted, completed, failed, missing, answers,
            completed / attempted if attempted else None, "UNVERIFIED"])
        efficiency.append([variant, mean(arm, "actions"), mean(arm, "http_requests"), mean(arm, "model_calls"), mean(arm, "input_tokens"),
            mean(arm, "output_tokens"), mean(arm, "seconds"), mean(arm, "usage_based_cost_upper_bound_usd"), mean(arm, "openalex_reported_cost_usd")])
        mechanisms = [r["mechanisms"] for r in arm if r.get("mechanisms")]
        mechanism.append([variant, *[mean(mechanisms, key) for key in ("graph_edges_stored", "graph_unique_edges_sent", "clm_edits_proposed",
            "clm_edits_committed", "committed_edits_in_later_requests", "accepted_recoveries")]])
    table(tables, "retrieval", ["Variant", "Planned", "Completed", "Failed/partial", "Not run", "Recall@10 available", "Unique sources", "Sources sent"], retrieval)
    table(tables, "reliability", ["Variant", "Attempted", "Completed", "Failed/partial", "Not run", "Validated answers", "Operational completion", "Semantic review"], reliability)
    table(tables, "efficiency", ["Variant", "Actions", "HTTP", "Model calls", "Input tokens", "Output tokens", "Seconds", "Hosted upper USD", "OpenAlex reported USD"], efficiency)
    table(tables, "mechanisms", ["Variant", "Edges stored", "Edges sent", "Edits proposed", "Edits committed", "Edits later sent", "Recoveries"], mechanism)
    review = []
    for record in records:
        syn = ((record.get("result") or {}).get("synthesis") or {})
        for section in ("answer", "foundations", "comparison", "conclusion"):
            for index, claim in enumerate((syn.get("answer") or {}).get(section, [])):
                review.append({"trajectory": record["id"], "section": section, "claim_index": index, "claim": claim["text"],
                    "citations": claim["citations"], "verdict": "UNVERIFIED", "reviewer": None,
                    "rubric": "SUPPORTED / PARTLY_SUPPORTED / UNSUPPORTED / CONTRADICTED / NOT_ASSESSABLE; read each cited excerpt and accessible source, assess population/method/outcome/scope and citation direction"})
    atomic_json(output / "claim-review.json", review)
    component_review = [{"trajectory": r["id"], "component": component, "verdict": "UNVERIFIED", "reviewer": None}
        for r in records if r["variant"] != "standard" for component in next(q for q in plan["questions"] if q["id"] == r["question_id"])["required_components"]]
    atomic_json(output / "component-review.json", component_review)
    genuine = sum(r["status"] == "COMPLETED" and r["variant"] != "standard" and r["evidence_mode"] == "REAL_LIVE_OPENALEX_AND_HOSTED" for r in rows)
    summary = {"evaluation": "Small exploratory end-to-end system evaluation; not an architecture superiority study",
        "planned": len(rows), "attempted": sum(r["attempted"] for r in rows), "real_hosted_completed": genuine,
        "semantic_claim_review": "UNVERIFIED", "rows": rows, "missing_values": "null; never-run is not a zero score",
        "standard_answer_metrics": "Not applicable; baseline receives no answer generator", "plan_sha256": sha(output / "plan.json")}
    summary["evidence_modes"] = sorted({r["evidence_mode"] for r in rows})
    summary["complete_hosted_question_pairs"] = sum(all(any(r["question_id"] == q["id"] and r["variant"] == v and
        r["status"] == "COMPLETED" for r in rows) for v in ("conventional", "combined")) for q in plan["questions"])
    atomic_json(output / "results.json", summary)
    issues = ["# Error analysis", "", "All planned cells are retained. Missing values are not inferred outcomes.", "",
        "Recall means known-target recall, not general relevance. Citation traceability and mechanism exposure do not establish scientific accuracy.",
        "Standard has no answer generator. Agent completion means a structurally validated answer, not an independently verified complete review.", ""]
    issues += [f"- {r['id']}: {r['status']}; stop={r.get('stop_reason')}; error={json.dumps(r.get('error'), ensure_ascii=False)}" for r in rows]
    (output / "ERROR_ANALYSIS.md").write_text("\n".join(issues) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--questions", type=Path, default=QUESTIONS); p.add_argument("--subset", help="Comma-separated fixed IDs; choose before outcomes")
    a = sub.add_parser("authorize"); a.add_argument("--output", type=Path, required=True); a.add_argument("--ceiling-usd", required=True)
    a.add_argument("--confirm", required=True); a.add_argument("--source", required=True)
    r = sub.add_parser("run"); r.add_argument("--output", type=Path, required=True); r.add_argument("--only", help="Comma-separated variants; standard needs no hosted authorization")
    t = sub.add_parser("report"); t.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("validate-targets"); v.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            value = prepare(args.output, load_live_settings(), questions_path=args.questions, subset=args.subset.split(",") if args.subset else None)
            print(json.dumps({k: value[k] for k in ("proposed_total_trajectories", "proposed_hosted_trajectories", "hosted_upper_bound_usd")}))
        elif args.command == "authorize":
            print(json.dumps(authorize(args.output, ceiling=args.ceiling_usd, confirmation=args.confirm, source=args.source)))
        elif args.command == "run":
            value = run(args.output, load_live_settings(), only=args.only.split(",") if args.only else None)
            print(json.dumps({k: value[k] for k in ("planned", "attempted", "real_hosted_completed")}))
            if any(r["status"] in ("FAILED", "PARTIAL", "INTERRUPTED") and
                    (not args.only or r["variant"] in args.only.split(",")) for r in value["rows"]):
                return 1
        elif args.command == "validate-targets":
            value = validate_targets(args.output, load_live_settings())
            print(json.dumps({"all_verified": value["all_verified"], "verified": sum(r["status"] == "VERIFIED" for r in value["targets"]), "paid_model_calls": 0}))
            return 0 if value["all_verified"] else 1
        else:
            value = report(args.output)
            print(json.dumps({k: value[k] for k in ("planned", "attempted", "real_hosted_completed")}))
        return 0
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
