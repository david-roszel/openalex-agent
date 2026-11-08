"""Controlled four-method comparisons, action sweeps, pools, and report artifacts."""

from dataclasses import asdict, replace
import hashlib
import json
import time
from pathlib import Path
from typing import Callable
import pandas as pd
from .actions import RELIABILITY_VERSION
from .controller import Budget, Controller, METHODS
from .corpus import LocalCorpus, atomic_json
from .evaluation import load_judgments, metrics
from .llm import DeterministicPlanner, Planner
from .selection import SELECTION_VERSION, Scoring
from .storage import Database
from .models import utcnow
from .git_provenance import git_provenance


def environment_record() -> dict:
    """Record runtime versions, without environment variables or credentials."""
    import importlib.metadata
    import platform
    packages = ["httpx", "networkx", "numpy", "pandas", "scikit-learn", "matplotlib", "python-dotenv"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}


def _capture_requests(planner, path: Path):
    """Capture whitelisted wire payloads, never URLs, headers or error bodies."""
    if not hasattr(planner, "http") or not hasattr(planner, "_redact"):
        return
    def capture(request):
        body = json.loads(request.content)
        safe = planner._redact({key: body[key] for key in ("model", "messages", "response_format",
            "max_tokens", "max_completion_tokens", "temperature", "reasoning_effort", "input", "tools",
            "max_output_tokens", "reasoning", "store", "text", "tool_choice", "parallel_tool_calls", "include") if key in body})
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(safe, ensure_ascii=False) + "\n")
    planner.http.event_hooks["request"].append(capture)


def run_experiments(*args, **kwargs):
    """Shared CLI/UI entry point; preserve stopped sweep manifests and raw runs."""
    from .execution_control import ExperimentStopped
    target=Path(kwargs["output"])
    owned=not target.exists() or target.is_dir() and not any(target.iterdir())
    try:
        return _run_experiments(*args, **kwargs)
    except ExperimentStopped as exc:
        path = Path(kwargs["output"]) / "experiment.json"
        if owned and path.exists():
            manifest = json.loads(path.read_text())
            manifest.update(status=exc.reason, stopped_at=utcnow())
            atomic_json(path, manifest)
        raise
    except Exception as exc:
        path=Path(kwargs["output"])/"experiment.json"
        if owned and path.exists():
            manifest=json.loads(path.read_text())
            manifest.update(status="failed",error_type=type(exc).__name__,stopped_at=utcnow())
            atomic_json(path,manifest)
        raise


def _run_experiments(corpus_path: str, questions_path: str, db: Database, planner_factory: Callable[[], Planner], *,
                    output: str, labels_path: str | None = None, action_budgets: tuple[int, ...] = (5, 10, 20),
                    trials: int = 3, budget: Budget | None = None, methods: tuple[str, ...] | None = None,
                    allow_deterministic: bool = False, ablations: bool = False, suite: str = "legacy",
                    inference_limits=None, require_evidence: bool = False, factual_questions_path: str | None = None,
                    trial_seeds: tuple[int, ...] | None = None, cancel_check=None) -> list[dict]:
    from .controlled import ARCHITECTURES, ALL_ARCHITECTURES, META_ARCHITECTURES, ControlledController, context_strategy
    from .context_planner import ContextPlanner, EngineeringContextModel, InferenceLimits
    from .research_metrics import efficiency_metrics, retrieval_diagnostics
    if suite not in ("legacy", "controlled-five", "controlled-seven"):
        raise ValueError("Choose legacy, controlled-five or controlled-seven experiment suite")
    available = METHODS if suite == "legacy" else ALL_ARCHITECTURES if suite == "controlled-seven" else ARCHITECTURES
    methods = available if methods is None else methods
    inference_limits = inference_limits or InferenceLimits()
    if type(trials) is not int or trials < 1 or not action_budgets or any(type(b) is not int or b < 1 for b in action_budgets):
        raise ValueError("Trials and action budgets must be positive")
    if trial_seeds is not None and (len(trial_seeds)!=trials or any(type(s) is not int or s<0 for s in trial_seeds)):
        raise ValueError("Provide one nonnegative trial seed per repeated trial")
    if not methods or len(set(methods)) != len(methods) or any(m not in available for m in methods):
        raise ValueError("Unknown retrieval method")
    if require_evidence and set(methods).intersection(META_ARCHITECTURES):
        raise ValueError("Meta uses native functions; use the common terminal factual protocol instead of legacy per-decision evidence envelopes")
    if len(set(action_budgets)) != len(action_budgets):
        raise ValueError("Action budgets must be unique")
    directory = Path(output)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("Experiment output already exists; use a fresh directory")
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions or any(
        not isinstance(q, dict) or not isinstance(q.get("id"), str) or not q["id"].strip() or
        not isinstance(q.get("question"), str) or not 1 <= len(q["question"].strip()) <= 1000 for q in questions):
        raise ValueError("Questions require nonempty string IDs and research questions of 1–1000 characters")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Questions must have unique IDs")
    corpus = LocalCorpus(corpus_path)
    judgments, scope = load_judgments(labels_path) if labels_path else ({}, "pooled")
    for q in questions:
        query_labels = judgments.get(q["id"], {})
        if labels_path and not query_labels:
            raise ValueError(f"No judgments for {q['id']}")
        if set(query_labels) - set(corpus.by_id):
            raise ValueError("Judged paper is outside the frozen corpus")
        if scope == "exhaustive" and set(query_labels) != set(corpus.by_id):
            raise ValueError("Exhaustive scope requires every candidate judged for each question")
    base_budget = budget or Budget(max_rounds=max(action_budgets))
    factual_questions=[]
    if factual_questions_path:
        if suite not in ("controlled-five", "controlled-seven") or base_budget.max_llm_calls<2:
            raise ValueError("Fixed factual questions require controlled-five and at least two total model calls")
        factual_questions=json.loads(Path(factual_questions_path).read_text(encoding="utf-8"))
        if (not isinstance(factual_questions,list) or any(not isinstance(f,dict) or set(f)!={"query_id","question_id","question"}
                or any(not isinstance(v,str) or not v.strip() or len(v)>1000 for v in f.values()) for f in factual_questions)
                or {f["query_id"] for f in factual_questions}!={q["id"] for q in questions}
                or len(factual_questions)!=len(questions)
                or len({f["question_id"] for f in factual_questions})!=len(factual_questions)):
            raise ValueError("Declare exactly one public factual question per research question, with no hidden answer fields")
    probe = planner_factory()
    planner_name, model = probe.name, probe.model
    provider_fingerprint = probe.fingerprint
    provider_configuration = getattr(probe, "configuration", {})
    if suite == "controlled-seven" and planner_name != "deterministic" and not hasattr(probe, "_call_native"):
        if hasattr(probe, "close"): probe.close()
        raise ValueError("controlled-seven requires explicit Responses transport for all six agent configurations")
    if hasattr(probe, "close"):
        probe.close()
    if planner_name == "deterministic" and any(m != "standard" for m in methods) and not allow_deterministic:
        raise ValueError("LLM comparisons require a hosted planner; --allow-deterministic is for engineering smoke tests")
    directory.mkdir(parents=True, exist_ok=True)
    variants = [("full", Scoring())]
    if ablations:
        variants += [("no_novelty", Scoring(novelty=0)), ("no_importance", Scoring(importance=0)),
                     ("no_coverage", Scoring(coverage=0))]
    systems = [(method, variant, scoring) for method in methods
               for variant, scoring in (variants if method in ("graph", "llm_graph", "clm_graph") else variants[:1])]
    schedule = []
    for qi, q in enumerate(questions):
        for bi, max_actions in enumerate(action_budgets):
            for trial in range(trials):
                offset = (qi + bi + trial) % len(systems)
                for method, variant, _ in systems[offset:] + systems[:offset]:
                    schedule.append({"query_id": q["id"], "action_budget": max_actions,
                                     "trial": trial, "method": method, "variant": variant})
    manifest = {"corpus": corpus.fingerprint, "corpus_manifest": corpus.manifest,
        "created_at": utcnow(), "environment": environment_record(), "git": git_provenance(), "questions": questions,
        "questions_sha256": hashlib.sha256(Path(questions_path).read_bytes()).hexdigest(),
        "judgments": json.loads(Path(labels_path).read_text()) if labels_path else None,
        "action_budgets": action_budgets, "trials": trials, "methods": methods, "base_budget": base_budget.__dict__,
        "planner": planner_name, "model": model, "deterministic_smoke": planner_name == "deterministic" or provider_configuration.get('mocked_generation',False),
        "suite": suite, "inference_limits": asdict(inference_limits) if suite in ("controlled-five", "controlled-seven") else None,
        "experimental_implementation_version": ("meta-conversation-reference-v1" if suite == "controlled-seven" else RELIABILITY_VERSION if suite == "controlled-five" else None),
        "require_evidence": require_evidence if suite in ("controlled-five", "controlled-seven") else False,
        "factual_questions":factual_questions,
        "trial_seeds":trial_seeds,
        "seed_role":"trial identifier; no hosted sampling seed is transmitted",
        "factual_protocol":("fixed-factual-opportunity-v1" if factual_questions else None),
        "factual_budget_policy":("one common total call reserved for final response; remaining calls for retrieval/edit/recovery" if factual_questions else None),
        "provider_fingerprint": provider_fingerprint, "provider_configuration": provider_configuration,
        "selection_version": SELECTION_VERSION,
        "scoring_variants": {name: scoring.__dict__ for name, scoring in variants},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
        "run_order": schedule, "planned_runs": len(schedule), "status": "running",
        "has_relevance_judgments": bool(labels_path)}
    if suite in ("controlled-five", "controlled-seven"):
        from .context_planner import CONTEXT_POLICY_VERSION
        manifest["legacy_context_policy_version"] = CONTEXT_POLICY_VERSION
    atomic_json(directory / "experiment.json", manifest)
    if suite == "controlled-seven":
        from .meta_context import META_VERSION, VENDOR, reference_provenance
        from shutil import copytree
        manifest.update(experimental_implementation_version=META_VERSION,meta_reference=reference_provenance(),
            context_strategies={method:context_strategy(method) for method in methods},
            analysis_version="factorial-2x3-descriptive-v1",
            transport_comparison="Responses for all six agents; native Meta functions versus existing JSON decision envelopes is a declared adapter difference",
            meta_adaptations=dict(version=META_VERSION,sandbox="single-file capability; no Bash",
                archived_observation_recovery=False,overflow="official gate plus unchanged common hard limits"))
        if provider_configuration.get('protocol','').startswith('gemini-'):
            manifest.update(provider_experiment_version=provider_configuration['protocol'],
                transport_comparison='Gemini Enterprise generate_content for all six agents; existing JSON envelopes versus native Meta functions; ADC only')
        else:
            manifest['provider_experiment_version']='terra-responses-seven-v1'
        atomic_json(directory / "experiment.json",manifest)
        copytree(VENDOR,directory / "reference-source",ignore=__import__('shutil').ignore_patterns('__pycache__'))
    (directory / "source").mkdir()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        (directory / "source" / path.name).write_bytes(path.read_bytes())
    rows = []
    curve_rows = []
    for qi, q in enumerate(questions):
        for bi, max_actions in enumerate(action_budgets):
            for trial in range(trials):
                offset = (qi + bi + trial) % len(systems)
                for method, variant, scoring in systems[offset:] + systems[:offset]:
                    if cancel_check is not None and cancel_check():
                        from .execution_control import ExperimentStopped
                        raise ExperimentStopped()
                    planner = planner_factory() if method != "standard" else DeterministicPlanner()
                    run_budget = replace(base_budget, max_actions=max_actions)
                    retrieval_budget = replace(run_budget,max_llm_calls=run_budget.max_llm_calls-1) if factual_questions and method!="standard" else run_budget
                    wall_started = time.perf_counter()
                    try:
                        if method != "standard" and (planner.name, planner.model, planner.fingerprint) != (planner_name, model, provider_fingerprint):
                            raise ValueError("Experiment provider configuration changed; use the frozen settings")
                        if suite in ("controlled-five", "controlled-seven") and method != "standard":
                            if method in META_ARCHITECTURES:
                                from .meta_planner import MetaContextPlanner, MetaEngineeringModel
                                if planner.name == "deterministic": planner = MetaEngineeringModel()
                                planner = MetaContextPlanner(planner, limits=inference_limits, require_evidence=require_evidence)
                            else:
                                if planner.name == "deterministic": planner = EngineeringContextModel()
                                planner = ContextPlanner(planner, mode=context_strategy(method), limits=inference_limits,
                                                         require_evidence=require_evidence)
                            controller = ControlledController(q["question"], corpus, db, architecture=method, planner=planner,
                                budget=retrieval_budget, scoring=scoring, output=str(directory / "runs"))
                        else:
                            controller = Controller(q["question"], corpus, db, method=method, planner=planner,
                                budget=run_budget, scoring=scoring, output=str(directory / "runs"))
                        _capture_requests(planner, controller.directory / "planner_requests.jsonl")
                        result = controller.run()
                        if factual_questions and method!="standard":
                            task=next(f for f in factual_questions if f["query_id"]==q["id"])
                            result["retrieval_llm_usage"]=result["llm_usage"]
                            planner.budget=run_budget
                            before=asdict(planner.usage)
                            started=time.perf_counter()
                            response=planner.factual_response(task["question_id"],task["question"])
                            result["factual_responses"]=[response]
                            result["factual_measurement_seconds"]=time.perf_counter()-started
                            result["factual_measurement_calls"]=planner.usage.calls-before["calls"]
                            result["factual_measurement_input_tokens"]=planner.usage.input_tokens-before["input_tokens"]
                            result["factual_measurement_output_tokens"]=planner.usage.output_tokens-before["output_tokens"]
                            result["total_model_call_ceiling"]=run_budget.max_llm_calls
                            result["llm_usage"]=asdict(planner.usage)
                            atomic_json(controller.directory/"factual-responses.json",result["factual_responses"])
                    finally:
                        if hasattr(planner, "close"):
                            planner.close()
                    execution_wall_seconds = time.perf_counter()-wall_started
                    if suite in ("controlled-five", "controlled-seven"):
                        result["end_to_end_seconds"] = execution_wall_seconds
                        atomic_json(Path(result["directory"]) / "result.json", result)
                    row = {"query_id": q["id"], "method": method, "variant": variant, "trial": trial,
                        "seed":trial_seeds[trial] if trial_seeds is not None else None,
                        "action_budget": max_actions, "run_id": result["run_id"], "directory": result["directory"],
                        "actions": result["actions"], "inspected": len(result["inspected_ids"]),
                        "api_requests": result["api_usage"]["requests"],
                        "api_cost_usd": result["api_usage"]["reported_cost_usd"],
                        "llm_calls": result["llm_usage"]["calls"], "input_tokens": result["llm_usage"]["input_tokens"],
                        "output_tokens": result["llm_usage"]["output_tokens"],
                        "estimated_input_tokens": result["llm_usage"]["estimated_input_tokens"],
                        "estimated_output_tokens": result["llm_usage"]["estimated_output_tokens"],
                        "llm_cost_usd": result["llm_usage"]["estimated_cost_usd"],
                        "seconds": result["seconds"], "api_seconds": result["api_usage"]["api_seconds"],
                        "llm_seconds": result["llm_usage"]["seconds"], "local_seconds": result["local_seconds"],
                        "wait_seconds": result["wait_seconds"],
                        "stop_reason": result["stop_reason"], "corpus": result["corpus"],
                        "planner": result["planner"], "model": result["model"],
                        "deterministic_smoke": planner_name == "deterministic" or provider_configuration.get('mocked_generation',False), "has_judgments": bool(labels_path)}
                    events = db.events(result["run_id"])
                    row["llm_failures"] = result["llm_usage"]["failures"]
                    row["fallbacks"] = sum(e["kind"] == "invalid_decision" for e in events)
                    row["valid_llm_run"] = (method == "standard" or
                        planner_name != "deterministic" and row["llm_calls"] > 0) and not row["llm_failures"] and not row["fallbacks"] and result["stop_reason"] != "retrieval_error"
                    if provider_configuration.get('mocked_generation'):row['valid_llm_run']=False
                    if factual_questions and method!="standard" and result["factual_responses"][0]["status"]!="valid":
                        row["valid_llm_run"]=False
                    if suite in ("controlled-five", "controlled-seven"):
                        row.update(graph_enabled=method.endswith("_graph"), context_management=context_strategy(method),
                            context_edits=result.get("context_edits", 0), rejected_context_edits=result.get("rejected_context_edits", 0),
                            context_snapshots=result.get("context_snapshots", 0), active_context_chars=result.get("active_context_chars", 0),
                            evidence_record_count=len(result.get("evidence_records", [])),
                            factual_response_status=(result.get("factual_responses") or [{}])[0].get("status"),
                            factual_measurement_calls=result.get("factual_measurement_calls",0),
                            factual_measurement_seconds=result.get("factual_measurement_seconds",0),
                            factual_measurement_input_tokens=result.get("factual_measurement_input_tokens",0),
                            factual_measurement_output_tokens=result.get("factual_measurement_output_tokens",0),
                            context_adapter_local_seconds=result.get("context_adapter_local_seconds", 0), end_to_end_seconds=execution_wall_seconds)
                        row.update(efficiency_metrics(result, [e for e in events if e["kind"] == "context_model_call"]))
                        row.update({k: result.get(k) for k in ("active_context_tokens", "max_active_context_tokens", "max_full_input_tokens")})
                        overhead = [e for e in events if e["kind"] == "context_model_call" and e.get("operation_kind") in ("edit", "recovery")]
                        row.update(context_management_seconds=sum(e.get("seconds", 0) for e in overhead),
                            context_management_input_tokens=sum(e.get("usage_delta", {}).get("input_tokens", 0) for e in overhead),
                            context_management_output_tokens=sum(e.get("usage_delta", {}).get("output_tokens", 0) for e in overhead))
                    if suite == "controlled-seven":
                        row.update({key: result.get(key) for key in ("meta_reference_revision", "meta_synchronization_seconds",
                            "meta_reference_syncs", "meta_reference_meaningful_syncs", "meta_recovery_calls", "meta_sandbox",
                            "meta_common_limit_rollbacks", "meta_reference_rollbacks")})
                    if labels_path:
                        row.update(metrics([p["paper"]["id"] for p in result["ranked"]], judgments[q["id"]],
                            k=run_budget.output_k, scope=scope, inspected=result["inspected_ids"], actions=result["actions"]))
                        if suite in ("controlled-five", "controlled-seven"):
                            at10 = metrics([p["paper"]["id"] for p in result["ranked"]], judgments[q["id"]], k=10, scope=scope)
                            row.update(precision_at_10=at10["precision_at_k"], ndcg_at_10=at10["ndcg_at_k"], recall_at_10=at10["recall_at_k"])
                            row.update(retrieval_diagnostics(result["inspected_ids"], judgments[q["id"]], result["actions"]))
                        for point in result["trace"]:
                            curve_rows.append({"query_id": q["id"], "method": method, "variant": variant,
                                "trial": trial, "action_budget": max_actions, **point,
                                **metrics([], judgments[q["id"]], k=run_budget.output_k, scope=scope,
                                          inspected=point["inspected_ids"], actions=point["actions"])})
                    rows.append(row)
                    # Persist every completed run so interruption does not lose a sweep.
                    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
                    if curve_rows:
                        pd.DataFrame(curve_rows).drop(columns=["inspected_ids", "graph"]).to_csv(directory / "curves.csv", index=False)
    manifest.update(status="complete", completed_runs=len(rows), completed_at=utcnow())
    atomic_json(directory / "experiment.json", manifest)
    return rows


def build_pool(experiment_directory: str, output: str) -> int:
    """Blind, deduplicated assessment pool without assigning any relevance labels."""
    directory = Path(experiment_directory)
    table = pd.read_csv(directory / "metrics.csv")
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    questions = {q["id"]: q["question"] for q in experiment["questions"]}
    records = {}
    for row in table.to_dict("records"):
        state = json.loads((Path(row["directory"]) / "state.json").read_text())
        # Include all inspected papers to enable drift and inspection-efficiency analysis.
        for paper in state["papers"]:
            key = row["query_id"], paper["id"]
            records[key] = {"query_id": row["query_id"], "question": questions[row["query_id"]],
                            "paper_id": paper["id"], "title": paper["title"],
                            "abstract": paper["abstract"], "grade": "", "assessor": ""}
    frame = pd.DataFrame(list(records.values()))
    if not frame.empty:
        frame = frame.sample(frac=1, random_state=42)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return len(frame)


def score_experiment(experiment_directory: str, labels_path: str) -> int:
    """Attach independent judgments to saved runs; no retrieval or LLM calls."""
    directory = Path(experiment_directory)
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    judgments, scope = load_judgments(labels_path)
    ids = set(experiment["corpus_manifest"]["ids"])
    rows = pd.read_csv(directory / "metrics.csv").to_dict("records")
    from .experiment_integrity import validate_schedule
    validate_schedule(experiment, rows)
    for query_id in {r["query_id"] for r in rows}:
        labels = judgments.get(query_id, {})
        if not labels or set(labels) - ids:
            raise ValueError("Every query needs judgments restricted to the frozen corpus")
        if scope == "exhaustive" and set(labels) != ids:
            raise ValueError("Exhaustive judgments must cover all candidate papers")
    curves = []
    for row in rows:
        result = json.loads((Path(row["directory"]) / "result.json").read_text(encoding="utf-8"))
        labels = judgments[row["query_id"]]
        row.update(metrics([p["paper"]["id"] for p in result["ranked"]], labels,
            k=result["budget"]["output_k"], scope=scope, inspected=result["inspected_ids"], actions=result["actions"]))
        row["has_judgments"] = True
        if experiment.get("suite") in ("controlled-five", "controlled-seven"):
            from .research_metrics import retrieval_diagnostics
            at10 = metrics([p["paper"]["id"] for p in result["ranked"]], labels, k=10, scope=scope)
            row.update(precision_at_10=at10["precision_at_k"], ndcg_at_10=at10["ndcg_at_k"], recall_at_10=at10["recall_at_k"])
            row.update(retrieval_diagnostics(result["inspected_ids"], labels, result["actions"]))
        for point in result["trace"]:
            curves.append({"query_id": row["query_id"], "method": row["method"], "variant": row["variant"],
                "trial": row["trial"], "action_budget": row["action_budget"], **point,
                **metrics([], labels, k=result["budget"]["output_k"], scope=scope,
                          inspected=point["inspected_ids"], actions=point["actions"])})
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    if curves:
        pd.DataFrame(curves).drop(columns=["inspected_ids", "graph"]).to_csv(directory / "curves.csv", index=False)
    experiment["judgments"] = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    experiment["has_relevance_judgments"] = True
    experiment["scored_at"] = utcnow()
    atomic_json(directory / "experiment.json", experiment)
    return len(rows)


def report(experiment_directory: str, *, final: bool = False) -> list[str]:
    """Generate tables/figures only from actual recorded, labeled runs."""
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/openalex-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(experiment_directory)
    frame = pd.read_csv(directory / "metrics.csv")
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    if "recall_at_k" not in frame or not frame["has_judgments"].all():
        raise ValueError("Performance figures require recorded relevance judgments")
    from .experiment_integrity import validate_schedule
    validate_schedule(experiment, frame.to_dict("records"))
    from .controlled import ARCHITECTURES, ALL_ARCHITECTURES
    expected_methods = ALL_ARCHITECTURES if experiment.get("suite") == "controlled-seven" else ARCHITECTURES if experiment.get("suite") == "controlled-five" else METHODS
    if final and (experiment.get("deterministic_smoke") or experiment.get("trials", 1) < 3 or
                  not 10 <= len(experiment["questions"]) <= 20 or set(experiment["methods"]) != set(expected_methods)):
        raise ValueError("Final comparisons require 10–20 independently judged questions, every method in the selected suite, and at least three hosted trials")
    if final and experiment.get("suite") in ("controlled-five", "controlled-seven") and any(q.get("split") != "heldout" for q in experiment["questions"]):
        raise ValueError("Final controlled comparisons require explicitly held-out evaluation questions, separate from development")
    # Separate within-question trial variability from variation across questions.
    repeatability = frame.groupby(["query_id", "method", "variant", "action_budget"])[
        ["recall_at_k", "precision_lower_bound", "seconds", "inspected", "actions"]].agg(["mean", "std", "count"])
    repeatability.to_csv(directory / "repeatability.csv")
    diagnostic_columns = [c for c in ("llm_failures", "fallbacks", "valid_llm_run") if c in frame]
    if diagnostic_columns:
        frame.groupby(["method", "variant", "action_budget"])[diagnostic_columns].agg(["sum", "count"]).to_csv(directory / "failures.csv")
    else:
        pd.DataFrame([{"status": "Legacy experiment: failure counters unavailable"}]).to_csv(directory / "failures.csv", index=False)
    summary = frame.groupby(["method", "variant", "action_budget"])[
        ["precision_lower_bound", "judged_precision", "recall_at_k", "ndcg_at_k", "seconds", "inspected", "actions"]].agg(["mean", "std", "count"])
    summary.to_csv(directory / "summary.csv")
    paths = [str(directory / "summary.csv"), str(directory / "repeatability.csv"), str(directory / "failures.csv")]
    if experiment.get("suite") in ("controlled-five", "controlled-seven"):
        from .research_metrics import paired_comparisons
        if experiment.get("suite") == "controlled-seven":
            from .factorial import paired_comparisons_2x3 as paired_comparisons
        contrasts = paired_comparisons(frame.to_dict("records"), "recall_at_k")
        atomic_json(directory / "paired-comparisons.json", contrasts)
        paths.append(str(directory / "paired-comparisons.json"))
    recall_population = "candidate corpus" if (frame["judgment_scope"] == "exhaustive").all() else "judged pool"
    for metric, ylabel in [("recall_at_k", "Recall@K relative to judged pool"),
                           ("precision_lower_bound", "Precision@K lower bound")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for (method, variant), group in frame.groupby(["method", "variant"]):
            values = group.groupby("action_budget")[metric].mean()
            ax.plot(values.index, values.values, marker="o", label=f"{method}/{variant}")
        ax.set(xlabel="Allowed retrieval actions", ylabel=f"Recall@K relative to {recall_population}" if metric == "recall_at_k" else ylabel)
        if frame["deterministic_smoke"].any():
            ax.set_title("ENGINEERING SMOKE TEST — deterministic planner")
        ax.legend(fontsize="small")
        fig.tight_layout()
        path = directory / f"{metric}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    curves_path = directory / "curves.csv"
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        fig, ax = plt.subplots(figsize=(7, 4))
        for (method, variant, action_budget), group in curves.groupby(["method", "variant", "action_budget"]):
            values = group.groupby("inspected")["inspected_known_recall"].mean()
            ax.plot(values.index, values.values, marker=".", label=f"{method}/{variant}/budget={action_budget}")
        ax.set(xlabel="Papers inspected", ylabel="Known relevant papers discovered / judged relevant")
        ax.legend(fontsize="small")
        fig.tight_layout()
        path = directory / "recall_vs_inspected.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths
