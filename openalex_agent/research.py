"""Local research-run adapter. All decisions and retrieval remain in Controller.

The worker owns its SQLite/HTTP clients. Streamlit only reads detached snapshots
of committed events; the worker never calls a UI API.
"""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import threading
import time
import math
from urllib.parse import urlsplit
import uuid

from .config import Settings
from .controller import Budget
from .context_planner import EngineeringContextModel, InferenceLimits
from .corpus import LocalCorpus, atomic_json
from .freeze_pilot import verify
from .graph import ScholarlyGraph
from .llm import DeterministicPlanner
from .models import canonical_id, utcnow
from .selection import Scoring
from .storage import Database
from .live_research import (LiveController, LiveControlledController, LiveContextPlanner,
                            LiveHostedPlanner, finish_research, VERSION as LIVE_POLICY_VERSION)
from .live_safety import diagnostic, SpendingLedger
from .live_profiles import priced_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def redact(value, secrets: tuple[str, ...]):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {redact(k, secrets): redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    return value


@dataclass(frozen=True)
class ResearchConfig:
    question: str
    mode: str = "live"
    planner: str = "hosted"
    graph_memory: bool = True
    corpus: str | None = None
    budget: Budget = field(default_factory=lambda: Budget(max_actions=5, max_rounds=5,
        max_inspected=30, max_api_requests=10, max_llm_calls=4, page_size=5))
    context_management: str = "conventional"
    controlled: bool = False
    inference_limits: InferenceLimits = field(default_factory=InferenceLimits)
    # Existing programmatic callers retain retrieval-only behavior. The UI
    # explicitly enables synthesis and reserves a call inside its chosen cap.
    final_synthesis: bool = False
    quality_aware: bool = False
    synthesis_revisions: int = 0
    max_seconds: float = 600.0

    def __post_init__(self):
        if not isinstance(self.question, str) or not 1 <= len(self.question.strip()) <= 1000:
            raise ValueError("Enter a research question of 1–1000 characters")
        if self.mode not in ("live", "snapshot") or self.planner not in ("hosted", "deterministic"):
            raise ValueError("Choose an available retrieval mode and planner")
        if type(self.graph_memory) is not bool or not isinstance(self.budget, Budget):
            raise ValueError("Invalid graph or retrieval budget configuration")
        if self.mode == "snapshot" and not self.corpus:
            raise ValueError("Select a frozen corpus directory")
        if self.mode == "live" and self.corpus:
            raise ValueError("Live retrieval cannot also use a fixed corpus")
        if self.context_management not in ("conventional", "clm") or type(self.controlled) is not bool:
            raise ValueError("Choose a supported context-management policy")
        if self.context_management == "clm" and not self.controlled:
            raise ValueError("CLM requires the matched six-tool research policy")
        if not isinstance(self.inference_limits, InferenceLimits):
            raise ValueError("Invalid inference resource limits")
        if type(self.final_synthesis) is not bool:
            raise ValueError("Invalid synthesis policy")
        if type(self.quality_aware) is not bool or type(self.synthesis_revisions) is not int or self.synthesis_revisions not in (0, 1):
            raise ValueError("Invalid quality/revision policy")
        if type(self.max_seconds) not in (int, float) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("Research time limit must be finite and positive")
        if self.final_synthesis and self.budget.max_llm_calls < 1 + self.synthesis_revisions:
            raise ValueError("The call ceiling must leave room for synthesis and its revision")


class ResearchJob:
    """One background run, with a credential-free public state."""

    def __init__(self, directory: Path, config: ResearchConfig):
        self.directory = directory
        self.config = config
        self._lock = threading.Lock()
        self._thread = None
        self._state = {"status": "starting", "mode": config.mode, "events": [],
                       "run_id": None, "result": None, "error": None,
                       "directory": str(directory), "comparison": {"performed": False}}

    def snapshot(self) -> dict:
        with self._lock:
            return deepcopy(self._state)

    def update(self, **values):
        with self._lock:
            self._state.update(values)

    def observe(self, event: dict):
        with self._lock:
            self._state["events"].append(event)
            if event.get("run_id"):
                self._state["run_id"] = event["run_id"]
        with (self.directory / "activity.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def join(self, timeout: float | None = None):
        if self._thread:
            self._thread.join(timeout)


class ObservedDatabase(Database):
    """Observe rows after the existing database has committed them."""

    def __init__(self, path: str, sink):
        super().__init__(path)
        self.sink = sink

    def event(self, run_id: str, kind: str, payload: dict) -> None:
        super().event(run_id, kind, payload)
        # This connection is used exclusively by the worker. Read the committed
        # row so timestamp/order/payload are actual stored evidence.
        seq, name, timestamp, data = self.connection.execute(
            "SELECT seq,kind,timestamp,payload FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        self.sink({"kind": name, "timestamp": timestamp, **json.loads(data), "database_seq": seq})


def pilot_reference(root: Path) -> tuple[set[str] | None, dict]:
    """Use only the verified, byte-preserved corpus as the comparison baseline."""
    receipt_path = root / "benchmarks/frozen_pilot_v1.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["snapshot"] = str(root / receipt["snapshot"])
        verify(receipt)
        corpus = LocalCorpus(Path(receipt["snapshot"]) / "evidence/graph-ablation-pilot-2026-11-02/corpus")
        return set(corpus.by_id), {"performed": False, "baseline_available": True, "baseline_papers": len(corpus.by_id),
            "corpus_sha256": corpus.fingerprint, "snapshot_manifest_sha256": receipt["manifest_sha256"]}
    except (OSError, ValueError, KeyError, TypeError):
        return None, {"performed": False, "reason": "Verified frozen pilot corpus is unavailable"}


def compare_ids(ids, baseline: set[str] | None, record: dict) -> dict:
    if baseline is None:
        return dict(record, performed=False)
    actual = {canonical_id(identifier) for identifier in ids}
    outside = sorted(actual - baseline)
    return {**record, "performed": True, "observed_works": len(actual),
            "outside_pilot_count": len(outside), "outside_pilot_ids": outside,
            "meaning": "Absent from this pilot snapshot; not a claim of novel scientific discovery"}


def start_research(config: ResearchConfig, settings: Settings, *, root: Path = PROJECT_ROOT,
                   openalex_transport=None, planner_transport=None, spending_ledger=None, trajectory_id=None) -> ResearchJob:
    """Start one isolated run. Transports are injectable for offline integration tests."""
    settings = priced_settings(settings)
    secrets = tuple(v for v in (settings.api_key, settings.llm_api_key, settings.email) if v)
    if any(secret in config.question for secret in secrets):
        raise ValueError("Remove configured credentials or contact details from the research question")
    settings.validate_retrieval()
    if config.planner == "hosted":
        missing = [name for name, value in (("LLM_BASE_URL", settings.llm_base_url),
            ("LLM_API_KEY", settings.llm_api_key), ("LLM_MODEL", settings.llm_model)) if not value.strip()]
        if missing:
            raise ValueError("Missing hosted configuration: " + ", ".join(missing) + ". Set these in the project's .env file.")
        if config.inference_limits.max_llm_cost_usd is not None and not (
                settings.llm_input_price > 0 and settings.llm_output_price > 0):
            raise ValueError("Dollar limits require LLM_INPUT_USD_PER_MILLION and LLM_OUTPUT_USD_PER_MILLION above zero. "
                             "Zero means unconfigured pricing, not free inference.")
    if config.mode == "live" and settings.max_requests == 0:
        raise ValueError("OPENALEX_MAX_REQUESTS is zero; live retrieval is unavailable")
    root = Path(root).resolve()
    base = root / "data/ui-runs"
    if base.resolve() != base:
        raise ValueError("UI output directory must not redirect into another directory")
    directory = base / str(uuid.uuid4())
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "activity.jsonl").touch()
    # A fresh per-run database cannot read a benchmark's cache or write to its files.
    effective = replace(settings, database=str(directory / "metadata.sqlite"),
                        max_requests=min(settings.max_requests, config.budget.max_api_requests))
    if config.quality_aware:
        effective = replace(effective, max_retries=0)
    job = ResearchJob(directory, config)
    job.spending_ledger = spending_ledger
    job.trajectory_id = trajectory_id or directory.name
    from .git_provenance import git_provenance
    request = {"schema": 1, "created_at": utcnow(), "git": git_provenance(root), "configuration": asdict(config),
               "live_policy_version": LIVE_POLICY_VERSION,
               "effective_api_request_limit": effective.max_requests,
               "configured_api_cost_ceiling_usd": effective.max_cost_usd,
               "planner_model": effective.llm_model if config.planner == "hosted" else "none"}
    request["pricing"] = {"input_upper_bound_per_million": settings.llm_input_price,
        "output_per_million": settings.llm_output_price, "discount_assumed": False}
    atomic_json(directory / "research.json", redact(request, secrets))
    job._thread = threading.Thread(target=_execute, args=(job, effective, root, secrets,
        openalex_transport, planner_transport), name=f"research-{directory.name}", daemon=True)
    job._thread.start()
    return job


def _execute(job, settings, root, secrets, openalex_transport, planner_transport):
    config = job.config
    try:
        # Historical pilot comparison is optional metadata, never a research
        # dependency. Thorough standalone runs do not open V1/V3 corpora.
        baseline, comparison = ((None, {"performed": False, "reason": "Independent standalone research"})
            if config.quality_aware else pilot_reference(root))
        job.update(status="running", comparison=comparison)
        with ExitStack() as stack:
            db = stack.enter_context(ObservedDatabase(settings.database,
                lambda event: job.observe(redact(event, secrets))))
            if config.mode == "snapshot":
                path = Path(config.corpus)
                retriever = LocalCorpus(path if path.is_absolute() else root / path)
            else:
                from .openalex import OpenAlexClient
                retriever = stack.enter_context(OpenAlexClient(db, settings, transport=openalex_transport))
            planner = (stack.enter_context(LiveHostedPlanner(settings, transport=planner_transport))
                       if config.planner == "hosted" else DeterministicPlanner())
            base_planner = planner
            if config.controlled:
                if planner.name == "deterministic":
                    planner = EngineeringContextModel()
                planner = LiveContextPlanner(planner, mode=config.context_management, limits=config.inference_limits)
                architecture = ("clm" if config.context_management == "clm" else "llm") + ("_graph" if config.graph_memory else "")
                controller = LiveControlledController(config.question, retriever, db, architecture=architecture,
                    planner=planner, budget=config.budget, scoring=Scoring(), output=str(job.directory / "runs"))
            else:
                controller = LiveController(config.question, retriever, db,
                    method="graph" if config.graph_memory else "iterative", planner=planner,
                    budget=config.budget, scoring=Scoring(), output=str(job.directory / "runs"))
            controller.final_synthesis = config.final_synthesis
            controller._redactor = lambda value: redact(value, secrets)
            controller.quality_aware = config.quality_aware
            controller.synthesis_revisions = config.synthesis_revisions
            controller.max_seconds = config.max_seconds
            controller.research_started = time.monotonic()
            started = time.perf_counter()
            controller.synthesis_reserve = 1 + config.synthesis_revisions if config.final_synthesis else 0
            if isinstance(base_planner, LiveHostedPlanner):
                base_planner.configure_live(controller, config.inference_limits, config.final_synthesis)
                if job.spending_ledger is None and config.inference_limits.max_llm_cost_usd is not None:
                    job.spending_ledger = SpendingLedger(job.directory / "spending.sqlite",
                        ceiling=config.inference_limits.max_llm_cost_usd, identity=job.directory.name)
                base_planner.spending_ledger = job.spending_ledger
                base_planner.trajectory_id = job.trajectory_id
                if config.quality_aware:
                    base_planner.analyze_question()
            job.update(run_id=controller.run_id)
            result = controller.run()
            job.update(phase="synthesis")
            try:
                synthesis = finish_research(controller, base_planner, enabled=config.final_synthesis)
            except Exception as exc:
                # Keep committed retrieval evidence even if terminal formatting
                # or an unexpected provider failure interrupts the answer stage.
                controller._log("live_synthesis_error", {"type": type(exc).__name__})
                synthesis = {"status": "unavailable", "reason": "unexpected_synthesis_error", "answer": None,
                    "sources": [], "metadata_admitted": len(controller.papers), "stop_reason": controller.stop_reason}
            if synthesis is not None:
                result["synthesis"] = redact(synthesis, secrets)
                atomic_json(controller.directory / "synthesis.json", result["synthesis"])
            result["llm_usage"] = asdict(planner.usage)
            result["live_policy_version"] = LIVE_POLICY_VERSION
            if config.quality_aware:
                from .live_quality import coverage_view
                result["research_plan"] = controller.research_plan
                result["research_coverage"] = coverage_view(controller)
            if job.spending_ledger:
                result["spending"] = job.spending_ledger.summary()
            result["mechanisms"] = mechanism_metrics(controller, base_planner, db.events(controller.run_id))
            result["effective_api_request_limit"] = settings.max_requests
            result["retrieval_seconds"] = result["seconds"]
            result["seconds"] = time.perf_counter() - started
            result["local_seconds"] = max(0.0, result["seconds"] - retriever.usage.api_seconds -
                retriever.usage.wait_seconds - planner.usage.seconds)
            result["live_accounting"] = {"unique_metadata_admitted": len(controller.papers),
                "unique_records_returned": len(controller.retrieved_ids),
                "returned_record_observations": sum(p.get("returned_papers", 0) for p in controller.trace),
                "sources_sent_to_model": len(getattr(base_planner, "visible_ids", set())),
                "abstracts_sent_to_model": len(getattr(base_planner, "excerpt_ids", set()))}
            result["binding_limits"] = binding_limits(result)
            controller._checkpoint(result["seconds"], finished=True)
            atomic_json(controller.directory / "result.json", redact(result, secrets))
            from .live_research import answer_markdown
            (job.directory / "answer.md").write_text(redact(answer_markdown(result), secrets), encoding="utf-8")
            comparison = compare_ids(result["retrieved_ids"], baseline, comparison)
            status = "failed" if result["stop_reason"] == "retrieval_error" else "completed"
            api_errors = [e for e in db.events(controller.run_id) if e["kind"] == "request_error"]
            error = ({"type": "retrieval_error", "message": retrieval_error_message(api_errors)}
                     if status == "failed" else None)
            job.update(status=status, result=redact(result, secrets), error=error, comparison=comparison)
    except Exception as exc:
        details = diagnostic(exc, secrets)
        atomic_json(job.directory / "error.json", details)
        message = details["message"] if isinstance(exc, (ValueError, FileNotFoundError)) else (
            f"Research failed ({type(exc).__name__}). A credential-redacted traceback was saved to error.json.")
        job.update(status="failed", error={"type": type(exc).__name__, "message": message,
                                          "diagnostic": "error.json"})
    finally:
        state = job.snapshot()
        atomic_json(job.directory / "summary.json", {k: v for k, v in state.items() if k != "events"})


def retrieval_error_message(events):
    status = next((e.get("status") for e in reversed(events) if e.get("status")), None)
    if status in (401, 403):
        return "Retrieval failed: OpenAlex rejected access. Check OPENALEX_API_KEY in this project's .env file and the account's API access."
    if status == 429:
        return "OpenAlex rate limit reached. The partial research artifacts are saved; try again after the service resets."
    if status:
        return f"Retrieval failed: OpenAlex returned HTTP {status}. Partial research artifacts are saved."
    return "Retrieval failed: unable to connect to OpenAlex. Check network access, DNS/proxy settings and OPENALEX_REQUEST_TIMEOUT. Partial artifacts are saved."


def mechanism_metrics(controller, base, events):
    edits = [e for e in events if e["kind"] == "context_edit" and e.get("source") == "model"]
    recoveries = [e for e in events if e["kind"] == "context_recovery"]
    inputs = getattr(base, "wire_payloads", [])
    later_edits = 0
    for edit in edits:
        call = (edit.get("usage") or {}).get("call_index", 0)
        after = edit.get("after", "")
        if edit.get("committed") and edit.get("applied") and after and any(
                p["call"] > call and after in p["input"].get("active_context", "") for p in inputs):
            later_edits += 1
    later_recoveries = 0
    for recovery in recoveries:
        request = recovery.get("request") or {}
        call = (recovery.get("usage") or {}).get("call_index", 0)
        if recovery.get("accepted") and any(p["call"] > call and
                p["input"].get("recovered_observation", {}).get("index") == request.get("index") and
                p["input"].get("recovered_observation", {}).get("start") == request.get("start") and
                bool(p["input"].get("recovered_observation", {}).get("text")) for p in inputs):
            later_recoveries += 1
    return {"graph_edges_stored": controller.graph.citation_graph().number_of_edges() if controller.graph else 0,
        "graph_unique_edges_sent": len(getattr(base, "graph_pairs", set())),
        "successful_followup_planning_calls": sum(p["stage"] == "planning" and p.get("successful", False) for p in inputs),
        "graph_edges_in_successful_planning": len({tuple(pair) for p in inputs
            if p["stage"] == "planning" and p.get("successful") for pair in p.get("graph_edge_pairs", [])}),
        "environment_context_updates": sum(e["kind"] == "context_observation" and e.get("source") != "model" for e in events),
        "clm_edits_proposed": len(edits), "clm_edits_accepted": sum(e.get("accepted") is True for e in edits),
        "clm_edits_committed": sum(bool(e.get("committed") and e.get("applied")) for e in edits),
        "committed_edits_in_later_requests": later_edits,
        "archive_observations": len(getattr(getattr(controller.planner, "store", None), "observations", [])),
        "recovery_operations": len(recoveries), "accepted_recoveries": sum(e.get("accepted") is True for e in recoveries),
        "recovered_evidence_in_later_requests": later_recoveries,
        "context_local_seconds": getattr(controller.planner, "context_local_seconds", 0),
        "interpretation": "Mechanism exposure and persistence; not evidence of improved research quality"}


def binding_limits(result: dict) -> list[str]:
    """All reached limits, in addition to the controller's first stop reason."""
    budget = dict(result.get("budget", {}))
    if "effective_api_request_limit" in result and "max_api_requests" in budget:
        budget["max_api_requests"] = min(budget["max_api_requests"], result["effective_api_request_limit"])
    counts = {"max_actions": result.get("actions", 0), "max_rounds": result.get("rounds", 0),
        "max_inspected": len(result.get("inspected_ids", [])),
        "max_api_requests": (result.get("api_usage") or {}).get("requests", 0),
        "max_llm_calls": (result.get("llm_usage") or {}).get("calls", 0)}
    if (result.get("retrieval") or {}).get("live_openalex_available") is False:
        counts.pop("max_api_requests")
    return [key for key, used in counts.items() if key in budget and used >= budget[key]]


def progress(state: dict) -> dict:
    """Summarize actual events, never infer successful HTTP access from mode alone."""
    events = state["events"]
    result = state.get("result") or {}
    observations = [e for e in events if e["kind"] == "observation"]
    last = observations[-1] if observations else {}
    request_events = [e for e in events if e["kind"] == "request"]
    request_errors = [e for e in events if e["kind"] == "request_error"]
    usage = result.get("api_usage") or (request_events[-1].get("usage", {}) if request_events else {})
    requests = usage.get("requests", 0)
    # Errors occurring after the last successful response still represent attempts.
    requests = max(requests, len(request_events) + len(request_errors))
    cached = usage.get("cache_hits", sum(e["kind"] == "cache_hit" for e in events))
    inspected = len(result["inspected_ids"]) if result else last.get("inspected", 0)
    retrieved = len(result["retrieved_ids"]) if result else last.get("retrieved", 0)
    if state["mode"] == "snapshot":
        evidence = "Fixed corpus; live OpenAlex retrieval is unavailable."
    elif state["status"] == "failed":
        evidence = "Live retrieval failed; any partial results are shown with the recorded errors."
    elif request_events:
        evidence = "Successful live OpenAlex HTTP responses are recorded."
    elif cached:
        evidence = "Only cache reuse is recorded; live HTTP retrieval has not been verified."
    else:
        evidence = "Live mode selected; no successful OpenAlex HTTP response is recorded yet."
    call_events = ([e for e in events if e["kind"] == "live_model_call"] or
                   [e for e in events if e["kind"] == "context_model_call"] or [e for e in events if e["kind"] == "planner_call"])
    usage_llm = result.get("llm_usage") or {}
    inputs = [e for e in events if e["kind"] == "live_model_input"]
    sources = {pid for e in inputs for pid in e.get("source_ids", [])}
    # Legacy saved runs contain the exact conventional planning context; count
    # those excerpts without pretending that every retrieved record was sent.
    if not inputs:
        sources = {p["id"] for e in events if e["kind"] == "context" for p in e.get("context", {}).get("top_papers", [])}
    source_count = len(sources)
    recorded_planner = result.get("planner") or next((e.get("planner") for e in events if e["kind"] == "start"), None)
    if recorded_planner == "deterministic":
        source_count = 0
    elif not inputs and any(e["kind"] == "context_model_call" for e in events):
        # Old controlled runs logged pre-selection context, not actual wire
        # excerpts. Do not mislabel an archived or pending source as delivered.
        source_count = None
    if "live_accounting" in result:
        source_count = result["live_accounting"]["sources_sent_to_model"]
    return {"http_attempts": requests, "successful_responses": len(request_events), "cache_hits": cached,
            "api_errors": max(usage.get("errors", 0), len(request_errors)),
            "fallbacks": sum(e["kind"] == "invalid_decision" for e in events),
            "inspected": inspected, "retrieved": retrieved, "graph": last.get("graph", {}),
            "returned_record_observations": sum(e.get("returned_papers", 0) for e in observations),
            "sources_sent_to_model": source_count,
            "actions": result.get("actions", last.get("actions", 0)),
            "rounds": result.get("rounds", last.get("round", 0)),
            "binding_limits": result.get("binding_limits", binding_limits(result)),
            "evidence": evidence, "stop_reason": result.get("stop_reason"),
            "planner_calls": usage_llm.get("calls", len(call_events)),
            "input_tokens": usage_llm.get("input_tokens", sum(e.get("usage_delta", {}).get("input_tokens", e.get("prompt_tokens", 0)) for e in call_events)),
            "output_tokens": usage_llm.get("output_tokens", sum(e.get("usage_delta", {}).get("output_tokens", e.get("completion_tokens", 0)) for e in call_events)),
            "context_edits": sum(e["kind"] == "context_edit" and e.get("accepted", False) for e in events)}


def paper_view(item: dict) -> dict:
    """Present only fields recorded in a ranked backend result, with safe links."""
    paper = item["paper"]
    authors = []
    for authorship in paper.get("authorships", []):
        author = authorship.get("author") or {}
        authors.append(author.get("name") or authorship.get("raw_name") or author.get("id") or "Unknown author")
    doi = paper.get("doi")
    doi_link = None
    if isinstance(doi, str):
        try:
            parsed = urlsplit(doi)
        except ValueError:
            parsed = urlsplit("")
        if (parsed.scheme in ("http", "https") and parsed.netloc.lower() == "doi.org" and
                parsed.path.startswith("/10.") and not parsed.query and not parsed.fragment and not any(c.isspace() for c in doi)):
            doi_link = "https://doi.org" + parsed.path
    return {"id": canonical_id(paper["id"]), "title": paper.get("title") or "Title unavailable",
            "authors": "; ".join(authors) or "Authors unavailable", "year": paper.get("year"),
            "venue": paper.get("venue"), "abstract": paper.get("abstract"), "citation_count": paper.get("citation_count"),
            "openalex_link": canonical_id(paper["id"]), "doi_link": doi_link,
            "score": item.get("algorithmic", {}).get("relevance"),
            "ranking_reason": item.get("algorithmic", {}).get("reason"),
            "discovery_path": item.get("algorithmic", {}).get("discovery_path", []),
            "provenance": paper.get("provenance"), "retrieved_at": paper.get("retrieved_at"),
            "limitations": item.get("limitations", [])}


def corpus_options(root: Path = PROJECT_ROOT) -> list[str]:
    """Discover ordinary snapshots, plus the one canonical frozen pilot copy."""
    paths = []
    for parent in (root / "data/corpus",):
        if parent.exists():
            paths += [str(p.parent.relative_to(root)) for p in parent.rglob("manifest.json")
                      if (p.parent / "papers.jsonl").is_file()]
    pilot = root / "data/frozen/three-question-pilot-v1/evidence/graph-ablation-pilot-2026-11-02/corpus"
    if (pilot / "manifest.json").is_file():
        paths.append(str(pilot.relative_to(root)))
    return sorted(set(paths))


def saved_runs(root: Path = PROJECT_ROOT) -> list[str]:
    base = Path(root).resolve() / "data/ui-runs"
    if base.resolve() != base:
        return []
    paths = sorted(base.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.parent.name for p in paths if not p.parent.is_symlink()][:100]


def load_research(identifier: str, *, root: Path = PROJECT_ROOT) -> ResearchJob:
    """Reopen saved UI evidence without a database connection or new API calls."""
    if str(uuid.UUID(identifier)) != identifier:
        raise ValueError("Choose an existing UI run")
    base = Path(root).resolve() / "data/ui-runs"
    directory = base / identifier
    if base.resolve() != base or directory.resolve() != directory:
        raise ValueError("UI run directory cannot redirect to another location")
    record = json.loads((directory / "research.json").read_text(encoding="utf-8"))
    configuration = dict(record["configuration"])
    configuration["budget"] = Budget(**configuration["budget"])
    if "inference_limits" in configuration:
        configuration["inference_limits"] = InferenceLimits(**configuration["inference_limits"])
    job = ResearchJob(directory, ResearchConfig(**configuration))
    state = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if state["status"] not in ("completed", "failed"):
        raise ValueError("Only ended UI runs can be reopened")
    if state.get("run_id"):
        uuid.UUID(state["run_id"])
    state["directory"] = str(directory)
    state["events"] = [json.loads(line) for line in (directory / "activity.jsonl").read_text(encoding="utf-8").splitlines()]
    job.update(**state)
    return job


def graph_view(directory: str, seed: str, *, max_nodes: int = 30, hops: int = 1,
               relations: list[str] | None = None) -> dict:
    """Read-only bounded traversal of the existing validated graph snapshot."""
    graph = ScholarlyGraph.load(Path(directory) / "graph.json")
    return graph.neighborhood([seed], hops=hops, max_nodes=max_nodes, relations=relations)
