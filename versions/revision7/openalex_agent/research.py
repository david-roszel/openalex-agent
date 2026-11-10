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
from urllib.parse import urlsplit
import uuid

from .config import Settings
from .controller import Budget, Controller
from .context_planner import ContextPlanner, EngineeringContextModel, InferenceLimits
from .controlled import ControlledController
from .corpus import LocalCorpus, atomic_json
from .freeze_pilot import verify
from .graph import ScholarlyGraph
from .llm import DeterministicPlanner, HostedJSONPlanner
from .models import canonical_id, utcnow
from .selection import Scoring
from .storage import Database

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
                   openalex_transport=None, planner_transport=None) -> ResearchJob:
    """Start one isolated run. Transports are injectable for offline integration tests."""
    secrets = tuple(v for v in (settings.api_key, settings.llm_api_key, settings.email) if v)
    if any(secret in config.question for secret in secrets):
        raise ValueError("Remove configured credentials or contact details from the research question")
    settings.validate_retrieval()
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
    job = ResearchJob(directory, config)
    from .git_provenance import git_provenance
    request = {"schema": 1, "created_at": utcnow(), "git": git_provenance(root), "configuration": asdict(config),
               "effective_api_request_limit": effective.max_requests,
               "configured_api_cost_ceiling_usd": effective.max_cost_usd,
               "planner_model": effective.llm_model if config.planner == "hosted" else "none"}
    atomic_json(directory / "research.json", redact(request, secrets))
    job._thread = threading.Thread(target=_execute, args=(job, effective, root, secrets,
        openalex_transport, planner_transport), name=f"research-{directory.name}", daemon=True)
    job._thread.start()
    return job


def _execute(job, settings, root, secrets, openalex_transport, planner_transport):
    config = job.config
    try:
        baseline, comparison = pilot_reference(root)
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
            planner = (stack.enter_context(HostedJSONPlanner(settings, transport=planner_transport))
                       if config.planner == "hosted" else DeterministicPlanner())
            if config.controlled:
                if planner.name == "deterministic":
                    planner = EngineeringContextModel()
                planner = ContextPlanner(planner, mode=config.context_management, limits=config.inference_limits)
                architecture = ("clm" if config.context_management == "clm" else "llm") + ("_graph" if config.graph_memory else "")
                controller = ControlledController(config.question, retriever, db, architecture=architecture,
                    planner=planner, budget=config.budget, scoring=Scoring(), output=str(job.directory / "runs"))
            else:
                controller = Controller(config.question, retriever, db,
                    method="graph" if config.graph_memory else "iterative", planner=planner,
                    budget=config.budget, scoring=Scoring(), output=str(job.directory / "runs"))
            job.update(run_id=controller.run_id)
            result = controller.run()
            comparison = compare_ids(result["retrieved_ids"], baseline, comparison)
            status = "failed" if result["stop_reason"] == "retrieval_error" else "completed"
            error = ({"type": "retrieval_error", "message": "Retrieval failed. Inspect the recorded API and controller errors; partial results may be available."}
                     if status == "failed" else None)
            job.update(status=status, result=redact(result, secrets), error=error, comparison=comparison)
    except Exception as exc:
        # Never forward exception text: transports/configuration can contain keys.
        job.update(status="failed", error={"type": type(exc).__name__,
            "message": "Run could not complete. Check the selected corpus, server-side configuration and recorded events."})
    finally:
        state = job.snapshot()
        atomic_json(job.directory / "summary.json", {k: v for k, v in state.items() if k != "events"})


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
    call_events = [e for e in events if e["kind"] == "context_model_call"] or [e for e in events if e["kind"] == "planner_call"]
    usage_llm = result.get("llm_usage") or {}
    return {"http_attempts": requests, "successful_responses": len(request_events), "cache_hits": cached,
            "api_errors": max(usage.get("errors", 0), len(request_errors)),
            "fallbacks": sum(e["kind"] == "invalid_decision" for e in events),
            "inspected": inspected, "retrieved": retrieved, "graph": last.get("graph", {}),
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
