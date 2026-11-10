"""Single-worker, source-frozen LiveDemo evaluation; no external campaign imports.

Uses the SAME start_research adapter as UI/CLI. Only supervision, admission,
accounting and descriptive reporting live here. No automatic campaign replay.
"""
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from urllib.parse import urlsplit

import httpx

from .corpus import atomic_json
from .live_benchmark import profile, environment, score, matches, now
from .live_profiles import load_live_settings, thorough_config
from .live_safety import SpendingLedger, diagnostic
from .models import Paper
from .openalex import OpenAlexClient, BudgetExceeded, RetrievalError
from .research import PROJECT_ROOT, start_research
from .storage import Database

VARIANTS = ("standard", "conventional", "graph", "clm", "combined")
DIAGNOSTIC = {
    "id": "D01", "domain": "NLP diagnostic", "previously_tested": True,
    "question": "How do ReAct and Reflexion differ in how they enable large language model agents to reason, take actions, and learn from feedback? Compare their original research papers.",
    "baseline_query": "ReAct Reflexion language agents reasoning acting feedback",
    "required_components": ["original ReAct publication", "original Reflexion publication", "reasoning and action mechanisms", "feedback and learning comparison", "findings and limitations"],
    "targets": [
        {"title": "ReAct: Synergizing Reasoning and Acting in Language Models", "doi": "10.48550/arxiv.2210.03629", "year": 2022, "openalex_id": None, "identity_source": "https://arxiv.org/abs/2210.03629"},
        {"title": "Reflexion: Language Agents with Verbal Reinforcement Learning", "doi": "10.48550/arxiv.2303.11366", "year": 2023, "openalex_id": None, "identity_source": "https://arxiv.org/abs/2303.11366"}],
}
AUTHORIZATION = "I want it to just run. I give approval, add in the prompt it is approved and can spend up to $50 in api if needed to get it done. It is approved."
API_REQUEST_BOUND = Decimal("0.001")
PER_TRAJECTORY = Decimal("1.50")
MODEL_TRAJECTORY = Decimal("1.46")  # .04 retained for 40 metadata HTTP requests


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(output):
    output = Path(output).resolve()
    config = json.loads((output / "EXPERIMENT_CONFIGURATION.json").read_text())
    questions = json.loads((output / "QUESTION_SET.json").read_text())["questions"]
    return output, config, questions


def configuration(question, variant):
    config = thorough_config(question["baseline_query"] if variant == "standard" else question["question"],
        variant=variant, max_seconds=300)
    return replace(config, inference_limits=replace(config.inference_limits,
        max_llm_cost_usd=float(MODEL_TRAJECTORY) if variant != "standard" else None))


def prepare(output, settings, *, root=PROJECT_ROOT):
    """Offline only; caller supplies the explicit approval at execution time."""
    output, root = Path(output).resolve(), Path(root).resolve()
    if (output / "EXPERIMENT_CONFIGURATION.json").exists():
        raise ValueError("Frozen configuration exists; never overwrite a campaign")
    output.mkdir(parents=True, exist_ok=True)
    original = root / "benchmarks/live_submission_v1/questions.json"
    primary = json.loads(original.read_text())["questions"]
    lookup = {q["id"]: q for q in primary}
    # Fixed before results: breadth, independent retrieval, methodological
    # comparison, multi-hop experiment/foundation connection, then other domains.
    questions = [lookup[k] for k in ("Q01",)] + [DIAGNOSTIC] + [lookup[k] for k in
        ("Q03", "Q07", "Q02", "Q05", "Q06", "Q08", "Q04")]
    snapshot = output / "source_snapshot"
    for path in sorted((root / "openalex_agent").rglob("*")):
        if path.is_file() and not path.is_symlink() and (path.suffix == ".py" or "tokenizer_data" in path.parts):
            dest = snapshot / path.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    hashes = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
    question_set = {"questions": questions, "primary_source_sha256": sha(original),
        "selection": "All eight frozen questions retained verbatim. D01 is previously tested diagnostic, excluded from primary aggregates. Fixed order emphasizes multiple retrieval and foundation-to-experiment connection before the other domains.",
        "trajectory_count_clarification": "Eight primary questions imply 40 trajectories; adding the requested diagnostic implies five additional trajectories (45 maximum), not a replacement chosen after outcomes."}
    atomic_json(output / "QUESTION_SET.json", question_set)
    cells = []
    # Cyclic Latin-square rotation: each method occupies each position once in
    # the first five blocks; imperfect balance in a time-limited prefix disclosed.
    for index, question in enumerate(questions):
        order = VARIANTS[index % 5:] + VARIANTS[:index % 5]
        cells.extend({"id": question["id"] + "-" + variant, "question_id": question["id"],
            "variant": variant, "block": index + 1, "position": pos + 1,
            "diagnostic": question["id"] == "D01"} for pos, variant in enumerate(order))
    api_reserve = API_REQUEST_BOUND * (sum(1 if c["variant"] == "standard" else 40 for c in cells) + 40)
    configs = {v: asdict(configuration(DIAGNOSTIC, v)) for v in VARIANTS}
    config = {"schema": "livedemo-authorized-hour-v1", "created_at": now(),
        "max_seconds": 3600, "trajectory_wall_seconds": 300, "workers": 1,
        "aggregate_ceiling_usd": "50.00", "per_hosted_total_ceiling_usd": str(PER_TRAJECTORY),
        "per_hosted_inference_ceiling_usd": str(MODEL_TRAJECTORY),
        "openalex_request_upper_bound_usd": str(API_REQUEST_BOUND),
        "openalex_total_reserve_usd": str(api_reserve), "inference_ledger_ceiling_usd": str(Decimal(50) - api_reserve),
        "provider": profile(settings), "environment": environment(), "research_configurations": configs,
        "cells": cells, "primary_trajectories": 40, "diagnostic_trajectories": 5,
        "source_sha256": hashes, "source_tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "question_set_sha256": sha(output / "QUESTION_SET.json"),
        "env_file": str(root / ".env"),
        "pricing": {"checked_at": now(), "model_source": "https://developers.openai.com/api/docs/models/gpt-6.1-terra",
            "model_basis": "Standard short context, conservative input upper bound includes 12-hour cache writes; no credits/discounts",
            "openalex_source": "https://help.openalex.org/access/example-costs/",
            "openalex_basis": "Metadata only: singleton free, list/filter .0001, keyword search .001; no rerank, semantic search, content or OQL. Retain .001 for every HTTP attempt including failed/unknown attempts."},
        "architecture_differences": {"standard": "Nonagentic fixed query; one action/request, up to 30 records, zero model calls/synthesis. Answer metrics N/A. Common original research question retained in manifest; fixed keyword query is the actual baseline acquisition input.",
            "agents": "Identical 18 actions/rounds, 144 admitted records, 40 requests, 28 ALL-stage model calls, 150000 tokens, page size 6, 24000 context chars, 300 seconds, inference cap1.46 plus metadata cap.04. CLM edits/recovery consume the shared model allowance. Graph expands the available memory only in graph variants."},
        "analysis": {"target_scoring": "Predefined DOI or exact normalized title/year or preflight-resolved OpenAlex ID; all-admitted and top-10 recall reported separately. Targets never supplied to research.",
            "semantic_answer_quality": "AWAITING_INDEPENDENT_REVIEW; no paid judge",
            "primary_comparison": "Complete matched PRIMARY question blocks; diagnostic separate. All partial/failed/never-run cells retained. One repetition; no significance or causal-superiority claim.",
            "coverage": "Separate exact target/excerpt/citation traceability from model-assessed component coverage. Neither establishes factual correctness."},
        "stop_policy": "One process at a time. Reserve entire remaining matched block's worst-case time and cost before starting it. Stop on uncertain billing, transport/HTTP failure, hard deadline, authorization exhaustion or unrecoverable exception. No automatic request or campaign replay.",
        "council": {"status": "NOT_RUN", "reason": "No existing operational five-agent council found in LiveDemo; primary trajectories have priority. No council inference charged."}}
    atomic_json(output / "EXPERIMENT_CONFIGURATION.json", config)
    for filename in ("EXPERIMENT_CONFIGURATION.json", "QUESTION_SET.json"):
        (output / filename).chmod(0o444)
    for cell in cells:
        path = output / "trajectories" / cell["id"] / "trajectory.json"
        path.parent.mkdir(parents=True)
        atomic_json(path, {**cell, "status": "NOT_RUN", "attempted": False, "result": None})
    report(output)
    return config


def check(output, settings):
    output, config, questions = load(output)
    if profile(settings) != config["provider"] or environment() != config["environment"]:
        raise ValueError("Provider/runtime settings differ from the frozen configuration")
    if sha(output / "QUESTION_SET.json") != config["question_set_sha256"]:
        raise ValueError("Frozen questions changed")
    if any(sha(output / "source_snapshot" / path) != digest for path, digest in config["source_sha256"].items()):
        raise ValueError("Frozen implementation changed")
    # Exact known direct endpoint/model; preserve configured conservatism.
    if (urlsplit(settings.llm_base_url).hostname != "api.openai.com" or
            settings.llm_model != "gpt-6.1-terra" or settings.llm_input_price < 2.5 or settings.llm_output_price < 5):
        raise ValueError("Cannot verify conservative pricing for this provider/model configuration")
    return output, config, questions


def authorize(output):
    """Explicit current user grant; never reads Revision 7 files or approvals."""
    output = Path(output)
    target = output / "AUTHORIZATION.json"
    if target.exists():
        raise ValueError("Do not replace an authorization or reset a ledger")
    atomic_json(target, {"approved": True, "ceiling_usd": "50.00", "user_statement": AUTHORIZATION,
        "scope": "This LiveDemo hour evaluation only; separate from Revision 7", "created_at": now(),
        "configuration_sha256": sha(output / "EXPERIMENT_CONFIGURATION.json")})


class DualLedger:
    """Same durable pre-HTTP accounting, enforcing BOTH local and shared caps."""
    def __init__(self, output, config, trajectory):
        self.shared = SpendingLedger(Path(output) / "inference-spending.sqlite",
            ceiling=config["inference_ledger_ceiling_usd"], identity=sha(Path(output) / "EXPERIMENT_CONFIGURATION.json"))
        self.local = SpendingLedger(Path(output) / "trajectories" / trajectory / "inference-spending.sqlite",
            ceiling=MODEL_TRAJECTORY, identity=trajectory)

    def reserve(self, *args, **kwargs):
        self.local.reserve(*args, **kwargs)
        return self.shared.reserve(*args, **kwargs)

    def ensure_affordable(self, *args, **kwargs):
        self.local.ensure_affordable(*args, **kwargs)
        self.shared.ensure_affordable(*args, **kwargs)

    def settle(self, *args, **kwargs):
        error = None
        for ledger in (self.local, self.shared):
            try:
                ledger.settle(*args, **kwargs)
            except BudgetExceeded as exc:
                error = exc
        if error:
            raise error

    def summary(self):
        return {**self.local.summary(), "shared": self.shared.summary()}


class MetadataTransport(httpx.BaseTransport):
    """Bounded API accounting without changing retrieval actions or algorithms."""
    def __init__(self, output, config, trajectory, transport=None):
        self.output, self.config, self.trajectory = Path(output), config, trajectory
        self.transport = transport or httpx.HTTPTransport(retries=0)
        self.path = self.output / "openalex-spending.sqlite"
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,trajectory TEXT,bound TEXT,status TEXT,reported TEXT)")

    @contextmanager
    def transaction(self):
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    def handle_request(self, request):
        if (request.method != "GET" or request.url.host != "api.openalex.org" or
                request.url.path.strip("/").split("/")[0] not in ("works", "topics", "authors", "institutions", "sources") or
                not set(request.url.params) <= {"search", "filter", "cursor", "per_page", "per-page", "mailto", "select", "sort"}):
            raise BudgetExceeded("Only pre-priced metadata API operations are allowed")
        with self.transaction() as db:
            rows = db.execute("SELECT trajectory,bound FROM attempts").fetchall()
            cap = 1 if self.trajectory.endswith("-standard") else 40
            if sum(r[0] == self.trajectory for r in rows) >= cap:
                raise BudgetExceeded("Metadata HTTP attempt cap")
            if sum((Decimal(r[1]) for r in rows), Decimal(0)) + API_REQUEST_BOUND > Decimal(self.config["openalex_total_reserve_usd"]):
                raise BudgetExceeded("Shared metadata reservation exhausted")
            key = db.execute("INSERT INTO attempts(trajectory,bound,status) VALUES(?,?,?)",
                (self.trajectory, str(API_REQUEST_BOUND), "RETAINED_BEFORE_HTTP")).lastrowid
        # Keep the maximum even on success: no assumption about free credits,
        # cache hits or whether a failed HTTP attempt was billed.
        response = self.transport.handle_request(request)
        response.read()
        cost = None
        if response.is_success:
            try:
                body = json.loads(response.content)
                cost = (body.get("meta") or {}).get("cost_usd")
            except (ValueError, AttributeError):
                pass
        with self.transaction() as db:
            db.execute("UPDATE attempts SET status=?,reported=? WHERE id=?", (str(response.status_code), str(cost) if cost is not None else None, key))
        if cost is not None and (not Decimal(str(cost)).is_finite() or Decimal(str(cost)) > API_REQUEST_BOUND):
            raise BudgetExceeded("OpenAlex reported price exceeds frozen operation bound; stop campaign")
        return response

    def close(self):
        self.transport.close()


def spending(output, config):
    inference = Path(output) / "inference-spending.sqlite"
    summary = SpendingLedger(inference, ceiling=config["inference_ledger_ceiling_usd"],
        identity=sha(Path(output) / "EXPERIMENT_CONFIGURATION.json")).summary() if inference.exists() else {
            "retained_upper_bound_usd": "0", "usage_based_upper_bound_usd": "0", "uncertain_attempts": 0, "attempts": 0}
    api = Path(output) / "openalex-spending.sqlite"
    with sqlite3.connect(api) if api.exists() else nullcontext(None) as db:
        rows = db.execute("SELECT bound,reported FROM attempts").fetchall() if db else []
    bound = sum((Decimal(r[0]) for r in rows), Decimal(0))
    return {"inference": summary, "openalex_attempts": len(rows), "openalex_retained_upper_bound_usd": str(bound),
        "total_retained_upper_bound_usd": str(bound + Decimal(summary["retained_upper_bound_usd"])),
        "basis": "Provider tokens at conservative prices plus worst-case metadata cost per HTTP attempt; not invoice cost"}


def admission(remaining_seconds, remaining_dollars, cells):
    if remaining_seconds < len(cells) * 300 + 5:
        return "insufficient_time_for_complete_matched_block"
    maximum = sum((PER_TRAJECTORY if c["variant"] != "standard" else API_REQUEST_BOUND for c in cells), Decimal(0))
    if remaining_dollars < maximum:
        return "insufficient_budget_for_complete_matched_block"
    return ""


def preflight(output, settings, config, questions):
    """Metadata-only identity validation; never inject these records into agents."""
    receipt = {"started_at": now(), "hosted_calls": 0, "targets": [], "status": "RUNNING"}
    path = Path(output) / "PREFLIGHT.json"
    atomic_json(path, receipt)
    effective = replace(settings, max_retries=0, max_requests=40, timeout=min(settings.timeout, 10))
    try:
        with Database(str(Path(output) / "preflight.sqlite")) as db, OpenAlexClient(db, effective,
                transport=MetadataTransport(output, config, "preflight")) as client:
            client.search("academic research", limit=1, refresh=True)
            for question in questions:
                for target in question["targets"]:
                    try:
                        paper = Paper.from_openalex(client._request("/works/https://doi.org/" + target["doi"]))
                    except RetrievalError:
                        errors = [e for e in db.events(client.run_id) if e["kind"] == "request_error"]
                        if not errors or errors[-1].get("status") != 404:
                            raise
                        paper = next((p for p in client.search(target["title"], limit=3) if matches(p.to_dict(), target)), None)
                    row = {"question_id": question["id"], "doi": target["doi"], "status": "VERIFIED" if paper and matches(paper.to_dict(), target) else "UNRESOLVED",
                        "openalex_id": paper.id if paper and matches(paper.to_dict(), target) else None}
                    receipt["targets"].append(row)
                    atomic_json(path, receipt)
            receipt["status"] = "PASSED" if all(t["status"] == "VERIFIED" for t in receipt["targets"]) else "BLOCKED_TARGET_IDENTITIES"
    except Exception as exc:
        receipt.update(status="BLOCKED_PREFLIGHT", error=diagnostic(exc, (settings.api_key, settings.llm_api_key, settings.email)))
        # Capture public-host DNS diagnostics without printing URLs, proxy
        # credentials, request headers or configured keys.
        import socket
        receipt["dns_checks"] = []
        for hostname in ("api.openalex.org", "api.openai.com"):
            try:
                socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
                receipt["dns_checks"].append({"host": hostname, "status": "RESOLVED"})
            except OSError as dns_error:
                receipt["dns_checks"].append({"host": hostname, "status": "FAILED",
                    "type": type(dns_error).__name__, "errno": dns_error.errno})
    receipt["ended_at"] = now()
    atomic_json(path, receipt)
    return receipt


def worker(output, cell_id, settings, *, transports=None):
    output, config, questions = check(output, settings)
    authorization = json.loads((output / "AUTHORIZATION.json").read_text())
    if authorization.get("approved") is not True or authorization.get("configuration_sha256") != sha(output / "EXPERIMENT_CONFIGURATION.json"):
        raise ValueError("Worker has no matching dedicated authorization")
    cell = next(c for c in config["cells"] if c["id"] == cell_id)
    path = output / "trajectories" / cell_id / "trajectory.json"
    previous = json.loads(path.read_text())
    if previous["status"] != "RUNNING":
        raise ValueError("Worker requires durable RUNNING intent; no replay")
    question = next(q for q in questions if q["id"] == cell["question_id"])
    ledger = DualLedger(output, config, cell_id) if cell["variant"] != "standard" else None
    record = dict(previous)
    if transports is not None:
        record["evidence_mode"] = "MOCKED_INTEGRATION_NO_REAL_INFERENCE"
    started = time.monotonic()
    try:
        job = start_research(configuration(question, cell["variant"]), replace(settings, max_retries=0),
            root=path.parent, spending_ledger=ledger, trajectory_id=cell_id,
            **(transports or {"openalex_transport": MetadataTransport(output, config, cell_id)}))
        record["research_directory"] = str(job.directory)
        atomic_json(path, record)
        while job.snapshot()["status"] in ("starting", "running"):
            job.join(.1)
        state = job.snapshot()
        result = state.get("result") or {}
        synthesis = result.get("synthesis") or {}
        record.update(result=result, events=state["events"], error=state.get("error"),
            status="COMPLETED" if state["status"] == "completed" else "FAILED")
        if state.get("error"):
            record["fatal"] = True
        if cell["variant"] != "standard" and synthesis.get("status") != "qualified":
            record["status"] = "PARTIAL" if result else "FAILED"
        if any(e["kind"] == "live_model_call" and e.get("error_code") in ("transport_error", "timeout", "http_error", "authentication_error", "rate_limited") for e in state["events"]):
            record["fatal"] = True
        if synthesis.get("question_completion") == "incomplete_evidence":
            record["research_outcome"] = "PARTIAL_EVIDENCE"
        if ledger and (ledger.local.summary()["uncertain_attempts"] or ledger.shared.summary()["uncertain_attempts"]):
            record["fatal"] = True
    except BaseException as exc:
        record.update(status="FAILED", fatal=True, error=diagnostic(exc, (settings.api_key, settings.llm_api_key, settings.email)))
    record.update(ended_at=now(), worker_wall_seconds=time.monotonic() - started)
    atomic_json(path, record)
    return record


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=.25)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=1)


def supervise(output, settings, *, resume_blocked=False):
    output, config, questions = check(output, settings)
    authorization = json.loads((output / "AUTHORIZATION.json").read_text())
    if authorization.get("approved") is not True or authorization.get("ceiling_usd") != "50.00" or authorization.get("configuration_sha256") != sha(output / "EXPERIMENT_CONFIGURATION.json"):
        raise ValueError("Missing/mismatched dedicated user authorization")
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "RUN_STATE.json"
        old = json.loads(state_path.read_text()) if state_path.exists() else None
        if old and (not resume_blocked or not old["status"].startswith("BLOCKED") or spending(output, config)["inference"]["attempts"]):
            raise ValueError("Campaign already attempted. No automatic restart or paid replay")
        started_wall = old["started_wall"] if old else time.time()
        deadline = time.monotonic() + max(0, config["max_seconds"] - (time.time() - started_wall))
        state = {"status": "PREFLIGHT", "pid": os.getpid(), "started_at": old["started_at"] if old else now(), "started_wall": started_wall,
            "deadline_wall": started_wall + config["max_seconds"], "current": None}
        atomic_json(state_path, state)
        child = None
        previous_handler = signal.getsignal(signal.SIGTERM)
        def terminated(signum, frame):
            raise KeyboardInterrupt("Supervisor terminated; preserve accounting and stop own worker")
        signal.signal(signal.SIGTERM, terminated)
        try:
            if deadline - time.monotonic() < 90:
                state.update(status="STOPPED", reason="preflight_timeout_cannot_fit_deadline")
                return
            # Metadata preflight is bounded by the same campaign deadline.
            with (output / "preflight.log").open("a") as log:
                child = subprocess.Popen(command(output, "preflight"), env=frozen_env(output), stdout=log, stderr=subprocess.STDOUT)
                try:
                    child.wait(timeout=min(90, max(.001, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    stop_child(child)
            preflight_record = json.loads((output / "PREFLIGHT.json").read_text()) if (output / "PREFLIGHT.json").exists() else {}
            if child.returncode != 0 or preflight_record.get("status") != "PASSED":
                state.update(status="BLOCKED_PREFLIGHT", reason=preflight_record.get("status", "Preflight process failed"))
                return
            state["status"] = "RUNNING"
            for block in range(1, len(questions) + 1):
                cells = [c for c in config["cells"] if c["block"] == block]
                dollars = spending(output, config)
                if dollars["inference"]["uncertain_attempts"]:
                    state.update(status="STOPPED", reason="uncertain_inference_accounting")
                    break
                reason = admission(deadline - time.monotonic(), Decimal(50) - Decimal(dollars["total_retained_upper_bound_usd"]), cells)
                if reason:
                    state.update(status="STOPPED", reason=reason)
                    break
                block_inference = sum(c["variant"] != "standard" for c in cells) * MODEL_TRAJECTORY
                if Decimal(dollars["inference"]["retained_upper_bound_usd"]) + block_inference > Decimal(config["inference_ledger_ceiling_usd"]):
                    state.update(status="STOPPED", reason="insufficient_inference_partition_for_complete_block")
                    break
                for cell in cells:
                    state["current"] = cell["id"]
                    atomic_json(state_path, state)
                    path = output / "trajectories" / cell["id"] / "trajectory.json"
                    record = json.loads(path.read_text())
                    if record["status"] != "NOT_RUN":
                        raise ValueError("Prior trajectory found; automatic replay forbidden")
                    # Inferential ceiling uses its own conservative partition.
                    bound = Decimal(spending(output, config)["inference"]["retained_upper_bound_usd"])
                    if cell["variant"] != "standard" and bound + MODEL_TRAJECTORY > Decimal(config["inference_ledger_ceiling_usd"]):
                        raise BudgetExceeded("Full hosted maximum cannot be reserved")
                    if deadline - time.monotonic() < 301:
                        raise BudgetExceeded("Full trajectory timeout cannot fit global deadline")
                    record.update(status="RUNNING", attempted=True, started_at=now(), evidence_mode="REAL_LIVE_OPENALEX_NO_LLM" if cell["variant"] == "standard" else "REAL_LIVE_OPENALEX_AND_HOSTED")
                    atomic_json(path, record)
                    with (path.parent / "worker.log").open("a") as log:
                        began = time.monotonic()
                        child = subprocess.Popen(command(output, "worker", "--cell", cell["id"]), env=frozen_env(output), stdout=log, stderr=subprocess.STDOUT)
                        try:
                            child.wait(timeout=min(299, max(.001, deadline - time.monotonic() - 1)))
                        except subprocess.TimeoutExpired:
                            stop_child(child)
                            record = json.loads(path.read_text())
                            record.update(status="TIMED_OUT", fatal=True, ended_at=now(), supervisor_wall_seconds=time.monotonic()-began,
                                error={"type": "hard_deadline", "message": "Own worker terminated; provider holds and partial artifacts retained; no retry"})
                            atomic_json(path, record)
                    record = json.loads(path.read_text())
                    if record["status"] == "RUNNING":
                        record.update(status="INTERRUPTED", fatal=True, ended_at=now(), error={"type": "worker_exit", "message": "Worker exited before terminal receipt; never replay"})
                        atomic_json(path, record)
                    record["supervisor_wall_seconds"] = time.monotonic() - began
                    atomic_json(path, record)
                    # Durable journal precedes report and next acquisition.
                    with (output / "TRAJECTORY_RESULTS.jsonl").open("a") as journal:
                        journal.write(json.dumps(record, ensure_ascii=False) + "\n"); journal.flush(); os.fsync(journal.fileno())
                    report(output)
                    if record.get("fatal") or spending(output, config)["inference"]["uncertain_attempts"]:
                        raise BudgetExceeded("Unrecoverable worker error or uncertain accounting; no retry")
            else:
                state.update(status="FINISHED", reason="all_planned_trajectories_attempted")
        except BaseException as exc:
            if child:
                stop_child(child)
            if state.get("current"):
                path = output / "trajectories" / state["current"] / "trajectory.json"
                record = json.loads(path.read_text())
                if record["status"] == "RUNNING":
                    record.update(status="INTERRUPTED", fatal=True, ended_at=now(), error={"type": type(exc).__name__, "message": "Supervisor stopped; artifacts/holds retained, no replay"})
                    atomic_json(path, record)
                    with (output / "TRAJECTORY_RESULTS.jsonl").open("a") as journal:
                        journal.write(json.dumps(record, ensure_ascii=False) + "\n"); journal.flush(); os.fsync(journal.fileno())
            state.update(status="STOPPED", reason=diagnostic(exc, (settings.api_key, settings.llm_api_key, settings.email)))
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            state.update(current=None, ended_at=now(), elapsed_seconds=max(0, time.time() - started_wall))
            atomic_json(state_path, state)
            report(output)


def frozen_env(output):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(output) / "source_snapshot")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def command(output, operation, *extra):
    return [sys.executable, "-m", "openalex_agent.hour_evaluation", operation, "--output", str(Path(output).resolve()), *extra]


def launch(output, *, resume_blocked=False):
    output = Path(output).resolve()
    with (output / "supervisor.log").open("a") as log:
        child = subprocess.Popen(command(output, "run", *(["--resume-blocked"] if resume_blocked else [])),
            cwd=output / "source_snapshot", env=frozen_env(output), start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    atomic_json(output / "LAUNCH.json", {"pid": child.pid, "submitted_at": now(), "startup_verified": False})
    return child.pid


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def report(output):
    output, config, questions = load(output)
    state = json.loads((output / "RUN_STATE.json").read_text()) if (output / "RUN_STATE.json").exists() else {"status": "PREPARED"}
    records, rows, review = [], [], []
    verification = json.loads((output / "PREFLIGHT.json").read_text()) if (output / "PREFLIGHT.json").exists() else {}
    for cell in config["cells"]:
        record = json.loads((output / "trajectories" / cell["id"] / "trajectory.json").read_text())
        question = dict(next(q for q in questions if q["id"] == cell["question_id"]))
        question["targets"] = [dict(t, openalex_id=next((v["openalex_id"] for v in verification.get("targets", [])
            if v["question_id"] == question["id"] and v["doi"] == t["doi"] and v["status"] == "VERIFIED"), t.get("openalex_id"))) for t in question["targets"]]
        records.append(record)
        row = score(record, question, config["provider"])
        result = record.get("result") or {}
        synthesis = result.get("synthesis") or {}
        answer = synthesis.get("answer") or {}
        papers = [r["paper"] for r in result.get("ranked", [])]
        source_ids = {p["id"] for p in synthesis.get("sources", []) if p.get("abstract_excerpt")}
        targets = [t["doi"] for t in question["targets"] if any(matches(p, t) for p in papers)] if result else None
        row.update(diagnostic=cell["diagnostic"], metadata_admitted=len(result.get("inspected_ids", [])) if result else None,
            targets_any_admitted=[t["doi"] for t in question["targets"] if t.get("openalex_id") in result.get("inspected_ids", [])
                or any(matches(p, t) for p in papers)] if result else None,
            targets_in_retained_ranking=targets, target_retained_recall=len(targets)/len(question["targets"]) if targets is not None else None,
            question_answered_independent=None, factual_accuracy_independent=None,
            question_completion=synthesis.get("question_completion"),
            pending_observation_chars=result.get("pending_observation_chars"),
            unadmitted_observation_indices=result.get("unadmitted_observation_indices"),
            retrieval_seconds=result.get("retrieval_seconds"), supervisor_wall_seconds=record.get("supervisor_wall_seconds"),
            retrieved_target_abstracts=len([t for t in question["targets"] if any(matches(p, t) and p["id"] in source_ids for p in papers)]) if result else None)
        local = output / "trajectories" / cell["id"] / "inference-spending.sqlite"
        if local.exists():
            with sqlite3.connect(local) as db:
                attempts = db.execute("SELECT held,status FROM attempts").fetchall()
            row["retained_inference_upper_bound_usd"] = str(sum((Decimal(a[0]) for a in attempts), Decimal(0)))
            row["uncertain_inference_requests"] = sum(a[1] != "SETTLED" for a in attempts)
        else:
            row["retained_inference_upper_bound_usd"] = "0" if record.get("attempted") and cell["variant"] == "standard" else None
            row["uncertain_inference_requests"] = None
        metadata = output / "openalex-spending.sqlite"
        if metadata.exists() and record.get("attempted"):
            with sqlite3.connect(metadata) as db:
                attempts = db.execute("SELECT bound,status FROM attempts WHERE trajectory=?", (cell["id"],)).fetchall()
            row["metadata_attempts_including_failed"] = len(attempts)
            row["retained_openalex_upper_bound_usd"] = str(sum((Decimal(a[0]) for a in attempts), Decimal(0)))
            row["metadata_failed_or_unknown_requests"] = sum(a[1] != "200" for a in attempts)
        else:
            row["metadata_attempts_including_failed"] = None
            row["retained_openalex_upper_bound_usd"] = None
            row["metadata_failed_or_unknown_requests"] = None
        row["http_rate_limits"] = sum(e.get("status", e.get("http_status")) == 429 for e in record.get("events", [])) if record.get("attempted") else None
        row["automatic_paid_retries"] = 0 if record.get("attempted") else None
        rows.append(row)
        for section in ("answer", "foundations", "comparison", "conclusion"):
            for i, claim in enumerate(answer.get(section, [])):
                review.append({"trajectory": cell["id"], "section": section, "index": i, "claim": claim,
                    "review_status": "AWAITING_INDEPENDENT_REVIEW", "required_question_components": question["required_components"]})
    atomic_json(output / "RUN_MANIFEST.json", {"configuration_sha256": sha(output / "EXPERIMENT_CONFIGURATION.json"),
        "state": state, "cells": [{k: r.get(k) for k in ("id", "question_id", "variant", "status", "attempted", "research_directory")} for r in records]})
    journal = output / "TRAJECTORY_RESULTS.jsonl"
    if not journal.exists():
        journal.touch()
    atomic_json(output / "RESULTS.json", {"state": state, "rows": rows, "spending": spending(output, config), "claim_review": review})
    identity = ("id", "question_id", "variant", "diagnostic", "status", "attempted")
    groups = {
        "RETRIEVAL_RESULTS.csv": ("http_requests", "actions", "metadata_admitted", "unique_publications", "targets_any_admitted", "target_recall_at_10", "targets_found_at_10", "target_retained_recall", "targets_in_retained_ranking", "retrieved_target_abstracts", "sources_sent_to_model"),
        "ANSWER_QUALITY.csv": ("final_answer_produced", "validated_synthesis_completed", "question_answered_independent", "factual_accuracy_independent", "question_completion", "traceable_citations", "untraceable_citations", "model_component_coverage", "unresolved_components_at_stop", "limitations_present", "scientifically_unsupported_claims", "independent_claim_review"),
        "COST_AND_LATENCY.csv": ("model_calls", "input_tokens", "output_tokens", "usage_based_cost_upper_bound_usd", "retained_inference_upper_bound_usd", "uncertain_inference_requests", "retained_openalex_upper_bound_usd", "openalex_reported_cost_usd", "metadata_attempts_including_failed", "metadata_failed_or_unknown_requests", "http_rate_limits", "automatic_paid_retries", "seconds", "supervisor_wall_seconds", "retrieval_seconds", "planning_failures", "invalid_actions", "repeated_action_rejections", "stop_reason", "error"),
        "GRAPH_CLM_ACTIVITY.csv": ("mechanisms", "pending_observation_chars", "unadmitted_observation_indices"),
    }
    for filename, keys in groups.items():
        write_csv(output / filename, [{k: row.get(k) for k in identity + keys} for row in rows])
    atomic_json(output / "CLAIM_REVIEW.json", review)
    table = ["| Architecture | Attempted | Validated answers* | Failures/partial | Mean target R@10 | Calls | Tokens | Runtime s | Inference upper bound $ |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        arm = [r for r in rows if r["variant"] == variant and not r["diagnostic"]]
        attempted = [r for r in arm if r["attempted"]]
        recalls = [r["target_recall_at_10"] for r in attempted if r["target_recall_at_10"] is not None]
        def sums(key):
            values = [r[key] for r in attempted if r.get(key) is not None]
            if not values:
                return "—"
            return str(round(sum(values), 6)) + (" + unknown" if len(values) < len(attempted) else "")
        table.append(f"| {variant} | {len(attempted)}/{len(arm)} | {'N/A' if variant == 'standard' else sum(r.get('validated_synthesis_completed') is True for r in attempted)} | {sum(r['status'] != 'COMPLETED' for r in attempted)} | {round(sum(recalls)/len(recalls),3) if recalls else '—'} | {sums('model_calls')} | {sum((r.get('input_tokens') or 0)+(r.get('output_tokens') or 0) for r in attempted) if attempted else '—'} | {sums('seconds')} | {sums('usage_based_cost_upper_bound_usd')} |")
    complete_blocks = [q["id"] for q in questions if all(r["status"] in ("COMPLETED", "PARTIAL", "FAILED", "TIMED_OUT", "INTERRUPTED") for r in rows if r["question_id"] == q["id"])]
    mechanism_table = ["| Architecture | Successful planner inputs | Directed pairs delivered | Model edits committed | Edits in later requests | Accepted recoveries | Planner-assessed adequate* |", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        arm = [r for r in rows if r["variant"] == variant and not r["diagnostic"] and r["attempted"]]
        def mechanism(key):
            values = [(r.get("mechanisms") or {}).get(key) for r in arm]
            return str(sum(v for v in values if v is not None)) if any(v is not None for v in values) else "—"
        mechanism_table.append(f"| {variant} | {mechanism('successful_followup_planning_calls')} | {mechanism('graph_edges_in_successful_planning')} | {mechanism('clm_edits_committed')} | {mechanism('committed_edits_in_later_requests')} | {mechanism('accepted_recoveries')} | {sum((r.get('model_component_coverage') or {}).get('adequate') is True for r in arm) if arm and variant != 'standard' else '—'} |")
    finances = spending(output, config)
    text = "\n".join(["# Exploratory LiveDemo comparison", "", f"Campaign state: {state['status']}. Planned: 40 primary + 5 diagnostic trajectories; complete matched blocks: {complete_blocks or 'none'}.", "", *table, "",
        "*Validated answers mean structural citation/quote validation, not independent research completeness or factual accuracy. Standard retrieval has no answer generator and is not scored for synthesis.", "",
        "Primary aggregate excludes D01. Individual outcomes, including every unrun cell, are in RESULTS.json and the CSV files. Incomplete blocks must not be used to imply matched superiority. Paired analysis is restricted to complete blocks; the table above also exposes all primary attempts for reliability accounting.", "",
        "Independent question completion, semantic evidence coverage, abstention appropriateness and factual accuracy are AWAITING INDEPENDENT REVIEW. Target recall and quote membership are narrower deterministic measurements. No paid judge was used.", "", *mechanism_table, "",
        "*Planner-assessed coverage is not independent grading. This mechanism table reports actual request/commit receipts, not effects on answer quality. Totals can include incomplete blocks for transparency; use per-question rows to assess matched outcomes. Diagnostic rows remain separate in CSVs.", "",
        "Graph benefit: not established. CLM benefit: not established. Combined superiority: not established. Mechanism exposure alone proves none of these. One repetition and few questions do not justify significance or causal claims. Provider/index drift and truncated method-order balance remain limitations.", "",
        "Financial accounting: " + json.dumps(finances), "",
        "All limits/architecture differences/pricing sources and question selection were fixed before hosted execution in EXPERIMENT_CONFIGURATION.json and QUESTION_SET.json."])
    (output / "COMPARISON_REPORT.md").write_text(text + "\n")
    (output / "COMPARISON.md").write_text(text + "\n")
    failures = ["# Failure analysis", "", f"Supervisor: {json.dumps(state)}", "", "Failures are retained, never excluded or automatically replayed.", ""]
    for r in records:
        if r["attempted"] and r["status"] != "COMPLETED":
            failures.append(f"- {r['id']}: {r['status']}; {json.dumps(r.get('error'))}")
    pre = output / "PREFLIGHT.json"
    if pre.exists():
        failures += ["", "Metadata preflight: " + pre.read_text()]
    (output / "FAILURE_ANALYSIS.md").write_text("\n".join(failures) + "\n")
    (output / "EVIDENCE_REVIEW.md").write_text("# Independent evidence review\n\nNo independent human claim adjudication has been completed. CLAIM_REVIEW.json lists every actual claim, its citations, trajectory and required question components. Check primary identity, abstract versus full-text scope, entailment, quantitative qualifiers, omissions, contradictions and abstention appropriateness. Assign supported/partly supported/unsupported/contradicted/not assessable; missing review stays unverified.\n\nEnvironment context admission is not a model CLM edit. Directed-edge delivery is not proof of use or quality improvement.\n")
    attempted = sum(bool(r["attempted"]) for r in records)
    final = ["# Final experiment report", "", f"State: {state['status']}", f"Planned: 45 (40 primary, 5 previously tested diagnostic)",
        f"Attempted: {attempted}; terminal completed: {sum(r['status']=='COMPLETED' for r in records)}; partial/failed/interrupted/timed out: {sum(r['attempted'] and r['status']!='COMPLETED' for r in records)}; never run: {sum(not r['attempted'] for r in records)}.",
        f"Elapsed: {state.get('elapsed_seconds', 'not started')} seconds.", "Conservative spending: " + json.dumps(finances), "",
        "No architecture performance conclusion is established without genuine trajectories and independent answer review. See comparison, per-question CSVs and preserved raw trajectories. Council: NOT RUN; no existing operational five-agent council was found. All five independent reviews must exist before claiming council execution.", "",
        "A blocked metadata preflight means the HOSTED EXPERIMENT DID NOT START. A submitted detached command alone is not startup evidence."]
    (output / "FINAL_EXPERIMENT_REPORT.md").write_text("\n".join(final) + "\n")
    return {"state": state, "rows": rows, "spending": finances}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "authorize", "launch", "run", "worker", "preflight", "report", "dry-run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell")
    parser.add_argument("--resume-blocked", action="store_true", help="Explicit preflight-only retry with original ledger/deadline, never after a hosted request")
    args = parser.parse_args(argv)
    if args.operation == "report":
        result = report(args.output)
        print(json.dumps({"state": result["state"], "spending": result["spending"]}, indent=2)); return 0
    if args.operation == "authorize":
        authorize(args.output); return 0
    if args.operation == "prepare":
        result = prepare(args.output, load_live_settings()); print(json.dumps({"planned": len(result["cells"]), "output": str(args.output)})); return 0
    output, config, questions = load(args.output)
    if args.operation in ("run", "worker", "preflight") and not Path(__file__).resolve().is_relative_to(output / "source_snapshot"):
        # Even a manual `run` must execute the frozen source, rather than a
        # possibly edited workspace whose snapshot still passes hash checks.
        os.chdir(output / "source_snapshot")
        os.execve(sys.executable, command(output, args.operation,
            *(["--cell", args.cell] if args.cell else []),
            *(["--resume-blocked"] if args.resume_blocked else [])), frozen_env(output))
    settings = load_live_settings(config["env_file"])
    check(output, settings)
    if args.operation == "dry-run":
        print(json.dumps({"paid_calls": 0, "network_calls": 0, "planned_primary": 40, "diagnostic": 5,
            "source_tree_sha256": config["source_tree_sha256"], "provider": config["provider"],
            "admission_allows_first_block": not admission(3600, Decimal(50), config["cells"][:5]),
            "caps": {k: config[k] for k in ("aggregate_ceiling_usd", "per_hosted_total_ceiling_usd", "max_seconds", "workers")}}, indent=2)); return 0
    if args.operation == "launch":
        print(json.dumps({"submitted_pid": launch(output, resume_blocked=args.resume_blocked), "startup_verified": False})); return 0
    if args.operation == "preflight":
        receipt = preflight(output, settings, config, questions); print(json.dumps(receipt, indent=2)); return 0 if receipt["status"] == "PASSED" else 2
    if args.operation == "worker":
        worker(output, args.cell, settings); return 0
    supervise(output, settings, resume_blocked=args.resume_blocked)
    return 0 if json.loads((output / "RUN_STATE.json").read_text())["status"] == "FINISHED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
