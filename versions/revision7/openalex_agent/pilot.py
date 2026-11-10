"""Small graph-information ablation with matched metadata and retrieval tools.

This isolated pilot deliberately does not use the search-only iterative baseline.
Both arms use the same metadata-based action eligibility and proposal order.
Only the graph arm constructs a graph and adds derived graph information.
"""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from .actions import ACTION_ARGUMENTS, Action, make_action
from .config import Settings
from .controller import Budget, Controller
from .corpus import LocalCorpus, atomic_json, freeze
from .experiments import environment_record
from .git_provenance import git_provenance
from .llm import HostedJSONPlanner
from .models import utcnow
from .openalex import OpenAlexClient
from .ranking import relevance
from .selection import score_candidates
from .storage import Database

GRAPH_FIELDS = {"graph", "candidates", "subgraph", "topic_coverage"}


class PilotController(Controller):
    """A context ablation, with an actual graph-free memory arm.

    Three actions bound any acquisition chain to at most two hops, below the
    existing depth limit. Shared metadata eligibility replaces graph-derived
    eligibility in BOTH arms, so tool access cannot reveal the treatment.
    Production graph and iterative policies are unchanged.
    """

    def __init__(self, *args, arm: str, **kwargs):
        if arm not in ("graph", "no_graph"):
            raise ValueError("Unknown pilot arm")
        budget = kwargs.get("budget") or Budget(max_actions=3, max_rounds=3, max_llm_calls=2, page_size=2)
        if budget.max_actions > 3:
            raise ValueError("This preliminary pilot supports at most three actions")
        self.arm = arm
        super().__init__(*args, method="graph" if arm == "graph" else "iterative", budget=budget,
                         **{k: v for k, v in kwargs.items() if k != "budget"})
        if self.scoring.max_depth < budget.max_actions:
            raise ValueError("Pilot depth limit must be nonbinding in both arms")

    def _log(self, kind, payload):
        super()._log(kind, {**payload, "pilot_arm": self.arm})

    def _targets(self):
        scores = relevance(self.question, list(self.papers.values()))
        anchors = [p for p in self.papers.values() if scores[p.id] >= self.scoring.min_relevance]
        known = {i for p in anchors for i in p.references + p.related} - self.papers.keys()
        topics = {t.id for p in anchors for t in p.topics}
        return {p.id for p in anchors}, known, topics

    def validate(self, action):
        action = Action.parse(action.to_dict())
        if action.signature in self.completed:
            raise ValueError("Repeated action provides no new information")
        if action.action == "STOP_RETRIEVAL":
            return
        if action.action == "SEARCH_PAPERS":
            if action.action not in self._common_context()["permitted_actions"]:
                raise ValueError("Repeated unproductive searches require another acquisition action")
            return
        anchors, known, topics = self._targets()
        if action.action in ("EXPAND_REFERENCES", "EXPAND_CITATIONS"):
            valid = action.arguments["paper_id"] in anchors
        elif action.action == "INSPECT_PAPER":
            valid = action.arguments["paper_id"] in known
        else:
            valid = action.arguments["topic_id"] in topics
        if not valid:
            raise ValueError("Target is not supported by relevant observed metadata")

    def _common_context(self):
        scores = relevance(self.question, list(self.papers.values()))
        papers = sorted(self.papers.values(), key=lambda p: (-scores[p.id], p.id))
        anchors, known, topics = self._targets()
        # Neutral ordering, identical in both arms; no graph scores leak here.
        proposals = []
        for name, ids in (("EXPAND_REFERENCES", anchors), ("EXPAND_CITATIONS", anchors),
                          ("INSPECT_PAPER", known), ("EXPLORE_TOPIC", topics)):
            for target in sorted(ids)[:4]:
                action = make_action(name, target, "Observed OpenAlex metadata target")
                if action.signature not in self.completed:
                    proposals.append(action)
        # Interleave tools, avoiding a long list of reference expansions only.
        proposals.sort(key=lambda a: (list(ACTION_ARGUMENTS).index(a.action),
                                     next(iter(a.arguments.values()), "")))
        selected = []
        groups = [[p for p in proposals if p.action == name] for name in ACTION_ARGUMENTS]
        for index in range(4):
            selected.extend(g[index] for g in groups if len(g) > index)
        feedback = self.search_feedback()
        paused = feedback["consecutive_searches_without_new_papers"] >= 2 and bool(selected)
        if not paused:
            selected += self._search_proposals()[:3]
        permitted = [n for n in ACTION_ARGUMENTS if n != "SEARCH_PAPERS" or not paused]
        result = {"question": self.question, "round": self.rounds,
            "retrieval": self.retrieval_environment(), "search_feedback": feedback,
            "guidance": "All six retrieval actions remain available regardless of graph summaries. "
                "Use only observed metadata IDs and permitted actions. Proposals have neutral metadata ordering.",
            "remaining": {"actions": self.budget.max_actions-self.actions,
                "inspections": self.budget.max_inspected-len(self.papers),
                "api_requests": self.budget.max_api_requests-self.retriever.usage.requests},
            "top_papers": [{"id": p.id, "title": (p.title or "")[:180],
                "abstract_excerpt": (p.abstract or "")[:250], "original_query_relevance": scores[p.id]}
                for p in papers[:6]],
            "observed_metadata": [{"id": p.id, "references": p.references[:6], "related": p.related[:3],
                "topics": [{"id": t.id, "name": t.name} for t in p.topics[:3]]} for p in papers[:6]],
            "recent_actions": self.history[-3:], "permitted_actions": permitted,
            "counts": {"retrieved": len(self.retrieved_ids), "inspected": len(self.papers)},
            "last_observation": {k: self.trace[-1].get(k) for k in
                ("returned_papers", "new_papers", "updated_papers", "new_promising")} if self.trace else {},
            "proposals": [a.to_dict() for a in selected[:12]]}
        # Reserve exactly the same base-context capacity for both arms.
        while len(json.dumps(result, ensure_ascii=False)) > self.budget.max_context_chars//2:
            for key in ("recent_actions", "observed_metadata", "top_papers", "proposals"):
                if len(result[key]) > (1 if key == "proposals" else 0):
                    result[key].pop()
                    break
            else:
                if result["search_feedback"]["recent_searches"]:
                    result["search_feedback"]["recent_searches"].pop(0)
                else:
                    raise ValueError("Pilot base context exceeds its shared capacity")
        return result

    def context(self):
        result = self._common_context()
        if self.graph is not None:
            top = [p["id"] for p in result["top_papers"][:3]]
            view = self.graph.neighborhood(top, max_nodes=12)
            result.update(graph=self.graph.stats(),
                candidates=score_candidates(self.question, self.papers, self.graph, self.scoring)[:8],
                subgraph=[{k: edge[k] for k in ("source", "target", "relation")} for edge in view["edges"][:12]],
                topic_coverage=dict(Counter(t.id for p in self.papers.values() for t in p.topics)))
            while len(json.dumps(result, ensure_ascii=False)) > self.budget.max_context_chars:
                for key in ("subgraph", "candidates"):
                    if result[key]:
                        result[key].pop()
                        break
                else:
                    raise ValueError("Pilot graph overlay exceeds capacity")
        return result

    def _export_result(self, seconds):
        result = super()._export_result(seconds)
        result.update(pilot_arm=self.arm, pilot_policy="matched metadata eligibility and neutral proposals")
        result["quota_wait_seconds"] = getattr(self.planner, "quota_wait_seconds", 0.0)
        result["local_processing_seconds"] = max(0.0, result["local_seconds"]-result["quota_wait_seconds"])
        atomic_json(self.directory / "result.json", result)
        return result


def acquire_pilot(questions, db, settings, output):
    """Bounded, shared acquisition before assignment; no arm sees extra papers."""
    records, papers = [], {}
    with OpenAlexClient(db, settings) as client:
        for q in questions:
            seeds = client.search(q["seed_query"], limit=8)
            if not seeds:
                raise ValueError("Pilot seed query returned no papers")
            anchor = min(seeds, key=lambda p: (-len(p.references), p.id))
            references = client.references(anchor.id, limit=3)
            citations = client.citations(anchor.id, limit=3)
            for paper in seeds + references + citations:
                papers[paper.id] = paper
            records.append({"query_id": q["id"], "query": q["seed_query"], "anchor": anchor.id,
                "seed_ids": [p.id for p in seeds], "reference_ids": [p.id for p in references],
                "citation_ids": [p.id for p in citations]})
        usage = asdict(client.usage)
    manifest = freeze(list(papers.values()), output, acquisition=records)
    atomic_json(Path(output)/"acquisition_usage.json", usage)
    return manifest


def run_pilot(corpus_path, questions, settings, output, planner_factory=None):
    if len(questions) != 3 or len({q["id"] for q in questions}) != 3:
        raise ValueError("This pilot requires exactly three distinct questions")
    output = Path(output)
    if (output/"experiment.json").exists():
        raise ValueError("Use a fresh pilot output directory")
    output.mkdir(parents=True, exist_ok=True)
    corpus = LocalCorpus(corpus_path)
    budget = Budget(max_actions=3, max_rounds=3, max_llm_calls=2, max_api_requests=1,
                    max_inspected=6, page_size=2, patience=3, output_k=6)
    factory = planner_factory or (lambda: HostedJSONPlanner(settings))
    with factory() as probe:
        configuration = probe.configuration
        model, fingerprint = probe.model, probe.fingerprint
    atomic_json(output/"experiment.json", {"created_at": utcnow(), "git": git_provenance(), "questions": questions,
        "corpus": corpus.manifest, "budget": asdict(budget), "model": model,
        "provider_configuration": configuration, "provider_fingerprint": fingerprint,
        "environment": environment_record(), "trials": 1, "arms": ["graph", "no_graph"],
        "run_order": [{"query_id": q["id"], "arms": ["graph", "no_graph"] if i % 2 == 0 else ["no_graph", "graph"]}
                      for i, q in enumerate(questions)],
        "intervention": "Graph store and derived graph summaries; shared flat metadata, tools, eligibility and proposals",
        "limitations": ["Single trial; no relevance judgments or quality inference.",
            "Both arms retrieve only from the same frozen, live-acquired corpus.",
            "Pilot common metadata policy differs from production graph and search-only iterative policies."]})
    rows, audits = [], []
    for index, q in enumerate(questions):
        # Explicit matched-state check BEFORE Gemini decisions can diverge.
        controllers = []
        for arm in ("graph", "no_graph"):
            with Database(":memory:") as audit_db:
                c = PilotController(q["question"], LocalCorpus(corpus_path), audit_db, arm=arm,
                                    budget=budget, output=str(output/"context-audits"/q["id"]))
                c._execute(make_action("SEARCH_PAPERS", q["question"]))
                controllers.append(c.context())
        full, flat = controllers
        base = {k: v for k, v in full.items() if k not in GRAPH_FIELDS}
        check = {"query_id": q["id"], "same_non_graph_context": base == flat,
            "same_action_schema": full["permitted_actions"] == flat["permitted_actions"] == list(ACTION_ARGUMENTS),
            "graph_fields_present": GRAPH_FIELDS <= full.keys(),
            "graph_fields_absent": not GRAPH_FIELDS.intersection(flat),
            "same_retrieval_scope": full["retrieval"] == flat["retrieval"],
            "initial_paper_ids": [p["id"] for p in flat["top_papers"]]}
        assert all(check[k] for k in ("same_non_graph_context", "same_action_schema",
                                     "graph_fields_present", "graph_fields_absent", "same_retrieval_scope"))
        atomic_json(output/"context-audits"/q["id"] / "contexts.json", {"graph": full, "no_graph": flat, "checks": check})
        audits.append(check)
        for arm in (("graph", "no_graph") if index % 2 == 0 else ("no_graph", "graph")):
            with Database(str(output/"pilot.sqlite")) as db, factory() as planner:
                if planner.fingerprint != fingerprint:
                    raise ValueError("Pilot provider configuration changed between arms")
                c = PilotController(q["question"], LocalCorpus(corpus_path), db, arm=arm, planner=planner,
                                    budget=budget, output=str(output/"runs"))
                wire_requests = []
                def capture(request):
                    # Capture only the actual JSON message payload, never URL,
                    # authorization headers, API keys, or provider response bodies.
                    body = json.loads(request.content)
                    safe = planner._redact({k: body[k] for k in ("model", "messages", "response_format",
                        "max_tokens", "max_completion_tokens", "temperature", "reasoning_effort") if k in body})
                    wire_requests.append(safe)
                    with (c.directory/"planner_requests.jsonl").open("a") as stream:
                        stream.write(json.dumps(safe, ensure_ascii=False)+"\n")
                planner.http.event_hooks["request"].append(capture)
                result = c.run()
                events = [json.loads(line) for line in (c.directory/"events.jsonl").read_text().splitlines()]
                contexts = [e["context"] for e in events if e["kind"] == "context"]
                wire_contexts = [json.loads(b["messages"][1]["content"]) for b in wire_requests]
                calls = [e for e in events if e["kind"] == "planner_call"]
                row = {"query_id": q["id"], "arm": arm, "run_id": c.run_id, "directory": str(c.directory),
                    "actions": result["actions"], "inspected": len(result["inspected_ids"]),
                    "llm_calls": result["llm_usage"]["calls"], "failures": result["llm_usage"]["failures"],
                    "fallbacks": sum(e["kind"] == "invalid_decision" for e in events),
                    "input_tokens": result["llm_usage"]["input_tokens"], "output_tokens": result["llm_usage"]["output_tokens"],
                    "seconds": result["seconds"], "llm_seconds": result["llm_usage"]["seconds"],
                    "quota_wait_seconds": result["quota_wait_seconds"],
                    "local_processing_seconds": result["local_processing_seconds"],
                    "mean_context_chars": statistics.mean(len(json.dumps(x,ensure_ascii=False)) for x in contexts) if contexts else 0,
                    "stop_reason": result["stop_reason"], "api_requests": result["api_usage"]["requests"],
                    "graph_memory_present": c.graph is not None,
                    "all_tools_offered": bool(contexts) and all(set(x["permitted_actions"]) == set(ACTION_ARGUMENTS) for x in contexts),
                    "graph_context_correct": all((GRAPH_FIELDS <= x.keys()) if arm == "graph" else not GRAPH_FIELDS.intersection(x) for x in contexts),
                    "wire_context_correct": len(wire_contexts) == len(contexts) and wire_contexts == contexts,
                    "reported_usage_calls": result["llm_usage"]["usage_reported_calls"],
                    "diagnostic_calls": len(calls)}
                row["valid_for_comparison"] = bool(contexts) and not row["failures"] and not row["fallbacks"] and \
                    row["wire_context_correct"] and row["graph_context_correct"] and row["all_tools_offered"]
                rows.append(row)
                atomic_json(output/"results.json", {"runs": rows, "context_audits": audits})
                with (output/"metrics.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader(); writer.writerows(rows)
                print(json.dumps(row), flush=True)
    return rows


def analyze_pilot(output):
    """Verify sent inputs independently from whether Gemini was available."""
    output = Path(output)
    experiment = json.loads((output/"experiment.json").read_text())
    saved = json.loads((output/"results.json").read_text())
    rows, checks, failures, rejections = saved["runs"], [], Counter(), Counter()
    for row in rows:
        directory = Path(row["directory"])
        result = json.loads((directory/"result.json").read_text())
        state = json.loads((directory/"state.json").read_text())
        events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
        for e in events:
            if e["kind"] == "invalid_decision":
                rejections[e.get("error_code", "unknown")] += 1
            if e["kind"] == "planner_call" and e.get("error_code"):
                failures[f"HTTP {e['http_status']}" if (e.get("http_status") or 0) >= 400 else e["error_code"]] += 1
        row["valid_for_comparison"] = (row["llm_calls"] > 0 and row["failures"] == row["fallbacks"] == 0
            and row["wire_context_correct"] and row["graph_context_correct"] and row["all_tools_offered"])
        row["graph_memory_correct"] = (state["graph_data"] is None and not (directory/"graph.json").exists()) \
            if row["arm"] == "no_graph" else state["graph_data"] is not None and (directory/"graph.json").exists()
        row["matched_budget_and_corpus"] = result["budget"] == experiment["budget"] and \
            result["corpus"] == experiment["corpus"]["sha256"]
        row["valid_for_comparison"] = row["valid_for_comparison"] and row["graph_memory_correct"] and row["matched_budget_and_corpus"]
    for q in experiment["questions"]:
        pair = {r["arm"]: r for r in rows if r["query_id"] == q["id"]}
        if set(pair) != {"graph", "no_graph"}:
            checks.append({"query_id": q["id"], "complete": False, "input_isolation_verified": False,
                           "valid_gemini_pair": False})
            continue
        requests = {}
        for arm, row in pair.items():
            p = Path(row["directory"])/"planner_requests.jsonl"
            requests[arm] = [json.loads(line) for line in p.read_text().splitlines()]
        if not all(requests.values()):
            raise ValueError("No actual request payloads captured for paired input audit")
        graph, flat = [json.loads(requests[arm][0]["messages"][1]["content"]) for arm in ("graph", "no_graph")]
        fields_equal = {k: v for k, v in graph.items() if k not in GRAPH_FIELDS} == flat
        same_system = requests["graph"][0]["messages"][0] == requests["no_graph"][0]["messages"][0]
        same_options = {k: v for k, v in requests["graph"][0].items() if k != "messages"} == \
                       {k: v for k, v in requests["no_graph"][0].items() if k != "messages"}
        six_tools = all(set(json.loads(b["messages"][1]["content"])["permitted_actions"]) == set(ACTION_ARGUMENTS)
                        for bodies in requests.values() for b in bodies)
        isolation = fields_equal and same_system and same_options and six_tools and GRAPH_FIELDS <= graph.keys() \
            and not GRAPH_FIELDS.intersection(flat) and all(r["graph_memory_correct"] and r["wire_context_correct"]
                                                         and r["matched_budget_and_corpus"] for r in pair.values())
        checks.append({"query_id": q["id"], "complete": True, "same_common_context": fields_equal,
            "same_system_schema": same_system, "same_request_options": same_options,
            "all_six_tools_in_every_sent_request": six_tools, "input_isolation_verified": isolation,
            "graph_initial_context_chars": len(requests["graph"][0]["messages"][1]["content"]),
            "no_graph_initial_context_chars": len(requests["no_graph"][0]["messages"][1]["content"]),
            "valid_gemini_pair": isolation and all(r["valid_for_comparison"] for r in pair.values())})
    summary = {"questions": len(experiment["questions"]), "completed_runs": len(rows),
        "model": experiment["model"],
        "input_isolation_verified": len(checks) == 3 and all(c["input_isolation_verified"] for c in checks),
        "valid_gemini_pairs": sum(c["valid_gemini_pair"] for c in checks), "provider_failures": dict(failures),
        "decision_rejections": dict(rejections),
        "question_checks": checks, "runs": rows,
        "quality_conclusions": "None: one trial and no independent relevance judgments; failed cells are excluded."}
    # Retain the legacy key for older Gemini pilot artifacts and callers.
    summary["valid_provider_pairs"] = summary["valid_gemini_pairs"]
    saved["runs"] = rows
    atomic_json(output/"results.json", saved)
    temporary = output/"metrics.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output/"metrics.csv")
    atomic_json(output/"audit.json", summary)
    lines = ["# Preliminary three-question graph pilot", "",
        f"Model: {experiment['model']}. Completed runs: {len(rows)}/6. Valid hosted-planner pairs: {summary['valid_provider_pairs']}/3.",
        f"Input isolation verified: {summary['input_isolation_verified']}.",
        f"Provider failures: {dict(failures)}.", "",
        "Failed cells include deterministic fallback and are invalid as hosted-planner behavior evidence.",
        "No retrieval-quality or graph-superiority conclusion is supported.", "",
        "| Question | Arm | Valid hosted cell | Actions | Inspected (includes fallback) | Fallbacks | Reported input tokens |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        tokens = r["input_tokens"] if r["reported_usage_calls"] else "unreported"
        lines.append(f"| {r['query_id']} | {r['arm']} | {r['valid_for_comparison']} | {r['actions']} | {r['inspected']} | {r['fallbacks']} | {tokens} |")
    lines += ["", "First actual request comparison (same observed metadata before decisions diverge):", "",
        "| Question | Same common input | Same system/schema | Six tools offered | Graph / no-graph context chars |",
        "| --- | --- | --- | --- | --- |"]
    for c in checks:
        if c["complete"]:
            lines.append(f"| {c['query_id']} | {c['same_common_context']} | {c['same_system_schema']} | {c['all_six_tools_in_every_sent_request']} | {c['graph_initial_context_chars']} / {c['no_graph_initial_context_chars']} |")
    (output/"report.md").write_text("\n".join(lines)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="benchmarks/graph_pilot_questions.json")
    parser.add_argument("--corpus", help="Reuse a frozen corpus; otherwise acquire a small shared OpenAlex sample")
    parser.add_argument("--output", required=True)
    parser.add_argument("--analyze-only", action="store_true", help="Audit saved requests/results without API calls")
    args = parser.parse_args()
    if args.analyze_only:
        summary = analyze_pilot(args.output)
        print(json.dumps({k: v for k, v in summary.items() if k not in ("runs", "question_checks")}))
        return
    settings = Settings.from_env()
    questions = json.loads(Path(args.questions).read_text())
    if len(questions) != 3 or len({q["id"] for q in questions}) != 3:
        raise ValueError("This pilot requires exactly three distinct questions")
    output = Path(args.output)
    if (output/"experiment.json").exists():
        raise ValueError("Use a fresh pilot output directory")
    corpus_path = args.corpus or str(output/"corpus")
    if not args.corpus:
        output.mkdir(parents=True, exist_ok=True)
        from dataclasses import replace
        with Database(str(output/"acquisition.sqlite")) as db:
            acquire_pilot(questions, db, replace(settings, max_requests=min(settings.max_requests, 20)), corpus_path)
    print("Both arms use the same frozen OpenAlex corpus; live retrieval during the paired runs is unavailable.", flush=True)
    run_pilot(corpus_path, questions, settings, output)
    analyze_pilot(output)


if __name__ == "__main__":
    main()
