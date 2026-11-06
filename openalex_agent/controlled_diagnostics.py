"""Seeded synthetic graph/context diagnostics, never a scientific quality study."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re

from .actions import make_action
from .context_memory import serialized
from .context_planner import ContextPlanner, EngineeringContextModel, InferenceLimits
from .controlled import ControlledController, GRAPH_FIELDS
from .controller import Budget
from .corpus import LocalCorpus, atomic_json, freeze
from .evaluation import metrics
from .models import Paper
from .openalex import Usage
from .research_metrics import context_metrics, retrieval_diagnostics
from .selection import Scoring
from .storage import Database


def fixture(case: str):
    """Labels belong to the evaluator; the policy receives only observations."""
    def work(number, title, refs=(), abstract=None):
        p = Paper.from_openalex({"id": f"W{number}", "title": "SYNTHETIC " + title,
            "referenced_works": [f"W{i}" for i in refs]}, provenance="SYNTHETIC DIAGNOSTIC FIXTURE ONLY")
        p.abstract = abstract
        return p
    if case == "G1":
        papers = [work(1, "retrieval therapy anchor", [2]), work(2, "retrieval bridge", [3]),
                  work(3, "zephyr evidence", abstract="Direct controlled therapeutic outcome evidence"),
                  *[work(i, "retrieval therapy retrieval therapy distractor") for i in range(4, 8)]]
        return "retrieval therapy", papers, [1], {1: 1, 2: 1, 3: 3}, 3
    if case in ("G2", "G5"):
        papers = [work(1, "transformer fault monitoring seed", [2, 4]),
                  work(5, "transformer fault monitoring case", [4]),
                  work(2, "transformer fault monitoring language model", [4]),
                  work(4, "oil insulation diagnosis", [2])]
        return "transformer fault monitoring", papers, [1, 5], {1: 2, 5: 1, 4: 3}, 2
    if case == "G3":
        papers = [work(1, "retrieval anchor", [2, 2, 3]), work(2, "retrieval duplicate paths", [1, 3, 3]),
                  work(3, "retrieval productive alternative", [1])]
        return "retrieval", papers, [1], {1: 1, 3: 3}, 3
    raise ValueError("Unknown graph diagnostic")


class SeededCorpus(LocalCorpus):
    """Identical declared initial observations; later search retains the full corpus."""
    def __init__(self, directory, seeds):
        super().__init__(directory)
        self.seeds, self.initial = seeds, True

    def search(self, query, limit=20):
        if self.initial:
            self.initial = False
            return [self.by_id[f"https://openalex.org/W{i}"] for i in self.seeds][:limit]
        return super().search(query, limit)


class GraphDiagnosticModel(EngineeringContextModel):
    """A declared deterministic decision rule; not evidence of hosted intelligence."""
    def _call(self, system, payload):
        body = payload.get("observation", {})
        proposals = payload["protected"]["proposals"]
        inspections = {p["arguments"]["paper_id"]: p for p in proposals if p["action"] == "INSPECT_PAPER"}
        for candidate in body.get("candidates", []):
            if not candidate["resolved"] and candidate["id"] in inspections:
                return self.respond(system, payload, {"decision": inspections[candidate["id"]]})
        if inspections:
            return self.respond(system, payload, {"decision": inspections[sorted(inspections)[0]]})
        return self.respond(system, payload, {"decision": proposals[0]})


class UninformativeController(ControlledController):
    def context(self):
        result = super().context()
        if self.graph is not None:
            # Deliberately uninformative derived view, only in this synthetic diagnostic.
            # Authoritative graph snapshots and OpenAlex citation evidence are unchanged.
            result.update(candidates=[], subgraph=[], topic_coverage={})
        return result


def run_graph_case(case: str, directory: Path) -> dict:
    question, papers, seeds, positives, actions = fixture(case)
    corpus = directory / "corpus"
    freeze(papers, corpus, acquisition=[{"source": "SYNTHETIC DIAGNOSTIC ONLY", "initial_seed_ids": seeds}])
    labels = {p.id: positives.get(int(p.id.rsplit("W", 1)[-1]), 0) for p in papers}
    rows, inputs = [], {}
    for arm in ("llm", "llm_graph"):
        model = ContextPlanner(GraphDiagnosticModel(), mode="conventional")
        with Database(":memory:") as db:
            cls = UninformativeController if case == "G5" else ControlledController
            controller = cls(question, SeededCorpus(corpus, seeds), db, architecture=arm, planner=model,
                budget=Budget(max_actions=actions, max_rounds=actions, page_size=2, output_k=10,
                              max_llm_calls=8, max_inspected=10), scoring=Scoring(min_relevance=0), output=str(directory / "runs"))
            result = controller.run()
            first = json.loads((model.store.directory / "request-00001.json").read_text())["input"]
            inputs[arm] = first
            row = {"method": arm, "actions": result["actions"], "history": controller.history,
                   "retrieved_ids": result["retrieved_ids"], "ranked_ids": [r["paper"]["id"] for r in result["ranked"]],
                   "graph_memory_present": controller.graph is not None, "directory": result["directory"],
                   **metrics([r["paper"]["id"] for r in result["ranked"]], labels, k=10, scope="exhaustive"),
                   **retrieval_diagnostics(result["inspected_ids"], labels, result["actions"])}
            row["relevant_discovery_actions"] = {identifier: next((point["actions"] for point in result["trace"] if identifier in point["inspected_ids"]), None)
                                                  for identifier, grade in labels.items() if grade > 0}
            rows.append(row)
    flat, graph = inputs["llm"], inputs["llm_graph"]
    graph_body = {k: v for k, v in graph["observation"].items() if k not in GRAPH_FIELDS}
    isolation = flat["protected"] == graph["protected"] and flat["observation"] == graph_body
    record = {"case": case, "kind": "SYNTHETIC SCRIPTED DIAGNOSTIC; NOT LLM EFFECTIVENESS", "rows": rows,
              "G4_matched_input_and_tools": isolation, "labels_used_by_policy": False,
              "behavior_changed": rows[0]["history"] != rows[1]["history"],
              "negative_control": case == "G5"}
    atomic_json(directory / "diagnostic.json", record)
    return record


class StreamRetriever:
    fingerprint = "SYNTHETIC-RETENTION-STREAM-v1"

    def __init__(self, horizon):
        self.usage, self.number, self.horizon = Usage(), 0, horizon

    def search(self, query, limit=20):
        self.number += 1
        p = Paper.from_openalex({"id": f"W{self.number}", "title": "retrieval evidence" if self.number == 1 else "retrieval"},
                                provenance="SYNTHETIC RETENTION STREAM ONLY")
        p.abstract = "FACT[W1:early-evidence-73]" if self.number == 1 else "irrelevant filler " + str(self.number)
        return [p]


class RetentionModel(EngineeringContextModel):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def _call(self, system, payload):
        protected = payload["protected"]
        if protected["context_management"] == "clm":
            active = payload["active_context"]
            # Derive retained evidence from actual request text, never the fixture oracle.
            needle = re.search(r"FACT\[[^\]]+\]", active)
            if needle and (len(active) > 1000 or "top_papers" in active):
                return self.respond(system, payload, {"context_edit": {"path": "active_context.txt", "op": "replace", "text": needle.group(0)}})
        if protected["round"] >= self.horizon:
            text = payload.get("active_context", serialized(payload.get("observation", {})))
            found = re.search(r"FACT\[([^:\]]+):[^\]]+\]", text)
            claim = {"statement": found.group(0) if found else "Insufficient retained evidence",
                     "work_id": found.group(1) if found else None, "evidence_quote": found.group(0) if found else "",
                     "insufficient": not bool(found)}
            return self.respond(system, payload, {"decision": {"action": "STOP_RETRIEVAL", "arguments": {}}, "evidence_records": [claim]})
        return self.respond(system, payload, {"decision": make_action("SEARCH_PAPERS", "stream observation " + str(protected["round"])).to_dict()})


def run_retention(directory: Path, *, horizon=24, max_chars=4000) -> dict:
    rows = []
    for mode in ("conventional", "clm"):
        base = RetentionModel(horizon)
        planner = ContextPlanner(base, mode=mode, limits=InferenceLimits(max_total_tokens=10000000))
        with Database(":memory:") as db:
            controller = ControlledController("retrieval", StreamRetriever(horizon), db,
                architecture="clm" if mode == "clm" else "llm", planner=planner,
                budget=Budget(max_actions=horizon+1, max_rounds=horizon+1, max_inspected=horizon+2,
                    page_size=1, max_llm_calls=horizon*3, patience=horizon+1, max_context_chars=max_chars), output=str(directory / mode))
            result = controller.run()
            request_files = sorted(planner.store.directory.glob("request-*.json"))
            last = json.loads(request_files[-1].read_text())["input"]
            active = last["active_context"] if mode == "clm" else serialized(last["observation"])
            rows.append({"mode": mode, "horizon": horizon, "context_limit": max_chars,
                "request_count": planner.usage.calls, "estimated_input_tokens": planner.usage.estimated_input_tokens,
                "estimated_output_tokens": planner.usage.estimated_output_tokens,
                "actions": result["actions"], "stop_reason": result["stop_reason"],
                **context_metrics(["FACT[W1:early-evidence-73]"], active, edits=planner.store.edits),
                "correct_fact_recovery": int(any(r["statement"] == "FACT[W1:early-evidence-73]" and not r["insufficient"] for r in result["evidence_records"])),
                "unsupported_claims": sum(not r["insufficient"] and r["statement"] != "FACT[W1:early-evidence-73]" for r in result["evidence_records"]),
                "abstentions": sum(r["insufficient"] for r in result["evidence_records"]),
                "every_request_within_limit": all(len(serialized(json.loads(p.read_text())["input"])) <= max_chars for p in request_files)})
    record = {"kind": "SYNTHETIC SCRIPTED RETENTION DIAGNOSTIC; NOT HOSTED EFFECTIVENESS", "rows": rows}
    atomic_json(directory / "diagnostic.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--saturation-observations", type=int, default=220)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a fresh diagnostics directory")
    args.output.mkdir(parents=True)
    graph = [run_graph_case(case, args.output / case) for case in ("G1", "G2", "G3", "G5")]
    context = [run_retention(args.output / f"C1-h{h}", horizon=h) for h in (12, 36)]
    context += [run_retention(args.output / f"C3-c{cap}", horizon=args.saturation_observations, max_chars=cap) for cap in (4000, 6000)]
    atomic_json(args.output / "diagnostics.json", {"graph": graph, "context": context,
        "note": "G4 isolation is audited in every graph case. C2/C4/C5/C6/C7 are separately tested with mocked hosted responses. No effectiveness claim."})
    print(serialized({"output": str(args.output), "graph_cases": len(graph), "context_cases": len(context),
                      "input_isolation": all(r["G4_matched_input_and_tools"] for r in graph)}))


if __name__ == "__main__":
    main()
