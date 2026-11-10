"""Committed retrieval memory; observation batches replace this atomically."""

from dataclasses import dataclass, field
from typing import Any
from .actions import Action
from .graph import ScholarlyGraph
from .models import Paper, canonical_id


@dataclass(frozen=True)
class AgentMemory:
    graph: ScholarlyGraph | None = None
    papers: dict[str, Paper] = field(default_factory=dict)
    retrieved_ids: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)
    history: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    pending: dict | None = None
    actions: int = 0
    rounds: int = 0
    stagnant: int = 0
    stop_reason: str = ""

    @classmethod
    def restore(cls, data: dict[str, Any], *, graph_enabled: bool) -> "AgentMemory":
        """Reject inconsistent checkpoints instead of silently repairing state."""
        try:
            papers = [Paper.from_dict(p) for p in data["papers"]]
            by_id = {p.id: p for p in papers}
            if len(by_id) != len(papers) or any(canonical_id(p.id) != p.id for p in papers):
                raise ValueError("Invalid checkpoint paper IDs")
            retrieved = set(data["retrieved_ids"])
            if not isinstance(data["retrieved_ids"], list) or len(retrieved) != len(data["retrieved_ids"]):
                raise ValueError("Duplicate or malformed retrieved IDs")
            if any(canonical_id(i) != i for i in retrieved) or not set(by_id) <= retrieved:
                raise ValueError("Inconsistent retrieved paper set")
            history = [Action.parse(a).to_dict() for a in data["history"]]
            if any(a["action"] == "STOP_RETRIEVAL" for a in history):
                raise ValueError("Stop decisions cannot count as retrieval actions")
            signatures = [Action.parse(a).signature for a in history]
            completed = set(data["completed"])
            if not isinstance(data["completed"], list) or len(completed) != len(data["completed"]):
                raise ValueError("Duplicate or malformed completed actions")
            pending = Action.parse(data["pending"]).to_dict() if data.get("pending") else None
            for key in ("actions", "rounds", "stagnant"):
                if type(data[key]) is not int or data[key] < 0:
                    raise ValueError("Invalid action counters")
            if data["actions"] != len(history) or data["rounds"] != data["actions"] or len(signatures) != len(set(signatures)):
                raise ValueError("Inconsistent action history")
            expected = signatures[:-1] if pending else signatures
            if pending and (not history or Action.parse(pending).signature != signatures[-1]):
                raise ValueError("Pending action does not match history")
            if completed != set(expected) or len(data["trace"]) != len(completed):
                raise ValueError("Inconsistent observation commits")
            if not isinstance(data["trace"], list):
                raise ValueError("Invalid observation trace")
            for index, point in enumerate(data["trace"], 1):
                if not isinstance(point, dict) or point.get("actions") != index or point.get("round") != index:
                    raise ValueError("Invalid observation trace counters")
                if not isinstance(point.get("inspected_ids"), list) or not set(point["inspected_ids"]) <= set(by_id):
                    raise ValueError("Trace contains unknown inspected papers")
            if data["trace"] and set(data["trace"][-1]["inspected_ids"]) != set(by_id):
                raise ValueError("Latest observation does not match committed memory")
            graph_data = data.get("graph_data")
            if graph_enabled != (graph_data is not None):
                raise ValueError("Graph memory does not match retrieval method")
            graph = ScholarlyGraph.from_dict(graph_data) if graph_enabled else None
            if graph:
                resolved = {n for n, a in graph.graph.nodes(data=True) if a.get("resolved")}
                if resolved != set(by_id):
                    raise ValueError("Graph contains unobserved papers or is missing observations")
                # Old checkpoints may have structural graph nodes without their
                # observations; the checkpoint paper set can hydrate them offline.
                for node in sorted(resolved):
                    if graph.paper(node) is None:
                        attrs = graph.graph.nodes[node]
                        graph.add(by_id[node], action="checkpoint_hydration", parent=attrs.get("parent"), depth=attrs["depth"])
            reason = data.get("stop_reason", "")
            allowed = {"", "single_pass", "planner_stop", "max_actions", "max_rounds", "max_inspected",
                       "max_api_requests", "max_llm_calls", "minimal_new_relevant_information", "retrieval_error", "budget_exhausted"}
            if reason not in allowed:
                raise ValueError("Unknown checkpoint stopping reason")
            return cls(graph=graph, papers=by_id, retrieved_ids=retrieved, completed=completed, history=history,
                trace=data["trace"], pending=pending, actions=data["actions"], rounds=data["rounds"], stagnant=data["stagnant"],
                stop_reason="" if reason == "retrieval_error" else reason)
        except (KeyError, TypeError, AttributeError):
            raise ValueError("Malformed agent memory checkpoint") from None
