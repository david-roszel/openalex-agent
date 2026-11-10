"""Factorial graph/context comparison with equal metadata and retrieval tools."""

from collections import Counter
import json

from .actions import ACTION_ARGUMENTS, Action, make_action, RELIABILITY_VERSION
from .context_planner import ContextPlanner
from .controller import Controller
from .corpus import atomic_json
from .ranking import relevance
from .selection import score_candidates

ARCHITECTURES = ("standard", "llm", "llm_graph", "clm", "clm_graph")
META_ARCHITECTURES = ("meta_clm", "meta_clm_graph")
ALL_ARCHITECTURES = ARCHITECTURES + META_ARCHITECTURES
GRAPH_FIELDS = {"graph", "candidates", "subgraph", "topic_coverage"}


def context_strategy(architecture):
    return ("none" if architecture == "standard" else "meta_clm" if architecture.startswith("meta_clm")
            else "clm" if architecture.startswith("clm") else "conventional")


class ControlledController(Controller):
    """Reuse execution, transactions, ranking and limits; isolate graph observations.

    Unlike the legacy graph method, all action eligibility comes from raw
    observed metadata and an identical acquisition-depth ledger in both arms.
    Graph information affects model decisions, not the availability of tools.
    """

    def __init__(self, *args, architecture: str, graph_policy: str = "legacy-v1", **kwargs):
        if architecture not in ALL_ARCHITECTURES[1:] or kwargs.get("resume"):
            raise ValueError("Choose a controlled agent architecture; interrupted research runs require fresh replay")
        if graph_policy not in ("legacy-v1", "graph-visible-v2"):
            raise ValueError("Unknown graph evidence policy")
        self.graph_policy = graph_policy
        self._graph_registry = {}
        self._graph_allocation = {}
        self.architecture = architecture
        self.depths = {}
        super().__init__(*args, method="graph" if architecture.endswith("_graph") else "iterative", **kwargs)
        if isinstance(self.planner, ContextPlanner):
            expected = context_strategy(architecture)
            if self.planner.mode != expected:
                raise ValueError("Planner context policy does not match architecture")
            self.planner.bind(self.directory, self.question, self.budget, self._log, lambda: self.papers)
            if self.graph is not None and self.graph_policy == "graph-visible-v2":
                self.planner.graph_evidence_resolver = lambda: {"architecture": self.architecture, "mode": self.planner.mode,
                    "registry": self._graph_registry, "allocation": self._graph_allocation}

    def _log(self, kind, payload):
        super()._log(kind, {**payload, "architecture": self.architecture})

    def _eligible(self):
        scores = relevance(self.question, list(self.papers.values()))
        anchors = {p.id for p in self.papers.values() if scores[p.id] >= self.scoring.min_relevance
                   and self.depths.get(p.id, 0) < self.scoring.max_depth}
        known = {target for parent in anchors for target in self.papers[parent].references + self.papers[parent].related} - self.papers.keys()
        topics = {t.id for parent in anchors for t in self.papers[parent].topics}
        return anchors, known, topics

    def _common_context(self):
        scores = relevance(self.question, list(self.papers.values()))
        papers = sorted(self.papers.values(), key=lambda p: (-scores[p.id], p.id))[:8]
        anchors, targets, topics = self._eligible()
        groups = []
        for name, ids in (("EXPAND_REFERENCES", anchors), ("EXPAND_CITATIONS", anchors),
                          ("INSPECT_PAPER", targets), ("EXPLORE_TOPIC", topics)):
            groups.append([make_action(name, identifier, "Observed metadata acquisition") for identifier in sorted(ids)
                           if make_action(name, identifier).signature not in self.completed][:4])
        proposals = [group[i] for i in range(4) for group in groups if len(group) > i]
        feedback = self.search_feedback()
        paused = feedback["consecutive_searches_without_new_papers"] >= 2 and bool(proposals)
        if not paused:
            proposals += self._search_proposals()[:4]
        if not proposals:
            proposals = [make_action("STOP_RETRIEVAL", reason="No unexplored acquisition proposals")]
        result = {"question": self.question, "round": self.rounds, "retrieval": self.retrieval_environment(),
            "remaining": {"actions": self.budget.max_actions-self.actions,
                          "inspections": self.budget.max_inspected-len(self.papers),
                          "api_requests": self.budget.max_api_requests-self.retriever.usage.requests},
            "permitted_actions": [a for a in ACTION_ARGUMENTS if a != "SEARCH_PAPERS" or not paused],
            "proposals": [a.to_dict() for a in proposals[:12]], "search_feedback": feedback,
            "counts": {"retrieved": len(self.retrieved_ids), "inspected": len(self.papers)},
            "top_papers": [{"id": p.id, "title": (p.title or "")[:200], "abstract_excerpt": (p.abstract or "")[:350],
                            "original_query_relevance": scores[p.id]} for p in papers],
            "observed_metadata": [{"id": p.id, "references": p.references[:8], "related": p.related[:4],
                "authors": [{"id": a.author.id, "name": a.author.name} for a in p.authorships[:4] if a.author],
                "topics": [{"id": t.id, "name": t.name} for t in p.topics[:4]], "year": p.year} for p in papers],
            "recent_actions": self.history[-4:], "last_observation": ({k: self.trace[-1].get(k) for k in
                ("new_papers", "updated_papers", "returned_papers", "new_promising")} if self.trace else {})}
        # Keep the original top-eight/recent-four selection policy. Reserve only
        # envelope/archive headroom, equally in all four cells, rather than
        # deliberately restricting the conventional arm to half its capacity.
        limit = self.budget.max_context_chars-1024
        def exceeds():
            if len(json.dumps(result, ensure_ascii=False)) > limit:
                return True
            if isinstance(self.planner, ContextPlanner) and self.planner.limits.max_active_tokens:
                from .token_budget import count
                from .context_memory import serialized
                return count(serialized(self.planner._parts(result)[1])) > self.planner.limits.max_active_tokens
            return False
        while exceeds():
            for key in ("recent_actions", "observed_metadata", "top_papers", "proposals"):
                if len(result[key]) > (1 if key == "proposals" else 0):
                    result[key].pop()
                    break
            else:
                if result["search_feedback"]["recent_searches"]:
                    result["search_feedback"]["recent_searches"].pop(0)
                else:
                    raise ValueError("Protected common context exceeds capacity")
        return result

    def context(self):
        if self.graph is not None and self.graph_policy == "graph-visible-v2":
            from .graph_evidence import pack
            return pack(self)
        context = self._common_context()
        if self.graph is not None:
            seeds = [p["id"] for p in context["top_papers"][:3]]
            view = self.graph.neighborhood(seeds, max_nodes=12, relations=["PAPER_CITES_PAPER"])
            context.update(graph=self.graph.stats(), candidates=score_candidates(self.question, self.papers, self.graph, self.scoring)[:8],
                subgraph=[{k: e[k] for k in ("source", "target", "relation")} for e in view["edges"][:12]],
                topic_coverage=dict(Counter(t.id for p in self.papers.values() for t in p.topics).most_common(8)))
            def exceeds():
                if len(json.dumps(context, ensure_ascii=False)) > self.budget.max_context_chars-512:
                    return True
                if isinstance(self.planner, ContextPlanner) and self.planner.limits.max_active_tokens:
                    from .token_budget import count
                    from .context_memory import serialized
                    return count(serialized(self.planner._parts(context)[1])) > self.planner.limits.max_active_tokens
                return False
            while exceeds():
                for key in ("subgraph", "candidates"):
                    if context[key]:
                        context[key].pop()
                        break
                else:
                    raise ValueError("Graph context exceeds capacity")
        return context

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
        anchors, known, topics = self._eligible()
        valid = (action.arguments["paper_id"] in anchors if action.action in ("EXPAND_REFERENCES", "EXPAND_CITATIONS") else
                 action.arguments["paper_id"] in known if action.action == "INSPECT_PAPER" else action.arguments["topic_id"] in topics)
        if not valid:
            raise ValueError("Target fails observed metadata, relevance or common path-depth guard")

    def _execute(self, action, *, single_pass=False):
        previous = set(self.papers)
        parent = action.arguments.get("paper_id")
        if action.action == "INSPECT_PAPER":
            sources = [p.id for p in self.papers.values() if parent in p.references + p.related]
            parent = min(sources, key=lambda p: (self.depths.get(p, 0), p)) if sources else None
        elif action.action == "EXPLORE_TOPIC":
            sources = [p.id for p in self.papers.values() if any(t.id == action.arguments["topic_id"] for t in p.topics)
                       and p.id in self._eligible()[0]]
            parent = min(sources, key=lambda p: (self.depths.get(p, 0), p)) if sources else None
        super()._execute(action, single_pass=single_pass)
        for identifier in self.papers.keys()-previous:
            self.depths[identifier] = self.depths.get(parent, 0)+1 if parent else 0

    def _export_result(self, seconds):
        result = super()._export_result(seconds)
        result.update(architecture=self.architecture, graph_enabled=self.graph is not None,
            experimental_implementation_version=RELIABILITY_VERSION,
            context_management=context_strategy(self.architecture),
            controlled_policy="matched-six-tools-v1")
        if isinstance(self.planner, ContextPlanner):
            if self.architecture not in META_ARCHITECTURES:
                result.update(experimental_implementation_version=self.planner.configuration["context_policy"],
                    context_policy_version=self.planner.configuration["context_policy"],
                    pending_observation_chars=len(self.planner._pending),
                    unadmitted_observation_indices=list(self.planner._pending_indices))
            records = self.planner.store.edits
            result.update(context_edits=sum(r["source"] == "model" and r["accepted"] for r in records),
                rejected_context_edits=sum(r["source"] == "model" and not r["accepted"] for r in records),
                context_snapshots=len(records), active_context_chars=len(self.planner.store.read()),
                inference_limits=self.planner.limits.__dict__, evidence_records=self.planner.evidence_records,
                context_adapter_local_seconds=self.planner.context_local_seconds)
            if self.architecture in META_ARCHITECTURES:
                from .meta_context import META_VERSION
                result.update(experimental_implementation_version=META_VERSION,
                    meta_reference_revision=self.planner.configuration["reference_revision"],
                    meta_synchronization_seconds=self.planner.sync_seconds,
                    meta_reference_syncs=self.planner.store.env.n_ctx_syncs,
                    meta_reference_meaningful_syncs=self.planner.store.env.n_ctx_syncs_real,
                    meta_common_limit_rollbacks=self.planner.common_limit_rollbacks,
                    meta_reference_rollbacks=self.planner.store.budget.n_retry_on_limit,
                    meta_recovery_calls=self.planner.recovery_calls,
                    meta_sandbox=self.planner.configuration["sandbox"])
            if self.planner.token_limited:
                measurements = [json.loads(p.read_text())["study_tokens"] for p in
                                sorted(self.planner.store.directory.glob("request-*.json"))]
                result.update(study_tokenizer=self.planner.configuration["study_tokenizer"],
                    active_context_tokens=measurements[-1]["active_tokens"] if measurements else None,
                    max_active_context_tokens=max((m["active_tokens"]+m["recovery_tokens"] for m in measurements),default=None),
                    max_full_input_tokens=max((m["full_input_tokens"] for m in measurements),default=None))
        if self.graph is not None and self.graph_policy == "graph-visible-v2":
            result.update(graph_evidence_policy=self.graph_policy, graph_allocation_last=self._graph_allocation)
        atomic_json(self.directory / "result.json", result)
        return result
