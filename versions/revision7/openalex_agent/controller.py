"""Observable autonomous controller with graph-free and graph-guided modes."""

from collections import Counter
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import time
import uuid
from .actions import Action, make_action
from .agent_state import AgentMemory
from .corpus import Retriever, LocalCorpus, atomic_json
from .graph import ScholarlyGraph
from .llm import DeterministicPlanner, Planner, PlannerError
from .models import Paper, utcnow, canonical_id, merge_observation
from .openalex import BudgetExceeded, OpenAlexClient, RetrievalError
from .ranking import relevance
from .selection import SELECTION_VERSION, Scoring, score_candidates
from .storage import Database

METHODS = ("standard", "reformulated", "iterative", "graph")


@dataclass(frozen=True)
class Budget:
    max_actions: int = 20
    max_rounds: int = 10
    max_inspected: int = 200
    max_api_requests: int = 50
    max_llm_calls: int = 20
    page_size: int = 20
    output_k: int = 20
    patience: int = 3
    min_new_relevant: int = 1
    max_context_chars: int = 16000

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in asdict(self).values()):
            raise ValueError("All budget values must be positive integers")
        if self.page_size > 100 or self.max_context_chars < 4000:
            raise ValueError("page_size must be <=100; context must allow at least 4000 characters")


class Controller:
    def __init__(self, question: str, retriever: Retriever, db: Database, *, method: str = "graph",
                 planner: Planner | None = None, budget: Budget | None = None, scoring: Scoring | None = None,
                 output: str = "data/results", resume: str | None = None):
        if method not in METHODS or not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError("Choose a valid method and a question of 1–1000 characters")
        self.question, self.retriever, self.db, self.method = question, retriever, db, method
        self.planner, self.budget, self.scoring = planner or DeterministicPlanner(), budget or Budget(), scoring or Scoring()
        self.run_id = str(uuid.uuid4())
        self.directory = Path(output) / self.run_id
        self._memory = AgentMemory(graph=ScholarlyGraph() if method == "graph" else None)
        self.prior_seconds = 0.0
        self._event_seq = 0
        self._has_run = False
        if resume:
            self._restore(Path(resume))
        self.directory.mkdir(parents=True, exist_ok=True)
        if isinstance(retriever, OpenAlexClient):
            retriever.run_id = self.run_id
            retriever.settings.max_requests = min(retriever.settings.max_requests, self.budget.max_api_requests)
        self._log("start" if not resume else "resume", {"question": self.question, "method": self.method,
            "planner": self.planner.name, "model": self.planner.model, "budget": asdict(self.budget),
            "scoring": asdict(self.scoring), "selection_version": SELECTION_VERSION if self.graph else None,
            "corpus": getattr(self.retriever, "fingerprint", "live:openalex"),
            "retrieval": self.retrieval_environment(),
            "provider_configuration": getattr(self.planner, "configuration", {})})

    def retrieval_environment(self) -> dict:
        if isinstance(self.retriever, LocalCorpus):
            return {"mode": "frozen_corpus", "live_openalex_available": False,
                "corpus_papers": len(self.retriever.papers),
                "description": "Only this fixed corpus is searchable. Live OpenAlex retrieval is unavailable. "
                    "Search uses lexical similarity; Boolean OR and quoted phrase operators are not interpreted. "
                    "Empty results do not establish absence from OpenAlex."}
        if isinstance(self.retriever, OpenAlexClient):
            return {"mode": "openalex", "live_openalex_available": True,
                "description": "OpenAlex API retrieval is enabled within request limits; cached responses may be reused."}
        return {"mode": "custom", "live_openalex_available": None,
                "description": "Custom retriever; live OpenAlex access is unspecified."}

    def search_feedback(self) -> dict:
        # Traces contain committed outcomes only; a pending action must not
        # acquire an invented result when resuming an interrupted run.
        completed = list(zip(self.history, self.trace))
        streak = 0
        for action, point in reversed(completed):
            if action["action"] != "SEARCH_PAPERS" or point["new_papers"] > 0:
                break
            streak += 1
        recent = [{"query": action["arguments"]["query"][:300],
                   "returned_papers": point.get("returned_papers"), "new_papers": point["new_papers"]}
                  for action, point in completed if action["action"] == "SEARCH_PAPERS"][-4:]
        return {"consecutive_searches_without_new_papers": streak, "recent_searches": recent,
                "guidance": "Choose another permitted acquisition action or STOP after repeated unproductive searches. "
                    "If search is the only option, broaden using fewer original-question terms. "
                    "A missing returned_papers count is unknown, not zero."}

    def _log_planner_call(self, stage: str) -> None:
        if getattr(self.planner, "last_call", None):
            self._log("planner_call", {"stage": stage, **self.planner.last_call})

    @property
    def graph(self) -> ScholarlyGraph | None:
        return self._memory.graph

    @property
    def papers(self) -> dict[str, Paper]:
        return self._memory.papers

    @property
    def retrieved_ids(self) -> set[str]:
        return self._memory.retrieved_ids

    @property
    def completed(self) -> set[str]:
        return self._memory.completed

    @property
    def history(self) -> list[dict]:
        return self._memory.history

    @property
    def trace(self) -> list[dict]:
        return self._memory.trace

    @property
    def pending(self) -> dict | None:
        return self._memory.pending

    @property
    def actions(self) -> int:
        return self._memory.actions

    @property
    def rounds(self) -> int:
        return self._memory.rounds

    @property
    def stagnant(self) -> int:
        return self._memory.stagnant

    @property
    def stop_reason(self) -> str:
        return self._memory.stop_reason

    def _log(self, kind: str, payload: dict) -> None:
        self._event_seq += 1
        record = {"run_id": self.run_id, "event_seq": self._event_seq, "timestamp": utcnow(), **payload}
        self.db.event(self.run_id, kind, record)
        with (self.directory / "events.jsonl").open("ab+") as stream:
            if stream.tell():
                stream.seek(-1, 2)
                if stream.read(1) != b"\n":
                    stream.write(b"\n")
            stream.write((json.dumps({"kind": kind, **record}, ensure_ascii=False) + "\n").encode())

    def _restore(self, directory: Path) -> None:
        try:
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            if not isinstance(state, dict) or state.get("checkpoint_schema", 1) not in (1, 2):
                raise ValueError("Unsupported agent checkpoint version")
            uuid.UUID(state["run_id"])
            if state["question"] != self.question or state["method"] != self.method or state["budget"] != asdict(self.budget) or state["scoring"] != asdict(self.scoring):
                raise ValueError("Resume requires the original question, method, budget and scoring")
            if self.method == "graph" and state.get("selection_version") != SELECTION_VERSION:
                raise ValueError("Resume requires the original graph selection version; start a fresh run")
            if state["corpus"] != getattr(self.retriever, "fingerprint", "live:openalex") or state["model"] != self.planner.model or state["planner"] != self.planner.name:
                raise ValueError("Resume requires the same corpus and planner/model")
            fingerprint = getattr(self.planner, "fingerprint", None)
            if state.get("provider_fingerprint") != fingerprint and (state.get("provider_fingerprint") or self.planner.name != "deterministic"):
                raise ValueError("Resume requires the same provider endpoint and generation settings")
            if state.get("finished"):
                raise ValueError("Run already finished; create a fresh run")
            memory = AgentMemory.restore(state, graph_enabled=self.method == "graph")
            if memory.actions > self.budget.max_actions or memory.rounds > self.budget.max_rounds or len(memory.papers) > self.budget.max_inspected:
                raise ValueError("Checkpoint exceeds its recorded budget")
            self.run_id, self.directory, self._memory = state["run_id"], directory, memory
            self.prior_seconds = state["seconds"]
            import math
            if isinstance(self.prior_seconds, bool) or not isinstance(self.prior_seconds, (int, float)) or not math.isfinite(self.prior_seconds) or self.prior_seconds < 0:
                raise ValueError("Invalid checkpoint elapsed time")
            for usage, values in ((self.retriever.usage, state["api_usage"]), (self.planner.usage, state["llm_usage"])):
                known = asdict(usage)
                if not isinstance(values, dict) or set(values) - known.keys():
                    raise ValueError("Invalid checkpoint usage fields")
                for key, value in values.items():
                    if value is None and key in ("remaining", "estimated_cost_usd"):
                        continue
                    numeric = key.endswith("seconds") or key.endswith("_usd") or key in ("remaining", "reported_credits")
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value) or not numeric and type(value) is not int:
                        raise ValueError("Invalid checkpoint usage counters")
                for key, value in values.items():
                    setattr(usage, key, value)
            if self.retriever.usage.requests > self.budget.max_api_requests or self.planner.usage.calls > self.budget.max_llm_calls:
                raise ValueError("Checkpoint usage exceeds its budget")
            self._event_seq = state.get("event_seq", 0)
            if type(self._event_seq) is not int or self._event_seq < 0:
                raise ValueError("Invalid checkpoint event sequence")
            sequences = [event.get("event_seq", 0) for event in self.db.events(self.run_id)]
            log = directory / "events.jsonl"
            if log.exists():
                for line in log.read_text(encoding="utf-8").splitlines():
                    try:
                        sequences.append(json.loads(line).get("event_seq", 0))
                    except (ValueError, AttributeError):
                        continue  # A final log write can be interrupted; state is authoritative.
            self._event_seq = max([self._event_seq] + [n for n in sequences if type(n) is int])
        except (KeyError, TypeError, AttributeError):
            raise ValueError("Malformed agent checkpoint") from None

    def _checkpoint(self, seconds: float, *, finished: bool = False) -> None:
        if self.graph:
            self.graph.save(self.directory / "graph.json")
        atomic_json(self.directory / "state.json", {"checkpoint_schema": 2,
            "run_id": self.run_id, "question": self.question,
            "method": self.method, "budget": asdict(self.budget), "scoring": asdict(self.scoring),
            "selection_version": SELECTION_VERSION if self.graph else None,
            "corpus": getattr(self.retriever, "fingerprint", "live:openalex"),
            "model": self.planner.model, "planner": self.planner.name,
            "provider_fingerprint": getattr(self.planner, "fingerprint", None),
            "papers": [p.to_dict() for p in self.papers.values()], "retrieved_ids": sorted(self.retrieved_ids),
            "completed": sorted(self.completed), "history": self.history, "trace": self.trace,
            "pending": self.pending, "graph_data": self.graph.to_dict() if self.graph else None,
            "actions": self.actions, "rounds": self.rounds, "stagnant": self.stagnant,
            "stop_reason": self.stop_reason, "event_seq": self._event_seq,
            "seconds": seconds, "api_usage": asdict(self.retriever.usage),
            "llm_usage": asdict(self.planner.usage), "finished": finished})

    def _hard_stop(self) -> str:
        if self.actions >= self.budget.max_actions:
            return "max_actions"
        if self.rounds >= self.budget.max_rounds:
            return "max_rounds"
        if len(self.papers) >= self.budget.max_inspected:
            return "max_inspected"
        if self.retriever.usage.requests >= self.budget.max_api_requests:
            return "max_api_requests"
        if self.stagnant >= self.budget.patience:
            return "minimal_new_relevant_information"
        return ""

    def _search_proposals(self) -> list[Action]:
        topics = Counter(t.name for p in self.papers.values() for t in p.topics if t.name)
        proposals = [make_action("SEARCH_PAPERS", f"{self.question} {topic}"[:1000], "Observed topic terminology")
                     for topic, _ in topics.most_common(10)]
        # Short lexical terms allow graph-free follow-up searches even when topics are missing.
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        import re
        terms = Counter(term.lower() for p in self.papers.values() for term in re.findall(r"[A-Za-z]{4,}", p.title or "")
                        if term.lower() not in ENGLISH_STOP_WORDS)
        proposals += [make_action("SEARCH_PAPERS", f"{self.question} {term}"[:1000], "Observed title terminology")
                      for term, _ in terms.most_common(10)]
        if self.search_feedback()["consecutive_searches_without_new_papers"] >= 2:
            original = [term for term in re.findall(r"[A-Za-z0-9]+", self.question)
                        if term.lower() not in ENGLISH_STOP_WORDS]
            if original:
                proposals.insert(0, make_action("SEARCH_PAPERS", " ".join(original[:max(1, len(original)//2)]),
                                                "Broaden after searches yielded no new papers"))
        return [p for p in proposals if p.signature not in self.completed]

    def context(self) -> dict:
        ranking = relevance(self.question, list(self.papers.values()))
        top = sorted(self.papers.values(), key=lambda p: (-ranking[p.id], p.id))[:8]
        proposals = []
        candidates = []
        subgraph = []
        coverage = Counter(t.id for p in self.papers.values() for t in p.topics)
        if self.graph:
            candidates = score_candidates(self.question, self.papers, self.graph, self.scoring)[:30]
            for c in candidates:
                target = c["id"]
                if c["resolved"] and c["depth"] < self.scoring.max_depth:
                    names = ["EXPAND_REFERENCES", "EXPAND_CITATIONS"]
                elif not c["resolved"]:
                    names = ["INSPECT_PAPER"]
                else:
                    names = []
                for name in names:
                    action = make_action(name, target, "Graph candidate acquisition priority")
                    if action.signature not in self.completed:
                        proposals.append(action)
            for topic, count in sorted(coverage.items(), key=lambda x: (x[1], x[0]))[:8]:
                parents = [p for p in self.papers.values() if any(t.id == topic for t in p.topics)]
                if not any(ranking[p.id] >= self.scoring.min_relevance and self.graph.graph.nodes[p.id]["depth"] < self.scoring.max_depth for p in parents):
                    continue
                action = make_action("EXPLORE_TOPIC", topic, "Underrepresented observed topic")
                if action.signature not in self.completed:
                    proposals.append(action)
            seeds = list(dict.fromkeys([p.id for p in top[:3]] + [c["id"] for c in candidates[:3]]))
            view = self.graph.neighborhood(seeds, max_nodes=12, relations=["PAPER_CITES_PAPER"])
            subgraph = [{"source": e["source"], "target": e["target"], "relation": e["relation"]} for e in view["edges"]][:16]
        feedback = self.search_feedback()
        search_paused = feedback["consecutive_searches_without_new_papers"] >= 2 and bool(proposals)
        if not search_paused:
            proposals += self._search_proposals()
        if not proposals:
            proposals = [make_action("STOP_RETRIEVAL", reason="No promising unexplored candidates")]
        permitted = ["SEARCH_PAPERS", "STOP_RETRIEVAL"]
        if search_paused:
            permitted.remove("SEARCH_PAPERS")
        if self.graph:
            permitted += ["EXPAND_REFERENCES", "EXPAND_CITATIONS", "EXPLORE_TOPIC", "INSPECT_PAPER"]
        result = {"question": self.question, "round": self.rounds,
            "retrieval": self.retrieval_environment(), "search_feedback": feedback,
            "remaining": {"actions": self.budget.max_actions-self.actions,
                          "inspections": self.budget.max_inspected-len(self.papers),
                          "api_requests": self.budget.max_api_requests-self.retriever.usage.requests},
            "top_papers": [{"id": p.id, "title": (p.title or "")[:200], "abstract_excerpt": (p.abstract or "")[:350],
                            "original_query_relevance": ranking[p.id]} for p in top],
            "recent_actions": deepcopy(self.history[-4:]), "permitted_actions": permitted,
            "counts": {"retrieved": len(self.retrieved_ids), "inspected": len(self.papers)},
            "last_observation": {k: self.trace[-1].get(k, 0) for k in ("new_papers", "updated_papers", "new_promising")} if self.trace else {},
            "proposals": [p.to_dict() for p in proposals[:12]]}
        if self.graph:
            result.update(candidates=candidates[:12], subgraph=subgraph, graph=self.graph.stats(),
                          topic_coverage=dict(sorted(coverage.items(), key=lambda x: (x[1], x[0]))[:8]),
                          selection_policy={"version": SELECTION_VERSION, "weights": asdict(self.scoring),
                              "guidance": "Candidates are ordered by observed graph acquisition utility. "
                                  "Consider supported citation and topic proposals alongside search. "
                                  "Unresolved targets have proxy scores, not verified relevance. "
                                  "Citation importance alone does not establish topical relevance."})
        # Bound the serialized representation itself, not just the number of nodes.
        while len(json.dumps(result, ensure_ascii=False)) > self.budget.max_context_chars:
            for key in ("subgraph", "candidates", "recent_actions", "top_papers", "proposals"):
                if len(result.get(key, [])) > (1 if key == "proposals" else 0):
                    result[key].pop()
                    break
            else:
                if result["search_feedback"]["recent_searches"]:
                    result["search_feedback"]["recent_searches"].pop(0)
                else:
                    raise ValueError("Context cannot fit configured character budget")
        return result

    def validate(self, action: Action) -> None:
        action = Action.parse(action.to_dict())
        if action.signature in self.completed:
            raise ValueError("Repeated action provides no new information")
        replay = self.pending is not None and Action.parse(self.pending).signature == action.signature
        if not replay and action.action == "SEARCH_PAPERS" and self.search_feedback()["consecutive_searches_without_new_papers"] >= 2:
            if "SEARCH_PAPERS" not in self.context()["permitted_actions"]:
                raise ValueError("Repeated unproductive searches: choose another acquisition action or stop")
        if action.action in ("SEARCH_PAPERS", "STOP_RETRIEVAL"):
            return
        if not self.graph:
            raise ValueError("Graph-free method may only search and stop")
        candidates = {c["id"]: c for c in score_candidates(self.question, self.papers, self.graph, self.scoring)}
        if action.action == "EXPLORE_TOPIC":
            topic = action.arguments["topic_id"]
            if topic not in self.graph.graph or self.graph.graph.nodes[topic]["type"] != "topic":
                raise ValueError("Unknown topic")
            if not any(p.id in candidates and candidates[p.id]["depth"] < self.scoring.max_depth and
                       any(t.id == topic for t in p.topics) for p in self.papers.values()):
                raise ValueError("Topic fails original-query drift guard")
            return
        target = action.arguments["paper_id"]
        if target not in candidates:
            raise ValueError("Unknown target, low relevance, or path exceeds depth limit")
        if action.action == "INSPECT_PAPER":
            if target in self.papers:
                raise ValueError("Paper already inspected")
        elif target not in self.papers or candidates[target]["depth"] >= self.scoring.max_depth:
            raise ValueError("Expansion needs an inspected paper within the path limit")

    def _execute(self, action: Action, *, single_pass: bool = False) -> None:
        action = Action.parse(action.to_dict())
        self.validate(action)
        name, args = action.action, dict(action.arguments)
        if name == "STOP_RETRIEVAL":
            self._memory = replace(self._memory, stop_reason="planner_stop")
            self._log("stop_decision", action.to_dict())
            return
        replay = self.pending is not None and Action.parse(self.pending).signature == action.signature
        if not replay:
            if self._hard_stop():
                raise BudgetExceeded("Controller retrieval limit reached")
            self._memory = replace(self._memory, actions=self.actions+1, rounds=self.rounds+1,
                                   history=self.history+[action.to_dict()], pending=action.to_dict())
            self._log("decision", {"round": self.rounds, **action.to_dict()})
            self._checkpoint(self.prior_seconds + time.perf_counter()-getattr(self, "started", time.perf_counter()))
        remaining = self.budget.max_inspected - len(self.papers)
        if remaining <= 0:
            raise BudgetExceeded("Inspected paper limit reached")
        limit = remaining if single_pass else min(self.budget.page_size, remaining)
        if name == "SEARCH_PAPERS":
            papers = self.retriever.search(args["query"], limit)
        elif name == "EXPAND_REFERENCES":
            papers = self.retriever.references(args["paper_id"], limit)
        elif name == "EXPAND_CITATIONS":
            papers = self.retriever.citations(args["paper_id"], limit)
        elif name == "EXPLORE_TOPIC":
            papers = self.retriever.topic(args["topic_id"], limit)
        else:
            try:
                papers = [self.retriever.get(args["paper_id"])]
            except ValueError:
                papers = []
                self._log("unavailable_target", {"paper_id": args["paper_id"]})
        if not isinstance(papers, list) or any(not isinstance(p, Paper) for p in papers):
            raise RetrievalError("Retriever returned invalid paper metadata")
        observed = {}
        for paper in papers:
            copied = Paper.from_dict(paper.to_dict())
            copied.id = canonical_id(copied.id)
            try:
                if datetime.fromisoformat(copied.retrieved_at).tzinfo is None:
                    raise ValueError("Missing timezone")
            except (ValueError, TypeError):
                raise RetrievalError("Retriever returned an invalid acquisition timestamp") from None
            observed[copied.id] = copied
        # Build an isolated observation transaction. An interrupt cannot persist
        # half a batch or mark its action completed before its graph is ready.
        staged_graph = ScholarlyGraph.from_dict(self.graph.to_dict()) if self.graph else None
        staged_papers = dict(self.papers)
        new, updated, committed = [], [], []
        for paper in observed.values():
            existing = staged_papers.get(paper.id)
            if existing is None and len(new) >= remaining:
                continue
            if existing and datetime.fromisoformat(paper.retrieved_at) < datetime.fromisoformat(existing.retrieved_at):
                continue
            if staged_graph:
                old = staged_graph.graph.nodes.get(paper.id, {})
                if old.get("resolved"):
                    parent, depth = old.get("parent"), old["depth"]
                else:
                    parent = args.get("paper_id")
                    if name == "INSPECT_PAPER":
                        parent = staged_graph.graph.nodes[args["paper_id"]].get("parent")
                    if name == "EXPLORE_TOPIC":
                        anchors = [p for p in staged_papers.values() if
                            staged_graph.graph.nodes[p.id]["depth"] < self.scoring.max_depth and
                            any(t.id == args["topic_id"] for t in p.topics)]
                        scores = relevance(self.question, anchors)
                        anchors = [p for p in anchors if scores[p.id] >= self.scoring.min_relevance]
                        parent = min(anchors, key=lambda p: (staged_graph.graph.nodes[p.id]["depth"], -scores[p.id], p.id)).id if anchors else None
                    depth = staged_graph.graph.nodes[parent]["depth"]+1 if parent else 0
                staged_graph.add(paper, action=name, parent=parent, depth=depth)
                merged = staged_graph.paper(paper.id)
            else:
                merged, _ = merge_observation(paper, existing.to_dict() if existing else None, {})
                merged = Paper.from_dict(merged)
            staged_papers[paper.id] = merged
            if existing is None:
                new.append(merged)
                committed.append(paper)
            elif existing.to_dict() != merged.to_dict():
                updated.append(merged)
                committed.append(paper)
        new_scores = relevance(self.question, new)
        promising = sum(s >= self.scoring.min_relevance for s in new_scores.values())
        stagnant = self.stagnant+1 if promising < self.budget.min_new_relevant else 0
        retrieved_ids = self.retrieved_ids | set(observed)
        trace = {"round": self.rounds, "actions": self.actions, "new_papers": len(new),
                 "returned_papers": len(observed),
                 "updated_papers": len(updated), "inspected": len(staged_papers), "inspected_ids": sorted(staged_papers),
                 "retrieved": len(retrieved_ids), "api_requests": self.retriever.usage.requests,
                 "api_cost_usd": self.retriever.usage.reported_cost_usd,
                 "seconds": self.prior_seconds + time.perf_counter()-getattr(self, "started", time.perf_counter()),
                 "llm_calls": self.planner.usage.calls, "input_tokens": self.planner.usage.input_tokens,
                 "output_tokens": self.planner.usage.output_tokens,
                 "graph": staged_graph.stats() if staged_graph else {}, "new_promising": promising}
        self._memory = replace(self._memory, graph=staged_graph, papers=staged_papers, retrieved_ids=retrieved_ids,
            completed=self.completed | {action.signature}, pending=None, stagnant=stagnant, trace=self.trace+[trace])
        self._log("observation", {**trace, "papers": [p.to_dict() for p in committed]})
        if name == "INSPECT_PAPER" and observed and args["paper_id"] not in observed:
            self._log("identifier_redirect", {"requested_id": args["paper_id"], "returned_ids": list(observed)})

    def _fallback(self, context: dict) -> Action:
        for proposal in context.get("proposals", []):
            try:
                action = Action.parse(proposal)
                self.validate(action)
                return action
            except ValueError:
                continue
        return make_action("STOP_RETRIEVAL", reason="No valid unexplored fallback action")

    def run(self) -> dict:
        if self._has_run:
            raise ValueError("Controller instance already ran; use a fresh instance with resume for interrupted work")
        self._has_run = True
        started = time.perf_counter()
        self.started = started
        try:
            if not self.stop_reason and self.pending:
                self._execute(Action.parse(self.pending), single_pass=self.method in ("standard", "reformulated"))
                self._checkpoint(self.prior_seconds + time.perf_counter()-started)
            if not self.stop_reason and not self.history:
                query = self.question
                if self.method == "reformulated":
                    if self.planner.usage.calls >= self.budget.max_llm_calls:
                        raise BudgetExceeded("LLM call limit reached before query reformulation")
                    try:
                        query = self.planner.reformulate(self.question)
                    except ValueError as exc:
                        self._log("invalid_decision", {"stage": "reformulation", "fallback": "original query",
                                  "error_code": exc.code if isinstance(exc, PlannerError) else "invalid_query"})
                    finally:
                        self._log_planner_call("reformulation")
                self._execute(make_action("SEARCH_PAPERS", query, "Initial retrieval"),
                              single_pass=self.method in ("standard", "reformulated"))
                self._checkpoint(self.prior_seconds + time.perf_counter()-started)
            if self.method in ("standard", "reformulated") and not self.stop_reason:
                self._memory = replace(self._memory, stop_reason="single_pass")
            while not self.stop_reason:
                reason = self._hard_stop()
                if reason:
                    self._memory = replace(self._memory, stop_reason=reason)
                    break
                if self.planner.usage.calls >= self.budget.max_llm_calls:
                    self._memory = replace(self._memory, stop_reason="max_llm_calls")
                    break
                context = self.context()
                self._log("context", {"characters": len(json.dumps(context, ensure_ascii=False)),
                    "approx_tokens": (len(json.dumps(context, ensure_ascii=False).encode())+3)//4, "context": context})
                attempted = None
                try:
                    action = Action.parse(self.planner.decide(deepcopy(context)))
                    attempted = {"action": action.action, "verification": "unverified planner proposal",
                                 "target_id": action.arguments.get("paper_id", action.arguments.get("topic_id"))}
                    self.validate(action)
                except ValueError as exc:
                    action = self._fallback(context)
                    self._log("invalid_decision", {"stage": "planning",
                        "error_code": exc.code if isinstance(exc, PlannerError) else "invalid_action",
                        "validation_rule": exc.code if isinstance(exc, PlannerError) else str(exc),
                        "rejected_action": attempted,
                        "fallback_action": action.to_dict()})
                finally:
                    self._log_planner_call("planning")
                self._execute(action)
                self._checkpoint(self.prior_seconds + time.perf_counter()-started)
        except (RetrievalError, ValueError) as exc:
            reason = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "retrieval_error"
            self._memory = replace(self._memory, stop_reason=reason)
            self._log("controller_error", {"reason": reason, "error_type": type(exc).__name__})
        except KeyboardInterrupt:
            self._checkpoint(self.prior_seconds + time.perf_counter()-started)
            raise
        total_seconds = self.prior_seconds + time.perf_counter()-started
        # Incomplete final exports can be regenerated by resuming this state;
        # a run is marked finished only after every result artifact is written.
        self._checkpoint(total_seconds, finished=False)
        try:
            return self._export_result(total_seconds)
        except KeyboardInterrupt:
            self._checkpoint(self.prior_seconds + time.perf_counter()-started, finished=False)
            raise

    def _export_result(self, total_seconds: float) -> dict:
        scores = relevance(self.question, list(self.papers.values()))
        ranked = []
        for p in sorted(self.papers.values(), key=lambda p: (-scores[p.id], p.id))[:self.budget.output_k]:
            ranked.append({"paper": p.to_dict(), "algorithmic": {"relevance": scores[p.id],
                "reason": "Lexical cosine similarity of observed title/abstract/topics to the original question",
                "discovery_path": self.graph.path(p.id) if self.graph else []},
                "llm_interpretation": None, "limitations": [] if p.abstract else ["Abstract unavailable"]})
        result = {"run_id": self.run_id, "question": self.question, "method": self.method,
            "planner": self.planner.name, "model": self.planner.model,
            "experiment_kind": getattr(self.retriever, "experiment_kind",
                                       "live" if isinstance(self.retriever, OpenAlexClient) else "fixed_corpus"),
            "provider_configuration": getattr(self.planner, "configuration", {}),
            "corpus": getattr(self.retriever, "fingerprint", "live:openalex"),
            "retrieval": self.retrieval_environment(),
            "budget": asdict(self.budget), "scoring": asdict(self.scoring), "stop_reason": self.stop_reason,
            "selection_version": SELECTION_VERSION if self.graph else None,
            "actions": self.actions, "rounds": self.rounds, "retrieved_ids": sorted(self.retrieved_ids),
            "inspected_ids": sorted(self.papers), "ranked": ranked, "trace": self.trace,
            "api_usage": asdict(self.retriever.usage), "llm_usage": asdict(self.planner.usage),
            "seconds": total_seconds,
            "wait_seconds": self.retriever.usage.wait_seconds,
            "local_seconds": max(0.0, total_seconds-self.retriever.usage.api_seconds-self.retriever.usage.wait_seconds-self.planner.usage.seconds),
            "directory": str(self.directory)}
        atomic_json(self.directory / "result.json", result)
        temporary_csv = self.directory / "ranked.csv.tmp"
        with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["rank", "id", "title", "authors", "year", "venue", "doi", "relevance", "reason", "discovery_path", "limitations"])
            writer.writeheader()
            for i, item in enumerate(ranked, 1):
                p, calc = item["paper"], item["algorithmic"]
                writer.writerow(dict(rank=i, id=p["id"], title=p["title"], year=p["year"], venue=p["venue"], doi=p["doi"],
                    authors="; ".join(a["author"]["name"] or a["author"]["id"] if a["author"] else a["raw_name"] or "[unknown]" for a in p["authorships"]),
                    relevance=calc["relevance"], reason=calc["reason"], discovery_path=" > ".join(calc["discovery_path"]),
                    limitations="; ".join(item["limitations"])))
        temporary_csv.replace(self.directory / "ranked.csv")
        self._log("stop", {"reason": self.stop_reason, "seconds": total_seconds})
        self._checkpoint(total_seconds, finished=self.stop_reason != "retrieval_error")
        return result
