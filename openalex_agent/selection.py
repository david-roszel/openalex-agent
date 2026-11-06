"""Graph-derived acquisition priorities with configurable exploration and drift."""

from dataclasses import asdict, dataclass
import math
from collections import Counter
import networkx as nx
import numpy as np
from .graph import ScholarlyGraph
from .models import Paper
from .ranking import VECTORIZER, relevance

SELECTION_VERSION = "graph-priority-v2"
NOVELTY_BATCH_SIZE = 64


@dataclass(frozen=True)
class Scoring:
    relevance: float = 0.45
    novelty: float = 0.15
    importance: float = 0.20
    coverage: float = 0.20
    cost: float = 0.10
    exploration: float = 0.15
    min_relevance: float = 0.03
    max_depth: int = 3

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or not math.isfinite(v)
               for v in asdict(self).values()):
            raise ValueError("Scoring values must be finite and nonnegative")
        if type(self.max_depth) is not int:
            raise ValueError("Maximum path depth must be a nonnegative integer")
        if not 0 <= self.exploration <= 1 or self.min_relevance > 1:
            raise ValueError("Exploration and minimum relevance must be in [0,1]")


def _novelty(matrix, positions: list[int]) -> dict[int, float]:
    """Exact nearest-observed cosine, with at most 64 similarity rows in memory.

    Never densify an N-by-N matrix. Worst-case time remains quadratic; only
    eligible anchors are scored and the agent's inspection budget bounds N.
    """
    result = {}
    for offset in range(0, len(positions), NOVELTY_BATCH_SIZE):
        batch = positions[offset:offset + NOVELTY_BATCH_SIZE]
        similarities = (matrix[batch] @ matrix.T).tocsr()
        # Only touch stored diagonal entries, including empty-text rows safely.
        for row, position in enumerate(batch):
            begin, end = similarities.indptr[row:row + 2]
            columns = similarities.indices[begin:end]
            similarities.data[begin:end][columns == position] = 0
        maximum = similarities.max(axis=1).toarray().ravel()
        result.update((position, float(np.clip(1 - maximum[i], 0, 1))) for i, position in enumerate(batch))
    return result


def score_candidates(query: str, papers: dict[str, Paper], graph: ScholarlyGraph,
                     config: Scoring) -> list[dict]:
    """Score resolved anchors and unresolved references using observed data only.

    Unknown references get a conservative relevance proxy from their observed
    supporters. No hidden target metadata is fetched to score a placeholder.
    """
    scores = relevance(query, list(papers.values()))
    citations = graph.citation_graph()
    personalization = {n: scores.get(n, 0.0) for n in citations}
    if citations and sum(personalization.values()):
        try:
            importance = nx.pagerank(citations, personalization=personalization, max_iter=200, tol=1e-6)
            importance_method = "personalized_pagerank"
        except nx.PowerIterationFailedConvergence:
            # Label-free local support is a deterministic, bounded fallback.
            importance = {n: sum(scores.get(parent, 0) for parent in citations.predecessors(n))
                          for n in citations}
            importance_method = "observed_incoming_support_fallback"
    else:
        importance = dict.fromkeys(citations, 0.0)
        importance_method = "no_relevant_observations"
    max_importance = max(importance.values(), default=0.0) or 1.0
    topic_counts = Counter(t.id for p in papers.values() for t in p.topics)
    observed = list(papers.values())
    matrix = VECTORIZER.transform([p.text for p in observed]) if observed else None
    positions = {p.id: i for i, p in enumerate(observed)}
    eligible = [positions[p.id] for p in observed if p.id in graph.graph and
                graph.graph.nodes[p.id].get("depth", 0) <= config.max_depth and
                scores[p.id] >= config.min_relevance]
    novelties = _novelty(matrix, eligible) if config.novelty and eligible else {}
    result = []
    for node, attrs in graph.graph.nodes(data=True):
        if attrs["type"] != "paper" or attrs.get("depth", 0) > config.max_depth:
            continue
        p = papers.get(node)
        parents = list(citations.predecessors(node))
        proxy = 0.5 * max((scores.get(n, 0) for n in parents), default=0)
        r = scores.get(node, proxy)
        if r < config.min_relevance:
            continue
        if p:
            novelty = novelties.get(positions[node], 0.0)
            coverage = sum(1 / topic_counts[t.id] for t in p.topics) / len(p.topics) if p.topics else 0.0
        else:
            novelty = 1.0  # uncertainty-driven exploration proxy, not verified content novelty
            parent_topics = {t.id for parent in parents if parent in papers for t in papers[parent].topics}
            coverage = sum(1 / topic_counts[t] for t in parent_topics) / len(parent_topics) if parent_topics else 0.0
        g = importance.get(node, 0) / max_importance
        cost = 0.5 if p else 1.0  # relative lookup/action cost; actual requests logged independently
        value = config.relevance*r + config.novelty*novelty + config.importance*g + config.coverage*coverage - config.cost*cost
        exploration_value = ((novelty if config.novelty else 0) + (coverage if config.coverage else 0))/2
        utility = (1 - config.exploration)*value + config.exploration*exploration_value
        result.append({"id": node, "resolved": bool(p), "score": utility,
                       "components": {"relevance": r, "novelty": novelty, "importance": g, "coverage": coverage, "cost": cost},
                       "relevance_is_proxy": not bool(p), "depth": attrs.get("depth", 0),
                       "importance_method": importance_method})
    return sorted(result, key=lambda x: (-x["score"], x["id"]))
