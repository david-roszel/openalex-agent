"""Manual, bounded scholarly relationship expansion; no agent or LLM logic."""

from typing import Protocol
from .graph import ScholarlyGraph, _paper_copy
from .models import Paper, canonical_id


class GraphRetriever(Protocol):
    def get(self, paper_id: str) -> Paper: ...
    def batch(self, ids: list[str]) -> list[Paper]: ...
    def citations(self, paper_id: str, limit: int = 100) -> list[Paper]: ...


def expand_graph(graph: ScholarlyGraph, retriever: GraphRetriever, paper_id: str, *,
                 direction: str = "references", limit: int = 20) -> dict:
    """Resolve references or acquire citing works using observed citation evidence.

    Work on a staged graph so retrieval/validation failures leave the supplied
    graph unchanged. Cached graph metadata avoids re-fetching the anchor.
    Incoming works must themselves report the citation; filter membership alone
    never creates a factual edge. Outgoing evidence comes from the anchor.
    """
    if direction not in ("references", "citations") or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Expansion needs references/citations and a limit between 1 and 100")
    paper_id = canonical_id(paper_id)
    node = graph.graph.nodes.get(paper_id)
    if not node or not node.get("resolved"):
        raise ValueError("Expansion anchor must be a resolved paper in the graph")
    staged = ScholarlyGraph.from_dict(graph.to_dict())
    anchor = staged.paper(paper_id)
    if anchor is None:
        # Compatibility with old snapshots, whose resolved nodes lacked metadata.
        anchor = _paper_copy(retriever.get(paper_id))
        if anchor.id != paper_id:
            raise ValueError("Retriever returned a different anchor identifier")
        staged.add(anchor, action="restore_anchor_metadata", parent=node.get("parent"), depth=node["depth"])
        if staged.paper(paper_id) is None:
            raise ValueError("Legacy graph is newer than cached anchor metadata; refresh the work and update the graph")
    if direction == "references":
        requested = anchor.references[:limit]
        retrieved = retriever.batch(requested) if requested else []
    else:
        requested = []
        retrieved = retriever.citations(paper_id, limit)
    normalized = [_paper_copy(p) for p in retrieved[:limit]]
    papers = {p.id: p for p in normalized}
    verified, skipped = [], []
    before = graph.stats()
    for target, paper in papers.items():
        supported = target in requested if direction == "references" else paper_id in paper.references
        if target == paper_id or not supported:
            skipped.append(target)
            continue
        staged.add(paper, action="expand_" + direction, parent=paper_id, depth=staged.graph.nodes[paper_id]["depth"] + 1)
        verified.append(target)
    staged.validate()
    graph.graph = staged.graph
    graph._indexed_graph = None
    return {"direction": direction, "anchor": paper_id, "limit": limit,
            "retrieved_ids": list(papers), "verified_ids": verified, "skipped_ids": skipped,
            "unavailable_ids": [i for i in requested if i not in papers],
            "new_resolved_papers": graph.stats()["resolved_papers"] - before["resolved_papers"],
            "before": before, "after": graph.stats()}
