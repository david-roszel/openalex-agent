from openalex_agent.graph import ScholarlyGraph
from openalex_agent.models import Paper


def test_graph_evidence_direction_dedup_and_persistence(raw_work, tmp_path):
    paper = Paper.from_openalex(raw_work)
    graph = ScholarlyGraph()
    assert graph.add(paper, action="search")
    assert not graph.add(paper)
    citation = graph.citation_graph()
    assert citation.has_edge(paper.id, "https://openalex.org/W2")
    assert not citation.has_edge("https://openalex.org/W2", paper.id)
    assert graph.stats() == dict(nodes=5, edges=4, paper_nodes=2, resolved_papers=1)
    data = graph.graph[paper.id]["https://openalex.org/W2"]["PAPER_CITES_PAPER"]
    assert data["evidence"] == [paper.id]
    assert graph.graph.nodes["https://openalex.org/W2"]["resolved"] is False
    graph.add(Paper(id="https://openalex.org/W2", title="SYNTHETIC TEST target"), parent=paper.id, depth=1)
    assert graph.path("W2") == [paper.id, "https://openalex.org/W2"]
    path = tmp_path / "graph.json"
    graph.save(path)
    loaded = ScholarlyGraph.load(path)
    assert loaded.stats() == graph.stats()
    assert loaded.path("W2") == graph.path("W2")
    assert len(loaded.neighborhood([paper.id], hops=2, max_nodes=3)["nodes"]) == 3


def test_unknown_authors_create_no_invented_edges():
    graph = ScholarlyGraph()
    graph.add(Paper(id="https://openalex.org/W1"))
    assert graph.stats()["edges"] == 0
