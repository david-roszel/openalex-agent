"""Graph snapshot validation uses synthetic metadata only."""

from copy import deepcopy
from dataclasses import replace
import json
import pytest
from openalex_agent.graph import ScholarlyGraph
from openalex_agent.models import Paper


@pytest.fixture
def snapshot(raw_work):
    graph = ScholarlyGraph()
    graph.add(Paper.from_openalex(raw_work))
    return graph.to_dict()


def test_roundtrip_preserves_metadata_and_allows_incremental_updates(snapshot, tmp_path):
    graph = ScholarlyGraph.from_dict(snapshot)
    path = tmp_path / "graph.json"
    graph.save(path)
    restored = ScholarlyGraph.load(path)
    assert restored.to_dict() == graph.to_dict() == snapshot
    work = restored.paper("W1")
    restored.add(replace(work, references=[], retrieved_at="2099-01-01T00:00:00+00:00"))
    assert "https://openalex.org/W2" not in restored.graph
    restored.save(path)
    assert ScholarlyGraph.load(path).paper("W1").references == []


def test_legacy_links_and_missing_metadata_are_readable(snapshot):
    snapshot["graph"] = {}
    snapshot["links"] = snapshot.pop("edges")
    for node in snapshot["nodes"]:
        node.pop("metadata", None)
    restored = ScholarlyGraph.from_dict(snapshot)
    assert restored.stats()["resolved_papers"] == 1
    assert restored.paper("W1") is None
    assert restored.graph.has_edge("https://openalex.org/W1", "https://openalex.org/W2", "PAPER_CITES_PAPER")
    assert not restored.citation_graph().has_edge("https://openalex.org/W1", "https://openalex.org/W2")
    assert restored.paper_state("W1") == "legacy_metadata_missing"


@pytest.mark.parametrize("corruption", ["undirected", "not_multi", "version", "type", "identifier", "duplicate_node",
    "dangling_edge", "duplicate_edge", "invented_relation", "reversed_edge", "missing_evidence", "wrong_evidence",
    "missing_snapshot", "metadata_mismatch", "unsupported_metadata", "parent", "depth", "source_timestamp", "missing_edges"])
def test_invalid_snapshots_are_rejected(snapshot, corruption):
    data = deepcopy(snapshot)
    paper = next(n for n in data["nodes"] if n["id"].endswith("W1"))
    target = next(n for n in data["nodes"] if n["id"].endswith("W2"))
    edge = next(e for e in data["edges"] if e["key"] == "PAPER_CITES_PAPER")
    if corruption == "undirected": data["directed"] = False
    elif corruption == "not_multi": data["multigraph"] = False
    elif corruption == "version": data["graph"]["scholarly_schema"] = 99
    elif corruption == "type": target["type"] = "invented"
    elif corruption == "identifier": target["id"] = "not-an-id"
    elif corruption == "duplicate_node": data["nodes"].append(deepcopy(paper))
    elif corruption == "dangling_edge": edge["target"] = "https://openalex.org/W999"
    elif corruption == "duplicate_edge": data["edges"].append(deepcopy(edge))
    elif corruption == "invented_relation": edge["key"] = edge["relation"] = "PAPER_PROVES_PAPER"
    elif corruption == "reversed_edge": edge["source"], edge["target"] = edge["target"], edge["source"]
    elif corruption == "missing_evidence": edge["evidence"] = []
    elif corruption == "wrong_evidence": edge["evidence"] = [target["id"]]
    elif corruption == "missing_snapshot": paper.pop("metadata")
    elif corruption == "metadata_mismatch": paper["metadata"]["references"] = []
    elif corruption == "unsupported_metadata": target["title"] = "INVENTED TEST TITLE"
    elif corruption == "parent": target["parent"] = "https://openalex.org/W999"
    elif corruption == "depth": target["depth"] = 99
    elif corruption == "source_timestamp": edge["observations"][paper["id"]]["retrieved_at"] = "wrong"
    elif corruption == "missing_edges": data.pop("edges")
    with pytest.raises(ValueError):
        ScholarlyGraph.from_dict(data)


def test_save_does_not_replace_valid_file_if_raw_graph_was_modified(snapshot, tmp_path):
    graph = ScholarlyGraph.from_dict(snapshot)
    path = tmp_path / "graph.json"
    graph.save(path)
    original = path.read_bytes()
    # Correctly typed but unsupported by the work's actual topic metadata.
    graph.graph.add_node("https://openalex.org/T99", type="topic")
    graph.graph.add_edge("https://openalex.org/W1", "https://openalex.org/T99", key="PAPER_HAS_TOPIC",
                         relation="PAPER_HAS_TOPIC", evidence=["https://openalex.org/W1"])
    with pytest.raises(ValueError):
        graph.save(path)
    assert path.read_bytes() == original


def test_mutating_export_does_not_change_graph(snapshot):
    graph = ScholarlyGraph.from_dict(snapshot)
    exported = graph.to_dict()
    exported["nodes"].clear()
    exported["edges"].clear()
    assert graph.to_dict() == snapshot


def test_malformed_json_is_a_controlled_load_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"nodes":')
    with pytest.raises(ValueError):
        ScholarlyGraph.load(path)
