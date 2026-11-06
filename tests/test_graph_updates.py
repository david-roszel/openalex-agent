"""Synthetic graph fixtures only; no real scientific claims or relevance labels."""

from dataclasses import replace
from copy import deepcopy
import pytest
from openalex_agent.graph import ScholarlyGraph
from openalex_agent.models import Authorship, Entity, Paper


def paper(number, **kwargs):
    if "references" in kwargs:
        item = Paper.from_openalex({"id": f"W{number}", "title": f"SYNTHETIC TEST {number}",
                                   "referenced_works": kwargs.pop("references")}, provenance="SYNTHETIC GRAPH TEST ONLY")
        return replace(item, **kwargs)
    return Paper(id=f"https://openalex.org/W{number}", title=f"SYNTHETIC TEST {number}",
                 provenance="SYNTHETIC GRAPH TEST ONLY", **kwargs)


def test_short_ids_duplicate_metadata_and_snapshot_copies(raw_work):
    raw_work["id"] = "W1"
    p = Paper.from_openalex(raw_work, provenance="SYNTHETIC GRAPH TEST ONLY")
    p.id = "W1"
    p.authorships[0].author.id = "A1"
    graph = ScholarlyGraph()
    assert graph.add(p)
    first = graph.to_dict()
    assert not graph.add(p)
    assert graph.to_dict() == first
    p.title = "CHANGED OUTSIDE GRAPH"
    snapshot = graph.paper("W1")
    assert snapshot.id == "https://openalex.org/W1" and snapshot.title != p.title
    snapshot.references.clear()
    assert graph.paper("W1").references == ["https://openalex.org/W2"]


def test_refresh_retracts_only_its_evidence_and_removes_orphan_entities(raw_work):
    first = Paper.from_openalex(raw_work, provenance="SYNTHETIC GRAPH TEST ONLY")
    first.retrieved_at = "2026-01-01T00:00:00+00:00"
    second = replace(first, id="https://openalex.org/W3", references=[], topics=[])
    graph = ScholarlyGraph()
    graph.add(first)
    graph.add(second)
    edge = ("https://openalex.org/A1", "https://openalex.org/I1", "AUTHOR_AFFILIATED_WITH_INSTITUTION")
    assert graph.graph.edges[edge]["evidence"] == [first.id, second.id]
    updated = replace(first, references=[], topics=[], authorships=[], retrieved_at="2026-01-02T00:00:00+00:00")
    graph.add(updated)
    assert graph.graph.edges[edge]["evidence"] == [second.id]
    assert set(graph.graph.edges[edge]["observations"]) == {second.id}
    assert "https://openalex.org/W2" not in graph.graph
    assert "https://openalex.org/T1" not in graph.graph
    assert "https://openalex.org/A1" in graph.graph
    graph.add(replace(second, authorships=[], retrieved_at="2026-01-03T00:00:00+00:00"))
    assert "https://openalex.org/A1" not in graph.graph and "https://openalex.org/I1" not in graph.graph
    graph.validate()


def test_old_observation_does_not_restore_retracted_edges(raw_work):
    old = Paper.from_openalex(raw_work)
    old.retrieved_at = "2026-01-01T00:00:00+00:00"
    latest = replace(old, references=[], retrieved_at="2026-01-02T01:00:00+01:00")
    graph = ScholarlyGraph()
    graph.add(latest)
    before = graph.to_dict()
    graph.add(old)
    assert graph.to_dict() == before
    assert graph.paper("W1").retrieved_at == "2026-01-02T00:00:00+00:00"


def test_shared_entity_name_is_not_erased_by_missing_or_older_name():
    graph = ScholarlyGraph()
    newest = paper(1, authorships=[Authorship(Entity("A1", "SYNTHETIC NEW NAME"))], retrieved_at="2026-01-02T00:00:00+00:00")
    graph.add(newest)
    graph.add(paper(2, authorships=[Authorship(Entity("A1", None))]))
    graph.add(paper(3, authorships=[Authorship(Entity("A1", "SYNTHETIC OLD NAME"))], retrieved_at="2026-01-01T00:00:00+00:00"))
    assert graph.graph.nodes["https://openalex.org/A1"]["name"] == "SYNTHETIC NEW NAME"


def test_placeholder_resolution_preserves_first_discovery(raw_work):
    graph = ScholarlyGraph()
    first = Paper.from_openalex(raw_work)
    graph.add(first)
    target = "https://openalex.org/W2"
    discovered = deepcopy(graph.graph.nodes[target])
    assert not discovered["resolved"] and "metadata" not in discovered
    graph.add(paper(2), parent=first.id, action="expand_references")
    node = graph.graph.nodes[target]
    assert node["discovered_at"] == discovered["discovered_at"]
    assert node["resolved_by"] == "expand_references" and node["resolved_at"]
    assert any(d["action"] == "expand_references" for d in node["discoveries"])
    assert graph.path("W2") == [first.id, target]


def test_shorter_discovery_path_updates_descendants():
    graph = ScholarlyGraph()
    graph.add(paper(1, references=["W2"]))
    graph.add(paper(2, references=["W3"]), parent="W1")
    third = paper(3, references=["W4"])
    graph.add(third, parent="W2")
    assert graph.graph.nodes["https://openalex.org/W4"]["depth"] == 3
    graph.add(third, action="independent_search")
    assert graph.path("W4") == ["https://openalex.org/W3", "https://openalex.org/W4"]
    assert graph.graph.nodes["https://openalex.org/W4"]["depth"] == 1
    graph.validate()


def test_existing_placeholder_can_be_reached_from_a_closer_anchor():
    graph = ScholarlyGraph()
    graph.add(paper(1, references=["W2"]))
    graph.add(paper(2, references=["W3"]), parent="W1")
    graph.add(paper(4, references=["W3"]))
    assert graph.path("W3") == ["https://openalex.org/W4", "https://openalex.org/W3"]
    assert graph.graph.nodes["https://openalex.org/W3"]["depth"] == 1


@pytest.mark.parametrize("kwargs", [{"parent": "W99"}, {"parent": "W2"}, {"depth": -1}, {"depth": True}, {"action": ""}])
def test_invalid_discovery_arguments_do_not_mutate_graph(kwargs):
    graph = ScholarlyGraph()
    graph.add(paper(1, references=["W2"]))
    before = graph.to_dict()
    with pytest.raises(ValueError):
        graph.add(paper(3), **kwargs)
    assert graph.to_dict() == before


def test_invalid_metadata_ids_fail_before_mutation():
    graph = ScholarlyGraph()
    with pytest.raises(ValueError):
        graph.add(Paper(id="https://openalex.org/W1", references=["not-an-id"]))
    assert graph.stats()["nodes"] == 0


def test_directed_and_typed_traversal(raw_work):
    graph = ScholarlyGraph()
    graph.add(Paper.from_openalex(raw_work))
    graph.add(paper(3, references=["W1"]))
    assert graph.neighbors("W1", relation="PAPER_CITES_PAPER", direction="out") == ["https://openalex.org/W2"]
    assert graph.neighbors("W1", relation="PAPER_CITES_PAPER", direction="in") == ["https://openalex.org/W3"]
    assert graph.neighbors("A1", relation="AUTHOR_AFFILIATED_WITH_INSTITUTION", direction="out") == ["https://openalex.org/I1"]
    view = graph.neighborhood("W1", hops=2, direction="out", relations=["PAPER_WRITTEN_BY_AUTHOR", "AUTHOR_AFFILIATED_WITH_INSTITUTION"])
    assert {n["id"] for n in view["nodes"]} == {"https://openalex.org/W1", "https://openalex.org/A1", "https://openalex.org/I1"}
    assert {e["key"] for e in view["edges"]} == {"PAPER_WRITTEN_BY_AUTHOR", "AUTHOR_AFFILIATED_WITH_INSTITUTION"}
    assert len(graph.neighborhood("W1", hops=0)["nodes"]) == 1
    assert graph.neighborhood([], hops=1)["nodes"] == []


def test_high_degree_neighborhood_is_bounded_and_deterministic():
    graph = ScholarlyGraph()
    graph.add(paper(1, references=[f"W{i}" for i in range(2, 1002)]))
    first = graph.neighborhood("W1", hops=2, max_nodes=5)
    assert len(first["nodes"]) == 5
    assert first == graph.neighborhood("W1", hops=2, max_nodes=5)


@pytest.mark.parametrize("kwargs", [{"hops": -1}, {"hops": True}, {"max_nodes": 0}, {"direction": "wrong"}, {"relations": ["PAPER_PROVES_PAPER"]}])
def test_invalid_traversal_is_rejected(kwargs):
    graph = ScholarlyGraph()
    graph.add(paper(1))
    with pytest.raises(ValueError):
        graph.neighborhood("W1", **kwargs)
