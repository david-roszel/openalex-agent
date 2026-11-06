"""State/provenance regression cases use synthetic OpenAlex-shaped fixtures."""

from dataclasses import replace
import pytest
from openalex_agent.graph import ScholarlyGraph
from openalex_agent.models import Paper


def observed(raw, date="2026-01-01T00:00:00+00:00"):
    work = Paper.from_openalex(raw, provenance="SYNTHETIC GRAPH TEST ONLY")
    work.retrieved_at = date
    return work


def test_retrieved_and_unresolved_states_and_promotion_use_one_canonical_node(raw_work):
    raw_work["referenced_works"] = ["w2", "W2", "http://openalex.org/W2"]
    graph = ScholarlyGraph()
    graph.add(observed(raw_work))
    assert graph.paper_state("w1") == "retrieved"
    assert graph.paper_state("https://openalex.org/W2") == "unresolved"
    assert graph.paper("W2") is None
    target = graph.graph.nodes["https://openalex.org/W2"]
    assert target["status"] == "unresolved" and not target["resolved"]
    assert not {"metadata", "title", "retrieved_at", "metadata_history"} & target.keys()
    assert graph.stats()["paper_nodes"] == 2
    discovered_at = target["discovered_at"]
    graph.add(observed({"id": "http://openalex.org/w2", "title": "SYNTHETIC acquired work"}), parent="W1")
    assert graph.stats()["paper_nodes"] == 2 and graph.stats()["resolved_papers"] == 2
    assert graph.paper_state("W2") == "retrieved"
    assert target["discovered_at"] == discovered_at
    assert graph.citation_graph(include_unresolved=False).has_edge("https://openalex.org/W1", "https://openalex.org/W2")


def test_titles_never_define_identity():
    graph = ScholarlyGraph()
    for work_id in ("W1", "W2", "https://openalex.org/w1"):
        graph.add(observed({"id": work_id, "title": "SYNTHETIC identical title"}))
    assert graph.stats()["paper_nodes"] == 2


def test_partial_response_preserves_known_fields_and_citation_source(raw_work):
    graph = ScholarlyGraph()
    first = observed(raw_work)
    graph.add(first)
    edge = (first.id, "https://openalex.org/W2", "PAPER_CITES_PAPER")
    original_source = dict(graph.graph.edges[edge]["observations"][first.id])
    partial = observed({"id": "W1", "title": "SYNTHETIC revised title"}, "2026-02-01T00:00:00+00:00")
    graph.add(partial)
    current = graph.paper("W1")
    assert current.title == partial.title and current.abstract == first.abstract
    assert current.authorships == first.authorships and current.topics == first.topics
    assert current.references == first.references
    assert graph.graph.edges[edge]["observations"][first.id] == original_source
    assert graph.graph.edges[edge]["source_field"] == "referenced_works"
    assert graph.paper_history("W1")[0]["metadata"]["title"] == first.title
    before = graph.to_dict()
    graph.add(partial)
    assert graph.to_dict() == before  # Replaying the same partial observation is idempotent.
    graph.add(graph.paper("W1"))
    assert graph.to_dict() == before  # Reading/reinserting the merged snapshot is also idempotent.


def test_missing_and_null_fields_do_not_clear_known_values(raw_work):
    graph = ScholarlyGraph()
    graph.add(observed(raw_work))
    graph.add(observed({"id": "W1", "title": None, "abstract_inverted_index": None,
                        "referenced_works": None, "authorships": None}, "2026-02-01T00:00:00+00:00"))
    work = graph.paper("W1")
    assert work.title == raw_work["title"] and work.abstract and work.authorships
    assert work.references == ["https://openalex.org/W2"]
    graph.validate()


def test_malformed_reference_field_cannot_erase_prior_metadata(raw_work):
    graph = ScholarlyGraph()
    graph.add(observed(raw_work))
    graph.add(observed({"id": "W1", "referenced_works": ["not-an-openalex-id"]},
                       "2026-02-01T00:00:00+00:00"))
    assert graph.paper("W1").references == ["https://openalex.org/W2"]
    assert graph.citation_graph().has_edge("https://openalex.org/W1", "https://openalex.org/W2")
    graph.validate()


def test_explicit_correction_preserves_historical_information_across_restart(raw_work, tmp_path):
    graph = ScholarlyGraph()
    first = observed(raw_work)
    graph.add(first)
    first_discovery = graph.graph.nodes["https://openalex.org/W2"]["discovered_at"]
    graph.add(observed({"id": "W1", "referenced_works": []}, "2026-02-01T00:00:00+00:00"))
    assert not graph.citation_graph().has_edge(first.id, "https://openalex.org/W2")
    assert graph.paper_history("W1")[0]["metadata"]["references"] == ["https://openalex.org/W2"]
    path = tmp_path / "graph.json"
    graph.save(path)
    restored = ScholarlyGraph.load(path)
    assert restored.paper_history("W1") == graph.paper_history("W1")
    assert restored.paper("W1").abstract == first.abstract
    assert restored.historical_entity("W2")["discovered_at"] == first_discovery
    assert restored.paper_state("W2") == "unresolved"
    history = restored.paper_history("W1")
    history[0]["metadata"]["references"].clear()
    assert restored.paper_history("W1")[0]["metadata"]["references"]


def test_reintroduced_targets_and_entities_keep_first_discovery_and_known_names(raw_work):
    graph = ScholarlyGraph()
    graph.add(observed(raw_work))
    discovered = graph.graph.nodes["https://openalex.org/W2"]["discovered_at"]
    graph.add(observed({"id": "W1", "referenced_works": [], "authorships": [], "topics": []},
                       "2026-02-01T00:00:00+00:00"))
    assert graph.historical_entity("A1")["name"] == "TEST AUTHOR"
    incoming = {"id": "W3", "referenced_works": ["W2"], "topics": [{"id": "T1"}],
                "authorships": [{"author": {"id": "A1"}, "institutions": [{"id": "I1"}]}]}
    graph.add(observed(incoming, "2026-03-01T00:00:00+00:00"))
    assert graph.graph.nodes["https://openalex.org/W2"]["discovered_at"] == discovered
    assert graph.graph.nodes["https://openalex.org/A1"]["name"] == "TEST AUTHOR"
    assert graph.historical_entity("A1") is None and graph.historical_entity("W2") is None
    assert graph.stats()["paper_nodes"] == 3
    graph.validate()


def test_inferred_citations_are_rejected_before_mutation():
    graph = ScholarlyGraph()
    with pytest.raises(ValueError, match="OpenAlex referenced_works"):
        graph.add(Paper(id="https://openalex.org/W1", references=["W2"], provenance="LLM-generated"))
    assert graph.stats()["nodes"] == 0
    actual = observed({"id": "W1", "referenced_works": ["W2"], "related_works": ["W3"]})
    graph.add(actual)
    assert "https://openalex.org/W3" not in graph.citation_graph()
    # A non-OpenAlex empty list cannot erase verified citation evidence either.
    graph.add(Paper(id=actual.id, retrieved_at="2026-02-01T00:00:00+00:00", provenance="other source"))
    assert graph.citation_graph().has_edge(actual.id, "https://openalex.org/W2")
    graph.validate()


def test_source_display_label_cannot_authorize_citations():
    graph = ScholarlyGraph()
    with pytest.raises(ValueError, match="OpenAlex referenced_works"):
        graph.add(Paper(id="https://openalex.org/W1", references=["W2"], provenance="OpenAlex API"))


def test_legacy_normalized_openalex_cache_is_identified_explicitly():
    old_record = Paper(id="https://openalex.org/W1", references=["https://openalex.org/W2"]).to_dict()
    old_record.pop("metadata_source")
    old_record.pop("metadata_fields")
    work = Paper.from_dict(old_record)
    assert work.metadata_source == "openalex_legacy"
    graph = ScholarlyGraph()
    graph.add(work)
    assert graph.detailed_stats()["unverified_citations"] == 0
    assert graph.graph.edges[work.id, work.references[0], "PAPER_CITES_PAPER"]["observations"][work.id]["metadata_source"] == "openalex_legacy"


def test_tampered_citation_source_is_rejected(raw_work):
    graph = ScholarlyGraph()
    graph.add(observed(raw_work))
    snapshot = graph.to_dict()
    work = next(n for n in snapshot["nodes"] if n["id"].endswith("W1"))
    work["field_sources"]["references"]["metadata_source"] = "LLM"
    edge = next(e for e in snapshot["edges"] if e["key"] == "PAPER_CITES_PAPER")
    edge["observations"][work["id"]]["metadata_source"] = "LLM"
    with pytest.raises(ValueError, match="OpenAlex"):
        ScholarlyGraph.from_dict(snapshot)
