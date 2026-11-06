"""Manual graph expansion over mocked scholarly metadata, never paid services."""

from dataclasses import replace
import httpx
import pytest
from openalex_agent.graph import ScholarlyGraph
from openalex_agent.graph_expansion import expand_graph
from openalex_agent.models import Paper
from openalex_agent.openalex import OpenAlexClient, RetrievalError


def test_reference_expansion_verifies_direction_reuses_cache_and_survives_restart(db, settings, raw_work, tmp_path):
    graph = ScholarlyGraph()
    anchor = Paper.from_openalex(raw_work)
    graph.add(anchor)
    calls = []
    def handle(request):
        calls.append(request)
        assert request.url.params["filter"] == "openalex:W2"
        return httpx.Response(200, json={"results": [dict(raw_work, id="W2", referenced_works=[])]})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle)) as client:
        result = expand_graph(graph, client, "W1", limit=1)
        assert result["new_resolved_papers"] == 1 and result["verified_ids"] == ["https://openalex.org/W2"]
        assert graph.graph.has_edge(anchor.id, "https://openalex.org/W2", "PAPER_CITES_PAPER")
        assert graph.path("W2") == [anchor.id, "https://openalex.org/W2"]
        path = tmp_path / "expanded.json"
        graph.save(path)
        restored = ScholarlyGraph.load(path)
        second = expand_graph(restored, client, "W1", limit=1)
        assert second["new_resolved_papers"] == 0
        assert len(calls) == 1
        assert restored.to_dict() == graph.to_dict()


def test_citation_expansion_requires_observed_incoming_link(db, settings, raw_work):
    graph = ScholarlyGraph()
    graph.add(Paper.from_openalex(raw_work))
    def handle(request):
        assert request.url.params["filter"] == "cites:W1"
        return httpx.Response(200, json={"results": [
            dict(raw_work, id="W3", referenced_works=["W1"]),
            dict(raw_work, id="W4", referenced_works=[])]})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle)) as client:
        result = expand_graph(graph, client, "W1", direction="citations", limit=2)
    assert result["verified_ids"] == ["https://openalex.org/W3"]
    assert result["skipped_ids"] == ["https://openalex.org/W4"]
    assert graph.graph.has_edge("https://openalex.org/W3", "https://openalex.org/W1", "PAPER_CITES_PAPER")
    assert not graph.graph.has_edge("https://openalex.org/W1", "https://openalex.org/W3", "PAPER_CITES_PAPER")
    assert "https://openalex.org/W4" not in graph.graph


def test_missing_references_are_recorded_and_do_not_fabricate_targets(db, settings, raw_work):
    graph = ScholarlyGraph()
    graph.add(Paper.from_openalex(raw_work))
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []}))) as client:
        result = expand_graph(graph, client, "W1")
    assert result["unavailable_ids"] == ["https://openalex.org/W2"]
    assert graph.graph.nodes["https://openalex.org/W2"]["resolved"] is False


def test_empty_bibliography_does_not_call_api(db, settings):
    graph = ScholarlyGraph()
    graph.add(Paper(id="https://openalex.org/W1", provenance="SYNTHETIC TEST ONLY"))
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected HTTP"))) as client:
        assert expand_graph(graph, client, "W1")["retrieved_ids"] == []
        assert client.usage.requests == 0


def test_failed_retrieval_leaves_graph_unchanged(db, settings, raw_work):
    graph = ScholarlyGraph()
    graph.add(Paper.from_openalex(raw_work))
    before = graph.to_dict()
    settings.max_retries = 0
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(503))) as client:
        with pytest.raises(RetrievalError):
            expand_graph(graph, client, "W1")
    assert graph.to_dict() == before


@pytest.mark.parametrize("kwargs", [{"paper_id": "W999"}, {"paper_id": "W2"}, {"paper_id": "W1", "limit": 0},
                                   {"paper_id": "W1", "limit": 101}, {"paper_id": "W1", "direction": "related"}])
def test_invalid_expansion_fails_before_api(db, settings, raw_work, kwargs):
    graph = ScholarlyGraph()
    graph.add(Paper.from_openalex(raw_work))
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected HTTP"))) as client:
        with pytest.raises(ValueError):
            expand_graph(graph, client, **kwargs)


def test_legacy_anchor_can_be_hydrated_without_inventing_edges(db, settings, raw_work):
    graph = ScholarlyGraph()
    graph.add(Paper.from_openalex(raw_work))
    legacy = graph.to_dict()
    legacy["graph"] = {}
    next(n for n in legacy["nodes"] if n["id"].endswith("W1")).pop("metadata")
    graph = ScholarlyGraph.from_dict(legacy)
    db.save_papers([Paper.from_openalex(raw_work)])
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": [{"id": "W2"}]}))) as client:
        result = expand_graph(graph, client, "W1")
        assert result["new_resolved_papers"] == 1
        assert graph.paper("W1") is not None
