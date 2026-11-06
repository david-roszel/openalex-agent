import json
import httpx
import pytest
from openalex_agent.corpus import LocalCorpus, acquire, freeze
from openalex_agent.graph import ScholarlyGraph
from openalex_agent.models import Paper
from openalex_agent.openalex import OpenAlexClient, RetrievalError
from openalex_agent.selection import Scoring, score_candidates


def test_freeze_integrity_and_local_actions(raw_work, tmp_path):
    p = Paper.from_openalex(raw_work)
    target = Paper(id="https://openalex.org/W2", title="SYNTHETIC retrieval target")
    freeze([p, target], tmp_path, acquisition=[{"source": "SYNTHETIC TEST ONLY"}])
    corpus = LocalCorpus(tmp_path)
    assert corpus.search("adaptive retrieval")[0].id == p.id
    assert corpus.references(p.id) == [target]
    assert corpus.citations(target.id) == [p]
    assert corpus.topic("T1") == [p]
    with pytest.raises(ValueError):
        freeze([p], tmp_path, acquisition=[])
    with (tmp_path / "papers.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        LocalCorpus(tmp_path)


def test_graph_scores_use_observed_support_not_hidden_metadata(raw_work):
    p = Paper.from_openalex(raw_work)
    graph = ScholarlyGraph()
    graph.add(p)
    scores = score_candidates("adaptive retrieval", {p.id: p}, graph, Scoring(min_relevance=0))
    target = next(s for s in scores if s["id"].endswith("W2"))
    assert target["relevance_is_proxy"]
    assert 0 <= target["components"]["importance"] <= 1
    graph_without_link = ScholarlyGraph()
    graph_without_link.add(Paper.from_dict(dict(p.to_dict(), references=[])))
    assert len(score_candidates("adaptive retrieval", {p.id: p}, graph_without_link, Scoring(min_relevance=0))) == 1
    assert score_candidates("completely unrelated astrophysics", {p.id: p}, graph, Scoring(min_relevance=0.9)) == []


def test_acquisition_recovers_after_failed_page(db, settings, raw_work, tmp_path):
    settings.max_retries = 0
    calls = []
    def first(request):
        calls.append(request.url.params["cursor"])
        if calls[-1] != "*":
            return httpx.Response(503)
        return httpx.Response(200, json={"results": [raw_work], "meta": {"next_cursor": "second"}})
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(first))
    with pytest.raises(RetrievalError):
        acquire(client, ["retrieval"], str(tmp_path), per_query=2)
    client.close()
    def second(request):
        assert request.url.params["cursor"] == "second"
        return httpx.Response(200, json={"results": [dict(raw_work, id="W2")], "meta": {"next_cursor": None}})
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(second))
    assert acquire(client, ["retrieval"], str(tmp_path), per_query=2)["count"] == 2
    assert client.usage.requests == 1
    client.close()
