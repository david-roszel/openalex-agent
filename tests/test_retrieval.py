import json
import httpx
import pytest
from openalex_agent.models import Paper, canonical_id
from openalex_agent.openalex import BudgetExceeded, OpenAlexClient, RetrievalError
from openalex_agent.storage import Database


def test_normalization_and_missing_fields(raw_work):
    paper = Paper.from_openalex(raw_work)
    assert paper.abstract == "Adaptive retrieval reduces retrieval"
    assert paper.authorships[0].institutions[0].name == "TEST INSTITUTION"
    assert Paper.from_openalex({"id": "W9"}).title is None
    assert Paper.from_openalex({"id": "W9"}).citation_count is None
    assert Paper.from_openalex({"id": "W9", "cited_by_count": 0}).citation_count == 0
    assert Paper.from_dict(paper.to_dict()) == paper
    with pytest.raises(ValueError):
        canonical_id("https://evil.example/W1")


def test_search_dedup_cache_and_restart(tmp_path, raw_work, settings):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"results": [raw_work, raw_work], "meta": {"next_cursor": None, "cost_usd": 0.001}})
    path = str(tmp_path / "test.sqlite")
    for iteration in range(2):
        with Database(path) as db:
            client = OpenAlexClient(db, settings, transport=httpx.MockTransport(handle))
            try:
                papers = client.search("adaptive retrieval", 2)
                assert len(papers) == len(db.papers()) == 1
                assert client.get("W1").id == papers[0].id
                if iteration:
                    assert client.usage.requests == 0
                else:
                    assert client.usage.reported_cost_usd == 0.001
            finally:
                client.close()
    assert len(calls) == 1
    with Database(path) as db:
        timestamp = db.get_paper("W1").retrieved_at
        client = OpenAlexClient(db, settings, transport=httpx.MockTransport(handle))
        assert client.search("adaptive retrieval", 2)[0].retrieved_at == timestamp
        client.close()


def test_incomplete_records_are_skipped_with_an_audit_event(db, settings, raw_work):
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda request: httpx.Response(200,
        json={"results": [{"title": "NO ID: SYNTHETIC"}, raw_work], "meta": {}})))
    assert len(client.search("retrieval")) == 1
    assert any(e["kind"] == "metadata_skipped" for e in db.events(client.run_id))
    client.close()


def test_cursor_pagination(db, settings, raw_work):
    def handle(request):
        second = request.url.params["cursor"] == "next"
        work = dict(raw_work, id="W2" if second else "W1")
        return httpx.Response(200, json={"results": [work], "meta": {"next_cursor": None if second else "next"}})
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(handle))
    assert len(client.search("retrieval", 2)) == 2
    assert client.usage.requests == 2
    client.close()


def test_backoff_and_hard_request_budget(db, settings, raw_work):
    settings.max_requests = 2
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "2"})
    delays = []
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(handle), sleep=delays.append)
    with pytest.raises(BudgetExceeded):
        client.search("retrieval")
    assert len(calls) == client.usage.requests == 2
    assert 2 in delays
    client.close()


def test_no_secret_in_errors_or_events(db, settings):
    settings.api_key = "TEST-SECRET"
    settings.email = "private@example.test"
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(RetrievalError) as error:
        client.search("query")
    serialized = json.dumps(db.events(client.run_id)) + str(error.value) + repr(settings)
    assert settings.api_key not in serialized
    assert settings.email not in serialized
    client.close()


def test_batch_reuses_persisted_metadata(db, settings, raw_work):
    db.save_papers([Paper.from_openalex(raw_work)])
    def handle(request):
        assert request.url.params["filter"] == "openalex:W2"
        return httpx.Response(200, json={"results": [{"id": "W2"}], "meta": {}})
    client = OpenAlexClient(db, settings, transport=httpx.MockTransport(handle))
    assert len(client.batch(["W1", "W2", "W1"])) == 2
    assert client.usage.requests == 1
    client.close()


def test_nonretryable_and_invalid_json(db, settings):
    for response in [httpx.Response(400), httpx.Response(200, text="not JSON")]:
        client = OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: response))
        with pytest.raises(RetrievalError):
            client.search("query")
        assert client.usage.requests == 1
        client.close()
