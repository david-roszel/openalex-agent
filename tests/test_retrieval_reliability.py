"""Milestone 1 regression tests; all HTTP uses synthetic local transports."""

from dataclasses import asdict, replace
from email.utils import formatdate
import json
import httpx
import pytest
from openalex_agent.config import Settings
from openalex_agent.models import Paper, reconstruct_abstract
from openalex_agent.openalex import BudgetExceeded, OpenAlexClient, RetrievalError
from openalex_agent.storage import Database


def test_partial_optional_entities_never_discard_usable_work(raw_work):
    raw_work.update(authorships=[None, {"author": {"id": "invalid", "display_name": "TEST RAW NAME"},
        "institutions": [None, {"id": "invalid"}, {"id": "I2", "display_name": "TEST INSTITUTION"}]}],
        topics=[None, {"id": "bad"}, {"id": "T2", "display_name": "TEST TOPIC"}],
        referenced_works=[None, "invalid", "W2", "https://openalex.org/W2"],
        related_works="not-an-array", publication_year=True, cited_by_count=-4,
        primary_location={"source": "not-an-object"}, biblio=[])
    paper = Paper.from_openalex(raw_work)
    assert paper.id.endswith("/W1") and paper.title
    assert paper.authorships[0].author is None
    assert paper.authorships[0].raw_name == "TEST RAW NAME"
    assert paper.authorships[0].institutions[0].id.endswith("/I2")
    assert paper.topics[0].id.endswith("/T2")
    assert paper.references == ["https://openalex.org/W2"]
    assert paper.related == [] and paper.year is None and paper.citation_count is None
    assert paper.venue is None and paper.biblio == {}


def test_malformed_abstract_entries_are_ignored():
    assert reconstruct_abstract(["not an index"]) is None
    assert reconstruct_abstract({"SYNTHETIC": [0, True, -1, "2"], "TEST": [1], "bad": 3}) == "SYNTHETIC TEST"


@pytest.mark.parametrize("payload", [
    {"meta": {}}, {"results": {}}, {"results": [], "meta": "bad"},
    {"results": [], "meta": {"next_cursor": 1}},
    {"results": [{"id": "W1"}], "meta": {"next_cursor": "*"}},
])
def test_malformed_schema_is_not_cached_and_next_call_recovers(db, settings, raw_work, payload):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=payload if len(calls) == 1 else {"results": [raw_work], "meta": {}})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(RetrievalError):
            client.search("test")
        assert client.search("test")[0].id.endswith("/W1")
        assert len(calls) == 2 and client.usage.errors == 1


def test_transport_error_and_transient_status_retry_then_succeed(db, settings, raw_work):
    calls, delays = [], []
    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("DO NOT LOG TEST-SECRET", request=request)
        return httpx.Response(503) if len(calls) == 2 else httpx.Response(200, json={"results": [raw_work]})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle), sleep=delays.append) as client:
        assert len(client.search("test")) == 1
        assert client.usage.requests == 3 and client.usage.retries == client.usage.errors == 2
        assert 1 in delays and 2 in delays
        assert "TEST-SECRET" not in json.dumps(db.events(client.run_id))


def test_retry_after_http_date_is_respected(db, settings, raw_work, monkeypatch):
    monkeypatch.setattr("openalex_agent.openalex.time.time", lambda: 1700000000)
    calls, delays = [], []
    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": formatdate(1700000012, usegmt=True)})
        return httpx.Response(200, json={"results": [raw_work]})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle), sleep=delays.append) as client:
        client.search("test")
    assert 12 in delays


def test_long_retry_after_returns_promptly_and_daily_exhaustion_does_not_retry(db, settings):
    for headers, expected in [({"Retry-After": "120"}, RetrievalError),
                              ({"X-RateLimit-Remaining": "0"}, BudgetExceeded)]:
        delays = []
        with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: httpx.Response(429, headers=headers)),
                            sleep=delays.append) as client:
            with pytest.raises(expected):
                client.search("test")
            assert client.usage.requests == 1 and client.usage.retries == 0
            assert not any(delay > 0 for delay in delays)


def test_reported_cost_ceiling_and_unknown_cost_are_distinct(db, settings, raw_work):
    settings.max_cost_usd = 0.001
    def handle(request):
        return httpx.Response(200, json={"results": [raw_work], "meta": {"cost_usd": 0.001}})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle)) as client:
        client.search("first")
        assert client.usage.cost_reported_requests == 1 and client.usage.cost_unreported_requests == 0
        with pytest.raises(BudgetExceeded):
            client.search("second")
        assert len(client.search("first")) == 1  # cache remains available when budget is exhausted
    def malformed_cost(request):
        return httpx.Response(200, headers={"X-RateLimit-Credits-Used": "nan", "X-RateLimit-Remaining": "inf"},
                              json={"results": [raw_work], "meta": {"cost_usd": "nan"}})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(malformed_cost)) as client:
        client.search("unknown cost")
        assert client.usage.cost_unreported_requests == 1 and client.usage.cost_reported_requests == 0
        assert client.usage.remaining is None
        json.dumps(asdict(client.usage), allow_nan=False)


def test_search_resumes_cached_pages_after_failure(tmp_path, settings, raw_work):
    path = str(tmp_path / "recovery.sqlite")
    settings.max_retries = 0
    def interrupted(request):
        if request.url.params["cursor"] == "*":
            return httpx.Response(200, json={"results": [raw_work], "meta": {"next_cursor": "second"}})
        return httpx.Response(503)
    with Database(path) as db, OpenAlexClient(db, settings, transport=httpx.MockTransport(interrupted)) as client:
        with pytest.raises(RetrievalError):
            client.search("test", limit=2)
        assert len(db.papers()) == 1
    def resumed(request):
        assert request.url.params["cursor"] == "second"
        return httpx.Response(200, json={"results": [dict(raw_work, id="W2")], "meta": {}})
    with Database(path) as db, OpenAlexClient(db, settings, transport=httpx.MockTransport(resumed)) as client:
        assert len(client.search("test", limit=2)) == 2
        assert client.usage.requests == client.usage.cache_hits == 1


def test_unusable_records_do_not_hide_later_valid_pages(db, settings, raw_work):
    def handle(request):
        if request.url.params["cursor"] == "*":
            return httpx.Response(200, json={"results": [{"title": "SYNTHETIC missing ID"}], "meta": {"next_cursor": "second"}})
        return httpx.Response(200, json={"results": [raw_work], "meta": {}})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle)) as client:
        assert len(client.search("test", limit=2)) == 1
        assert client.usage.requests == 2


def test_cursor_cycle_is_reported_instead_of_silent_truncation(db, settings, raw_work):
    def handle(request):
        cursor = request.url.params["cursor"]
        return httpx.Response(200, json={"results": [raw_work], "meta": {"next_cursor": "second" if cursor == "*" else "*"}})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(RetrievalError, match="cycle"):
            client.search("test", limit=3)
        assert client.usage.requests == 2


def test_anonymous_cursor_authentication_error_has_actionable_message(db, settings, raw_work):
    def handle(request):
        if request.url.params["cursor"] == "*":
            return httpx.Response(200, json={"results": [raw_work], "meta": {"next_cursor": "second"}})
        return httpx.Response(401)
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(RetrievalError, match="activated OPENALEX_API_KEY"):
            client.search("test", limit=2)


def test_historical_cache_cannot_overwrite_newer_persisted_metadata(db):
    latest = Paper(id="https://openalex.org/W1", title="SYNTHETIC updated", retrieved_at="2026-01-02T00:00:00+00:00")
    db.save_papers([latest])
    db.save_papers([replace(latest, title="SYNTHETIC old", retrieved_at="2026-01-01T00:00:00+00:00")])
    assert db.get_paper("W1").title == latest.title


def test_endpoint_filters_and_entity_lookups_are_canonical(db, settings, raw_work):
    seen = []
    def handle(request):
        seen.append(request)
        if request.url.path == "/works/W1":
            return httpx.Response(200, json=dict(raw_work, related_works=["W3"]))
        for endpoint, kind in [("authors", "A"), ("institutions", "I"), ("topics", "T")]:
            if request.url.path.startswith(f"/{endpoint}/"):
                return httpx.Response(200, json={"id": f"https://openalex.org/{kind}1"})
        filter_value = request.url.params.get("filter", "")
        target = "W3" if filter_value == "openalex:W3" else "W2"
        return httpx.Response(200, json={"results": [dict(raw_work, id=target)], "meta": {}})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle)) as client:
        assert client.get("W1").id.endswith("W1")
        assert client.references("W1")[0].id.endswith("W2")
        assert client.related("W1")[0].id.endswith("W3")
        client.citations("W1", 1)
        client.topic("T1", 1)
        client.search("test", 1, filters="publication_year:2024")
        for kind in "AIT":
            client.entity(kind + "1", kind)
    filters = {r.url.params.get("filter") for r in seen}
    assert {"openalex:W2", "openalex:W3", "cites:W1", "topics.id:T1", "publication_year:2024"} <= filters


def test_batch_chunks_at_one_hundred_and_preserves_input_order(db, settings, raw_work):
    sizes = []
    def handle(request):
        ids = request.url.params["filter"].removeprefix("openalex:").split("|")
        sizes.append(len(ids))
        return httpx.Response(200, json={"results": [dict(raw_work, id=i) for i in reversed(ids)]})
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(handle)) as client:
        papers = client.batch([f"W{i}" for i in range(1, 102)])
    assert sizes == [100, 1]
    assert [p.id.rsplit("/", 1)[-1] for p in papers] == [f"W{i}" for i in range(1, 102)]


@pytest.mark.parametrize("kwargs", [{"timeout": 0}, {"timeout": float("nan")}, {"min_interval": float("inf")},
                                   {"max_requests": -1}, {"max_retries": True}, {"max_cost_usd": -1}])
def test_invalid_configuration_is_rejected_before_network(db, kwargs):
    with pytest.raises(ValueError):
        OpenAlexClient(db, Settings(**kwargs))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limits_do_not_contact_api(db, settings, limit):
    with OpenAlexClient(db, settings, transport=httpx.MockTransport(lambda r: pytest.fail("Unexpected HTTP"))) as client:
        with pytest.raises(ValueError):
            client.search("test", limit)
        with pytest.raises(ValueError):
            client.references("W1", limit)
        with pytest.raises(ValueError):
            client.page(per_page=limit)
