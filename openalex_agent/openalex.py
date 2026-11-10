"""Synchronous, paced OpenAlex client with durable caching and bounded retries.

Official API rules checked 2026-10-29; see docs/api.md. No raw URLs,
authorization headers, response bodies, or exception strings enter audit logs.
"""

from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import math
import time
from typing import Callable, Iterable
import uuid
import httpx
from .config import Settings
from .models import Paper, canonical_id, utcnow
from .storage import Database

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Safe, credential-free retrieval error."""


class BudgetExceeded(RetrievalError):
    """Hard request limit or observed external usage ceiling reached."""


@dataclass
class Usage:
    requests: int = 0
    cache_hits: int = 0
    retries: int = 0
    errors: int = 0
    api_seconds: float = 0.0
    wait_seconds: float = 0.0
    reported_cost_usd: float = 0.0
    reported_credits: float = 0.0
    cost_reported_requests: int = 0
    cost_unreported_requests: int = 0
    remaining: float | None = None


class OpenAlexClient:
    def __init__(self, db: Database, settings: Settings | None = None, *, transport=None,
                 sleep: Callable[[float], None] = time.sleep, run_id: str | None = None):
        self.db = db
        self.settings = settings or Settings.from_env()
        self.settings.validate_retrieval()
        self.run_id = run_id or str(uuid.uuid4())
        self.usage = Usage()
        self.sleep = sleep
        self.last_request = 0.0
        self.acquired_at = utcnow()
        headers = {"User-Agent": "openalex-agent/0.1 (academic retrieval research)"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        self.http = httpx.Client(base_url="https://api.openalex.org", headers=headers,
                                 timeout=self.settings.timeout, transport=transport)

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "OpenAlexClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @staticmethod
    def _number(value) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    @staticmethod
    def _positive_limit(limit: int) -> None:
        if type(limit) is not int or limit < 1:
            raise ValueError("Retrieval limit must be a positive integer")

    @staticmethod
    def _validate_payload(endpoint: str, payload: dict, request_cursor: str | None = None) -> None:
        """Check the endpoint contract before persisting a successful response."""
        if not isinstance(payload, dict):
            raise RetrievalError("OpenAlex returned a malformed JSON object")
        meta = payload.get("meta")
        if meta is not None and not isinstance(meta, dict):
            raise RetrievalError("OpenAlex returned malformed response metadata")
        if endpoint == "/works":
            if not isinstance(payload.get("results"), list):
                raise RetrievalError("OpenAlex list response has no results array")
            cursor = (meta or {}).get("next_cursor")
            if cursor is not None and (not isinstance(cursor, str) or not cursor):
                raise RetrievalError("OpenAlex returned an invalid pagination cursor")
            if request_cursor is not None and cursor == request_cursor and payload["results"]:
                raise RetrievalError("OpenAlex returned a repeated pagination cursor")
        else:
            kinds = {"works": "W", "authors": "A", "institutions": "I", "topics": "T"}
            kind = kinds.get(endpoint.strip("/").split("/")[0])
            if kind:
                try:
                    canonical_id(payload.get("id"), kind)
                except ValueError:
                    raise RetrievalError("OpenAlex singleton has no valid identifier") from None

    def _request(self, endpoint: str, params: dict | None = None, *, refresh: bool = False) -> dict:
        params = dict(params or {})
        key = hashlib.sha256(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
        if not refresh and (cached := self.db.cache_get(key)) is not None:
            self._validate_payload(endpoint, cached, params.get("cursor"))
            self.acquired_at = self.db.cache_acquired(key)
            self.usage.cache_hits += 1
            self.db.event(self.run_id, "cache_hit", {"endpoint": endpoint, "key": key})
            return cached
        if self.settings.email:
            params["mailto"] = self.settings.email
        for attempt in range(self.settings.max_retries + 1):
            if self.usage.requests >= self.settings.max_requests:
                raise BudgetExceeded("OpenAlex request limit reached")
            if self.usage.reported_cost_usd >= self.settings.max_cost_usd:
                raise BudgetExceeded("Observed OpenAlex cost ceiling reached")
            if self.usage.remaining is not None and self.usage.remaining <= 0:
                raise BudgetExceeded("OpenAlex reports exhausted daily budget")
            wait_start = time.perf_counter()
            self.sleep(max(0.0, self.settings.min_interval - (time.monotonic() - self.last_request)))
            self.usage.wait_seconds += time.perf_counter() - wait_start
            self.last_request = time.monotonic()
            started = time.perf_counter()
            self.usage.requests += 1
            self.usage.cost_unreported_requests += 1
            delay = min(2 ** attempt, 30)
            status = None
            response = None
            transport_error = None
            try:
                response = self.http.get(endpoint, params=params)
                status = response.status_code
            except httpx.TransportError as exc:
                transport_error = {"type": type(exc).__name__, "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None}
            finally:
                elapsed = time.perf_counter() - started
                self.usage.api_seconds += elapsed
            if response is not None:
                credits = self._number(response.headers.get("X-RateLimit-Credits-Used"))
                if credits is not None:
                    self.usage.reported_credits += credits
                remaining = self._number(response.headers.get("X-RateLimit-Remaining"))
                if remaining is not None:
                    self.usage.remaining = remaining
                if response.is_success:
                    try:
                        payload = response.json()
                        if not isinstance(payload, dict):
                            raise ValueError("Expected object")
                    except (ValueError, TypeError):
                        self.usage.errors += 1
                        self.db.event(self.run_id, "request_error", {"endpoint": endpoint, "error": "invalid_json"})
                        raise RetrievalError("OpenAlex returned invalid JSON") from None
                    meta = payload.get("meta")
                    cost = self._number(meta.get("cost_usd")) if isinstance(meta, dict) else None
                    if cost is not None:
                        self.usage.reported_cost_usd += cost
                        self.usage.cost_reported_requests += 1
                        self.usage.cost_unreported_requests -= 1
                    try:
                        self._validate_payload(endpoint, payload, params.get("cursor"))
                    except RetrievalError:
                        self.usage.errors += 1
                        self.db.event(self.run_id, "request_error", {"endpoint": endpoint, "error": "invalid_schema"})
                        raise
                    self.acquired_at = self.db.cache_put(key, payload)
                    self.db.event(self.run_id, "request", {"endpoint": endpoint, "key": key, "status": status,
                                                          "seconds": elapsed, "usage": asdict(self.usage)})
                    return payload
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        seconds = float(retry_after)
                        if math.isfinite(seconds):
                            delay = max(0.0, seconds)
                    except ValueError:
                        try:
                            delay = max(0.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
                        except (TypeError, ValueError, OverflowError):
                            pass
            self.usage.errors += 1
            self.db.event(self.run_id, "request_error", {"endpoint": endpoint, "status": status, "attempt": attempt + 1,
                **({"transport_error": transport_error} if transport_error else {})})
            logger.warning("OpenAlex request failed: endpoint=%s status=%s attempt=%s", endpoint, status, attempt + 1)
            if status is not None and status not in (408, 429, 500, 502, 503, 504):
                if status == 401 and params.get("cursor") not in (None, "*") and not self.settings.api_key:
                    raise RetrievalError("Further cursor pages require an activated OPENALEX_API_KEY; "
                                         "acquired pages are cached, so configure the key and rerun") from None
                raise RetrievalError(f"OpenAlex HTTP {status}; check credentials and query parameters") from None
            if self.usage.remaining == 0:
                raise BudgetExceeded("OpenAlex reports exhausted daily budget; retry after its reset")
            if attempt < self.settings.max_retries:
                # Long Retry-After means leave resumable work on disk instead of blocking indefinitely.
                if delay > 60:
                    raise RetrievalError("OpenAlex requested a long pause; retry this resumable command later")
                self.usage.retries += 1
                wait_start = time.perf_counter()
                self.sleep(delay)
                self.usage.wait_seconds += time.perf_counter() - wait_start
        raise RetrievalError("OpenAlex request failed after bounded retries") from None

    def normalize(self, raw: Iterable[dict]) -> list[Paper]:
        papers = {}
        for item in raw:
            try:
                paper = Paper.from_openalex(item)
            except (ValueError, TypeError, AttributeError):
                self.db.event(self.run_id, "metadata_skipped", {"reason": "invalid OpenAlex record"})
                continue
            paper.retrieved_at = self.acquired_at
            papers[paper.id] = paper
        self.db.save_papers(papers.values())
        return list(papers.values())

    def page(self, *, query: str = "", filters: str = "", cursor: str = "*", per_page: int = 100,
             refresh: bool = False) -> tuple[list[Paper], str | None]:
        if type(per_page) is not int or not 1 <= per_page <= 100:
            raise ValueError("per_page must be between 1 and 100")
        if not isinstance(cursor, str) or not cursor.strip():
            raise ValueError("Cursor must be a nonempty string")
        params: dict = {"cursor": cursor, "per_page": per_page}
        if query:
            params["search"] = query
        if filters:
            params["filter"] = filters
        # Cache is checked before contacting the API, including without a current key.
        payload = self._request("/works", params, refresh=refresh)
        next_cursor = (payload.get("meta") or {}).get("next_cursor") if payload["results"] else None
        if next_cursor == cursor:
            raise RetrievalError("OpenAlex returned a repeated cursor; acquired pages remain cached")
        return self.normalize(payload["results"]), next_cursor

    def search(self, query: str, limit: int = 20, *, filters: str = "", refresh: bool = False) -> list[Paper]:
        self._positive_limit(limit)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Provide a nonempty query and positive limit")
        return self._collect(query=query, filters=filters, limit=limit, refresh=refresh)

    def _collect(self, *, query: str = "", filters: str = "", limit: int = 100,
                 refresh: bool = False) -> list[Paper]:
        papers: dict[str, Paper] = {}
        self._positive_limit(limit)
        cursor = "*"
        seen = set()
        while cursor and len(papers) < limit:
            if cursor in seen:
                raise RetrievalError("OpenAlex returned a cursor cycle; acquired pages remain cached")
            seen.add(cursor)
            batch, cursor = self.page(query=query, filters=filters, cursor=cursor,
                                      per_page=min(100, limit), refresh=refresh)
            papers.update((p.id, p) for p in batch)
        return list(papers.values())[:limit]

    def get(self, paper_id: str, *, refresh: bool = False) -> Paper:
        paper_id = canonical_id(paper_id)
        if not refresh and (paper := self.db.get_paper(paper_id)):
            self.usage.cache_hits += 1
            return paper
        normalized = self.normalize([self._request("/works/" + paper_id.rsplit("/", 1)[-1], refresh=refresh)])
        if not normalized:
            raise RetrievalError("OpenAlex singleton has invalid metadata")
        return normalized[0]

    def batch(self, ids: Iterable[str], *, refresh: bool = False) -> list[Paper]:
        ids = list(dict.fromkeys(canonical_id(i) for i in ids))
        found: dict[str, Paper] = {}
        if not refresh:
            for paper_id in ids:
                if paper := self.db.get_paper(paper_id):
                    found[paper_id] = paper
                    self.usage.cache_hits += 1
        missing = [i for i in ids if i not in found]
        for start in range(0, len(missing), 100):
            chunk = missing[start:start + 100]
            batch = self._collect(filters="openalex:" + "|".join(i.rsplit("/", 1)[-1] for i in chunk),
                                  limit=len(chunk), refresh=refresh)
            found.update((p.id, p) for p in batch)
        return [found[i] for i in ids if i in found]

    def references(self, paper_id: str, limit: int = 100) -> list[Paper]:
        self._positive_limit(limit)
        return self.batch(self.get(paper_id).references[:limit])

    def related(self, paper_id: str, limit: int = 100) -> list[Paper]:
        self._positive_limit(limit)
        return self.batch(self.get(paper_id).related[:limit])

    def citations(self, paper_id: str, limit: int = 100) -> list[Paper]:
        return self._collect(filters="cites:" + canonical_id(paper_id).rsplit("/", 1)[-1], limit=limit)

    def topic(self, topic_id: str, limit: int = 100) -> list[Paper]:
        return self._collect(filters="topics.id:" + canonical_id(topic_id, "T").rsplit("/", 1)[-1], limit=limit)

    def entity(self, entity_id: str, kind: str) -> dict:
        endpoint = {"A": "authors", "I": "institutions", "T": "topics"}[kind]
        return self._request(f"/{endpoint}/" + canonical_id(entity_id, kind).rsplit("/", 1)[-1])
