"""Provider quota waiting must not be reported as fast model inference or hidden retries."""

import json
import httpx
import pytest
from openalex_agent.config import Settings
from openalex_agent.groq_pilot import duration_seconds, TokenPacer, PacedPlanner
from openalex_agent.llm import PlannerError


@pytest.mark.parametrize(("value", "expected"), [("27.502s", 27.502), ("2m3.5s", 123.5),
    ("450ms", .45), ("1h2m3s", 3723), ("2", 2), (None, None), ("-2s", None), ("bad", None)])
def test_duration_headers(value, expected):
    assert duration_seconds(value) == expected


def test_pacing_uses_token_reset_not_daily_request_reset_and_excludes_secrets(tmp_path):
    now = [0.0]
    def sleep(seconds):
        now[0] += seconds
    p = TokenPacer(tmp_path/"headers.jsonl", clock=lambda: now[0], sleep=sleep)
    p.observe(httpx.Response(200, headers={"x-ratelimit-reset-tokens": "27.5s",
        "x-ratelimit-reset-requests": "5m45s", "Authorization": "PRIVATE"}))
    assert p.wait() == 28.5
    assert "PRIVATE" not in p.path.read_text()
    assert p.wait() == 0


def test_rate_limit_is_not_retried_and_next_call_wait_is_separate(tmp_path):
    now, requests = [0.0], []
    def sleep(seconds):
        now[0] += seconds
    p = TokenPacer(tmp_path/"headers.jsonl", clock=lambda: now[0], sleep=sleep)
    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"retry-after": "7", "x-ratelimit-reset-tokens": "3s"})
        return httpx.Response(200, headers={"x-ratelimit-reset-tokens": "3s"}, json={
            "choices": [{"message": {"content": json.dumps({"action": "STOP_RETRIEVAL", "arguments": {}})}}]})
    settings = Settings(llm_base_url="https://provider.example/v1", llm_api_key="PRIVATE", llm_model="test")
    with PacedPlanner(settings, p, transport=httpx.MockTransport(handle)) as planner:
        with pytest.raises(PlannerError, match="rate_limited"):
            planner.decide({})
        assert len(requests) == planner.usage.calls == 1
        assert planner.decide({})["action"] == "STOP_RETRIEVAL"
        assert len(requests) == 2 and planner.usage.failures == 1
        assert planner.quota_wait_seconds == planner.last_call["quota_wait_seconds"] == 8
        assert planner.usage.seconds < planner.quota_wait_seconds
