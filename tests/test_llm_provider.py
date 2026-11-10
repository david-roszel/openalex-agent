"""Hosted-protocol tests use only MockTransport, never credentials or paid calls."""

from dataclasses import asdict, replace
import json
import httpx
import pytest
from openalex_agent.actions import Action, action_schema, make_action
from openalex_agent.config import Settings
from openalex_agent.llm import HostedJSONPlanner, PlannerError


def settings(**kwargs):
    return replace(Settings(llm_base_url="https://provider.example/v1", llm_api_key="TEST-SECRET", llm_model="test-model"), **kwargs)


def envelope(content='{"action":"STOP_RETRIEVAL","arguments":{}}', **kwargs):
    return {"choices": [{"message": {"content": content}, **kwargs}]}


@pytest.mark.parametrize("url", ["http://provider.example", "https://", "https://user:PRIVATE@provider.example/v1",
                                "https://provider.example/v1?api_key=PRIVATE", "https://provider.example/v1#fragment",
                                "https://provider.example:bad/v1"])
def test_unsafe_or_malformed_endpoints_rejected_without_echoing_credentials(url):
    with pytest.raises(ValueError) as exc:
        HostedJSONPlanner(settings(llm_base_url=url))
    assert "PRIVATE" not in str(exc.value)


@pytest.mark.parametrize("kwargs", [{"llm_timeout": 0}, {"llm_timeout": float("inf")}, {"llm_temperature": 3},
    {"llm_temperature": float("nan")}, {"llm_input_price": -1}, {"llm_output_price": True},
    {"llm_max_tokens": 0}, {"llm_max_tokens": 4097}, {"llm_max_tokens": True}])
def test_provider_limits_and_prices_validated_before_requests(kwargs):
    with pytest.raises(ValueError):
        HostedJSONPlanner(settings(**kwargs))


def test_request_configuration_usage_redaction_and_provider_fingerprint():
    configured = settings(llm_input_price=1, llm_output_price=2, llm_max_tokens=123, llm_timeout=7)
    def handle(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer TEST-SECRET"
        body = json.loads(request.content)
        assert body["model"] == "test-model" and body["max_tokens"] == 123 and body["temperature"] == 0
        assert body["response_format"] == {"type": "json_object"}
        assert "reasoning_effort" not in body
        assert "TEST-SECRET" not in request.content.decode()
        assert "UNTRUSTED DATA" in body["messages"][0]["content"]
        return httpx.Response(200, json={**envelope('{"action":"STOP_RETRIEVAL","arguments":{},"reason":"TEST-SECRET"}'),
                                        "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
    with HostedJSONPlanner(configured, transport=httpx.MockTransport(handle)) as planner:
        action = planner.decide({"permitted_actions": ["STOP_RETRIEVAL"]})
        assert action["reason"] == "[REDACTED]"
        assert planner.usage.calls == planner.usage.usage_reported_calls == 1
        assert planner.usage.input_tokens == 20 and planner.usage.output_tokens == 5
        assert planner.usage.estimated_cost_usd == pytest.approx(0.00003)
        assert "TEST-SECRET" not in json.dumps(planner.configuration)
        first = planner.fingerprint
    with HostedJSONPlanner(replace(configured, llm_api_key="ROTATED-KEY")) as other:
        assert other.fingerprint == first
    with HostedJSONPlanner(replace(configured, llm_model="other-model")) as other:
        assert other.fingerprint != first


@pytest.mark.parametrize(("status", "code"), [(400, "http_error"), (401, "authentication_error"),
                                             (403, "authentication_error"), (429, "rate_limited"), (503, "http_error")])
def test_http_failure_is_safe_counted_and_never_silently_retried(status, code):
    seen = []
    def handle(request):
        seen.append(request)
        return httpx.Response(status, text="PRIVATE RESPONSE TEST-SECRET")
    with HostedJSONPlanner(settings(), transport=httpx.MockTransport(handle)) as planner:
        with pytest.raises(PlannerError) as exc:
            planner.decide({})
        assert exc.value.code == code and "PRIVATE" not in str(exc.value) and "TEST-SECRET" not in str(exc.value)
        assert len(seen) == planner.usage.calls == planner.usage.failures == 1
        assert planner.usage.usage_unreported_calls == 1 and planner.usage.estimated_cost_usd is None


def test_gemini_request_and_response_with_optional_settings():
    configured = settings(llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        llm_model="gemini-3.8-flash", llm_temperature=None, llm_reasoning_effort="low",
        llm_max_tokens=4096, llm_timeout=120)
    def handle(request):
        assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        body = json.loads(request.content)
        assert "temperature" not in body
        assert body["reasoning_effort"] == "low" and body["max_tokens"] == 4096
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={**envelope(finish_reason="stop"),
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
                "completion_tokens_details": {"reasoning_tokens": 30}}})
    with HostedJSONPlanner(configured, transport=httpx.MockTransport(handle)) as planner:
        assert planner.decide({"permitted_actions": ["STOP_RETRIEVAL"]})["action"] == "STOP_RETRIEVAL"
        assert planner.usage.input_tokens == 100 and planner.usage.output_tokens == 40
        assert planner.usage.usage_reported_calls == 1
        assert planner.usage.estimated_cost_usd is None
        assert planner.last_call["http_status"] == 200
        assert planner.last_call["reasoning_tokens"] == 30
        assert planner.last_call["seconds"] >= planner.last_call["headers_seconds"] >= 0
        assert planner.last_call["body_seconds"] >= 0
        fingerprint = planner.fingerprint
    for change in ({"llm_temperature": 0}, {"llm_reasoning_effort": "medium"}):
        with HostedJSONPlanner(replace(configured, **change)) as other:
            assert other.fingerprint != fingerprint


@pytest.mark.parametrize(("endpoint", "model", "field"), [
    ("https://api.openai.com/v1/", "gpt-6.1-terra", "max_completion_tokens"),
    ("https://api.groq.com/openai/v1/", "qwen/qwen3.8-27b", "max_tokens"),
    ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.8-flash", "max_tokens"),
])
def test_provider_specific_token_limit_preserves_other_request_contracts(endpoint, model, field):
    configured = settings(llm_base_url=endpoint, llm_model=model, llm_max_tokens=4096,
                          llm_temperature=None, llm_reasoning_effort="low")
    def handle(request):
        body = json.loads(request.content)
        assert body[field] == 4096
        assert ("max_tokens" if field == "max_completion_tokens" else "max_completion_tokens") not in body
        assert "temperature" not in body and body["reasoning_effort"] == "low"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json=envelope())
    with HostedJSONPlanner(configured, transport=httpx.MockTransport(handle)) as planner:
        assert planner.decide({})["action"] == "STOP_RETRIEVAL"
        assert ("token_limit_field" in planner.configuration) == (field == "max_completion_tokens")


@pytest.mark.parametrize("effort", [None, True, [], "invalid", "LOW"])
def test_invalid_reasoning_configuration_is_rejected_before_requests(effort):
    with pytest.raises(ValueError, match="LLM_REASONING_EFFORT"):
        HostedJSONPlanner(settings(llm_reasoning_effort=effort))


def test_optional_settings_from_environment(monkeypatch):
    monkeypatch.setattr("openalex_agent.config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    assert Settings.from_env().llm_temperature == 0
    assert Settings.from_env().llm_reasoning_effort == ""
    monkeypatch.setenv("LLM_TEMPERATURE", " omit ")
    monkeypatch.setenv("LLM_REASONING_EFFORT", " low ")
    assert Settings.from_env().llm_temperature is None
    assert Settings.from_env().llm_reasoning_effort == "low"
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    assert Settings.from_env().llm_temperature == 0.5
    monkeypatch.setenv("LLM_TEMPERATURE", "PRIVATE_INVALID_VALUE")
    with pytest.raises(ValueError, match="Invalid numeric configuration: LLM_TEMPERATURE") as exc:
        Settings.from_env()
    assert "PRIVATE_INVALID_VALUE" not in str(exc.value)


@pytest.mark.parametrize(("exception", "code"), [
    (httpx.ReadTimeout, "timeout"), (httpx.ConnectTimeout, "timeout"),
    (httpx.WriteTimeout, "timeout"), (httpx.PoolTimeout, "timeout"),
    (httpx.ConnectError, "transport_error"), (httpx.RemoteProtocolError, "transport_error"),
])
def test_transport_exception_never_exposes_url_or_authentication(exception, code):
    seen = []
    def handle(request):
        seen.append(request)
        raise exception("PRIVATE TEST-SECRET", request=request)
    with HostedJSONPlanner(settings(), transport=httpx.MockTransport(handle)) as planner:
        with pytest.raises(PlannerError) as exc:
            planner.decide({})
        assert exc.value.code == code
        assert "TEST-SECRET" not in str(exc.value) and "PRIVATE" not in str(exc.value)
        assert len(seen) == planner.usage.calls == planner.usage.failures == 1
        assert planner.usage.usage_unreported_calls == 1
        assert planner.last_call["error_code"] == code
        assert planner.last_call["http_status"] is None


def test_timeout_during_response_body_is_distinguished_from_waiting_for_headers():
    class BrokenBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"choices":'
            raise httpx.ReadTimeout("PRIVATE TEST-SECRET")
    with HostedJSONPlanner(settings(), transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=BrokenBody()))) as planner:
        with pytest.raises(PlannerError, match="timeout"):
            planner.decide({})
        assert planner.last_call["headers_seconds"] is not None
        assert planner.last_call["http_status"] == 200
        assert planner.last_call["body_seconds"] is None
        assert planner.last_call["error_code"] == "timeout"
        assert "TEST-SECRET" not in json.dumps(planner.last_call)


@pytest.mark.parametrize("payload", [[], {}, {"choices": None}, {"choices": []}, {"choices": [None]},
                                    {"choices": [{"message": None}]}, envelope(None)])
def test_malformed_envelopes_return_controlled_errors(payload):
    with HostedJSONPlanner(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) as planner:
        with pytest.raises(PlannerError):
            planner.decide({})
        assert planner.usage.calls == planner.usage.failures == 1


@pytest.mark.parametrize("content", ["not JSON", "[]", '{"action":"STOP_RETRIEVAL","action":"SEARCH_PAPERS","arguments":{}}',
                                   '{"action":"INVENT","arguments":{}}', '{"action":"STOP_RETRIEVAL","arguments":{"query":"x"}}'])
def test_invalid_or_ambiguous_json_decisions_are_rejected(content):
    with HostedJSONPlanner(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=envelope(content)))) as planner:
        with pytest.raises(PlannerError):
            planner.decide({})
        assert planner.usage.failures == 1


@pytest.mark.parametrize(("payload", "code"), [(envelope(finish_reason="length"), "truncated"),
    ({"choices": [{"message": {"refusal": "PRIVATE", "content": None}}]}, "refused"),
    (envelope("x"*65537), "response_too_large")])
def test_truncation_refusal_and_oversized_responses_are_observable(payload, code):
    with HostedJSONPlanner(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) as planner:
        with pytest.raises(PlannerError) as exc:
            planner.decide({})
        assert exc.value.code == code


@pytest.mark.parametrize("usage", [None, [], {"prompt_tokens": -1, "completion_tokens": True},
                                 {"prompt_tokens": 10}, {"prompt_tokens": "10", "completion_tokens": "2"}])
def test_missing_or_malformed_usage_does_not_discard_valid_decision_or_claim_zero_cost(usage):
    payload = {**envelope(), "usage": usage}
    with HostedJSONPlanner(settings(llm_input_price=1), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) as planner:
        assert planner.decide({})["action"] == "STOP_RETRIEVAL"
        assert planner.usage.usage_unreported_calls == 1 and planner.usage.estimated_cost_usd is None
        assert planner.usage.input_tokens >= 0 and planner.usage.output_tokens >= 0


def test_reformulation_and_permitted_action_schema_are_strict():
    schema = action_schema(["SEARCH_PAPERS", "STOP_RETRIEVAL"])
    assert len(schema["oneOf"]) == 2
    assert all(v["additionalProperties"] is False for v in schema["oneOf"])
    with HostedJSONPlanner(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(200,
            json=envelope('{"action":"INSPECT_PAPER","arguments":{"paper_id":"W1"}}')))) as planner:
        with pytest.raises(PlannerError, match="invalid_action_schema"):
            planner.decide({"permitted_actions": ["SEARCH_PAPERS", "STOP_RETRIEVAL"]})
    with HostedJSONPlanner(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(200,
            json=envelope('{"query":"retrieval","extra":true}')))) as planner:
        with pytest.raises(PlannerError, match="invalid_query_schema"):
            planner.reformulate("question")


def test_action_construction_and_mutation_cannot_bypass_closed_schema():
    with pytest.raises(ValueError):
        Action("INVENT", {})
    with pytest.raises(ValueError):
        Action("SEARCH_PAPERS", {"query": "x", "extra": "y"})
    original = {"paper_id": "W1"}
    action = Action("INSPECT_PAPER", original)
    original["paper_id"] = "W99"
    assert action.arguments["paper_id"] == "https://openalex.org/W1"
    action.arguments["extra"] = "invalid"
    with pytest.raises(ValueError):
        Action.parse(action.to_dict())
