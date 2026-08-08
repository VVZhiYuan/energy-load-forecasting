import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from src.agent_contract import MAX_RESPONSE_BYTES, validate_agent_response
from src.ai_config import AISettings
from src.ai_provider import (
    AIProviderError,
    AgentContext,
    AgentResponse,
    DisabledAIProvider,
    MockAIProvider,
    _NO_REDIRECT_OPENER,
    _NoRedirectHandler,
    OpenAICompatibleAIProvider,
    build_provider,
)


def make_context():
    return AgentContext(
        summary={
            "selected_model": "Ridge",
            "horizon": "1h",
            "peak_load": 123.4,
        },
        forecast_rows=[
            {"step": 1, "prediction": 100.0, "p10": 95.0, "p50": 100.0, "p90": 105.0},
            {"step": 2, "prediction": 101.0, "p10": 96.0, "p50": 101.0, "p90": 106.0},
        ],
        comparison_rows=[
            {"model": "Ridge", "validation_mae": 1.2, "selected": True},
            {"model": "LightGBM", "validation_mae": 1.3, "selected": False},
        ],
        recent_load_rows=[
            {"timestamp": "2026-08-06T12:00:00+08:00", "load": 98.0},
            {"timestamp": "2026-08-06T12:15:00+08:00", "load": 99.0},
        ],
    )


def valid_content() -> dict[str, object]:
    return {
        "status": "ok",
        "summary": "Forecast load remains broadly stable.",
        "risk_level": "low",
        "evidence": ["The peak remains within the expected interval."],
        "recommendations": [
            {
                "action": "Review flexible load before the forecast peak.",
                "reason": "The forecast identifies a higher-demand window.",
                "priority": "medium",
                "requires_human_approval": True,
            }
        ],
        "forecast_unchanged": True,
        "execution_enabled": False,
    }


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self, size=-1):
        return self.payload if size < 0 else self.payload[:size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def openai_payload(content: object) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"content": json.dumps(content)}}]}
    ).encode("utf-8")


def test_settings_from_env_defaults(monkeypatch):
    for key in (
        "ENERGY_AI_PROVIDER",
        "ENERGY_AI_BASE_URL",
        "ENERGY_AI_MODEL",
        "ENERGY_AI_API_KEY",
        "ENERGY_AI_ALLOWED_HOSTS",
        "ENERGY_AI_TIMEOUT_SECONDS",
        "ENERGY_AI_MAX_RESPONSE_BYTES",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = AISettings.from_env()

    assert settings.provider == "disabled"
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.model == "gpt-4o-mini"
    assert settings.api_key is None
    assert settings.allowed_hosts == ()
    assert settings.timeout_seconds == 30.0
    assert settings.max_response_bytes == MAX_RESPONSE_BYTES


def test_settings_from_env_parses_provider_transport_limits(monkeypatch):
    monkeypatch.setenv("ENERGY_AI_ALLOWED_HOSTS", " LLM.EXAMPLE, local.example ")
    monkeypatch.setenv("ENERGY_AI_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("ENERGY_AI_MAX_RESPONSE_BYTES", "4096")

    settings = AISettings.from_env()

    assert settings.allowed_hosts == ("llm.example", "local.example")
    assert settings.timeout_seconds == 12.5
    assert settings.max_response_bytes == 4096


@pytest.mark.parametrize(
    ("environment", "value"),
    [
        ("ENERGY_AI_TIMEOUT_SECONDS", "not-a-number"),
        ("ENERGY_AI_MAX_RESPONSE_BYTES", "12.5"),
    ],
)
def test_settings_from_env_rejects_malformed_transport_values(monkeypatch, environment, value):
    monkeypatch.setenv(environment, value)

    with pytest.raises(ValueError, match=environment):
        AISettings.from_env()


@pytest.mark.parametrize(
    "provider_name, expected_type",
    [
        ("disabled", DisabledAIProvider),
        ("mock", MockAIProvider),
        ("openai-compatible", OpenAICompatibleAIProvider),
    ],
)
def test_build_provider_selects_expected_implementation(provider_name, expected_type):
    settings = AISettings(
        provider=provider_name,
        base_url="https://example.invalid/v1",
        model="demo-model",
        api_key="secret",
        allowed_hosts=("example.invalid",),
    )

    provider = build_provider(settings)

    assert isinstance(provider, expected_type)


def test_disabled_provider_does_not_make_network_requests(monkeypatch):
    provider = build_provider(AISettings(provider="disabled"))

    def forbidden_request(*args, **kwargs):
        raise AssertionError("disabled provider must not make network requests")

    monkeypatch.setattr("src.ai_provider.urlopen", forbidden_request)

    response = provider.analyze(make_context())

    assert isinstance(response, AgentResponse)
    assert response.provider == "disabled"
    assert response.model == "none"
    assert response.content == validate_agent_response(response.content)
    assert response.content["status"] == "disabled"
    assert "disabled" in response.content["summary"].lower()


def test_mock_provider_is_deterministic_offline_and_contract_valid(monkeypatch):
    provider = build_provider(AISettings(provider="mock"))
    monkeypatch.setattr(
        "src.ai_provider.urlopen",
        lambda *args, **kwargs: pytest.fail("mock provider must not make network requests"),
    )

    first = provider.analyze(make_context())
    second = provider.analyze(make_context())

    assert first.provider == "mock"
    assert first.model == "mock"
    assert first.content == second.content
    assert first.content == validate_agent_response(first.content)
    assert first.content["status"] == "ok"


def test_non_loopback_http_url_is_rejected_before_request(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="http://llm.example/v1",
        model="forecast-writer",
        allowed_hosts=("llm.example",),
    )
    monkeypatch.setattr(
        "src.ai_provider.urlopen",
        lambda *args, **kwargs: pytest.fail("unsafe endpoint must not make a request"),
    )

    with pytest.raises(ValueError, match="HTTPS"):
        build_provider(settings)


def test_remote_host_requires_explicit_allowlist(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="https://llm.example/v1",
        model="forecast-writer",
    )
    monkeypatch.setattr(
        "src.ai_provider.urlopen",
        lambda *args, **kwargs: pytest.fail("unsafe endpoint must not make a request"),
    )

    with pytest.raises(ValueError, match="allowlist"):
        build_provider(settings).analyze(make_context())


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://llm.example/v1",
        "https://user:secret@llm.example/v1",
        "https://llm.example/v1?debug=true",
        "https://llm.example/v1#fragment",
        "https:///v1",
    ],
)
def test_openai_provider_rejects_unsafe_base_urls(monkeypatch, base_url):
    settings = AISettings(
        provider="openai-compatible",
        base_url=base_url,
        model="forecast-writer",
        allowed_hosts=("llm.example",),
    )
    monkeypatch.setattr(
        "src.ai_provider.urlopen",
        lambda *args, **kwargs: pytest.fail("unsafe endpoint must not make a request"),
    )

    with pytest.raises(ValueError, match="base_url|HTTPS|credentials|query|fragment"):
        build_provider(settings)


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        (
            AISettings(
                base_url="http://remote.example/v1",
                model="forecast-writer",
                allowed_hosts=("remote.example",),
            ),
            "HTTPS",
        ),
        (
            AISettings(
                base_url="https://remote.example/v1",
                model="forecast-writer",
            ),
            "allowlist",
        ),
        (AISettings(timeout_seconds=0), "positive"),
    ],
)
def test_direct_openai_provider_rejects_unsafe_settings_before_network(
    monkeypatch, settings, error
):
    monkeypatch.setattr(
        "src.ai_provider.urlopen",
        lambda *args, **kwargs: pytest.fail("unsafe settings must not make a request"),
    )

    with pytest.raises(ValueError, match=error):
        OpenAICompatibleAIProvider(settings)


def test_loopback_http_is_allowed_for_local_model(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="http://127.0.0.1:11434/v1",
        model="local-model",
    )
    provider = build_provider(settings)
    monkeypatch.setattr(
        "src.ai_provider.urlopen", lambda *args, **kwargs: FakeResponse(openai_payload(valid_content()))
    )

    assert provider.analyze(make_context()).content["execution_enabled"] is False


@pytest.mark.parametrize(
    "redirect_target",
    [
        "https://unallowed.example/v1/chat/completions",
        "http://allowed.example/v1/chat/completions",
        "http://127.0.0.1:11434/v1/chat/completions",
    ],
)
def test_provider_redirects_fail_closed_for_unsafe_targets(redirect_target):
    request = Request("https://allowed.example/v1/chat/completions")

    assert any(
        isinstance(handler, _NoRedirectHandler)
        for handler in _NO_REDIRECT_OPENER.handlers
    )
    with pytest.raises(HTTPError):
        _NoRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            redirect_target,
        )


def test_openai_provider_posts_only_model_and_messages_and_parses_contract(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="https://llm.example/v1/",
        model="forecast-writer",
        api_key="super-secret",
        allowed_hosts=("llm.example",),
        timeout_seconds=12.5,
    )
    provider = build_provider(settings)
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(openai_payload(valid_content()))

    monkeypatch.setattr("src.ai_provider.urlopen", fake_urlopen)

    response = provider.analyze(make_context())

    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 12.5
    assert set(captured["body"]) == {"model", "messages"}
    assert captured["body"]["model"] == "forecast-writer"
    assert "tools" not in captured["body"]
    assert "tool_choice" not in captured["body"]
    assert captured["body"]["messages"][1]["role"] == "user"
    assert "forecast_rows" in captured["body"]["messages"][1]["content"]
    assert "read-only" in captured["body"]["messages"][0]["content"]
    assert "fixed JSON contract" in captured["body"]["messages"][0]["content"]
    assert captured["headers"]["Authorization"] == "Bearer super-secret"
    assert response.provider == "openai-compatible"
    assert response.model == "forecast-writer"
    assert response.content == valid_content()
    assert response.raw_content is not None


def test_oversized_response_is_rejected(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="http://127.0.0.1:11434/v1",
        model="local-model",
        max_response_bytes=32,
    )
    provider = build_provider(settings)

    class OversizedResponse:
        def read(self, size=-1):
            assert size == 33
            return b"x" * size

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("src.ai_provider.urlopen", lambda *args, **kwargs: OversizedResponse())

    with pytest.raises(AIProviderError, match="size"):
        provider.analyze(make_context())


@pytest.mark.parametrize(
    "settings",
    [
        AISettings(provider="openai-compatible", timeout_seconds=0),
        AISettings(provider="openai-compatible", timeout_seconds=float("nan")),
        AISettings(provider="openai-compatible", max_response_bytes=0),
    ],
)
def test_openai_provider_rejects_invalid_transport_limits(settings):
    with pytest.raises(ValueError, match="positive"):
        build_provider(settings)


def test_invalid_provider_is_rejected():
    with pytest.raises(ValueError, match="provider"):
        build_provider(AISettings(provider="wat"))


def test_openai_provider_rejects_non_object_content(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="https://llm.example/v1",
        model="forecast-writer",
        allowed_hosts=("llm.example",),
    )
    provider = build_provider(settings)
    monkeypatch.setattr(
        "src.ai_provider.urlopen",
        lambda *args, **kwargs: FakeResponse(openai_payload("not an object")),
    )

    with pytest.raises(AIProviderError, match="JSON object"):
        provider.analyze(make_context())


def test_openai_provider_rejects_incomplete_agent_contract(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="https://llm.example/v1",
        model="forecast-writer",
        allowed_hosts=("llm.example",),
    )
    provider = build_provider(settings)
    monkeypatch.setattr(
        "src.ai_provider.urlopen",
        lambda *args, **kwargs: FakeResponse(openai_payload({"status": "ok"})),
    )

    with pytest.raises(AIProviderError, match="contract"):
        provider.analyze(make_context())


@pytest.mark.parametrize(
    "choice_update",
    [
        {"message": {"tool_calls": [{"id": "call_1"}]}},
        {"message": {"function_call": {"name": "dispatch"}}},
        {"finish_reason": "tool_calls"},
    ],
)
def test_openai_provider_rejects_response_envelope_tool_calls(
    monkeypatch, choice_update
):
    settings = AISettings(
        provider="openai-compatible",
        base_url="https://llm.example/v1",
        model="forecast-writer",
        allowed_hosts=("llm.example",),
    )
    provider = build_provider(settings)
    choice = {
        "message": {"content": json.dumps(valid_content())},
        "finish_reason": "stop",
    }
    for key, value in choice_update.items():
        if isinstance(value, dict):
            choice[key].update(value)
        else:
            choice[key] = value
    monkeypatch.setattr(
        "src.ai_provider.urlopen",
        lambda *args, **kwargs: FakeResponse(
            json.dumps({"choices": [choice]}).encode("utf-8")
        ),
    )

    with pytest.raises(AIProviderError, match="tool or function call"):
        provider.analyze(make_context())


@pytest.mark.parametrize(
    "wrapped_content",
    [
        f"```json\\n{json.dumps(valid_content())}\\n```",
        f"Analysis follows: {json.dumps(valid_content())}",
    ],
)
def test_openai_provider_accepts_wrapped_json_object(monkeypatch, wrapped_content):
    settings = AISettings(
        provider="openai-compatible",
        base_url="https://llm.example/v1",
        model="forecast-writer",
        allowed_hosts=("llm.example",),
    )
    provider = build_provider(settings)
    payload = json.dumps({"choices": [{"message": {"content": wrapped_content}}]}).encode(
        "utf-8"
    )
    monkeypatch.setattr("src.ai_provider.urlopen", lambda *args, **kwargs: FakeResponse(payload))

    assert provider.analyze(make_context()).content == valid_content()
