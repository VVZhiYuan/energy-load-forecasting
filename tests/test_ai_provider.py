import json

import pytest

from src.ai_config import AISettings
from src.ai_provider import (
    AIProviderError,
    AgentContext,
    AgentResponse,
    DisabledAIProvider,
    MockAIProvider,
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


def test_settings_from_env_defaults(monkeypatch):
    for key in (
        "ENERGY_AI_PROVIDER",
        "ENERGY_AI_BASE_URL",
        "ENERGY_AI_MODEL",
        "ENERGY_AI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = AISettings.from_env()

    assert settings.provider == "disabled"
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.model == "gpt-4o-mini"
    assert settings.api_key is None


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
    )

    provider = build_provider(settings)

    assert isinstance(provider, expected_type)


def test_disabled_provider_does_not_make_network_requests(monkeypatch):
    settings = AISettings(
        provider="disabled",
        base_url="https://example.invalid/v1",
        model="demo-model",
        api_key="secret",
    )
    provider = build_provider(settings)

    def forbidden_request(*args, **kwargs):
        raise AssertionError("disabled provider must not make network requests")

    monkeypatch.setattr("src.ai_provider.urlopen", forbidden_request)

    response = provider.analyze(make_context())

    assert isinstance(response, AgentResponse)
    assert response.provider == "disabled"
    assert response.model == "demo-model"
    assert response.content["status"] == "disabled"
    assert "disabled" in response.content["message"].lower()


def test_mock_provider_returns_deterministic_response_shape():
    settings = AISettings(
        provider="mock",
        base_url="https://example.invalid/v1",
        model="demo-model",
        api_key=None,
    )
    provider = build_provider(settings)

    response = provider.analyze(make_context())

    assert response.provider == "mock"
    assert response.model == "demo-model"
    assert response.content["status"] == "mock"
    assert response.content["selected_model"] == "Ridge"
    assert response.content["forecast_steps"] == 2
    json.dumps(response.content)


def test_openai_provider_posts_chat_completions_and_parses_json_content(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="https://llm.example/v1/",
        model="forecast-writer",
        api_key="super-secret",
    )
    provider = build_provider(settings)
    captured = {}

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        response_body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "ok",
                                    "analysis": "Forecast looks steady.",
                                    "recommendations": ["Keep current operating plan."],
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")
        return FakeResponse(response_body)

    monkeypatch.setattr("src.ai_provider.urlopen", fake_urlopen)

    response = provider.analyze(make_context())

    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["method"] == "POST"
    assert captured["body"]["model"] == "forecast-writer"
    assert captured["body"]["messages"][1]["role"] == "user"
    assert "forecast_rows" in captured["body"]["messages"][1]["content"]
    assert captured["headers"]["Authorization"] == "Bearer super-secret"
    assert response.provider == "openai-compatible"
    assert response.model == "forecast-writer"
    assert response.content["status"] == "ok"
    assert response.content["analysis"] == "Forecast looks steady."
    assert response.raw_content is not None


def test_invalid_provider_is_rejected():
    settings = AISettings(
        provider="wat",
        base_url="https://example.invalid/v1",
        model="demo-model",
        api_key=None,
    )

    with pytest.raises(ValueError, match="provider"):
        build_provider(settings)


def test_openai_provider_rejects_non_object_content(monkeypatch):
    settings = AISettings(
        provider="openai-compatible",
        base_url="https://llm.example/v1",
        model="forecast-writer",
        api_key=None,
    )
    provider = build_provider(settings)

    class FakeResponse:
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "\"not an object\""}}]}
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("src.ai_provider.urlopen", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(AIProviderError, match="JSON object"):
        provider.analyze(make_context())
