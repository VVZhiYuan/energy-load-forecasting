"""Provider contract and implementations for AI-based forecast analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.ai_config import AISettings


PROVIDER_DISABLED = "disabled"
PROVIDER_MOCK = "mock"
PROVIDER_OPENAI_COMPATIBLE = "openai-compatible"
SUPPORTED_PROVIDERS = {
    PROVIDER_DISABLED,
    PROVIDER_MOCK,
    PROVIDER_OPENAI_COMPATIBLE,
}


class AIProviderError(RuntimeError):
    """Raised when an AI provider cannot complete an analysis request."""


@dataclass(frozen=True)
class AgentContext:
    """JSON-safe context passed to a provider for analysis."""

    summary: dict[str, object]
    forecast_rows: list[dict[str, object]]
    comparison_rows: list[dict[str, object]]
    recent_load_rows: list[dict[str, object]]


@dataclass(frozen=True)
class AgentResponse:
    """Provider response contract returned to the agent layer."""

    provider: str
    model: str
    content: dict[str, object]
    raw_content: str | None = None


@runtime_checkable
class AIProvider(Protocol):
    """Provider interface for forecast analysis."""

    def analyze(self, context: AgentContext) -> AgentResponse:
        """Analyze a forecast context and return a structured response."""


def _normalize_provider(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _context_payload(context: AgentContext) -> dict[str, object]:
    return asdict(context)


def _json_dumps(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


class DisabledAIProvider:
    """Provider that returns a deterministic disabled response."""

    def __init__(self, settings: AISettings):
        self._settings = settings

    def analyze(self, context: AgentContext) -> AgentResponse:
        content = {
            "status": PROVIDER_DISABLED,
            "message": "AI analysis is disabled.",
            "selected_model": context.summary.get("selected_model"),
        }
        return AgentResponse(
            provider=PROVIDER_DISABLED,
            model="none",
            content=content,
            raw_content=_json_dumps(content),
        )


class MockAIProvider:
    """Provider that returns a deterministic offline demonstration response."""

    def __init__(self, settings: AISettings):
        self._settings = settings

    def analyze(self, context: AgentContext) -> AgentResponse:
        content = {
            "status": PROVIDER_MOCK,
            "selected_model": context.summary.get("selected_model"),
            "horizon": context.summary.get("horizon"),
            "forecast_steps": len(context.forecast_rows),
            "comparison_models": [
                row.get("model") for row in context.comparison_rows
            ],
            "recent_points": len(context.recent_load_rows),
            "peak_timestamp": context.summary.get("peak_timestamp"),
            "peak_prediction": context.summary.get("peak_prediction"),
            "mean_interval_width": context.summary.get("mean_interval_width"),
            "recommendations": [
                "Review the peak window and keep load flexibility available."
            ],
        }
        return AgentResponse(
            provider=PROVIDER_MOCK,
            model="mock",
            content=content,
            raw_content=_json_dumps(content),
        )


class OpenAICompatibleAIProvider:
    """Provider that calls an OpenAI-compatible chat completions endpoint."""

    def __init__(self, settings: AISettings, timeout_seconds: float = 30.0):
        self._settings = settings
        self._timeout_seconds = timeout_seconds

    def analyze(self, context: AgentContext) -> AgentResponse:
        url = f"{self._settings.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self._settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You analyze electricity load forecasts. "
                        "Return a single JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": _json_dumps(_context_payload(context)),
                },
            ],
        }
        request = Request(
            url,
            data=_json_dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self._settings.api_key:
            request.add_header("Authorization", f"Bearer {self._settings.api_key}")

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AIProviderError("OpenAI-compatible provider request failed.") from exc

        try:
            content_text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIProviderError(
                "OpenAI-compatible response did not include message content."
            ) from exc

        if not isinstance(content_text, str):
            raise AIProviderError("OpenAI-compatible message content must be text.")

        content = _parse_json_object(content_text)

        if not isinstance(content, dict):
            raise AIProviderError("OpenAI-compatible message content must be a JSON object.")

        return AgentResponse(
            provider=PROVIDER_OPENAI_COMPATIBLE,
            model=self._settings.model,
            content=content,
            raw_content=content_text,
        )


def _parse_json_object(content_text: str) -> dict[str, object]:
    """Parse a JSON object from bare text, a code fence, or a short preamble."""

    stripped = content_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()

    try:
        content = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise AIProviderError(
                "OpenAI-compatible message content must include a JSON object."
            )
        try:
            content, _ = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError as exc:
            raise AIProviderError(
                "OpenAI-compatible message content must include a valid JSON object."
            ) from exc

    if not isinstance(content, dict):
        raise AIProviderError("OpenAI-compatible message content must be a JSON object.")
    return content


def build_provider(settings: AISettings) -> AIProvider:
    """Construct the provider implementation requested by settings."""

    provider_name = _normalize_provider(settings.provider)
    if provider_name not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{settings.provider}'. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )

    if provider_name == PROVIDER_OPENAI_COMPATIBLE:
        if not settings.base_url or not settings.base_url.strip():
            raise ValueError("openai-compatible provider requires a base_url.")
        if not settings.model or not settings.model.strip():
            raise ValueError("openai-compatible provider requires a model.")
        return OpenAICompatibleAIProvider(settings)
    if provider_name == PROVIDER_MOCK:
        return MockAIProvider(settings)
    return DisabledAIProvider(settings)
