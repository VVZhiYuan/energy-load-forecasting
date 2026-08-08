"""Provider contract and implementations for AI-based forecast analysis."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from src.ai_config import AISettings
from src.agent_contract import AgentContractError, disabled_response, validate_agent_response


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


class _NoRedirectHandler(HTTPRedirectHandler):
    """Fail closed when an endpoint attempts to redirect the provider request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req, code, "Provider redirects are disabled.", headers, fp)


_NO_REDIRECT_OPENER = build_opener(_NoRedirectHandler())


def urlopen(request, timeout=None):
    """Open a provider request without following redirects."""
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


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


def _validate_transport_settings(settings: AISettings) -> None:
    if (
        not isinstance(settings.timeout_seconds, (int, float))
        or isinstance(settings.timeout_seconds, bool)
        or not math.isfinite(settings.timeout_seconds)
        or settings.timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number.")
    if (
        not isinstance(settings.max_response_bytes, int)
        or isinstance(settings.max_response_bytes, bool)
        or settings.max_response_bytes <= 0
    ):
        raise ValueError("max_response_bytes must be a positive integer.")


def _validate_base_url(settings: AISettings) -> None:
    parsed = urlparse(settings.base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("openai-compatible base_url must use HTTP or HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("openai-compatible base_url must not include credentials.")
    if parsed.query:
        raise ValueError("openai-compatible base_url must not include a query string.")
    if parsed.fragment:
        raise ValueError("openai-compatible base_url must not include a fragment.")

    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("openai-compatible base_url must include a host.")
    hostname = hostname.lower()
    is_loopback = hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "http" and not is_loopback:
        raise ValueError("openai-compatible remote base_url must use HTTPS.")
    if not is_loopback:
        allowed_hosts = {host.strip().lower() for host in settings.allowed_hosts}
        if hostname not in allowed_hosts:
            raise ValueError(
                "openai-compatible remote base_url host must be in the explicit allowlist."
            )


class DisabledAIProvider:
    """Provider that returns a deterministic disabled response."""

    def __init__(self, settings: AISettings):
        self._settings = settings

    def analyze(self, context: AgentContext) -> AgentResponse:
        content = disabled_response(selected_model=context.summary.get("selected_model"))
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
            "status": "ok",
            "summary": "Mock analysis generated from the supplied forecast context.",
            "risk_level": "low",
            "evidence": [],
            "recommendations": [
                {
                    "action": "Review flexible load before the forecast peak.",
                    "reason": "The mock provider does not change the forecast.",
                    "priority": "medium",
                    "requires_human_approval": True,
                }
            ],
            "forecast_unchanged": True,
            "execution_enabled": False,
        }
        return AgentResponse(
            provider=PROVIDER_MOCK,
            model="mock",
            content=content,
            raw_content=_json_dumps(content),
        )


class OpenAICompatibleAIProvider:
    """Provider that calls an OpenAI-compatible chat completions endpoint."""

    def __init__(self, settings: AISettings):
        _validate_transport_settings(settings)
        _validate_base_url(settings)
        self._settings = settings

    def analyze(self, context: AgentContext) -> AgentResponse:
        url = f"{self._settings.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self._settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a read-only electricity load forecast analyst. "
                        "Provide analysis only; do not change forecasts or execute actions. "
                        "Do not call tools, functions, MCP, or shell commands. "
                        "Return only a single JSON object matching the fixed JSON contract: "
                        "status, summary, risk_level, evidence, recommendations, "
                        "forecast_unchanged=true, and execution_enabled=false."
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
            with urlopen(request, timeout=self._settings.timeout_seconds) as response:
                response_body = response.read(self._settings.max_response_bytes + 1)
            if len(response_body) > self._settings.max_response_bytes:
                raise AIProviderError(
                    "OpenAI-compatible provider response exceeded the size limit."
                )
            payload = json.loads(response_body.decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AIProviderError("OpenAI-compatible provider request failed.") from exc

        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIProviderError(
                "OpenAI-compatible response did not include message content."
            ) from exc

        if not isinstance(choice, dict) or not isinstance(message, dict):
            raise AIProviderError(
                "OpenAI-compatible response did not include message content."
            )
        if message.get("tool_calls") or message.get("function_call"):
            raise AIProviderError(
                "OpenAI-compatible response included a tool or function call."
            )
        if choice.get("finish_reason") == "tool_calls":
            raise AIProviderError(
                "OpenAI-compatible response included a tool or function call."
            )

        try:
            content_text = message["content"]
        except KeyError as exc:
            raise AIProviderError(
                "OpenAI-compatible response did not include message content."
            ) from exc

        if not isinstance(content_text, str):
            raise AIProviderError("OpenAI-compatible message content must be text.")

        content = _parse_json_object(content_text)
        try:
            content = validate_agent_response(content)
        except AgentContractError as exc:
            raise AIProviderError(
                "OpenAI-compatible message content violates the agent response contract."
            ) from exc

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
