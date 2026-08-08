"""Configuration for the agentic AI provider layer."""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.agent_contract import MAX_RESPONSE_BYTES


DEFAULT_PROVIDER = "disabled"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RESPONSE_BYTES = MAX_RESPONSE_BYTES

ENV_PROVIDER = "ENERGY_AI_PROVIDER"
ENV_BASE_URL = "ENERGY_AI_BASE_URL"
ENV_MODEL = "ENERGY_AI_MODEL"
ENV_API_KEY = "ENERGY_AI_API_KEY"
ENV_ALLOWED_HOSTS = "ENERGY_AI_ALLOWED_HOSTS"
ENV_TIMEOUT_SECONDS = "ENERGY_AI_TIMEOUT_SECONDS"
ENV_MAX_RESPONSE_BYTES = "ENERGY_AI_MAX_RESPONSE_BYTES"


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_provider(value: str | None) -> str:
    cleaned = _clean(value)
    if cleaned is None:
        return DEFAULT_PROVIDER
    return cleaned.lower().replace("_", "-")


def _parse_allowed_hosts(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(
        host.lower()
        for item in value.split(",")
        if (host := item.strip())
    )


def _parse_timeout_seconds(value: str | None) -> float:
    cleaned = _clean(value)
    if cleaned is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"{ENV_TIMEOUT_SECONDS} must be a positive finite number."
        ) from exc


def _parse_max_response_bytes(value: str | None) -> int:
    cleaned = _clean(value)
    if cleaned is None:
        return DEFAULT_MAX_RESPONSE_BYTES
    try:
        return int(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"{ENV_MAX_RESPONSE_BYTES} must be a positive integer."
        ) from exc


@dataclass(frozen=True)
class AISettings:
    """Runtime settings for choosing and configuring an AI provider."""

    provider: str = DEFAULT_PROVIDER
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    allowed_hosts: tuple[str, ...] = ()
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES

    @classmethod
    def from_env(cls) -> "AISettings":
        """Build settings from environment variables."""

        return cls(
            provider=_normalize_provider(os.getenv(ENV_PROVIDER)),
            base_url=_clean(os.getenv(ENV_BASE_URL)) or DEFAULT_BASE_URL,
            model=_clean(os.getenv(ENV_MODEL)) or DEFAULT_MODEL,
            api_key=_clean(os.getenv(ENV_API_KEY)),
            allowed_hosts=_parse_allowed_hosts(os.getenv(ENV_ALLOWED_HOSTS)),
            timeout_seconds=_parse_timeout_seconds(os.getenv(ENV_TIMEOUT_SECONDS)),
            max_response_bytes=_parse_max_response_bytes(
                os.getenv(ENV_MAX_RESPONSE_BYTES)
            ),
        )
