"""Configuration for the agentic AI provider layer."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_PROVIDER = "disabled"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

ENV_PROVIDER = "ENERGY_AI_PROVIDER"
ENV_BASE_URL = "ENERGY_AI_BASE_URL"
ENV_MODEL = "ENERGY_AI_MODEL"
ENV_API_KEY = "ENERGY_AI_API_KEY"


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


@dataclass(frozen=True)
class AISettings:
    """Runtime settings for choosing and configuring an AI provider."""

    provider: str = DEFAULT_PROVIDER
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "AISettings":
        """Build settings from environment variables."""

        return cls(
            provider=_normalize_provider(os.getenv(ENV_PROVIDER)),
            base_url=_clean(os.getenv(ENV_BASE_URL)) or DEFAULT_BASE_URL,
            model=_clean(os.getenv(ENV_MODEL)) or DEFAULT_MODEL,
            api_key=_clean(os.getenv(ENV_API_KEY)),
        )
