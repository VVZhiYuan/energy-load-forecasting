"""Strict, read-only response validation for Agent providers."""

from __future__ import annotations

from typing import Final


MAX_CONTEXT_ROWS: Final = 96
MAX_LIST_ITEMS: Final = 5
MAX_TEXT_LENGTH: Final = 500
MAX_RESPONSE_BYTES: Final = 65536

_RESPONSE_KEYS: Final = frozenset(
    {
        "status",
        "summary",
        "risk_level",
        "evidence",
        "recommendations",
        "forecast_unchanged",
        "execution_enabled",
    }
)
_RECOMMENDATION_KEYS: Final = frozenset(
    {"action", "reason", "priority", "requires_human_approval"}
)
_STATUS_VALUES: Final = frozenset({"ok", "disabled", "error"})
_RISK_LEVEL_VALUES: Final = frozenset({"low", "medium", "high"})
_PRIORITY_VALUES: Final = frozenset({"low", "medium", "high"})


class AgentContractError(ValueError):
    """Raised when an Agent response violates the read-only contract."""


def validate_agent_response(content: object) -> dict[str, object]:
    """Validate and copy an Agent response with no executable affordances."""
    if not isinstance(content, dict):
        raise AgentContractError("Agent response must be an object")
    _require_exact_keys(content, _RESPONSE_KEYS, "Agent response")

    status = _require_enum(content["status"], _STATUS_VALUES, "status")
    summary = _require_text(content["summary"], "summary")
    risk_level = _require_enum(content["risk_level"], _RISK_LEVEL_VALUES, "risk_level")
    evidence = _validate_evidence(content["evidence"])
    recommendations = _validate_recommendations(content["recommendations"])

    if content["forecast_unchanged"] is not True:
        raise AgentContractError("forecast_unchanged must be true")
    if content["execution_enabled"] is not False:
        raise AgentContractError("execution_enabled must be false")

    return {
        "status": status,
        "summary": summary,
        "risk_level": risk_level,
        "evidence": evidence,
        "recommendations": recommendations,
        "forecast_unchanged": True,
        "execution_enabled": False,
    }


def disabled_response(
    selected_model: object = None,
    message: str = "Agent analysis is disabled.",
) -> dict[str, object]:
    """Return a valid, deterministic response for the disabled provider."""
    summary = _require_text(message, "message")
    if isinstance(selected_model, str) and selected_model:
        summary = _require_text(f"{summary} Selected model: {selected_model}.", "summary")

    return validate_agent_response(
        {
            "status": "disabled",
            "summary": summary,
            "risk_level": "low",
            "evidence": [],
            "recommendations": [],
            "forecast_unchanged": True,
            "execution_enabled": False,
        }
    )


def _require_exact_keys(value: dict[object, object], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unexpected keys: {', '.join(sorted(str(key) for key in extra))}")
        raise AgentContractError(f"{label} has {'; '.join(details)}")


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AgentContractError(f"{label} must be a string")
    if len(value) > MAX_TEXT_LENGTH:
        raise AgentContractError(f"{label} must be at most {MAX_TEXT_LENGTH} characters")
    return value


def _require_enum(value: object, allowed: frozenset[str], label: str) -> str:
    text = _require_text(value, label)
    if text not in allowed:
        raise AgentContractError(f"{label} has an unsupported value")
    return text


def _validate_evidence(value: object) -> list[str]:
    if not isinstance(value, list):
        raise AgentContractError("evidence must be a list")
    if len(value) > MAX_LIST_ITEMS:
        raise AgentContractError(f"evidence must contain at most {MAX_LIST_ITEMS} items")
    return [_require_text(item, "evidence item") for item in value]


def _validate_recommendations(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise AgentContractError("recommendations must be a list")
    if len(value) > MAX_LIST_ITEMS:
        raise AgentContractError(
            f"recommendations must contain at most {MAX_LIST_ITEMS} items"
        )

    validated: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise AgentContractError("recommendation must be an object")
        _require_exact_keys(item, _RECOMMENDATION_KEYS, "recommendation")
        if item["requires_human_approval"] is not True:
            raise AgentContractError("recommendation requires human approval")
        validated.append(
            {
                "action": _require_text(item["action"], "recommendation action"),
                "reason": _require_text(item["reason"], "recommendation reason"),
                "priority": _require_enum(
                    item["priority"], _PRIORITY_VALUES, "recommendation priority"
                ),
                "requires_human_approval": True,
            }
        )
    return validated
