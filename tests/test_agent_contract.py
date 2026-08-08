import pytest

from src.agent_contract import AgentContractError, disabled_response, validate_agent_response


def valid_content() -> dict[str, object]:
    return {
        "status": "ok",
        "summary": "Future load remains broadly stable.",
        "risk_level": "low",
        "evidence": ["Peak occurs near 09:00"],
        "recommendations": [
            {
                "action": "Review flexible loads before the peak window",
                "reason": "The forecast indicates higher dispatch pressure",
                "priority": "medium",
                "requires_human_approval": True,
            }
        ],
        "forecast_unchanged": True,
        "execution_enabled": False,
    }


def test_validate_agent_response_accepts_read_only_contract():
    result = validate_agent_response(valid_content())

    assert result["forecast_unchanged"] is True
    assert result["execution_enabled"] is False


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "summary",
        "risk_level",
        "evidence",
        "recommendations",
        "forecast_unchanged",
        "execution_enabled",
    ],
)
def test_validate_agent_response_rejects_missing_required_field(field):
    content = valid_content()
    content.pop(field)

    with pytest.raises(AgentContractError):
        validate_agent_response(content)


@pytest.mark.parametrize("field", ["status", "risk_level"])
def test_validate_agent_response_rejects_invalid_enum(field):
    content = valid_content()
    content[field] = "unsupported"

    with pytest.raises(AgentContractError):
        validate_agent_response(content)


@pytest.mark.parametrize(
    ("field", "value"),
    [("forecast_unchanged", 1), ("execution_enabled", 0)],
)
def test_validate_agent_response_rejects_non_boolean_safety_flags(field, value):
    content = valid_content()
    content[field] = value

    with pytest.raises(AgentContractError):
        validate_agent_response(content)


def test_validate_agent_response_rejects_too_many_evidence_items():
    content = valid_content()
    content["evidence"] = ["Evidence"] * 6

    with pytest.raises(AgentContractError):
        validate_agent_response(content)


def test_validate_agent_response_rejects_too_many_recommendations():
    content = valid_content()
    recommendation = content["recommendations"][0]
    content["recommendations"] = [recommendation] * 6

    with pytest.raises(AgentContractError):
        validate_agent_response(content)


def test_validate_agent_response_rejects_unapproved_recommendation():
    content = valid_content()
    content["recommendations"][0]["requires_human_approval"] = False

    with pytest.raises(AgentContractError, match="human approval"):
        validate_agent_response(content)


def test_validate_agent_response_rejects_execution_field():
    content = valid_content()
    content["recommendations"][0]["tool_call"] = {"name": "dispatch"}

    with pytest.raises(AgentContractError):
        validate_agent_response(content)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("summary", 1),
        ("summary", "x" * 501),
        ("evidence", [1]),
        ("recommendations", ["review loads"]),
        ("recommendations", [{"action": "review"}]),
    ],
)
def test_validate_agent_response_rejects_invalid_types_and_text_bounds(field, value):
    content = valid_content()
    content[field] = value

    with pytest.raises(AgentContractError):
        validate_agent_response(content)


def test_validate_agent_response_rejects_extra_top_level_field():
    content = valid_content()
    content["tool_call"] = {"name": "dispatch"}

    with pytest.raises(AgentContractError):
        validate_agent_response(content)


def test_validate_agent_response_returns_copied_containers():
    content = valid_content()
    result = validate_agent_response(content)
    content["evidence"].append("Changed after validation")
    content["recommendations"][0]["action"] = "Changed after validation"

    assert result["evidence"] == ["Peak occurs near 09:00"]
    assert result["recommendations"][0]["action"] == "Review flexible loads before the peak window"


def test_disabled_response_returns_a_complete_valid_response():
    response = disabled_response()

    assert response == validate_agent_response(response)
    assert response["status"] == "disabled"
    assert response["risk_level"] == "low"
    assert response["evidence"] == []
    assert response["recommendations"] == []
    assert response["forecast_unchanged"] is True
    assert response["execution_enabled"] is False
