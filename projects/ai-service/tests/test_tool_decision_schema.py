import pytest
from pydantic import ValidationError

from app.schemas.tool_decision import (
    ToolCallCandidate,
    ToolDecisionResponse,
    ToolDecisionType,
)


def test_tool_decision_response_accepts_direct_reply() -> None:
    response = ToolDecisionResponse(
        decision=ToolDecisionType.ANSWER_DIRECTLY,
        reply="请提供订单号后我再帮你查询。",
    )

    assert response.decision == ToolDecisionType.ANSWER_DIRECTLY
    assert response.reply == "请提供订单号后我再帮你查询。"
    assert response.tool_call is None


def test_tool_decision_response_accepts_tool_call() -> None:
    response = ToolDecisionResponse(
        decision=ToolDecisionType.CALL_TOOL,
        tool_call=ToolCallCandidate(
            name="query_order",
            arguments={"order_id": "A1001"},
            call_id="call_001",
        ),
    )

    assert response.decision == ToolDecisionType.CALL_TOOL
    assert response.reply is None
    assert response.tool_call is not None
    assert response.tool_call.name == "query_order"


def test_direct_reply_requires_reply() -> None:
    with pytest.raises(ValidationError):
        ToolDecisionResponse(decision=ToolDecisionType.ANSWER_DIRECTLY)


def test_tool_call_requires_tool_call() -> None:
    with pytest.raises(ValidationError):
        ToolDecisionResponse(decision=ToolDecisionType.CALL_TOOL)


def test_tool_call_candidate_rejects_invalid_tool_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolCallCandidate(name="QueryOrder", arguments={})

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("name",)
    assert error["type"] == "string_pattern_mismatch"
