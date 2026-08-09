import pytest
from pydantic import ValidationError

from app.schemas.tool_confirmation import (
    ConfirmToolConfirmationRequest,
    ToolConfirmationRequest,
)


def test_tool_confirmation_request_normalizes_identifiers() -> None:
    request = ToolConfirmationRequest(
        actor_id="  demo_user_001  ",
        tool_name="  create_ticket  ",
        arguments={"title": "订单未发货", "order_id": "A1001"},
    )

    assert request.actor_id == "demo_user_001"
    assert request.tool_name == "create_ticket"
    assert request.arguments == {"title": "订单未发货", "order_id": "A1001"}


def test_tool_confirmation_request_rejects_empty_arguments() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolConfirmationRequest(
            actor_id="demo_user_001",
            tool_name="create_ticket",
            arguments={},
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("arguments",)
    assert error["type"] == "too_short"


def test_confirm_tool_confirmation_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ConfirmToolConfirmationRequest.model_validate(
            {
                "actor_id": "demo_user_001",
                "arguments": {"title": "不允许在确认时替换参数"},
            }
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("arguments",)
    assert error["type"] == "extra_forbidden"
