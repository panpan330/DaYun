from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.tool_confirmation import (
    ToolConfirmationRequest,
    ToolConfirmationStatus,
)
from app.services.tool_confirmation_service import ToolConfirmationService
from app.tools.idempotency import build_arguments_fingerprint
from app.tools.tool_confirmation import ToolConfirmationStore


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 12, 10, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, *, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def make_service(
    *,
    clock: MutableClock | None = None,
    ttl_seconds: int = 300,
) -> ToolConfirmationService:
    return ToolConfirmationService(
        Settings(tool_confirmation_ttl_seconds=ttl_seconds, _env_file=None),
        ToolConfirmationStore(clock=clock or MutableClock()),
    )


def make_request(**overrides: object) -> ToolConfirmationRequest:
    data: dict[str, object] = {
        "actor_id": "demo_user_001",
        "tool_name": "create_ticket",
        "arguments": {
            "title": "订单 A1001 未发货",
            "description": "用户反馈订单迟迟未发货。",
            "order_id": "A1001",
        },
    }
    data.update(overrides)
    return ToolConfirmationRequest.model_validate(data)


def test_request_confirmation_binds_actor_tool_and_arguments() -> None:
    service = make_service()
    request = make_request()

    response = service.request_confirmation(request)

    assert response.status == ToolConfirmationStatus.PENDING
    assert response.actor_id == "demo_user_001"
    assert response.tool_name == "create_ticket"
    assert response.arguments == request.arguments
    assert response.arguments_fingerprint == build_arguments_fingerprint(
        "create_ticket",
        request.arguments,
    )
    assert response.expires_at > response.created_at
    assert "不会执行" in response.message


def test_confirm_confirmation_returns_same_stored_arguments() -> None:
    service = make_service()
    pending = service.request_confirmation(make_request())

    confirmed = service.confirm(
        pending.confirmation_id,
        actor_id="demo_user_001",
    )

    assert confirmed.status == ToolConfirmationStatus.CONFIRMED
    assert confirmed.tool_name == "create_ticket"
    assert confirmed.arguments == pending.arguments
    assert confirmed.arguments_fingerprint == pending.arguments_fingerprint
    assert "专用执行接口" in confirmed.message


def test_confirm_confirmation_is_idempotent_for_same_actor() -> None:
    service = make_service()
    pending = service.request_confirmation(make_request())

    first = service.confirm(pending.confirmation_id, actor_id="demo_user_001")
    second = service.confirm(pending.confirmation_id, actor_id="demo_user_001")

    assert first.status == ToolConfirmationStatus.CONFIRMED
    assert second == first


def test_confirmation_rejects_other_actor() -> None:
    service = make_service()
    pending = service.request_confirmation(make_request())

    with pytest.raises(AppException) as exc_info:
        service.confirm(pending.confirmation_id, actor_id="other_user_002")

    assert exc_info.value.code == "TOOL_CONFIRMATION_FORBIDDEN"
    assert exc_info.value.status_code == 403


def test_confirmation_rejects_expired_plan() -> None:
    clock = MutableClock()
    service = make_service(clock=clock, ttl_seconds=30)
    pending = service.request_confirmation(make_request())
    clock.advance(seconds=30)

    with pytest.raises(AppException) as exc_info:
        service.confirm(pending.confirmation_id, actor_id="demo_user_001")

    assert exc_info.value.code == "TOOL_CONFIRMATION_EXPIRED"
    assert exc_info.value.status_code == 409


def test_confirmation_rejects_read_tool_that_does_not_need_confirmation() -> None:
    service = make_service()

    with pytest.raises(AppException) as exc_info:
        service.request_confirmation(
            make_request(tool_name="query_order", arguments={"order_id": "A1001"})
        )

    assert exc_info.value.code == "TOOL_CONFIRMATION_NOT_REQUIRED"
    assert exc_info.value.status_code == 409


def test_confirmation_rejects_disabled_tool(disabled_sensitive_tool: str) -> None:
    service = make_service()

    with pytest.raises(AppException) as exc_info:
        service.request_confirmation(
            make_request(tool_name=disabled_sensitive_tool, arguments={"order_id": "A1001"})
        )

    assert exc_info.value.code == "TOOL_NOT_ALLOWED"
    assert exc_info.value.status_code == 403


def test_confirmation_accepts_refund_order_sensitive_tool() -> None:
    service = make_service()

    response = service.request_confirmation(
        make_request(
            tool_name="refund_order",
            arguments={"order_id": "A1001", "reason": "不想要了"},
        )
    )

    assert response.status == ToolConfirmationStatus.PENDING
    assert response.tool_name == "refund_order"
    assert response.arguments == {"order_id": "A1001", "reason": "不想要了"}
