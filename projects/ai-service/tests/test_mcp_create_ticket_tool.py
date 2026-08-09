import asyncio
import json

import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from app.core.exceptions import AppException
from app.mcp_servers import ticket_tool
from app.mcp_servers.minimal_server import mcp
from app.mcp_servers.ticket_tool import create_ticket_for_mcp
from app.tools.idempotency import clear_idempotency_store
from tests.tool_fakes import FakeTicketCreator


CONFIRMATION_ID = "9f4d0b2f5b0c4f2a9d6c8b1e0a3f7c11"


@pytest.fixture(autouse=True)
def clear_idempotency() -> None:
    clear_idempotency_store()
    yield
    clear_idempotency_store()


def make_create_ticket_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "requester_id": "demo_user_001",
        "title": "订单 A1001 一直未发货",
        "description": "订单 A1001 已付款一周仍未发货，请帮我处理。",
        "category": "complaint",
        "priority": "high",
        "related_order_id": "A1001",
        "confirmation_id": CONFIRMATION_ID,
        "user_confirmed": True,
    }
    payload.update(overrides)
    return payload


def test_create_ticket_mcp_tool_schema_exposes_write_contract() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            tools = await client.list_tools()

        tool = next(tool for tool in tools.tools if tool.name == "create_ticket")
        properties = tool.input_schema["properties"]
        assert {
            "requester_id",
            "title",
            "description",
            "category",
            "confirmation_id",
        }.issubset(set(tool.input_schema["required"]))
        assert properties["confirmation_id"]["pattern"] == r"^[a-f0-9]{32}$"
        assert properties["user_confirmed"]["default"] is False
        assert properties["category"]["$ref"] == "#/$defs/TicketCategory"
        assert tool.input_schema["$defs"]["TicketCategory"]["enum"] == [
            "refund",
            "order_query",
            "logistics",
            "complaint",
            "policy_gap",
        ]
        assert properties["priority"]["$ref"] == "#/$defs/TicketPriority"
        assert tool.input_schema["$defs"]["TicketPriority"]["enum"] == [
            "low",
            "normal",
            "high",
        ]

    asyncio.run(run())


def test_create_ticket_for_mcp_blocks_without_confirmation() -> None:
    creator = FakeTicketCreator()

    response = create_ticket_for_mcp(
        **make_create_ticket_payload(user_confirmed=False),
        creator=creator,
    )

    assert response["ok"] is False
    assert response["allowed"] is False
    assert response["requires_confirmation"] is True
    assert response["error_code"] == "TOOL_CONFIRMATION_REQUIRED"
    assert response["ticket"] is None
    assert creator.calls == []


def test_create_ticket_for_mcp_creates_ticket_after_confirmation() -> None:
    creator = FakeTicketCreator()

    response = create_ticket_for_mcp(
        **make_create_ticket_payload(),
        creator=creator,
    )

    assert response["ok"] is True
    assert response["allowed"] is True
    assert response["action_type"] == "write"
    assert response["confirmation_checked"] is True
    assert response["confirmation_id"] == CONFIRMATION_ID
    assert response["security_checks"]["idempotency_key_source"] == "confirmation_id"
    assert response["ticket"] == {
        "ticket_id": "T1001",
        "title": "订单 A1001 一直未发货",
        "category": "complaint",
        "priority": "high",
        "related_order_id": "A1001",
        "created_at": "2026-07-12T10:00:00+00:00",
    }
    assert creator.idempotency_keys == [CONFIRMATION_ID]


def test_create_ticket_for_mcp_reuses_idempotent_result() -> None:
    creator = FakeTicketCreator()

    first = create_ticket_for_mcp(**make_create_ticket_payload(), creator=creator)
    second = create_ticket_for_mcp(**make_create_ticket_payload(), creator=creator)

    assert first["ticket"] == second["ticket"]
    assert len(creator.calls) == 1


def test_create_ticket_for_mcp_reports_idempotency_conflict_as_business_error() -> None:
    creator = FakeTicketCreator()
    create_ticket_for_mcp(**make_create_ticket_payload(title="第一次确认"), creator=creator)

    response = create_ticket_for_mcp(
        **make_create_ticket_payload(title="第二次改了标题"),
        creator=creator,
    )

    assert response["ok"] is False
    assert response["error_code"] == "IDEMPOTENCY_KEY_CONFLICT"
    assert response["message"] == "同一个幂等键不能用于不同的工具调用参数。"
    assert len(creator.calls) == 1


def test_create_ticket_for_mcp_returns_invalid_arguments_safely() -> None:
    creator = FakeTicketCreator()

    response = create_ticket_for_mcp(
        **make_create_ticket_payload(title="   "),
        creator=creator,
    )

    assert response["ok"] is False
    assert response["allowed"] is False
    assert response["error_code"] == "INVALID_TOOL_ARGUMENTS"
    assert response["ticket"] is None
    assert response["errors"][0]["loc"] == ("title",)
    assert creator.calls == []


def test_create_ticket_for_mcp_returns_business_error_without_tool_error() -> None:
    creator = FakeTicketCreator(
        error=AppException(
            code="TICKET_ALREADY_EXISTS",
            message="已经存在相似工单，请不要重复提交。",
            status_code=409,
        )
    )

    response = create_ticket_for_mcp(**make_create_ticket_payload(), creator=creator)

    assert response["ok"] is False
    assert response["allowed"] is True
    assert response["error_code"] == "TICKET_ALREADY_EXISTS"
    assert response["message"] == "已经存在相似工单，请不要重复提交。"


def test_create_ticket_for_mcp_raises_safe_tool_error_for_timeout() -> None:
    creator = FakeTicketCreator(
        error=AppException(
            code="TOOL_TIMEOUT",
            message="internal timeout http://java-business-service/tickets",
            status_code=504,
        )
    )

    with pytest.raises(ToolError) as exc_info:
        create_ticket_for_mcp(**make_create_ticket_payload(), creator=creator)

    error_text = str(exc_info.value)
    assert "TOOL_TIMEOUT" in error_text
    assert "java-business-service" not in error_text


def test_create_ticket_for_mcp_raises_safe_tool_error_for_untrusted_result() -> None:
    creator = FakeTicketCreator(
        error=AppException(
            code="TOOL_RESULT_VALIDATION_FAILED",
            message="response contains internal field database_password",
            status_code=502,
        )
    )

    with pytest.raises(ToolError) as exc_info:
        create_ticket_for_mcp(**make_create_ticket_payload(), creator=creator)

    error_text = str(exc_info.value)
    assert "TOOL_RESULT_VALIDATION_FAILED" in error_text
    assert "database_password" not in error_text


def test_create_ticket_for_mcp_wraps_unexpected_errors_safely() -> None:
    creator = FakeTicketCreator(error=RuntimeError("database password leaked in stack"))

    with pytest.raises(ToolError) as exc_info:
        create_ticket_for_mcp(**make_create_ticket_payload(), creator=creator)

    error_text = str(exc_info.value)
    assert "TICKET_CREATION_TOOL_ERROR" in error_text
    assert "database password" not in error_text


def test_create_ticket_mcp_client_call_can_use_fake_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_create_ticket_for_mcp(**kwargs: object) -> dict[str, object]:
        return {
            "ok": True,
            "allowed": True,
            "action": "create_ticket",
            "action_type": "write",
            "requires_confirmation": True,
            "confirmation_checked": kwargs["user_confirmed"],
            "confirmation_id": kwargs["confirmation_id"],
            "error_code": None,
            "message": "工单创建成功。",
            "retryable": False,
            "security_checks": {
                "input_validated": True,
                "user_confirmed": kwargs["user_confirmed"],
                "idempotency_key_checked": True,
                "idempotency_key_source": "confirmation_id",
                "output_allowlist_applied": True,
                "sensitive_fields_returned": False,
            },
            "ticket": {"ticket_id": "T1001"},
        }

    async def run() -> None:
        monkeypatch.setattr(
            ticket_tool,
            "create_ticket_for_mcp",
            fake_create_ticket_for_mcp,
        )
        async with Client(mcp) as client:
            result = await client.call_tool(
                "create_ticket",
                make_create_ticket_payload(),
            )

        assert result.is_error is False
        assert result.structured_content["ok"] is True
        assert result.structured_content["ticket"] == {"ticket_id": "T1001"}

    asyncio.run(run())


def test_create_ticket_for_mcp_does_not_return_requester_or_description() -> None:
    response = create_ticket_for_mcp(
        **make_create_ticket_payload(description="用户手机号 13800000000 不应重复返回"),
        creator=FakeTicketCreator(),
    )

    raw_response = json.dumps(response, ensure_ascii=False)
    assert "demo_user_001" not in raw_response
    assert "13800000000" not in raw_response
