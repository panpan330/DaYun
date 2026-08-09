import pytest

from app.agents.mcp_tool_adapters import (
    McpRefundExecutor,
    McpTicketCreator,
    create_mcp_order_query_executor,
    create_mcp_refund_executor,
    create_mcp_ticket_creator,
    mcp_order_query_executor,
    register_ticket_confirmation,
)
from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.refund import RefundOrderArgs
from app.schemas.ticket import (
    CreateTicketArgs,
    CreatedTicket,
    TicketCategory,
    TicketPriority,
)
from app.schemas.tool import QueryOrderArgs, QueryOrderResult
from tests.tool_fakes import FakeMcpToolCaller


def _settings() -> Settings:
    return Settings(_env_file=None, tool_confirmation_backend="memory")


def test_mcp_ticket_creator_creates_ticket_from_ok_response() -> None:
    caller = FakeMcpToolCaller(
        responses={
            "create_ticket": {
                "ok": True,
                "confirmation_checked": True,
                "confirmation_id": "a" * 16,
                "error_code": None,
                "message": "工单创建成功。",
                "ticket": {
                    "ticket_id": "T1000001",
                    "requester_id": "user_001",
                    "title": "退款申请",
                    "description": "订单破损",
                    "category": "refund",
                    "priority": "high",
                    "related_order_id": "A1001",
                    "created_at": "2026-01-01T00:00:00Z",
                },
            }
        }
    )
    creator = McpTicketCreator(caller, settings=_settings())
    arguments = CreateTicketArgs(
        requester_id="user_001",
        title="退款申请",
        description="订单破损",
        category=TicketCategory.REFUND,
        priority=TicketPriority.HIGH,
        related_order_id="A1001",
    )
    ticket = creator.create_ticket(arguments, idempotency_key="a" * 16)
    assert ticket.ticket_id == "T1000001"
    assert caller.calls[0][0] == "create_ticket"
    assert caller.calls[0][1]["confirmation_id"] == "a" * 16
    assert caller.calls[0][1]["user_confirmed"] is True
    # The authenticated actor (agent ticket_actor_id) is forwarded to the
    # product MCP server so Java ownership checks see the real caller.
    assert caller.calls[0][1]["user_id"] == "user_001"
    # No business context / no injected tenant: nothing is forwarded, and the
    # MCP server falls back to the configured default tenant.
    assert caller.calls[0][1]["tenant_id"] is None


def test_mcp_ticket_creator_raises_app_exception_on_ok_false() -> None:
    caller = FakeMcpToolCaller(
        responses={
            "create_ticket": {
                "ok": False,
                "confirmation_checked": True,
                "confirmation_id": "b" * 16,
                "error_code": "ORDER_NOT_FOUND",
                "message": "订单不存在",
                "ticket": None,
            }
        }
    )
    creator = McpTicketCreator(caller, settings=_settings())
    arguments = CreateTicketArgs(
        requester_id="user_001",
        title="t",
        description="d",
        category=TicketCategory.REFUND,
    )
    with pytest.raises(AppException) as exc:
        creator.create_ticket(arguments, idempotency_key="b" * 16)
    assert exc.value.code == "ORDER_NOT_FOUND"


def test_mcp_order_query_executor_parses_result() -> None:
    caller = FakeMcpToolCaller(
        responses={
            "query_order": {
                "ok": True,
                "allowed": True,
                "action": "query_order",
                "action_type": "read",
                "requires_confirmation": False,
                "error_code": None,
                "message": "订单查询成功。",
                "retryable": False,
                "security_checks": {"input_validated": True},
                "result": {
                    "order_id": "A1001",
                    "order_status": "waiting_shipment",
                    "payment_status": "paid",
                    "logistics_message": "商家已接单。",
                    "latest_event": "仓库准备出库。",
                    "can_create_ticket": True,
                    "source": "java_business_service",
                },
            }
        }
    )
    executor = mcp_order_query_executor(caller)
    result = executor(QueryOrderArgs(order_id="A1001"))
    assert isinstance(result, QueryOrderResult)
    assert result.order_id == "A1001"
    # Without a business context the forwarded identity is empty.
    assert caller.calls == [
        ("query_order", {"order_id": "A1001", "user_id": None, "tenant_id": None})
    ]


def test_mcp_order_query_executor_forwards_business_context_identity() -> None:
    from app.core.business_context import (
        reset_business_context,
        set_business_context,
    )

    caller = FakeMcpToolCaller(
        responses={
            "query_order": {
                "ok": True,
                "result": {
                    "order_id": "A1001",
                    "order_status": "waiting_shipment",
                    "payment_status": "paid",
                    "logistics_message": "商家已接单。",
                    "latest_event": "仓库准备出库。",
                    "can_create_ticket": True,
                    "source": "java_business_service",
                },
            }
        }
    )
    executor = mcp_order_query_executor(caller)
    tokens = set_business_context(user_id="U1001", tenant_id="tenant-7")
    try:
        executor(QueryOrderArgs(order_id="A1001"))
    finally:
        reset_business_context(tokens)
    assert caller.calls == [
        (
            "query_order",
            {"order_id": "A1001", "user_id": "U1001", "tenant_id": "tenant-7"},
        )
    ]


def test_mcp_order_query_executor_factory_injects_identity() -> None:
    caller = FakeMcpToolCaller(
        responses={
            "query_order": {
                "ok": True,
                "result": {
                    "order_id": "A1001",
                    "order_status": "waiting_shipment",
                    "payment_status": "paid",
                    "logistics_message": "商家已接单。",
                    "latest_event": "仓库准备出库。",
                    "can_create_ticket": True,
                    "source": "java_business_service",
                },
            }
        }
    )
    executor = create_mcp_order_query_executor(
        caller=caller, user_id="U2002", tenant_id="tenant-9"
    )
    executor(QueryOrderArgs(order_id="A1001"))
    assert caller.calls == [
        (
            "query_order",
            {"order_id": "A1001", "user_id": "U2002", "tenant_id": "tenant-9"},
        )
    ]


def test_register_ticket_confirmation_registers_confirmed_record() -> None:
    from app.agents.ticket_agent import build_pending_ticket_confirmation

    fields = {
        "issue_type": "refund",
        "order_id": "A1001",
        "description": "订单破损",
        "user_request": "申请退款",
        "urgency": "high",
        "need_human_review": False,
    }
    confirmation_id = register_ticket_confirmation(
        actor_id="user_001",
        fields=fields,
        settings=_settings(),
        is_refund_execution=True,
    )
    expected_id = build_pending_ticket_confirmation(fields)["confirmation_id"]
    assert confirmation_id == expected_id

    from app.tools.tool_confirmation import create_tool_confirmation_store

    store = create_tool_confirmation_store(_settings())
    record = store.require_confirmed(confirmation_id, actor_id="user_001")
    assert record.status.value == "confirmed"
    # A refund *execution* draft registers under refund_order so the
    # confirmation is attributed to the refund tool rather than create_ticket.
    assert record.tool_name == "refund_order"


def test_register_ticket_confirmation_uses_create_ticket_for_non_refund_draft() -> None:
    from app.agents.ticket_agent import build_pending_ticket_confirmation

    fields = {
        "issue_type": "complaint",
        "order_id": "A1001",
        "description": "物流一直不动",
        "user_request": "投诉处理",
        "urgency": "high",
        "need_human_review": True,
    }
    confirmation_id = register_ticket_confirmation(
        actor_id="user_001",
        fields=fields,
        settings=_settings(),
    )
    expected_id = build_pending_ticket_confirmation(fields)["confirmation_id"]
    assert confirmation_id == expected_id

    from app.tools.tool_confirmation import create_tool_confirmation_store

    store = create_tool_confirmation_store(_settings())
    record = store.require_confirmed(confirmation_id, actor_id="user_001")
    assert record.tool_name == "create_ticket"


def test_register_ticket_confirmation_refund_draft_without_execution_uses_create_ticket() -> None:
    """A refund-typed ticket draft (ticket flow, not refund execution) keeps
    create_ticket: only is_refund_execution=True registers under refund_order."""
    from app.agents.ticket_agent import build_pending_ticket_confirmation

    fields = {
        "issue_type": "refund",
        "order_id": "A1001",
        "description": "订单破损，需要建退款工单跟进",
        "user_request": "创建工单",
        "urgency": "high",
        "need_human_review": True,
    }
    confirmation_id = register_ticket_confirmation(
        actor_id="user_001",
        fields=fields,
        settings=_settings(),
    )

    from app.tools.tool_confirmation import create_tool_confirmation_store

    store = create_tool_confirmation_store(_settings())
    record = store.require_confirmed(confirmation_id, actor_id="user_001")
    assert record.tool_name == "create_ticket"


def test_register_ticket_confirmation_cancel_execution_uses_cancel_order() -> None:
    """A cancel *execution* draft registers under cancel_order so the
    confirmation is attributed to the cancel tool rather than create_ticket."""
    from app.agents.ticket_agent import build_pending_ticket_confirmation

    fields = {
        "issue_type": "cancel",
        "order_id": "A1002",
        "description": "不想要了，取消订单 A1002",
        "user_request": "订单取消处理",
        "urgency": "normal",
        "need_human_review": True,
    }
    confirmation_id = register_ticket_confirmation(
        actor_id="user_001",
        fields=fields,
        settings=_settings(),
        is_cancel_execution=True,
    )
    expected_id = build_pending_ticket_confirmation(fields)["confirmation_id"]
    assert confirmation_id == expected_id

    from app.tools.tool_confirmation import create_tool_confirmation_store

    store = create_tool_confirmation_store(_settings())
    record = store.require_confirmed(confirmation_id, actor_id="user_001")
    assert record.status.value == "confirmed"
    assert record.tool_name == "cancel_order"


def test_create_mcp_ticket_creator_builds_from_settings() -> None:
    creator = create_mcp_ticket_creator(
        Settings(
            _env_file=None,
            mcp_product_base_url="http://127.0.0.1:9100/mcp",
            mcp_product_auth_token="token",
        )
    )
    assert isinstance(creator, McpTicketCreator)


def test_mcp_ticket_creator_forwards_injected_tenant_over_requester() -> None:
    caller = FakeMcpToolCaller(
        responses={
            "create_ticket": {
                "ok": True,
                "confirmation_id": "c" * 16,
                "error_code": None,
                "message": "工单创建成功。",
                "ticket": {
                    "ticket_id": "T2000001",
                    "requester_id": "user_007",
                    "title": "t",
                    "description": "d",
                    "category": "refund",
                    "priority": "normal",
                    "related_order_id": None,
                    "created_at": "2026-01-01T00:00:00Z",
                },
            }
        }
    )
    creator = McpTicketCreator(
        caller,
        settings=_settings(),
        user_id="injected-user",
        tenant_id="injected-tenant",
    )
    arguments = CreateTicketArgs(
        requester_id="user_007",
        title="t",
        description="d",
        category=TicketCategory.REFUND,
    )
    creator.create_ticket(arguments, idempotency_key="c" * 16)
    assert caller.calls[0][1]["user_id"] == "injected-user"
    assert caller.calls[0][1]["tenant_id"] == "injected-tenant"


def test_create_mcp_refund_executor_builds_from_settings() -> None:
    executor = create_mcp_refund_executor(
        Settings(
            _env_file=None,
            mcp_product_base_url="http://127.0.0.1:9100/mcp",
            mcp_product_auth_token="token",
        )
    )
    assert isinstance(executor, McpRefundExecutor)


def test_mcp_refund_executor_calls_refund_order_with_confirmation_and_identity() -> None:
    caller = FakeMcpToolCaller(
        responses={
            "refund_order": {
                "ok": True,
                "confirmation_id": "c" * 16,
                "error_code": None,
                "message": "退款成功。",
                "refund": {
                    "order_id": "A1001",
                    "refund_status": "succeeded",
                },
            }
        }
    )
    executor = McpRefundExecutor(
        caller,
        settings=_settings(),
        user_id="injected-user",
        tenant_id="injected-tenant",
    )
    arguments = RefundOrderArgs(
        order_id="A1001",
        reason="商品有质量问题",
        requester_id="user_007",
    )
    result = executor.refund_order(arguments, idempotency_key="c" * 16)

    assert result["refund_status"] == "succeeded"
    assert caller.calls == [
        (
            "refund_order",
            {
                "order_id": "A1001",
                "reason": "商品有质量问题",
                "confirmation_id": "c" * 16,
                "requester_id": "user_007",
                "user_confirmed": True,
                "user_id": "injected-user",
                "tenant_id": "injected-tenant",
            },
        )
    ]


def test_mcp_refund_executor_maps_failure_payload_to_app_exception() -> None:
    caller = FakeMcpToolCaller(
        responses={
            "refund_order": {
                "ok": False,
                "confirmation_id": "c" * 16,
                "error_code": "ORDER_REFUND_NOT_ALLOWED",
                "message": "当前订单状态不支持退款。",
            }
        }
    )
    executor = McpRefundExecutor(caller, settings=_settings())
    arguments = RefundOrderArgs(
        order_id="A1001",
        reason="不想要了",
        requester_id="user_007",
    )

    with pytest.raises(AppException) as exc_info:
        executor.refund_order(arguments, idempotency_key="c" * 16)

    assert exc_info.value.code == "ORDER_REFUND_NOT_ALLOWED"
    assert exc_info.value.message == "当前订单状态不支持退款。"


class _FakeJavaRefundClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def refund_order(
        self,
        order_id: str,
        reason: str,
        *,
        idempotency_key: str | None = None,
        trace_context: object | None = None,
    ) -> dict[str, object]:
        self.calls.append((order_id, reason, idempotency_key))
        return {"order_id": order_id, "refund_status": "succeeded"}


def test_java_refund_executor_delegates_to_java_client() -> None:
    from app.agents.ticket_agent import JavaRefundExecutor

    client = _FakeJavaRefundClient()
    executor = JavaRefundExecutor(client)
    arguments = RefundOrderArgs(
        order_id="A1002",
        reason="不想要了",
        requester_id="demo_user_001",
    )

    result = executor.refund_order(arguments, idempotency_key="k" * 16)

    assert result == {"order_id": "A1002", "refund_status": "succeeded"}
    assert client.calls == [("A1002", "不想要了", "k" * 16)]
