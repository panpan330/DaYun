"""MCP-backed adapters that satisfy the ticket agent's executor/creator protocols."""

from typing import Any

from app.agents.ticket_agent import TicketFields
from app.core.business_context import get_business_context
from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.mcp_clients.product_client import McpToolCaller, create_product_mcp_client
from app.schemas.cancel import CancelOrderArgs
from app.schemas.refund import RefundOrderArgs
from app.schemas.ticket import CreateTicketArgs, CreatedTicket
from app.schemas.tool import QueryOrderArgs, QueryOrderResult
from app.tools.tool_confirmation import create_tool_confirmation_store
from app.tools.tool_registry import CANCEL_ORDER_TOOL_NAME, REFUND_ORDER_TOOL_NAME


CREATE_TICKET_TOOL_NAME = "create_ticket"
QUERY_ORDER_TOOL_NAME = "query_order"


def _resolve_tenant_id(*, injected: str | None) -> str | None:
    """Tenant comes from the executor/creator injection or the business context
    that the console agent service set for the authenticated actor."""
    if injected is not None:
        return injected or None
    return get_business_context()[1]


def _require_ok(payload: dict[str, Any], *, fallback_code: str) -> None:
    if payload.get("ok") is True:
        return
    raise AppException(
        code=payload.get("error_code") or fallback_code,
        message=payload.get("message") or "工具调用失败，请稍后重试。",
        status_code=502,
    )


class McpTicketCreator:
    """TicketCreator that executes create_ticket through the product MCP client."""

    def __init__(
        self,
        caller: McpToolCaller,
        *,
        settings: Settings | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self._caller = caller
        self._settings = settings or get_settings()
        self._user_id = user_id
        self._tenant_id = tenant_id

    def create_ticket(
        self,
        arguments: CreateTicketArgs,
        *,
        idempotency_key: str,
    ) -> CreatedTicket:
        # The requester id is the agent's ticket_actor_id (the authenticated
        # user). The tenant comes from the injected value or the business
        # context; both are forwarded to the standalone MCP server so Java
        # ownership checks see the real caller, not the default fallback.
        user_id = self._user_id or arguments.requester_id
        tenant_id = _resolve_tenant_id(injected=self._tenant_id)
        payload = self._caller.call_tool(
            CREATE_TICKET_TOOL_NAME,
            {
                "requester_id": arguments.requester_id,
                "title": arguments.title,
                "description": arguments.description,
                "category": arguments.category.value,
                "priority": arguments.priority.value,
                "related_order_id": arguments.related_order_id,
                "confirmation_id": idempotency_key,
                "user_confirmed": True,
                "user_id": user_id,
                "tenant_id": tenant_id,
            },
        )
        _require_ok(payload, fallback_code="TICKET_CREATION_TOOL_ERROR")
        ticket_payload = payload.get("ticket")
        if not isinstance(ticket_payload, dict):
            raise AppException(
                code="MCP_RESULT_INVALID",
                message="AI 工具服务返回了无法解析的工单结果。",
                status_code=502,
            )
        return CreatedTicket.model_validate(ticket_payload)


def mcp_order_query_executor(
    caller: McpToolCaller,
    *,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> Any:
    """Return an OrderQueryExecutor (Callable[[QueryOrderArgs], QueryOrderResult]).

    ``user_id``/``tenant_id`` are forwarded to the product MCP server. They can
    be injected at construction (factory style) or, by default, read from the
    business context that the console agent service set for the authenticated
    actor at request time.
    """

    def execute(arguments: QueryOrderArgs) -> QueryOrderResult:
        ctx_user_id, _ = get_business_context()
        forwarded_user_id = user_id or ctx_user_id
        forwarded_tenant_id = _resolve_tenant_id(injected=tenant_id)
        payload = caller.call_tool(
            QUERY_ORDER_TOOL_NAME,
            {
                "order_id": arguments.order_id,
                "user_id": forwarded_user_id,
                "tenant_id": forwarded_tenant_id,
            },
        )
        _require_ok(payload, fallback_code="TOOL_CALL_FAILED")
        result_payload = payload.get("result")
        if not isinstance(result_payload, dict):
            raise AppException(
                code="MCP_RESULT_INVALID",
                message="AI 工具服务返回了无法解析的订单结果。",
                status_code=502,
            )
        return QueryOrderResult.model_validate(result_payload)

    return execute


def create_mcp_ticket_creator(
    settings: Settings | None = None,
    *,
    caller: McpToolCaller | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> McpTicketCreator:
    resolved_settings = settings or get_settings()
    return McpTicketCreator(
        caller or create_product_mcp_client(resolved_settings),
        settings=resolved_settings,
        user_id=user_id,
        tenant_id=tenant_id,
    )


def create_mcp_order_query_executor(
    settings: Settings | None = None,
    *,
    caller: McpToolCaller | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> Any:
    resolved_settings = settings or get_settings()
    return mcp_order_query_executor(
        caller or create_product_mcp_client(resolved_settings),
        user_id=user_id,
        tenant_id=tenant_id,
    )


class McpRefundExecutor:
    """RefundExecutor that executes refund_order through the product MCP client."""

    def __init__(
        self,
        caller: McpToolCaller,
        *,
        settings: Settings | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self._caller = caller
        self._settings = settings or get_settings()
        self._user_id = user_id
        self._tenant_id = tenant_id

    def refund_order(
        self,
        arguments: RefundOrderArgs,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        # The requester id is the agent's ticket_actor_id (the authenticated
        # user). The tenant comes from the injected value or the business
        # context; both are forwarded to the standalone MCP server so Java
        # ownership checks see the real caller, not the default fallback.
        user_id = self._user_id or arguments.requester_id
        tenant_id = _resolve_tenant_id(injected=self._tenant_id)
        payload = self._caller.call_tool(
            REFUND_ORDER_TOOL_NAME,
            {
                "order_id": arguments.order_id,
                "reason": arguments.reason,
                "confirmation_id": idempotency_key,
                "requester_id": arguments.requester_id,
                "user_confirmed": True,
                "user_id": user_id,
                "tenant_id": tenant_id,
            },
        )
        _require_ok(payload, fallback_code="REFUND_TOOL_ERROR")
        refund_payload = payload.get("refund")
        if not isinstance(refund_payload, dict):
            raise AppException(
                code="MCP_RESULT_INVALID",
                message="AI 工具服务返回了无法解析的退款结果。",
                status_code=502,
            )
        return refund_payload


def create_mcp_refund_executor(
    settings: Settings | None = None,
    *,
    caller: McpToolCaller | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> McpRefundExecutor:
    resolved_settings = settings or get_settings()
    return McpRefundExecutor(
        caller or create_product_mcp_client(resolved_settings),
        settings=resolved_settings,
        user_id=user_id,
        tenant_id=tenant_id,
    )


class McpCancelExecutor:
    """CancelExecutor that executes cancel_order through the product MCP client."""

    def __init__(
        self,
        caller: McpToolCaller,
        *,
        settings: Settings | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self._caller = caller
        self._settings = settings or get_settings()
        self._user_id = user_id
        self._tenant_id = tenant_id

    def cancel_order(
        self,
        arguments: CancelOrderArgs,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        # The requester id is the agent's ticket_actor_id (the authenticated
        # user). The tenant comes from the injected value or the business
        # context; both are forwarded to the standalone MCP server so Java
        # ownership checks see the real caller, not the default fallback.
        user_id = self._user_id or arguments.requester_id
        tenant_id = _resolve_tenant_id(injected=self._tenant_id)
        payload = self._caller.call_tool(
            CANCEL_ORDER_TOOL_NAME,
            {
                "order_id": arguments.order_id,
                "reason": arguments.reason,
                "confirmation_id": idempotency_key,
                "requester_id": arguments.requester_id,
                "user_confirmed": True,
                "user_id": user_id,
                "tenant_id": tenant_id,
            },
        )
        _require_ok(payload, fallback_code="CANCEL_TOOL_ERROR")
        cancel_payload = payload.get("cancel")
        if not isinstance(cancel_payload, dict):
            raise AppException(
                code="MCP_RESULT_INVALID",
                message="AI 工具服务返回了无法解析的取消订单结果。",
                status_code=502,
            )
        return cancel_payload


def create_mcp_cancel_executor(
    settings: Settings | None = None,
    *,
    caller: McpToolCaller | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> McpCancelExecutor:
    resolved_settings = settings or get_settings()
    return McpCancelExecutor(
        caller or create_product_mcp_client(resolved_settings),
        settings=resolved_settings,
        user_id=user_id,
        tenant_id=tenant_id,
    )


def register_ticket_confirmation(
    actor_id: str,
    fields: TicketFields,
    *,
    settings: Settings | None = None,
    is_refund_execution: bool = False,
    is_cancel_execution: bool = False,
) -> str:
    """Register the agent's pending confirmation as confirmed in the shared store.

    Returns the confirmation_id that the MCP server will verify.

    The registered tool_name reflects the path the agent will actually execute:
    ``cancel_order`` when the confirmation belongs to the cancel *execution*
    flow (cancel_request intent → execute_cancel_request), ``refund_order`` when
    it belongs to the refund *execution* flow (refund_request intent →
    execute_refund_request), and ``create_ticket`` otherwise — including
    cancel/refund-typed tickets created through the ordinary ticket flow
    (ticket_request intent → create_ticket).  ``is_cancel_execution`` /
    ``is_refund_execution`` are derived by the console agent service from the
    graph snapshot's ``cancel_request_active`` / ``refund_request_active`` flags;
    they cannot be inferred from the draft fields alone because a typed ticket
    draft and an execution draft share the same TicketFields shape.  The shared
    store's require_confirmed gate only checks confirmation_id + actor_id, so
    this is an audit/attribution concern rather than a functional one.
    """
    from app.agents.ticket_agent import build_pending_ticket_confirmation

    resolved_settings = settings or get_settings()
    confirmation_id = build_pending_ticket_confirmation(fields)["confirmation_id"]
    store = create_tool_confirmation_store(resolved_settings)
    tool_name = (
        CANCEL_ORDER_TOOL_NAME
        if is_cancel_execution
        else REFUND_ORDER_TOOL_NAME if is_refund_execution else CREATE_TICKET_TOOL_NAME
    )
    store.register_confirmed(
        confirmation_id=confirmation_id,
        actor_id=actor_id,
        tool_name=tool_name,
        arguments=dict(fields),
        ttl_seconds=resolved_settings.tool_confirmation_ttl_seconds,
    )
    return confirmation_id
