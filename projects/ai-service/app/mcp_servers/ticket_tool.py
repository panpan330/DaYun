"""MCP create_ticket tool adapter for the existing ticket creation chain."""

from typing import Annotated, Any, Protocol

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.schemas.ticket import (
    CreateTicketArgs,
    CreatedTicket,
    TicketCategory,
    TicketPriority,
)
from app.services.java_ticket_client import JavaTicketClient
from app.tools.idempotency import run_idempotent_tool
from app.tools.tool_registry import authorize_tool_call


RequesterId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Temporary demo requester id. Production code derives it from authentication.",
    ),
]
TicketTitle = Annotated[
    str,
    Field(min_length=1, max_length=200, description="Support ticket title."),
]
TicketDescription = Annotated[
    str,
    Field(min_length=1, max_length=1000, description="Support ticket description."),
]
RelatedOrderId = Annotated[
    str | None,
    Field(default=None, max_length=64, description="Optional related order id."),
]
ConfirmationId = Annotated[
    str,
    Field(
        pattern=r"^[a-f0-9]{32}$",
        description="Confirmation id approved by the user before the write operation.",
    ),
]

BUSINESS_TICKET_ERROR_CODES = {
    "ORDER_NOT_SUPPORT_TICKET",
    "TICKET_ALREADY_EXISTS",
    "TICKET_REQUEST_INVALID",
    "IDEMPOTENCY_KEY_CONFLICT",
}


class TicketCreator(Protocol):
    def create_ticket(
        self,
        arguments: CreateTicketArgs,
        *,
        idempotency_key: str,
    ) -> CreatedTicket: ...


class McpCreateTicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requester_id: RequesterId
    title: TicketTitle
    description: TicketDescription
    category: TicketCategory
    confirmation_id: ConfirmationId
    priority: TicketPriority = TicketPriority.NORMAL
    related_order_id: RelatedOrderId = None
    user_confirmed: bool = False

    @field_validator(
        "requester_id",
        "title",
        "description",
        "related_order_id",
        "confirmation_id",
        mode="before",
    )
    @classmethod
    def strip_string_fields(cls, value: object) -> object:
        if isinstance(value, str):
            stripped_value = value.strip()
            return stripped_value or None
        return value


def create_ticket_creator(settings: Settings | None = None) -> JavaTicketClient:
    return JavaTicketClient.from_settings(settings or get_settings())


def sanitize_created_ticket(ticket: CreatedTicket) -> dict[str, Any]:
    """Return the minimum ticket fields the model needs for a final answer."""
    return {
        "ticket_id": ticket.ticket_id,
        "title": ticket.title,
        "category": ticket.category.value,
        "priority": ticket.priority.value,
        "related_order_id": ticket.related_order_id,
        "created_at": ticket.created_at.isoformat(),
    }


def _create_ticket_response(
    *,
    ok: bool,
    allowed: bool,
    confirmation_checked: bool,
    confirmation_id: str | None,
    error_code: str | None,
    message: str,
    ticket: dict[str, Any] | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "allowed": allowed,
        "action": "create_ticket",
        "action_type": "write",
        "requires_confirmation": True,
        "confirmation_checked": confirmation_checked,
        "confirmation_id": confirmation_id,
        "error_code": error_code,
        "message": message,
        "retryable": retryable,
        "security_checks": {
            "input_validated": error_code != "INVALID_TOOL_ARGUMENTS",
            "user_confirmed": confirmation_checked,
            "idempotency_key_checked": confirmation_id is not None,
            "idempotency_key_source": "confirmation_id" if confirmation_id else None,
            "output_allowlist_applied": ticket is not None,
            "sensitive_fields_returned": False,
        },
        "ticket": ticket,
    }


def _confirmation_required_response(confirmation_id: str | None) -> dict[str, Any]:
    return _create_ticket_response(
        ok=False,
        allowed=False,
        confirmation_checked=False,
        confirmation_id=confirmation_id,
        error_code="TOOL_CONFIRMATION_REQUIRED",
        message="创建工单是写操作，必须先拿到用户确认，本次请求不会执行。",
    )


def _invalid_arguments_response(exc: ValidationError) -> dict[str, Any]:
    return {
        "ok": False,
        "allowed": False,
        "action": "create_ticket",
        "action_type": "write",
        "requires_confirmation": True,
        "confirmation_checked": False,
        "confirmation_id": None,
        "error_code": "INVALID_TOOL_ARGUMENTS",
        "message": "创建工单参数不正确，请重新整理后再确认。",
        "retryable": False,
        "security_checks": {
            "input_validated": False,
            "user_confirmed": False,
            "idempotency_key_checked": False,
            "idempotency_key_source": None,
            "output_allowlist_applied": False,
            "sensitive_fields_returned": False,
        },
        "errors": exc.errors(include_url=False, include_input=False),
        "ticket": None,
    }


def _business_error_response(
    exc: AppException,
    *,
    confirmation_id: str,
) -> dict[str, Any]:
    return _create_ticket_response(
        ok=False,
        allowed=True,
        confirmation_checked=True,
        confirmation_id=confirmation_id,
        error_code=exc.code,
        message=exc.message,
        ticket=None,
        retryable=False,
    )


def _raise_safe_tool_error(exc: AppException) -> None:
    if exc.code == "TOOL_TIMEOUT":
        raise ToolError(
            "TOOL_TIMEOUT: 创建工单工具响应超时，已使用幂等键保护，请稍后查询结果或重试同一确认。"
        ) from exc

    raise ToolError(f"{exc.code}: 创建工单工具暂时不可用，请稍后重试或联系人工处理。") from exc


def create_ticket_for_mcp(
    *,
    requester_id: str,
    title: str,
    description: str,
    category: TicketCategory,
    confirmation_id: str,
    priority: TicketPriority = TicketPriority.NORMAL,
    related_order_id: str | None = None,
    user_confirmed: bool = False,
    creator: TicketCreator | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Create a ticket through the existing Java adapter with MCP write boundaries."""
    if not user_confirmed:
        return _confirmation_required_response(confirmation_id.strip() or None)

    try:
        request = McpCreateTicketRequest(
            requester_id=requester_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            related_order_id=related_order_id,
            confirmation_id=confirmation_id,
            user_confirmed=user_confirmed,
        )
        arguments = CreateTicketArgs(
            requester_id=request.requester_id,
            title=request.title,
            description=request.description,
            category=request.category,
            priority=request.priority,
            related_order_id=request.related_order_id,
        )
    except ValidationError as exc:
        return _invalid_arguments_response(exc)

    try:
        authorize_tool_call("create_ticket", user_confirmed=True)
        ticket_creator = creator or create_ticket_creator(settings)
        ticket = run_idempotent_tool(
            "create_ticket",
            arguments,
            request.confirmation_id,
            lambda: ticket_creator.create_ticket(
                arguments,
                idempotency_key=request.confirmation_id,
            ),
        )
    except AppException as exc:
        if exc.code in BUSINESS_TICKET_ERROR_CODES:
            return _business_error_response(
                exc,
                confirmation_id=request.confirmation_id,
            )
        _raise_safe_tool_error(exc)
    except Exception as exc:
        raise ToolError(
            "TICKET_CREATION_TOOL_ERROR: 创建工单工具暂时不可用，请稍后重试或联系人工处理。"
        ) from exc

    return _create_ticket_response(
        ok=True,
        allowed=True,
        confirmation_checked=True,
        confirmation_id=request.confirmation_id,
        error_code=None,
        message="工单创建成功。",
        ticket=sanitize_created_ticket(ticket),
    )
