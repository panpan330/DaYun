"""MCP query_order tool adapter for the existing order lookup chain."""

from typing import Annotated, Any

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field, ValidationError

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.tool import QueryOrderArgs
from app.tools.fake_order_tool import OrderLookupClient, query_order as run_query_order


OrderId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Order id to query, for example A1001.",
    ),
]

BUSINESS_ORDER_ERROR_CODES = {
    "ORDER_ID_INVALID",
    "ORDER_NOT_FOUND",
    "ORDER_ACCESS_DENIED",
}


def _mcp_query_order_response(
    *,
    ok: bool,
    error_code: str | None,
    message: str,
    result: dict[str, Any] | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "allowed": True,
        "action": "query_order",
        "action_type": "read",
        "requires_confirmation": False,
        "error_code": error_code,
        "message": message,
        "retryable": retryable,
        "security_checks": {
            "input_validated": True,
            "output_allowlist_applied": result is not None,
            "sensitive_fields_returned": False,
        },
        "result": result,
    }


def _invalid_arguments_response(exc: ValidationError) -> dict[str, Any]:
    return {
        "ok": False,
        "allowed": False,
        "action": "query_order",
        "action_type": "read",
        "requires_confirmation": False,
        "error_code": "INVALID_TOOL_ARGUMENTS",
        "message": "订单号参数不正确，请确认后重新输入。",
        "retryable": False,
        "security_checks": {
            "input_validated": False,
            "output_allowlist_applied": False,
            "sensitive_fields_returned": False,
        },
        "errors": exc.errors(include_url=False, include_input=False),
        "result": None,
    }


def _business_error_response(exc: AppException) -> dict[str, Any]:
    return _mcp_query_order_response(
        ok=False,
        error_code=exc.code,
        message=exc.message,
        result=None,
        retryable=False,
    )


def _raise_safe_tool_error(exc: AppException) -> None:
    if exc.code == "TOOL_TIMEOUT":
        raise ToolError(f"{exc.code}: 订单查询工具调用超时，请稍后重试。") from exc

    raise ToolError(f"{exc.code}: 订单查询工具暂时不可用，请稍后重试。") from exc


def query_order_for_mcp(
    order_id: str,
    *,
    client: OrderLookupClient | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Query an order through the existing Java adapter and return MCP-safe output."""
    try:
        arguments = QueryOrderArgs(order_id=order_id)
    except ValidationError as exc:
        return _invalid_arguments_response(exc)

    try:
        result = run_query_order(arguments, client=client, settings=settings)
    except AppException as exc:
        if exc.code in BUSINESS_ORDER_ERROR_CODES:
            return _business_error_response(exc)
        _raise_safe_tool_error(exc)

    return _mcp_query_order_response(
        ok=True,
        error_code=None,
        message="订单查询成功。",
        result=result.model_dump(mode="json"),
        retryable=False,
    )
