"""Tool error handling helpers for MCP learning examples."""

from typing import Any, Literal

from mcp.server.mcpserver.exceptions import ToolError


ToolErrorScenario = Literal[
    "success",
    "business_not_found",
    "permission_denied",
    "upstream_timeout",
    "unexpected_failure",
]


def business_error(
    *,
    error_code: str,
    message: str,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": error_code,
        "message": message,
        "retryable": retryable,
        "details": details or {},
    }


def simulate_tool_error_response(scenario: ToolErrorScenario) -> dict[str, Any]:
    if scenario == "success":
        return {
            "ok": True,
            "error_code": None,
            "message": "Tool completed successfully.",
            "retryable": False,
            "details": {"example_result": "ticket_error_handling_smoke"},
        }

    if scenario == "business_not_found":
        return business_error(
            error_code="ORDER_NOT_FOUND",
            message="没有找到符合条件的订单，请确认订单号是否正确。",
            details={"safe_reason": "order_not_found"},
        )

    if scenario == "permission_denied":
        return business_error(
            error_code="ORDER_ACCESS_DENIED",
            message="当前用户无权查看或操作该订单。",
            details={"safe_reason": "permission_denied"},
        )

    if scenario == "upstream_timeout":
        raise ToolError(
            "UPSTREAM_TIMEOUT: 订单服务暂时没有响应，请稍后重试。"
        )

    try:
        raise RuntimeError("simulated internal dependency failure")
    except RuntimeError as exc:
        raise ToolError(
            "INTERNAL_TOOL_ERROR: 工具执行失败，请稍后重试或联系人工处理。"
        ) from exc
