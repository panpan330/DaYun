from dataclasses import dataclass
from typing import Literal

import httpx

from app.core.exceptions import AppException


JavaOperation = Literal["order_query", "ticket_creation", "order_refund", "order_cancel"]


@dataclass(frozen=True)
class JavaErrorMapping:
    code: str
    message: str
    status_code: int


ORDER_QUERY_UNAVAILABLE = "订单查询服务暂时不可用，请稍后重试。"
TICKET_CREATION_UNAVAILABLE = "工单业务服务暂时不可用，请稍后重试。"
TICKET_CONTRACT_REJECTED = "工单业务服务拒绝了已经校验过的请求，请联系管理员排查接口契约。"
REFUND_UNAVAILABLE = "退款服务暂时不可用，请稍后重试。"
CANCEL_UNAVAILABLE = "取消订单服务暂时不可用，请稍后重试。"


USER_SAFE_JAVA_ERROR_MAPPINGS: dict[str, JavaErrorMapping] = {
    "ORDER_ID_INVALID": JavaErrorMapping(
        code="ORDER_ID_INVALID",
        message="订单号格式不正确，请确认后重新输入。",
        status_code=422,
    ),
    "ORDER_NOT_FOUND": JavaErrorMapping(
        code="ORDER_NOT_FOUND",
        message="订单不存在，请确认订单号是否正确。",
        status_code=404,
    ),
    "ORDER_ACCESS_DENIED": JavaErrorMapping(
        code="ORDER_ACCESS_DENIED",
        message="当前账号无权查看或操作该订单。",
        status_code=403,
    ),
    "ORDER_NOT_SUPPORT_TICKET": JavaErrorMapping(
        code="ORDER_NOT_SUPPORT_TICKET",
        message="当前订单暂不支持创建这类工单，如需帮助可以联系人工客服。",
        status_code=409,
    ),
    "TICKET_REQUEST_INVALID": JavaErrorMapping(
        code="TICKET_REQUEST_INVALID",
        message="工单信息不完整或格式不正确，请补充后重新提交。",
        status_code=422,
    ),
    "TICKET_ALREADY_EXISTS": JavaErrorMapping(
        code="TICKET_ALREADY_EXISTS",
        message="已经存在相似工单，请不要重复提交。",
        status_code=409,
    ),
    "ORDER_NOT_REFUNDABLE": JavaErrorMapping(
        code="ORDER_NOT_REFUNDABLE",
        message="当前订单状态不支持退款。",
        status_code=409,
    ),
    "REFUND_ALREADY_EXISTS": JavaErrorMapping(
        code="REFUND_ALREADY_EXISTS",
        message="订单已退款，请勿重复操作。",
        status_code=409,
    ),
    "REFUND_REASON_REQUIRED": JavaErrorMapping(
        code="REFUND_REASON_REQUIRED",
        message="退款原因不能为空，请补充退款原因后重试。",
        status_code=422,
    ),
    "REFUND_REASON_TOO_LONG": JavaErrorMapping(
        code="REFUND_REASON_TOO_LONG",
        message="退款原因不能超过 200 字，请精简后重试。",
        status_code=422,
    ),
    "ORDER_NOT_CANCELABLE": JavaErrorMapping(
        code="ORDER_NOT_CANCELABLE",
        message="当前订单状态不支持取消。",
        status_code=409,
    ),
    "CANCEL_ALREADY_EXISTS": JavaErrorMapping(
        code="CANCEL_ALREADY_EXISTS",
        message="订单已取消，请勿重复操作。",
        status_code=409,
    ),
    "CANCEL_REASON_REQUIRED": JavaErrorMapping(
        code="CANCEL_REASON_REQUIRED",
        message="取消原因不能为空，请补充取消原因后重试。",
        status_code=422,
    ),
    "CANCEL_REASON_TOO_LONG": JavaErrorMapping(
        code="CANCEL_REASON_TOO_LONG",
        message="取消原因不能超过 200 字，请精简后重试。",
        status_code=422,
    ),
    "IDEMPOTENCY_KEY_CONFLICT": JavaErrorMapping(
        code="IDEMPOTENCY_KEY_CONFLICT",
        message="本次提交和已确认的工单请求不一致，请重新确认后再提交。",
        status_code=409,
    ),
    "TOOL_RATE_LIMITED": JavaErrorMapping(
        code="TOOL_RATE_LIMITED",
        message="业务服务请求过于频繁，请稍后再试。",
        status_code=429,
    ),
}


def extract_java_error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, dict):
        return None

    code = payload.get("code")
    if isinstance(code, str) and code.strip():
        return code.strip()
    return None


def build_java_error_app_exception(
    response: httpx.Response,
    *,
    operation: JavaOperation,
    fallback_code: str,
    fallback_message: str,
    fallback_status_code: int,
) -> AppException:
    upstream_code = extract_java_error_code(response)
    if upstream_code in USER_SAFE_JAVA_ERROR_MAPPINGS:
        mapping = USER_SAFE_JAVA_ERROR_MAPPINGS[upstream_code]
        return AppException(
            code=mapping.code,
            message=mapping.message,
            status_code=mapping.status_code,
        )

    if upstream_code in {"INTERNAL_AUTH_FAILED", "JAVA_SERVICE_ERROR"}:
        return AppException(
            code="TOOL_UPSTREAM_ERROR",
            message=_unavailable_message_for(operation),
            status_code=502,
        )

    if upstream_code in {"IDEMPOTENCY_KEY_REQUIRED", "IDEMPOTENCY_KEY_INVALID"}:
        return AppException(
            code="TICKET_UPSTREAM_REJECTED",
            message=TICKET_CONTRACT_REJECTED,
            status_code=502,
        )

    if response.status_code >= 500:
        return AppException(
            code="TOOL_UPSTREAM_ERROR",
            message=_unavailable_message_for(operation),
            status_code=502,
        )

    return AppException(
        code=fallback_code,
        message=fallback_message,
        status_code=fallback_status_code,
    )


def _unavailable_message_for(operation: JavaOperation) -> str:
    if operation == "ticket_creation":
        return TICKET_CREATION_UNAVAILABLE
    if operation == "order_refund":
        return REFUND_UNAVAILABLE
    if operation == "order_cancel":
        return CANCEL_UNAVAILABLE
    return ORDER_QUERY_UNAVAILABLE
