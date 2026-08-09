from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.schemas.tool import QueryOrderArgs, QueryOrderResult
from app.services.java_order_client import JavaOrderClient


class OrderLookupClient(Protocol):
    def get_order(self, order_id: str) -> Mapping[str, Any]:
        """Return raw order data from a business service."""


def map_java_order_to_query_order_payload(
    raw_order: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "order_id": raw_order.get("order_id"),
        "order_status": raw_order.get("order_status"),
        "payment_status": raw_order.get("payment_status"),
        "logistics_message": raw_order.get("logistics_message"),
        "latest_event": raw_order.get("latest_event"),
        "can_create_ticket": raw_order.get("can_create_ticket"),
        "source": "java_business_service",
    }


def validate_query_order_result(raw_result: Mapping[str, Any]) -> QueryOrderResult:
    try:
        return QueryOrderResult.model_validate(raw_result)
    except ValidationError as exc:
        raise AppException(
            code="TOOL_RESULT_VALIDATION_FAILED",
            message="工具返回结果校验失败，请稍后重试。",
            status_code=502,
            details=exc.errors(include_url=False, include_input=False),
        ) from exc


def map_query_order_error(exc: Exception) -> AppException:
    if isinstance(exc, AppException):
        return exc
    return AppException(
        code="TOOL_CALL_FAILED",
        message="工具调用失败，请稍后重试。",
        status_code=502,
    )


def create_order_lookup_client(settings: Settings | None = None) -> JavaOrderClient:
    return JavaOrderClient.from_settings(settings or get_settings())


def query_order(
    arguments: QueryOrderArgs,
    *,
    client: OrderLookupClient | None = None,
    settings: Settings | None = None,
) -> QueryOrderResult:
    try:
        order_client = client or create_order_lookup_client(settings)
        raw_order = order_client.get_order(arguments.order_id)
        tool_payload = map_java_order_to_query_order_payload(raw_order)
        return validate_query_order_result(tool_payload)
    except AppException:
        raise
    except Exception as exc:
        raise map_query_order_error(exc) from exc
