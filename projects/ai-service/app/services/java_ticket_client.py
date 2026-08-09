import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.business_context import build_java_internal_headers
from app.core.config import Settings
from app.core.exceptions import AppException
from app.core.trace import TRACE_ID_HEADER, build_trace_headers, generate_trace_id
from app.schemas.ticket import CreateTicketArgs, CreatedTicket
from app.services.java_error_mapping import build_java_error_app_exception


logger = logging.getLogger(__name__)


class JavaTicketClient:
    """HTTP adapter for the ticket-creation API owned by the Java service."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.settings = settings
        self.transport = transport

    @classmethod
    def from_settings(cls, settings: Settings) -> "JavaTicketClient":
        return cls(
            base_url=settings.resolved_java_business_service_base_url,
            timeout_seconds=settings.resolved_java_business_service_timeout_seconds,
            settings=settings,
        )

    def create_ticket(
        self,
        arguments: CreateTicketArgs,
        *,
        idempotency_key: str,
    ) -> CreatedTicket:
        from app.agents.tracing_spans import start_java_span

        with start_java_span(path="/internal/tickets", method="POST"):
            return self._create_ticket_inner(
                arguments,
                idempotency_key=idempotency_key,
            )

    def _create_ticket_inner(
        self,
        arguments: CreateTicketArgs,
        *,
        idempotency_key: str,
    ) -> CreatedTicket:
        path = "/internal/tickets"
        start_time = perf_counter()
        logger.info(
            (
                "java_ticket_create_started method=POST path=%s category=%s "
                "priority=%s related_order_id=%s"
            ),
            path,
            arguments.category,
            arguments.priority,
            arguments.related_order_id,
        )
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post(
                    path,
                    json=_build_java_ticket_payload(
                        arguments,
                        confirmation_id=idempotency_key,
                    ),
                    headers=self._build_headers(idempotency_key=idempotency_key),
                )
        except httpx.TimeoutException as exc:
            elapsed_ms = (perf_counter() - start_time) * 1000
            logger.warning(
                "java_ticket_create_failed method=POST path=%s code=%s elapsed_ms=%.2f",
                path,
                "TOOL_TIMEOUT",
                elapsed_ms,
            )
            raise AppException(
                code="TOOL_TIMEOUT",
                message="创建工单工具调用超时，请稍后重试。",
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            elapsed_ms = (perf_counter() - start_time) * 1000
            logger.warning(
                "java_ticket_create_failed method=POST path=%s code=%s elapsed_ms=%.2f",
                path,
                "TOOL_UPSTREAM_ERROR",
                elapsed_ms,
            )
            raise AppException(
                code="TOOL_UPSTREAM_ERROR",
                message="工单业务服务暂时不可用，请稍后重试。",
                status_code=502,
            ) from exc

        elapsed_ms = (perf_counter() - start_time) * 1000
        logger.info(
            (
                "java_ticket_create_finished method=POST path=%s status_code=%s "
                "upstream_trace_id=%s elapsed_ms=%.2f"
            ),
            path,
            response.status_code,
            response.headers.get(TRACE_ID_HEADER, "-"),
            elapsed_ms,
        )

        if response.status_code != 201:
            raise build_java_error_app_exception(
                response,
                operation="ticket_creation",
                fallback_code="TICKET_UPSTREAM_REJECTED",
                fallback_message="工单业务服务拒绝了已经校验过的请求，请联系管理员排查接口契约。",
                fallback_status_code=502,
            )

        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise AppException(
                code="TOOL_RESULT_VALIDATION_FAILED",
                message="工单业务服务返回的 JSON 格式不正确。",
                status_code=502,
            ) from exc

        try:
            ticket = CreatedTicket.model_validate(
                _map_java_ticket_response_to_created_ticket(payload, arguments)
            )
        except ValidationError as exc:
            raise AppException(
                code="TOOL_RESULT_VALIDATION_FAILED",
                message="工单业务服务返回的数据不符合约定。",
                status_code=502,
                details=exc.errors(include_url=False),
            ) from exc
        logger.info(
            "java_ticket_create_validated ticket_id=%s category=%s priority=%s",
            ticket.ticket_id,
            ticket.category,
            ticket.priority,
        )
        return ticket

    def _build_headers(self, *, idempotency_key: str) -> dict[str, str]:
        headers = {
            **build_trace_headers(),
            "Idempotency-Key": idempotency_key,
        }
        headers.setdefault(TRACE_ID_HEADER, generate_trace_id())
        if self.settings is not None:
            headers.update(build_java_internal_headers(self.settings))
        return headers


def _build_java_ticket_payload(
    arguments: CreateTicketArgs,
    *,
    confirmation_id: str,
) -> dict[str, Any]:
    return {
        "title": arguments.title,
        "description": arguments.description,
        "category": arguments.category.value,
        "priority": arguments.priority.value,
        "related_order_id": arguments.related_order_id,
        "source": "ai_agent",
        "confirmation_id": confirmation_id,
    }


def _map_java_ticket_response_to_created_ticket(
    payload: Any,
    arguments: CreateTicketArgs,
) -> dict[str, Any]:
    data = _unwrap_java_api_response_data(payload)
    return {
        "ticket_id": data.get("ticket_id"),
        "requester_id": data.get("requester_id") or arguments.requester_id,
        "title": data.get("title"),
        "description": data.get("description") or arguments.description,
        "category": data.get("category"),
        "priority": data.get("priority"),
        "related_order_id": data.get("related_order_id"),
        "created_at": data.get("created_at"),
    }


def _unwrap_java_api_response_data(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise AppException(
            code="TOOL_RESULT_VALIDATION_FAILED",
            message="工单业务服务返回的数据结构不正确。",
            status_code=502,
        )

    if "success" not in payload and "data" not in payload:
        return payload

    if payload.get("success") is True and isinstance(payload.get("data"), Mapping):
        return payload["data"]

    raise AppException(
        code="TOOL_RESULT_VALIDATION_FAILED",
        message="工单业务服务返回的数据结构不正确。",
        status_code=502,
    )
