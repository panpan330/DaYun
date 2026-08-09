from collections.abc import Mapping
import logging
from time import perf_counter
from typing import Any
from urllib.parse import quote

import httpx

from app.core.config import Settings
from app.core.business_context import build_java_internal_headers
from app.core.exceptions import AppException
from app.core.trace import TRACE_ID_HEADER, build_trace_headers, generate_trace_id
from app.services.java_error_mapping import build_java_error_app_exception


logger = logging.getLogger(__name__)


class JavaOrderClient:
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
    def from_settings(cls, settings: Settings) -> "JavaOrderClient":
        return cls(
            base_url=settings.resolved_java_business_service_base_url,
            timeout_seconds=settings.resolved_java_business_service_timeout_seconds,
            settings=settings,
        )

    def get_order(self, order_id: str) -> Mapping[str, Any]:
        from app.agents.tracing_spans import start_java_span

        with start_java_span(path=f"/internal/orders/{order_id}", method="GET"):
            return self._get_order_inner(order_id)

    def _get_order_inner(self, order_id: str) -> Mapping[str, Any]:
        path = f"/internal/orders/{order_id}"
        start_time = perf_counter()
        logger.info(
            "java_order_request_started method=GET path=%s order_id=%s",
            path,
            order_id,
        )
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.get(path, headers=self._build_headers())
        except httpx.TimeoutException as exc:
            elapsed_ms = (perf_counter() - start_time) * 1000
            logger.warning(
                "java_order_request_failed method=GET path=%s order_id=%s code=%s elapsed_ms=%.2f",
                path,
                order_id,
                "TOOL_TIMEOUT",
                elapsed_ms,
            )
            raise AppException(
                code="TOOL_TIMEOUT",
                message="订单查询工具调用超时，请稍后重试。",
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            elapsed_ms = (perf_counter() - start_time) * 1000
            logger.warning(
                "java_order_request_failed method=GET path=%s order_id=%s code=%s elapsed_ms=%.2f",
                path,
                order_id,
                "TOOL_UPSTREAM_ERROR",
                elapsed_ms,
            )
            raise AppException(
                code="TOOL_UPSTREAM_ERROR",
                message="订单查询服务暂时不可用，请稍后重试。",
                status_code=502,
            ) from exc

        elapsed_ms = (perf_counter() - start_time) * 1000
        logger.info(
            (
                "java_order_request_finished method=GET path=%s order_id=%s "
                "status_code=%s upstream_trace_id=%s elapsed_ms=%.2f"
            ),
            path,
            order_id,
            response.status_code,
            response.headers.get(TRACE_ID_HEADER, "-"),
            elapsed_ms,
        )

        if response.status_code != 200:
            raise build_java_error_app_exception(
                response,
                operation="order_query",
                fallback_code="TOOL_UPSTREAM_ERROR",
                fallback_message="订单查询服务返回了无法处理的状态，请稍后重试。",
                fallback_status_code=502,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise AppException(
                code="TOOL_RESULT_VALIDATION_FAILED",
                message="订单查询服务返回的 JSON 格式不正确。",
                status_code=502,
            ) from exc

        if not isinstance(data, dict):
            raise AppException(
                code="TOOL_RESULT_VALIDATION_FAILED",
                message="订单查询服务返回的数据结构不正确。",
                status_code=502,
            )

        return _unwrap_java_api_response_data(data)

    def refund_order(
        self,
        order_id: str,
        reason: str,
        *,
        idempotency_key: str | None = None,
        trace_context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        from app.agents.tracing_spans import start_java_span

        path = f"/internal/orders/{quote(order_id)}/refund"
        with start_java_span(path=path, method="POST"):
            return self._refund_order_inner(
                path,
                order_id,
                reason,
                idempotency_key=idempotency_key,
                trace_context=trace_context,
            )

    def _refund_order_inner(
        self,
        path: str,
        order_id: str,
        reason: str,
        *,
        idempotency_key: str | None,
        trace_context: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        start_time = perf_counter()
        logger.info(
            "java_order_request_started method=POST path=%s order_id=%s",
            path,
            order_id,
        )
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post(
                    path,
                    json={"reason": reason},
                    headers=self._build_headers(
                        trace_context=trace_context,
                        idempotency_key=idempotency_key,
                    ),
                )
        except httpx.TimeoutException as exc:
            elapsed_ms = (perf_counter() - start_time) * 1000
            logger.warning(
                "java_order_request_failed method=POST path=%s order_id=%s code=%s elapsed_ms=%.2f",
                path,
                order_id,
                "TOOL_TIMEOUT",
                elapsed_ms,
            )
            raise AppException(
                code="TOOL_TIMEOUT",
                message="退款工具调用超时，请稍后重试。",
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            elapsed_ms = (perf_counter() - start_time) * 1000
            logger.warning(
                "java_order_request_failed method=POST path=%s order_id=%s code=%s elapsed_ms=%.2f",
                path,
                order_id,
                "TOOL_UPSTREAM_ERROR",
                elapsed_ms,
            )
            raise AppException(
                code="TOOL_UPSTREAM_ERROR",
                message="退款服务暂时不可用，请稍后重试。",
                status_code=502,
            ) from exc

        elapsed_ms = (perf_counter() - start_time) * 1000
        logger.info(
            (
                "java_order_request_finished method=POST path=%s order_id=%s "
                "status_code=%s upstream_trace_id=%s elapsed_ms=%.2f"
            ),
            path,
            order_id,
            response.status_code,
            response.headers.get(TRACE_ID_HEADER, "-"),
            elapsed_ms,
        )

        if response.status_code != 200:
            raise build_java_error_app_exception(
                response,
                operation="order_refund",
                fallback_code="TOOL_UPSTREAM_ERROR",
                fallback_message="退款服务返回了无法处理的状态，请稍后重试。",
                fallback_status_code=502,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise AppException(
                code="TOOL_RESULT_VALIDATION_FAILED",
                message="退款服务返回的 JSON 格式不正确。",
                status_code=502,
            ) from exc

        if not isinstance(data, dict):
            raise AppException(
                code="TOOL_RESULT_VALIDATION_FAILED",
                message="退款服务返回的数据结构不正确。",
                status_code=502,
            )

        return _unwrap_java_api_response_data(data)

    def cancel_order(
        self,
        order_id: str,
        reason: str,
        *,
        idempotency_key: str | None = None,
        trace_context: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        from app.agents.tracing_spans import start_java_span

        path = f"/internal/orders/{quote(order_id)}/cancel"
        with start_java_span(path=path, method="POST"):
            return self._cancel_order_inner(
                path,
                order_id,
                reason,
                idempotency_key=idempotency_key,
                trace_context=trace_context,
            )

    def _cancel_order_inner(
        self,
        path: str,
        order_id: str,
        reason: str,
        *,
        idempotency_key: str | None,
        trace_context: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        start_time = perf_counter()
        logger.info(
            "java_order_request_started method=POST path=%s order_id=%s",
            path,
            order_id,
        )
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post(
                    path,
                    json={"reason": reason},
                    headers=self._build_headers(
                        trace_context=trace_context,
                        idempotency_key=idempotency_key,
                    ),
                )
        except httpx.TimeoutException as exc:
            elapsed_ms = (perf_counter() - start_time) * 1000
            logger.warning(
                "java_order_request_failed method=POST path=%s order_id=%s code=%s elapsed_ms=%.2f",
                path,
                order_id,
                "TOOL_TIMEOUT",
                elapsed_ms,
            )
            raise AppException(
                code="TOOL_TIMEOUT",
                message="取消订单工具调用超时，请稍后重试。",
                status_code=504,
            ) from exc
        except httpx.RequestError as exc:
            elapsed_ms = (perf_counter() - start_time) * 1000
            logger.warning(
                "java_order_request_failed method=POST path=%s order_id=%s code=%s elapsed_ms=%.2f",
                path,
                order_id,
                "TOOL_UPSTREAM_ERROR",
                elapsed_ms,
            )
            raise AppException(
                code="TOOL_UPSTREAM_ERROR",
                message="取消订单服务暂时不可用，请稍后重试。",
                status_code=502,
            ) from exc

        elapsed_ms = (perf_counter() - start_time) * 1000
        logger.info(
            (
                "java_order_request_finished method=POST path=%s order_id=%s "
                "status_code=%s upstream_trace_id=%s elapsed_ms=%.2f"
            ),
            path,
            order_id,
            response.status_code,
            response.headers.get(TRACE_ID_HEADER, "-"),
            elapsed_ms,
        )

        if response.status_code != 200:
            raise build_java_error_app_exception(
                response,
                operation="order_cancel",
                fallback_code="TOOL_UPSTREAM_ERROR",
                fallback_message="取消订单服务返回了无法处理的状态，请稍后重试。",
                fallback_status_code=502,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise AppException(
                code="TOOL_RESULT_VALIDATION_FAILED",
                message="取消订单服务返回的 JSON 格式不正确。",
                status_code=502,
            ) from exc

        if not isinstance(data, dict):
            raise AppException(
                code="TOOL_RESULT_VALIDATION_FAILED",
                message="取消订单服务返回的数据结构不正确。",
                status_code=502,
            )

        return _unwrap_java_api_response_data(data)

    def _build_headers(
        self,
        *,
        trace_context: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, str]:
        headers = build_trace_headers()
        headers.setdefault(TRACE_ID_HEADER, generate_trace_id())
        if trace_context is not None:
            trace_id = trace_context.get("trace_id") or trace_context.get(TRACE_ID_HEADER)
            if trace_id:
                headers[TRACE_ID_HEADER] = str(trace_id)
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if self.settings is not None:
            headers.update(build_java_internal_headers(self.settings))
        return headers


def _unwrap_java_api_response_data(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "success" not in payload and "data" not in payload:
        return payload

    if payload.get("success") is True and isinstance(payload.get("data"), Mapping):
        return payload["data"]

    raise AppException(
        code="TOOL_RESULT_VALIDATION_FAILED",
        message="订单查询服务返回的数据结构不正确。",
        status_code=502,
    )
