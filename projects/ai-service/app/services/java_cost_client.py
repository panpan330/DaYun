"""Internal adapter for LLM cost aggregation records owned by Java (MySQL)."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

import httpx

from app.core.business_context import build_java_internal_headers
from app.core.config import Settings
from app.core.trace import build_trace_headers

logger = logging.getLogger(__name__)


class JavaCostClientError(Exception):
    """Cost record sync with Java failed."""

    def __init__(self, message: str, *, status_code: int | None = None, code: str = "COST_SYNC_FAILED") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class JavaCostClient:
    """Internal adapter for LLM cost aggregation (model x intent) owned by Java (MySQL)."""

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
    def from_settings(cls, settings: Settings) -> "JavaCostClient":
        return cls(
            base_url=settings.resolved_java_business_service_base_url,
            timeout_seconds=settings.resolved_java_business_service_timeout_seconds,
            settings=settings,
        )

    def sync_cost_records(self, records: list[dict[str, Any]]) -> None:
        self._request("post", "/internal/ai-cost-records", json={"records": records})

    def fetch_overview(self) -> dict[str, Any]:
        data = self._request("get", "/internal/ai-cost-records/overview")
        return dict(data)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
    ) -> Any:
        started_at = perf_counter()
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.request(method, path, json=json, headers=self._build_headers())
        except httpx.RequestError as exc:
            logger.warning(
                "java_cost_request_failed path=%s error_type=%s elapsed_ms=%.2f",
                path,
                type(exc).__name__,
                (perf_counter() - started_at) * 1000,
            )
            raise JavaCostClientError(
                f"Cost sync request to {path} failed: {type(exc).__name__}",
                status_code=None,
            ) from exc

        if response.status_code != 200:
            logger.warning(
                "java_cost_request_rejected path=%s status_code=%s elapsed_ms=%.2f",
                path,
                response.status_code,
                (perf_counter() - started_at) * 1000,
            )
            raise JavaCostClientError(
                f"Cost sync request to {path} rejected with {response.status_code}",
                status_code=response.status_code,
            )
        try:
            payload = response.json()
            if not payload.get("success"):
                raise JavaCostClientError(
                    f"Cost sync failed: {payload.get('message', 'unknown')}",
                    status_code=response.status_code,
                    code=str(payload.get("code", "COST_SYNC_FAILED")),
                )
            return payload["data"]
        except (KeyError, TypeError, ValueError) as exc:
            raise JavaCostClientError(
                "Cost sync response was invalid",
                status_code=response.status_code,
            ) from exc

    def _build_headers(self) -> dict[str, str]:
        headers = build_trace_headers()
        if self.settings is not None:
            headers.update(build_java_internal_headers(self.settings))
        if "X-Trace-Id" not in headers:
            # Background tasks (e.g. cost sync) have no request trace context;
            # Java requires X-Trace-Id (8-128 chars) for internal auth.
            headers["X-Trace-Id"] = f"sync-{uuid4().hex[:24]}"
        return headers
