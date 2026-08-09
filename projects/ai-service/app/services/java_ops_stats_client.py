"""Internal adapter for Java ops stats aggregation (MySQL)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.business_context import build_java_internal_headers
from app.core.config import Settings
from app.core.trace import build_trace_headers

logger = logging.getLogger(__name__)


class JavaOpsStatsClientError(Exception):
    """Raised when the Java ops-stats service rejects or is unreachable."""

    def __init__(self, message: str, *, status_code: int | None = None, code: str = "OPS_STATS_UNAVAILABLE") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class JavaOpsStatsClient:
    """Fetch aggregated ops stats from the Java business service (MySQL)."""

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
    def from_settings(cls, settings: Settings) -> "JavaOpsStatsClient":
        return cls(
            base_url=settings.resolved_java_business_service_base_url,
            timeout_seconds=settings.resolved_java_business_service_timeout_seconds,
            settings=settings,
        )

    def get_summary(self, days: int) -> dict[str, Any]:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.get(
                    "/internal/ai-ops-stats/summary",
                    params={"days": days},
                    headers=self._build_headers(),
                )
        except httpx.RequestError as exc:
            logger.warning("java_ops_stats_request_failed error_type=%s", type(exc).__name__)
            raise JavaOpsStatsClientError(f"Ops stats request failed: {type(exc).__name__}") from exc

        if response.status_code != 200:
            logger.warning(
                "java_ops_stats_request_rejected status_code=%s",
                response.status_code,
            )
            raise JavaOpsStatsClientError(
                f"Ops stats request rejected with {response.status_code}",
                status_code=response.status_code,
            )
        payload = response.json()
        if not payload.get("success"):
            raise JavaOpsStatsClientError(
                f"Ops stats failed: {payload.get('message', 'unknown')}",
                status_code=response.status_code,
                code=str(payload.get("code", "OPS_STATS_UNAVAILABLE")),
            )
        return payload["data"]

    def _build_headers(self) -> dict[str, str]:
        headers = build_trace_headers()
        if self.settings is not None:
            headers.update(build_java_internal_headers(self.settings))
        return headers
