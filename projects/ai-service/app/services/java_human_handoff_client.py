"""Internal adapter for the human handoff queue owned by Java (MySQL)."""

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


class JavaHumanHandoffClientError(Exception):
    """Human handoff enqueue with Java failed."""

    def __init__(self, message: str, *, status_code: int | None = None, code: str = "HANDOFF_SYNC_FAILED") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class JavaHumanHandoffClient:
    """Internal adapter for the AI-to-human handoff queue (Java/MySQL)."""

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
    def from_settings(cls, settings: Settings) -> "JavaHumanHandoffClient":
        return cls(
            base_url=settings.resolved_java_business_service_base_url,
            timeout_seconds=settings.resolved_java_business_service_timeout_seconds,
            settings=settings,
        )

    def create_handoff(
        self,
        *,
        conversation_id: str,
        user_id: str,
        tenant_id: str,
        reason: str,
        related_order_id: str | None = None,
        emotion: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "reason": reason,
        }
        if related_order_id:
            payload["related_order_id"] = related_order_id
        if emotion:
            payload["emotion"] = emotion
        self._request("post", "/internal/ai-human-handoffs", json=payload)

    def get_active_handoff(self, conversation_id: str) -> dict[str, Any] | None:
        """Return the active (not closed) handoff for a conversation, or None."""
        data = self._request(
            "GET",
            "/internal/ai-human-handoffs/active-by-conversation",
            params={"conversationId": conversation_id},
        )
        return data or None

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        started_at = perf_counter()
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.request(
                    method, path, json=json, params=params, headers=self._build_headers()
                )
        except httpx.RequestError as exc:
            logger.warning(
                "java_handoff_request_failed path=%s error_type=%s elapsed_ms=%.2f",
                path,
                type(exc).__name__,
                (perf_counter() - started_at) * 1000,
            )
            raise JavaHumanHandoffClientError(
                f"Handoff request to {path} failed: {type(exc).__name__}",
                status_code=None,
            ) from exc

        if response.status_code != 200:
            logger.warning(
                "java_handoff_request_rejected path=%s status_code=%s elapsed_ms=%.2f",
                path,
                response.status_code,
                (perf_counter() - started_at) * 1000,
            )
            raise JavaHumanHandoffClientError(
                f"Handoff request to {path} rejected with {response.status_code}",
                status_code=response.status_code,
            )
        try:
            payload = response.json()
            if not payload.get("success"):
                raise JavaHumanHandoffClientError(
                    f"Handoff request failed: {payload.get('message', 'unknown')}",
                    status_code=response.status_code,
                    code=str(payload.get("code", "HANDOFF_SYNC_FAILED")),
                )
            return payload["data"]
        except (KeyError, TypeError, ValueError) as exc:
            raise JavaHumanHandoffClientError(
                "Handoff response was invalid",
                status_code=response.status_code,
            ) from exc

    def _build_headers(self) -> dict[str, str]:
        headers = build_trace_headers()
        if self.settings is not None:
            headers.update(build_java_internal_headers(self.settings))
        if "X-Trace-Id" not in headers:
            # Background tasks (e.g. handoff enqueue) have no request trace context;
            # Java requires X-Trace-Id (8-128 chars) for internal auth.
            headers["X-Trace-Id"] = f"sync-{uuid4().hex[:24]}"
        return headers
