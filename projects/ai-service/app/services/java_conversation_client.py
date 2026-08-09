import logging
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

import httpx

from app.core.business_context import build_java_internal_headers
from app.core.config import Settings
from app.core.trace import build_trace_headers

logger = logging.getLogger(__name__)


class JavaConversationClientError(Exception):
    """Conversation persistence sync with Java failed."""

    def __init__(self, message: str, *, status_code: int | None = None, code: str = "CONVERSATION_SYNC_FAILED") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class JavaConversation:
    conversation_id: str
    user_id: str
    title: str
    conversation_status: str
    updated_at: str


@dataclass(frozen=True)
class JavaConversationMessage:
    message_id: str
    conversation_id: str
    sender_type: str
    content: str
    trace_id: str
    created_at: str


class JavaConversationClient:
    """Internal adapter for durable conversation transcripts owned by Java (MySQL)."""

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
    def from_settings(cls, settings: Settings) -> "JavaConversationClient":
        return cls(
            base_url=settings.resolved_java_business_service_base_url,
            timeout_seconds=settings.resolved_java_business_service_timeout_seconds,
            settings=settings,
        )

    def upsert_conversation(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        user_id: str,
        title: str,
        conversation_status: str,
    ) -> None:
        self._request(
            "post",
            "/internal/ai-conversations",
            json={
                "conversation_id": conversation_id,
                "user_id": user_id,
                "title": title,
                "conversation_status": conversation_status,
            },
        )

    def batch_write_messages(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        messages: list[dict[str, str]],
    ) -> int:
        data = self._request(
            "post",
            "/internal/ai-conversations/messages",
            json={
                "conversation_id": conversation_id,
                "messages": messages,
            },
        )
        return int(data["inserted"])

    def list_conversations(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> list[JavaConversation]:
        data = self._request(
            "get",
            "/internal/ai-conversations",
            params={"limit": limit},
        )
        return [
            JavaConversation(
                conversation_id=str(item["conversation_id"]),
                user_id=str(item["user_id"]),
                title=str(item["title"]),
                conversation_status=str(item["conversation_status"]),
                updated_at=str(item["updated_at"]),
            )
            for item in data
        ]

    def get_messages(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
    ) -> list[JavaConversationMessage]:
        data = self._request(
            "get",
            f"/internal/ai-conversations/{conversation_id}/messages",
        )
        return [
            JavaConversationMessage(
                message_id=str(item["message_id"]),
                conversation_id=str(item["conversation_id"]),
                sender_type=str(item["sender_type"]),
                content=str(item["content"]),
                trace_id=str(item["trace_id"]),
                created_at=str(item["created_at"]),
            )
            for item in data
        ]

    def request_cleanup(self, *, older_than_days: int = 30) -> int:
        data = self._request(
            "post",
            "/internal/ai-conversations/cleanup",
            params={"olderThanDays": older_than_days},
        )
        return int(data["deleted"])

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        params: dict[str, object] | None = None,
    ) -> object:
        started_at = perf_counter()
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.request(method, path, json=json, params=params, headers=self._build_headers())
        except httpx.RequestError as exc:
            logger.warning(
                "java_conversation_request_failed path=%s error_type=%s elapsed_ms=%.2f",
                path,
                type(exc).__name__,
                (perf_counter() - started_at) * 1000,
            )
            raise JavaConversationClientError(
                f"Conversation sync request to {path} failed: {type(exc).__name__}",
                status_code=None,
            ) from exc

        if response.status_code != 200:
            logger.warning(
                "java_conversation_request_rejected path=%s status_code=%s elapsed_ms=%.2f",
                path,
                response.status_code,
                (perf_counter() - started_at) * 1000,
            )
            raise JavaConversationClientError(
                f"Conversation sync request to {path} rejected with {response.status_code}",
                status_code=response.status_code,
            )
        try:
            payload = response.json()
            if not payload.get("success"):
                raise JavaConversationClientError(
                    f"Conversation sync failed: {payload.get('message', 'unknown')}",
                    status_code=response.status_code,
                    code=str(payload.get("code", "CONVERSATION_SYNC_FAILED")),
                )
            return payload["data"]
        except (KeyError, TypeError, ValueError) as exc:
            raise JavaConversationClientError(
                "Conversation sync response was invalid",
                status_code=response.status_code,
            ) from exc

    def _build_headers(self) -> dict[str, str]:
        headers = build_trace_headers()
        if self.settings is not None:
            headers.update(build_java_internal_headers(self.settings))
        if "X-Trace-Id" not in headers:
            # Background tasks (e.g. conversation sync) have no request trace
            # context; Java requires X-Trace-Id (8-128 chars) for internal auth.
            headers["X-Trace-Id"] = f"sync-{uuid4().hex[:24]}"
        return headers
