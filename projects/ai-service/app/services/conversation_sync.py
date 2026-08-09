import logging
import threading
import time
from types import SimpleNamespace

from redis import Redis

from app.services.java_conversation_client import JavaConversationClient

logger = logging.getLogger(__name__)

PENDING_SYNC_KEY = "ai-service:pending-sync"
CONVERSATION_STATUS_ACTIVE = "active"


class PendingConversationSync:
    """Background batch sync of conversation transcripts from Redis to Java (MySQL).

    Writes are append-only and idempotent on the Java side (message_id unique
    key), so a failed marker is simply retried on the next cycle.
    """

    def __init__(
        self,
        *,
        store: object,
        client: JavaConversationClient,
        redis: Redis,
        interval_seconds: float = 30.0,
    ) -> None:
        self.store = store
        self.client = client
        self.redis = redis
        self.interval_seconds = interval_seconds

    def mark_pending(self, *, tenant_id: str, user_id: str, conversation_id: str) -> None:
        marker = _marker(tenant_id, user_id, conversation_id)
        try:
            self.redis.sadd(PENDING_SYNC_KEY, marker)
        except Exception:  # pragma: no cover - redis outage must not break chat
            logger.warning("conversation_sync_mark_failed marker=%s", marker)

    def sync_once(self) -> int:
        try:
            markers = self.redis.smembers(PENDING_SYNC_KEY)
        except Exception:
            logger.warning("conversation_sync_read_markers_failed", exc_info=True)
            return 0
        if not markers:
            return 0

        synced = 0
        for marker in markers:
            if self._sync_one(marker):
                synced += 1
        return synced

    def _sync_one(self, marker: str) -> bool:
        if isinstance(marker, bytes):  # redis-py returns bytes on real connections
            marker = marker.decode("utf-8")
        tenant_id, user_id, conversation_id = marker.split(":", 2)
        actor = SimpleNamespace(tenant_id=tenant_id, user_id=user_id)
        try:
            conversation = self.store.get(actor=actor, conversation_id=conversation_id)
            if conversation is None:
                # Transcript already expired in Redis; nothing durable to write.
                self.redis.srem(PENDING_SYNC_KEY, marker)
                return False
            self.client.upsert_conversation(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                user_id=user_id,
                title=conversation.title,
                conversation_status=CONVERSATION_STATUS_ACTIVE,
            )
            messages = [
                {
                    "message_id": message.id,
                    "sender_type": message.role,
                    "content": message.content,
                    "trace_id": message.trace_id or "sync",
                }
                for message in conversation.messages
            ]
            if messages:
                self.client.batch_write_messages(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    messages=messages,
                )
            self.redis.srem(PENDING_SYNC_KEY, marker)
            return True
        except Exception as exc:
            # Keep the marker so the next cycle retries; Java-side idempotency
            # makes a partially written conversation safe to re-apply.
            logger.warning(
                "conversation_sync_item_failed marker=%s error_type=%s",
                marker,
                type(exc).__name__,
            )
            return False

    def run_forever(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            started = time.monotonic()
            self.sync_once()
            elapsed = time.monotonic() - started
            stop_event.wait(max(0.0, self.interval_seconds - elapsed))

    def close(self) -> None:
        try:
            self.redis.close()
        except Exception:  # pragma: no cover
            logger.debug("conversation_sync_redis_close_failed", exc_info=True)


def _marker(tenant_id: str, user_id: str, conversation_id: str) -> str:
    return f"{tenant_id}:{user_id}:{conversation_id}"
