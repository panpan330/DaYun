import hashlib
import json
import logging
from datetime import datetime, timezone
from time import time_ns
from typing import Protocol
from uuid import uuid4
from threading import Lock

from redis import Redis
from redis.exceptions import RedisError
from pydantic import ValidationError

from app.core.ai_security_boundary import redact_sensitive_text
from app.core.config import Settings
from app.schemas.console_agent import (
    ConsoleAgentConversation,
    ConsoleAgentConversationMessage,
    ConsoleAgentConversationSummary,
    ConsoleAgentResponse,
)
from app.services.java_conversation_client import JavaConversationClient


logger = logging.getLogger(__name__)
MAX_CONVERSATION_MESSAGES = 100


class ConversationActor(Protocol):
    user_id: str
    tenant_id: str


class ConsoleAgentConversationStore:
    """Redis store for the client-visible transcript, separate from LangGraph checkpoints."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: Redis | None = None,
        java_client: JavaConversationClient | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or Redis.from_url(
            settings.resolved_agent_redis_url,
            decode_responses=True,
        )
        self._java_client = java_client
        self._score_lock = Lock()
        self._last_index_score = 0.0

    def close(self) -> None:
        self._client.close()

    def append_exchange(
        self,
        *,
        actor: ConversationActor,
        conversation_id: str,
        user_message: str,
        response: ConsoleAgentResponse,
    ) -> None:
        conversation = self.get(actor=actor, conversation_id=conversation_id)
        now = datetime.now(timezone.utc)
        messages = list(conversation.messages) if conversation is not None else []
        messages.extend(
            [
                ConsoleAgentConversationMessage(
                    id=uuid4().hex,
                    role="user",
                    content=redact_sensitive_text(user_message),
                    created_at=now,
                ),
                ConsoleAgentConversationMessage(
                    id=uuid4().hex,
                    role="assistant",
                    content=redact_sensitive_text(response.reply),
                    created_at=now,
                    trace_id=response.trace_id,
                    route=response.route,
                    citations=response.citations,
                    suggestions=[redact_sensitive_text(item) for item in response.suggestions],
                    pending_ticket_confirmation=response.pending_ticket_confirmation,
                    created_ticket=response.created_ticket,
                    human_handoff=response.human_handoff,
                ),
            ]
        )
        stored = ConsoleAgentConversation(
            conversation_id=conversation_id,
            title=conversation.title if conversation is not None else _build_title(user_message),
            updated_at=now,
            messages=messages[-MAX_CONVERSATION_MESSAGES:],
        )
        self._save(actor=actor, conversation=stored)

    def append_human_reply(
        self,
        *,
        actor: ConversationActor,
        conversation_id: str,
        content: str,
    ) -> None:
        """Append a single human-agent reply to the conversation."""
        conversation = self.get(actor=actor, conversation_id=conversation_id)
        now = datetime.now(timezone.utc)
        messages = list(conversation.messages) if conversation is not None else []
        messages.append(
            ConsoleAgentConversationMessage(
                id=uuid4().hex,
                role="human_agent",
                content=redact_sensitive_text(content),
                created_at=now,
            )
        )
        stored = ConsoleAgentConversation(
            conversation_id=conversation_id,
            title=conversation.title if conversation is not None else "人工回复",
            updated_at=now,
            messages=messages[-MAX_CONVERSATION_MESSAGES:],
        )
        self._save(actor=actor, conversation=stored)

    def list_recent(
        self,
        *,
        actor: ConversationActor,
        limit: int,
    ) -> list[ConsoleAgentConversationSummary]:
        conversation_ids = self._client.zrevrange(self._index_key(actor), 0, limit - 1)
        if not conversation_ids:
            return self._backfill_index(actor=actor, limit=limit)
        summaries: list[ConsoleAgentConversationSummary] = []
        stale_ids: list[str] = []
        for conversation_id in conversation_ids:
            conversation = self.get(actor=actor, conversation_id=conversation_id)
            if conversation is None:
                stale_ids.append(conversation_id)
                continue
            summaries.append(
                ConsoleAgentConversationSummary(
                    conversation_id=conversation.conversation_id,
                    title=conversation.title,
                    updated_at=conversation.updated_at,
                )
            )
        if stale_ids:
            self._client.zrem(self._index_key(actor), *stale_ids)
        return summaries

    def set_assistant_feedback(
        self,
        *,
        actor: ConversationActor,
        conversation_id: str,
        trace_id: str,
        rating: str,
        reason: str | None,
    ) -> bool:
        conversation = self.get(actor=actor, conversation_id=conversation_id)
        if conversation is None:
            return False
        messages = list(conversation.messages)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role == "assistant" and message.trace_id == trace_id:
                messages[index] = message.model_copy(
                    update={"feedback_rating": rating, "feedback_reason": reason}
                )
                self._save(
                    actor=actor,
                    conversation=conversation.model_copy(
                        update={"messages": messages, "updated_at": datetime.now(timezone.utc)}
                    ),
                )
                return True
        return False

    def get(
        self,
        *,
        actor: ConversationActor,
        conversation_id: str,
    ) -> ConsoleAgentConversation | None:
        raw = self._client.get(self._conversation_key(actor, conversation_id))
        if raw is not None:
            try:
                return ConsoleAgentConversation.model_validate_json(raw)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("console_agent_conversation_record_invalid error_type=%s", type(exc).__name__)
                return None
        return self._backfill_conversation(actor=actor, conversation_id=conversation_id)

    def _backfill_conversation(
        self,
        *,
        actor: ConversationActor,
        conversation_id: str,
        title: str | None = None,
    ) -> ConsoleAgentConversation | None:
        if self._java_client is None:
            return None
        try:
            messages = self._java_client.get_messages(
                tenant_id=actor.tenant_id,
                conversation_id=conversation_id,
            )
        except Exception:
            logger.warning(
                "console_agent_conversation_backfill_failed conversation_id=%s error_type=%s",
                conversation_id,
                "client_error",
            )
            return None
        if not messages:
            return None
        now = datetime.now(timezone.utc)
        converted = [
            ConsoleAgentConversationMessage(
                id=message.message_id,
                role=message.sender_type,
                content=message.content,
                created_at=_parse_java_time(message.created_at, now),
                trace_id=message.trace_id,
            )
            for message in messages
        ]
        resolved_title = title or _build_title(next((m.content for m in converted if m.role == "user"), "历史会话"))
        conversation = ConsoleAgentConversation(
            conversation_id=conversation_id,
            title=resolved_title,
            updated_at=converted[-1].created_at if converted else now,
            messages=converted[-MAX_CONVERSATION_MESSAGES:],
        )
        self._save(actor=actor, conversation=conversation)
        return conversation

    def _backfill_index(
        self,
        *,
        actor: ConversationActor,
        limit: int,
    ) -> list[ConsoleAgentConversationSummary]:
        if self._java_client is None:
            return []
        try:
            conversations = self._java_client.list_conversations(
                tenant_id=actor.tenant_id,
                user_id=actor.user_id,
                limit=limit,
            )
        except Exception:
            logger.warning(
                "console_agent_conversation_index_backfill_failed actor_id=%s",
                actor.user_id,
            )
            return []
        if not conversations:
            return []
        # Restore oldest first so _save()'s ascending index scores keep newest on top.
        restored: list[ConsoleAgentConversationSummary] = []
        for conversation in reversed(conversations):
            restored_conversation = self._backfill_conversation(
                actor=actor,
                conversation_id=conversation.conversation_id,
                title=conversation.title,
            )
            if restored_conversation is None:
                continue
            restored.append(
                ConsoleAgentConversationSummary(
                    conversation_id=restored_conversation.conversation_id,
                    title=restored_conversation.title,
                    updated_at=restored_conversation.updated_at,
                )
            )
        return list(reversed(restored))

    def _save(self, *, actor: ConversationActor, conversation: ConsoleAgentConversation) -> None:
        ttl_seconds = self._settings.agent_checkpoint_ttl_minutes * 60
        conversation_key = self._conversation_key(actor, conversation.conversation_id)
        index_key = self._index_key(actor)
        payload = conversation.model_dump_json()
        with self._client.pipeline(transaction=True) as pipeline:
            pipeline.set(conversation_key, payload, ex=ttl_seconds)
            pipeline.zadd(index_key, {conversation.conversation_id: self._next_index_score()})
            pipeline.expire(index_key, ttl_seconds)
            pipeline.execute()

    def _conversation_key(self, actor: ConversationActor, conversation_id: str) -> str:
        return f"{self._prefix}:conversation:{self._digest(actor, conversation_id)}"

    def _index_key(self, actor: ConversationActor) -> str:
        return f"{self._prefix}:conversation-index:{self._digest(actor)}"

    @property
    def _prefix(self) -> str:
        return self._settings.resolved_agent_checkpoint_key_prefix

    @staticmethod
    def _digest(actor: ConversationActor, conversation_id: str = "") -> str:
        payload = f"{actor.tenant_id}\x1f{actor.user_id}\x1f{conversation_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _next_index_score(self) -> float:
        with self._score_lock:
            candidate = time_ns() / 1_000_000_000
            self._last_index_score = max(candidate, self._last_index_score + 0.001)
            return self._last_index_score


def _build_title(user_message: str) -> str:
    normalized = " ".join(redact_sensitive_text(user_message).split())
    return (normalized[:117] + "...") if len(normalized) > 120 else normalized or "新会话"


def _parse_java_time(value: str, fallback: datetime) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return fallback
