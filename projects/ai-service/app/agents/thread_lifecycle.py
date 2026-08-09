from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4


TicketAgentThreadStatus = Literal[
    "active",
    "waiting_confirmation",
    "completed",
    "closed",
]
TicketAgentThreadResumeReason = Literal[
    "ok",
    "actor_mismatch",
    "expired",
    "completed",
    "closed",
]
TicketAgentThreadResumeAction = Literal[
    "resume_existing",
    "start_new",
    "reject",
]

TICKET_AGENT_THREAD_ID_PREFIX = "ticket-thread-"
TICKET_AGENT_THREAD_ID_MAX_LENGTH = 255
TICKET_AGENT_THREAD_TTL_SECONDS = 24 * 60 * 60
TICKET_AGENT_THREAD_CONFIRMATION_TTL_SECONDS = 30 * 60
TICKET_AGENT_THREAD_ID_EMPTY_MESSAGE = "thread_id 不能为空。"
TICKET_AGENT_THREAD_ID_TOO_LONG_MESSAGE = "thread_id 不能超过 255 个字符。"
TICKET_AGENT_THREAD_ID_UNSAFE_MESSAGE = (
    "thread_id 只能包含字母、数字、下划线、短横线和点，并且必须以字母或数字开头。"
)

_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def normalize_ticket_agent_thread_id(thread_id: str) -> str:
    normalized_thread_id = thread_id.strip()
    if not normalized_thread_id:
        raise ValueError(TICKET_AGENT_THREAD_ID_EMPTY_MESSAGE)
    if len(normalized_thread_id) > TICKET_AGENT_THREAD_ID_MAX_LENGTH:
        raise ValueError(TICKET_AGENT_THREAD_ID_TOO_LONG_MESSAGE)
    if not _THREAD_ID_PATTERN.fullmatch(normalized_thread_id):
        raise ValueError(TICKET_AGENT_THREAD_ID_UNSAFE_MESSAGE)
    return normalized_thread_id


def generate_ticket_agent_thread_id() -> str:
    return normalize_ticket_agent_thread_id(
        f"{TICKET_AGENT_THREAD_ID_PREFIX}{uuid4().hex}"
    )


@dataclass(frozen=True)
class TicketAgentThreadBinding:
    thread_id: str
    actor_id: str
    status: TicketAgentThreadStatus
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    session_id: str | None = None
    ticket_id: str | None = None
    pending_confirmation_id: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "actor_id": self.actor_id,
            "session_id": self.session_id,
            "ticket_id": self.ticket_id,
            "pending_confirmation_id": self.pending_confirmation_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass(frozen=True)
class TicketAgentThreadResumeDecision:
    allowed: bool
    reason: TicketAgentThreadResumeReason
    action: TicketAgentThreadResumeAction


def create_ticket_agent_thread_binding(
    *,
    actor_id: str,
    session_id: str | None = None,
    thread_id: str | None = None,
    now: datetime | None = None,
    ttl_seconds: int = TICKET_AGENT_THREAD_TTL_SECONDS,
) -> TicketAgentThreadBinding:
    current_time = ensure_utc_datetime(now or utc_now())
    selected_thread_id = (
        normalize_ticket_agent_thread_id(thread_id)
        if thread_id is not None
        else generate_ticket_agent_thread_id()
    )
    return TicketAgentThreadBinding(
        thread_id=selected_thread_id,
        actor_id=actor_id,
        session_id=session_id,
        status="active",
        created_at=current_time,
        updated_at=current_time,
        expires_at=current_time + timedelta(seconds=ttl_seconds),
    )


def mark_ticket_agent_thread_waiting_confirmation(
    binding: TicketAgentThreadBinding,
    *,
    pending_confirmation_id: str,
    now: datetime | None = None,
    ttl_seconds: int = TICKET_AGENT_THREAD_CONFIRMATION_TTL_SECONDS,
) -> TicketAgentThreadBinding:
    current_time = ensure_utc_datetime(now or utc_now())
    return replace(
        binding,
        status="waiting_confirmation",
        pending_confirmation_id=pending_confirmation_id,
        updated_at=current_time,
        expires_at=current_time + timedelta(seconds=ttl_seconds),
    )


def complete_ticket_agent_thread(
    binding: TicketAgentThreadBinding,
    *,
    ticket_id: str | None = None,
    now: datetime | None = None,
) -> TicketAgentThreadBinding:
    current_time = ensure_utc_datetime(now or utc_now())
    return replace(
        binding,
        status="completed",
        ticket_id=ticket_id,
        updated_at=current_time,
        expires_at=None,
        pending_confirmation_id=None,
    )


def close_ticket_agent_thread(
    binding: TicketAgentThreadBinding,
    *,
    now: datetime | None = None,
) -> TicketAgentThreadBinding:
    current_time = ensure_utc_datetime(now or utc_now())
    return replace(
        binding,
        status="closed",
        updated_at=current_time,
        expires_at=None,
        pending_confirmation_id=None,
    )


def is_ticket_agent_thread_expired(
    binding: TicketAgentThreadBinding,
    *,
    now: datetime | None = None,
) -> bool:
    if binding.expires_at is None:
        return False
    current_time = ensure_utc_datetime(now or utc_now())
    return current_time >= ensure_utc_datetime(binding.expires_at)


def evaluate_ticket_agent_thread_resume(
    binding: TicketAgentThreadBinding,
    *,
    actor_id: str,
    now: datetime | None = None,
) -> TicketAgentThreadResumeDecision:
    if binding.actor_id != actor_id:
        return TicketAgentThreadResumeDecision(
            allowed=False,
            reason="actor_mismatch",
            action="reject",
        )
    if binding.status == "completed":
        return TicketAgentThreadResumeDecision(
            allowed=False,
            reason="completed",
            action="start_new",
        )
    if binding.status == "closed":
        return TicketAgentThreadResumeDecision(
            allowed=False,
            reason="closed",
            action="start_new",
        )
    if is_ticket_agent_thread_expired(binding, now=now):
        return TicketAgentThreadResumeDecision(
            allowed=False,
            reason="expired",
            action="start_new",
        )
    return TicketAgentThreadResumeDecision(
        allowed=True,
        reason="ok",
        action="resume_existing",
    )
