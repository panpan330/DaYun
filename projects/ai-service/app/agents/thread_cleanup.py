from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from app.agents.thread_lifecycle import (
    TicketAgentThreadBinding,
    TicketAgentThreadStatus,
    ensure_utc_datetime,
    is_ticket_agent_thread_expired,
    utc_now,
)


TicketAgentThreadCleanupAction = Literal["keep", "expire", "archive"]
TicketAgentCheckpointCleanupAction = Literal["keep", "delete_after_archive"]
TicketAgentThreadCleanupReason = Literal[
    "active_not_expired",
    "waiting_confirmation_not_expired",
    "expired_active_grace_period",
    "expired_confirmation_grace_period",
    "expired_cleanup_due",
    "completed_retention_active",
    "completed_retention_elapsed",
    "closed_retention_active",
    "closed_retention_elapsed",
]

TICKET_AGENT_EXPIRED_THREAD_GRACE_SECONDS = 24 * 60 * 60
TICKET_AGENT_COMPLETED_THREAD_RETENTION_SECONDS = 7 * 24 * 60 * 60
TICKET_AGENT_CLOSED_THREAD_RETENTION_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True)
class TicketAgentThreadCleanupPolicy:
    expired_thread_grace_seconds: int = TICKET_AGENT_EXPIRED_THREAD_GRACE_SECONDS
    completed_thread_retention_seconds: int = (
        TICKET_AGENT_COMPLETED_THREAD_RETENTION_SECONDS
    )
    closed_thread_retention_seconds: int = TICKET_AGENT_CLOSED_THREAD_RETENTION_SECONDS


@dataclass(frozen=True)
class TicketAgentThreadCleanupDecision:
    thread_id: str
    status: TicketAgentThreadStatus
    action: TicketAgentThreadCleanupAction
    reason: TicketAgentThreadCleanupReason
    checkpoint_action: TicketAgentCheckpointCleanupAction
    archive_required: bool
    eligible_at: datetime | None = None


@dataclass(frozen=True)
class TicketAgentThreadCleanupPlan:
    decisions: list[TicketAgentThreadCleanupDecision] = field(default_factory=list)

    def count_by_action(self) -> dict[TicketAgentThreadCleanupAction, int]:
        counts = Counter(decision.action for decision in self.decisions)
        return {
            "keep": counts["keep"],
            "expire": counts["expire"],
            "archive": counts["archive"],
        }

    def archive_required_thread_ids(self) -> list[str]:
        return [
            decision.thread_id
            for decision in self.decisions
            if decision.archive_required
        ]

    def checkpoint_delete_thread_ids(self) -> list[str]:
        return [
            decision.thread_id
            for decision in self.decisions
            if decision.checkpoint_action == "delete_after_archive"
        ]


def _deadline_from(
    started_at: datetime,
    *,
    retention_seconds: int,
) -> datetime:
    return ensure_utc_datetime(started_at) + timedelta(seconds=retention_seconds)


def _is_deadline_elapsed(deadline: datetime, *, now: datetime) -> bool:
    return ensure_utc_datetime(now) >= ensure_utc_datetime(deadline)


def evaluate_ticket_agent_thread_cleanup(
    binding: TicketAgentThreadBinding,
    *,
    policy: TicketAgentThreadCleanupPolicy | None = None,
    now: datetime | None = None,
) -> TicketAgentThreadCleanupDecision:
    selected_policy = policy or TicketAgentThreadCleanupPolicy()
    current_time = ensure_utc_datetime(now or utc_now())

    if binding.status in ("active", "waiting_confirmation"):
        if not is_ticket_agent_thread_expired(binding, now=current_time):
            reason: TicketAgentThreadCleanupReason = (
                "waiting_confirmation_not_expired"
                if binding.status == "waiting_confirmation"
                else "active_not_expired"
            )
            return TicketAgentThreadCleanupDecision(
                thread_id=binding.thread_id,
                status=binding.status,
                action="keep",
                reason=reason,
                checkpoint_action="keep",
                archive_required=False,
                eligible_at=binding.expires_at,
            )

        expires_at = ensure_utc_datetime(binding.expires_at or binding.updated_at)
        cleanup_at = _deadline_from(
            expires_at,
            retention_seconds=selected_policy.expired_thread_grace_seconds,
        )
        if not _is_deadline_elapsed(cleanup_at, now=current_time):
            reason = (
                "expired_confirmation_grace_period"
                if binding.status == "waiting_confirmation"
                else "expired_active_grace_period"
            )
            return TicketAgentThreadCleanupDecision(
                thread_id=binding.thread_id,
                status=binding.status,
                action="expire",
                reason=reason,
                checkpoint_action="keep",
                archive_required=binding.status == "waiting_confirmation",
                eligible_at=cleanup_at,
            )

        return TicketAgentThreadCleanupDecision(
            thread_id=binding.thread_id,
            status=binding.status,
            action="archive",
            reason="expired_cleanup_due",
            checkpoint_action="delete_after_archive",
            archive_required=True,
            eligible_at=cleanup_at,
        )

    if binding.status == "completed":
        completed_cleanup_at = _deadline_from(
            binding.updated_at,
            retention_seconds=selected_policy.completed_thread_retention_seconds,
        )
        if not _is_deadline_elapsed(completed_cleanup_at, now=current_time):
            return TicketAgentThreadCleanupDecision(
                thread_id=binding.thread_id,
                status=binding.status,
                action="keep",
                reason="completed_retention_active",
                checkpoint_action="keep",
                archive_required=False,
                eligible_at=completed_cleanup_at,
            )
        return TicketAgentThreadCleanupDecision(
            thread_id=binding.thread_id,
            status=binding.status,
            action="archive",
            reason="completed_retention_elapsed",
            checkpoint_action="delete_after_archive",
            archive_required=True,
            eligible_at=completed_cleanup_at,
        )

    closed_cleanup_at = _deadline_from(
        binding.updated_at,
        retention_seconds=selected_policy.closed_thread_retention_seconds,
    )
    if not _is_deadline_elapsed(closed_cleanup_at, now=current_time):
        return TicketAgentThreadCleanupDecision(
            thread_id=binding.thread_id,
            status=binding.status,
            action="keep",
            reason="closed_retention_active",
            checkpoint_action="keep",
            archive_required=False,
            eligible_at=closed_cleanup_at,
        )
    return TicketAgentThreadCleanupDecision(
        thread_id=binding.thread_id,
        status=binding.status,
        action="archive",
        reason="closed_retention_elapsed",
        checkpoint_action="delete_after_archive",
        archive_required=True,
        eligible_at=closed_cleanup_at,
    )


def build_ticket_agent_thread_cleanup_plan(
    bindings: list[TicketAgentThreadBinding],
    *,
    policy: TicketAgentThreadCleanupPolicy | None = None,
    now: datetime | None = None,
) -> TicketAgentThreadCleanupPlan:
    current_time = ensure_utc_datetime(now or utc_now())
    return TicketAgentThreadCleanupPlan(
        decisions=[
            evaluate_ticket_agent_thread_cleanup(
                binding,
                policy=policy,
                now=current_time,
            )
            for binding in bindings
        ]
    )
