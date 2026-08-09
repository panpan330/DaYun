from datetime import datetime, timedelta, timezone

from app.agents.thread_cleanup import (
    TicketAgentThreadCleanupPolicy,
    build_ticket_agent_thread_cleanup_plan,
    evaluate_ticket_agent_thread_cleanup,
)
from app.agents.thread_lifecycle import (
    close_ticket_agent_thread,
    complete_ticket_agent_thread,
    create_ticket_agent_thread_binding,
    mark_ticket_agent_thread_waiting_confirmation,
)


def test_cleanup_keeps_active_thread_before_expiration() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-active-cleanup-001",
        now=now,
        ttl_seconds=3600,
    )

    decision = evaluate_ticket_agent_thread_cleanup(
        binding,
        now=now + timedelta(minutes=10),
    )

    assert decision.action == "keep"
    assert decision.reason == "active_not_expired"
    assert decision.checkpoint_action == "keep"
    assert decision.archive_required is False
    assert decision.eligible_at == now + timedelta(seconds=3600)


def test_cleanup_expires_waiting_confirmation_during_grace_period() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-confirm-cleanup-001",
        now=now,
    )
    waiting = mark_ticket_agent_thread_waiting_confirmation(
        binding,
        pending_confirmation_id="confirmation-001",
        now=now,
        ttl_seconds=60,
    )

    decision = evaluate_ticket_agent_thread_cleanup(
        waiting,
        policy=TicketAgentThreadCleanupPolicy(expired_thread_grace_seconds=3600),
        now=now + timedelta(minutes=10),
    )

    assert decision.action == "expire"
    assert decision.reason == "expired_confirmation_grace_period"
    assert decision.checkpoint_action == "keep"
    assert decision.archive_required is True
    assert decision.eligible_at == now + timedelta(seconds=3660)


def test_cleanup_archives_expired_thread_after_grace_period() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-expired-cleanup-001",
        now=now,
        ttl_seconds=60,
    )

    decision = evaluate_ticket_agent_thread_cleanup(
        binding,
        policy=TicketAgentThreadCleanupPolicy(expired_thread_grace_seconds=3600),
        now=now + timedelta(hours=2),
    )

    assert decision.action == "archive"
    assert decision.reason == "expired_cleanup_due"
    assert decision.checkpoint_action == "delete_after_archive"
    assert decision.archive_required is True


def test_cleanup_keeps_completed_thread_during_retention() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-completed-retain-001",
        now=now,
    )
    completed = complete_ticket_agent_thread(
        binding,
        ticket_id="T1001",
        now=now + timedelta(minutes=5),
    )

    decision = evaluate_ticket_agent_thread_cleanup(
        completed,
        policy=TicketAgentThreadCleanupPolicy(
            completed_thread_retention_seconds=3600,
        ),
        now=now + timedelta(minutes=30),
    )

    assert decision.action == "keep"
    assert decision.reason == "completed_retention_active"
    assert decision.checkpoint_action == "keep"
    assert decision.eligible_at == now + timedelta(minutes=65)


def test_cleanup_archives_completed_thread_after_retention() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-completed-archive-001",
        now=now,
    )
    completed = complete_ticket_agent_thread(
        binding,
        ticket_id="T1001",
        now=now + timedelta(minutes=5),
    )

    decision = evaluate_ticket_agent_thread_cleanup(
        completed,
        policy=TicketAgentThreadCleanupPolicy(
            completed_thread_retention_seconds=3600,
        ),
        now=now + timedelta(hours=2),
    )

    assert decision.action == "archive"
    assert decision.reason == "completed_retention_elapsed"
    assert decision.checkpoint_action == "delete_after_archive"
    assert decision.archive_required is True


def test_cleanup_archives_closed_thread_after_retention() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-closed-archive-001",
        now=now,
    )
    closed = close_ticket_agent_thread(
        binding,
        now=now + timedelta(minutes=5),
    )

    decision = evaluate_ticket_agent_thread_cleanup(
        closed,
        policy=TicketAgentThreadCleanupPolicy(closed_thread_retention_seconds=3600),
        now=now + timedelta(hours=2),
    )

    assert decision.action == "archive"
    assert decision.reason == "closed_retention_elapsed"
    assert decision.checkpoint_action == "delete_after_archive"
    assert decision.archive_required is True


def test_cleanup_plan_summarizes_actions_and_checkpoint_deletes() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    active = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-plan-active-001",
        now=now,
        ttl_seconds=4 * 3600,
    )
    expired = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-plan-expired-001",
        now=now,
        ttl_seconds=60,
    )
    completed = complete_ticket_agent_thread(
        create_ticket_agent_thread_binding(
            actor_id="demo_user_001",
            thread_id="ticket-thread-plan-completed-001",
            now=now,
        ),
        ticket_id="T1001",
        now=now,
    )

    plan = build_ticket_agent_thread_cleanup_plan(
        [active, expired, completed],
        policy=TicketAgentThreadCleanupPolicy(
            expired_thread_grace_seconds=3600,
            completed_thread_retention_seconds=3600,
        ),
        now=now + timedelta(hours=2),
    )

    assert plan.count_by_action() == {
        "keep": 1,
        "expire": 0,
        "archive": 2,
    }
    assert plan.archive_required_thread_ids() == [
        "ticket-thread-plan-expired-001",
        "ticket-thread-plan-completed-001",
    ]
    assert plan.checkpoint_delete_thread_ids() == [
        "ticket-thread-plan-expired-001",
        "ticket-thread-plan-completed-001",
    ]
