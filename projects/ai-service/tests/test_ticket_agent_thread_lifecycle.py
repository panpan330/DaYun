from datetime import datetime, timedelta, timezone

import pytest

from app.agents.thread_lifecycle import (
    TICKET_AGENT_THREAD_CONFIRMATION_TTL_SECONDS,
    TICKET_AGENT_THREAD_ID_MAX_LENGTH,
    TICKET_AGENT_THREAD_ID_PREFIX,
    TICKET_AGENT_THREAD_ID_TOO_LONG_MESSAGE,
    TICKET_AGENT_THREAD_ID_UNSAFE_MESSAGE,
    TICKET_AGENT_THREAD_TTL_SECONDS,
    close_ticket_agent_thread,
    complete_ticket_agent_thread,
    create_ticket_agent_thread_binding,
    ensure_utc_datetime,
    evaluate_ticket_agent_thread_resume,
    generate_ticket_agent_thread_id,
    is_ticket_agent_thread_expired,
    mark_ticket_agent_thread_waiting_confirmation,
    normalize_ticket_agent_thread_id,
)
from app.agents.ticket_agent import build_ticket_agent_thread_config


def test_generate_ticket_agent_thread_id_uses_safe_prefix_and_length() -> None:
    thread_id = generate_ticket_agent_thread_id()

    assert thread_id.startswith(TICKET_AGENT_THREAD_ID_PREFIX)
    assert len(thread_id) <= TICKET_AGENT_THREAD_ID_MAX_LENGTH
    assert normalize_ticket_agent_thread_id(thread_id) == thread_id


def test_normalize_ticket_agent_thread_id_rejects_unsafe_values() -> None:
    assert normalize_ticket_agent_thread_id(" ticket-thread-001 ") == (
        "ticket-thread-001"
    )

    with pytest.raises(ValueError, match="thread_id 不能为空"):
        normalize_ticket_agent_thread_id("   ")

    with pytest.raises(ValueError, match=TICKET_AGENT_THREAD_ID_UNSAFE_MESSAGE):
        normalize_ticket_agent_thread_id("../ticket-thread-001")

    with pytest.raises(ValueError, match=TICKET_AGENT_THREAD_ID_TOO_LONG_MESSAGE):
        normalize_ticket_agent_thread_id("t" * (TICKET_AGENT_THREAD_ID_MAX_LENGTH + 1))


def test_build_ticket_agent_thread_config_uses_lifecycle_thread_id_validation() -> None:
    assert build_ticket_agent_thread_config(" ticket-thread-001 ") == {
        "configurable": {"thread_id": "ticket-thread-001"}
    }

    with pytest.raises(ValueError, match=TICKET_AGENT_THREAD_ID_UNSAFE_MESSAGE):
        build_ticket_agent_thread_config("ticket/thread-001")


def test_create_ticket_agent_thread_binding_sets_owner_and_expiration() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)

    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        session_id="browser-session-001",
        thread_id="ticket-thread-fixed-001",
        now=now,
    )

    assert binding.thread_id == "ticket-thread-fixed-001"
    assert binding.actor_id == "demo_user_001"
    assert binding.session_id == "browser-session-001"
    assert binding.status == "active"
    assert binding.created_at == now
    assert binding.updated_at == now
    assert binding.expires_at == now + timedelta(seconds=TICKET_AGENT_THREAD_TTL_SECONDS)


def test_waiting_confirmation_shortens_thread_ttl_and_records_confirmation() -> None:
    created_at = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    waiting_at = datetime(2026, 7, 26, 9, 5, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-confirm-001",
        now=created_at,
    )

    waiting = mark_ticket_agent_thread_waiting_confirmation(
        binding,
        pending_confirmation_id="confirmation-001",
        now=waiting_at,
    )

    assert waiting.status == "waiting_confirmation"
    assert waiting.pending_confirmation_id == "confirmation-001"
    assert waiting.created_at == created_at
    assert waiting.updated_at == waiting_at
    assert waiting.expires_at == waiting_at + timedelta(
        seconds=TICKET_AGENT_THREAD_CONFIRMATION_TTL_SECONDS
    )


def test_resume_decision_allows_same_actor_before_expiration() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-active-001",
        now=now,
    )

    decision = evaluate_ticket_agent_thread_resume(
        binding,
        actor_id="demo_user_001",
        now=now + timedelta(minutes=10),
    )

    assert decision.allowed is True
    assert decision.reason == "ok"
    assert decision.action == "resume_existing"


def test_resume_decision_rejects_other_actor_to_prevent_thread_leak() -> None:
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-owner-001",
        now=datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc),
    )

    decision = evaluate_ticket_agent_thread_resume(
        binding,
        actor_id="other_user",
    )

    assert decision.allowed is False
    assert decision.reason == "actor_mismatch"
    assert decision.action == "reject"


def test_resume_decision_starts_new_thread_after_expiration() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-expired-001",
        now=now,
        ttl_seconds=60,
    )

    decision = evaluate_ticket_agent_thread_resume(
        binding,
        actor_id="demo_user_001",
        now=now + timedelta(seconds=60),
    )

    assert is_ticket_agent_thread_expired(
        binding,
        now=now + timedelta(seconds=60),
    )
    assert decision.allowed is False
    assert decision.reason == "expired"
    assert decision.action == "start_new"


def test_completed_and_closed_threads_should_not_be_resumed() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        thread_id="ticket-thread-finished-001",
        now=now,
    )

    completed = complete_ticket_agent_thread(
        binding,
        ticket_id="T1001",
        now=now + timedelta(minutes=2),
    )
    closed = close_ticket_agent_thread(
        binding,
        now=now + timedelta(minutes=3),
    )

    completed_decision = evaluate_ticket_agent_thread_resume(
        completed,
        actor_id="demo_user_001",
    )
    closed_decision = evaluate_ticket_agent_thread_resume(
        closed,
        actor_id="demo_user_001",
    )

    assert completed.status == "completed"
    assert completed.ticket_id == "T1001"
    assert completed.pending_confirmation_id is None
    assert completed_decision.allowed is False
    assert completed_decision.reason == "completed"
    assert completed_decision.action == "start_new"
    assert closed.status == "closed"
    assert closed_decision.allowed is False
    assert closed_decision.reason == "closed"
    assert closed_decision.action == "start_new"


def test_thread_binding_metadata_is_json_friendly() -> None:
    now = datetime(2026, 7, 26, 9, 0, tzinfo=timezone.utc)
    binding = create_ticket_agent_thread_binding(
        actor_id="demo_user_001",
        session_id="browser-session-001",
        thread_id="ticket-thread-metadata-001",
        now=now,
    )

    metadata = binding.to_metadata()

    assert metadata == {
        "thread_id": "ticket-thread-metadata-001",
        "actor_id": "demo_user_001",
        "session_id": "browser-session-001",
        "ticket_id": None,
        "pending_confirmation_id": None,
        "status": "active",
        "created_at": "2026-07-26T09:00:00+00:00",
        "updated_at": "2026-07-26T09:00:00+00:00",
        "expires_at": "2026-07-27T09:00:00+00:00",
    }


def test_ensure_utc_datetime_treats_naive_datetime_as_utc() -> None:
    naive = datetime(2026, 7, 26, 9, 0)

    assert ensure_utc_datetime(naive) == datetime(
        2026,
        7,
        26,
        9,
        0,
        tzinfo=timezone.utc,
    )
