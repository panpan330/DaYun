from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import AppException
from app.services.console_agent_service import (
    ConsoleAgentActor,
    ConsoleAgentService,
)

ACTIVE = {
    "id": 1,
    "conversation_id": "conv-r",
    "user_id": "U1001",
    "status": "in_progress",
    "assigned_agent": "A1001",
}
ACTOR_AGENT = ConsoleAgentActor(user_id="A1001", tenant_id="default", roles=("staff",))
ACTOR_OTHER = ConsoleAgentActor(user_id="A9999", tenant_id="default", roles=("staff",))
ACTOR_SUPERVISOR = ConsoleAgentActor(user_id="S1001", tenant_id="default", roles=("supervisor",))
ACTOR_CUSTOMER = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))


def make_service() -> ConsoleAgentService:
    return ConsoleAgentService(
        Settings(_env_file=None),
        graph=MagicMock(),
        conversation_store=MagicMock(),
    )


def test_human_reply_ok_for_assigned_agent(monkeypatch) -> None:
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)

    response = service.human_reply(
        actor=ACTOR_AGENT,
        conversation_id="conv-r",
        content="您的订单已加急",
    )

    assert response.reply == "您的订单已加急"
    assert response.route == "human_handoff"
    service.conversation_store.append_human_reply.assert_called_once_with(
        actor=ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",)),
        conversation_id="conv-r",
        content="您的订单已加急",
    )


def test_human_reply_rejects_when_not_active(monkeypatch) -> None:
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: None)

    with pytest.raises(AppException) as exc:
        service.human_reply(actor=ACTOR_AGENT, conversation_id="conv-r", content="x")
    assert exc.value.code == "HANDOFF_NOT_ACTIVE"
    assert exc.value.status_code == 409


def test_human_reply_rejects_unassigned_agent(monkeypatch) -> None:
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)

    with pytest.raises(AppException) as exc:
        service.human_reply(actor=ACTOR_OTHER, conversation_id="conv-r", content="x")
    assert exc.value.code == "HANDOFF_NOT_ASSIGNED"
    assert exc.value.status_code == 403


def test_human_reply_supervisor_can_cover(monkeypatch) -> None:
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)

    response = service.human_reply(actor=ACTOR_SUPERVISOR, conversation_id="conv-r", content="代答")

    assert response.reply == "代答"


def test_human_reply_customer_rejected(monkeypatch) -> None:
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)

    with pytest.raises(AppException) as exc:
        service.human_reply(actor=ACTOR_CUSTOMER, conversation_id="conv-r", content="x")
    assert exc.value.code == "HANDOFF_NOT_ASSIGNED"
