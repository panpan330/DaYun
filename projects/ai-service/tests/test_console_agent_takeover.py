from types import SimpleNamespace
from unittest.mock import MagicMock

from app.core.config import Settings
from app.services.console_agent_service import (
    ConsoleAgentActor,
    ConsoleAgentService,
    HUMAN_TAKEOVER_REPLY,
)
from app.services.java_human_handoff_client import JavaHumanHandoffClientError

ACTIVE = {"id": 1, "conversation_id": "conv-t", "status": "in_progress", "assigned_agent": "A1001"}
ACTOR = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))


class FakeGraph:
    def __init__(self, state: dict[str, object]) -> None:
        self.state = state

    def get_state(self, _config: object) -> SimpleNamespace:
        return SimpleNamespace(values=self.state, next=())


def make_service() -> ConsoleAgentService:
    return ConsoleAgentService(
        Settings(_env_file=None),
        graph=FakeGraph({}),
        conversation_store=MagicMock(),
    )


def test_reply_intercepts_when_takeover_active(monkeypatch) -> None:
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)

    response = service.reply(actor=ACTOR, conversation_id="conv-t", message="你好，还在吗")

    assert response.reply == HUMAN_TAKEOVER_REPLY
    assert response.route == "human_handoff"
    assert response.conversation_id == "conv-t"
    service.conversation_store.append_exchange.assert_called_once()


def test_reply_pending_handoff_not_intercepted(monkeypatch) -> None:
    service = make_service()
    pending = {"id": 1, "conversation_id": "conv-t", "status": "pending"}
    monkeypatch.setattr(service.handoff_client, "get_active_handoff", lambda _cid: pending)

    def fake_run_ticket_agent_in_thread(_graph, _message, **kwargs):
        return {"intent": "query_order", "final_answer": "订单正在配送中"}

    monkeypatch.setattr(
        "app.services.console_agent_service.run_ticket_agent_in_thread",
        fake_run_ticket_agent_in_thread,
    )

    response = service.reply(actor=ACTOR, conversation_id="conv-t", message="查下订单")

    assert response.reply == "订单正在配送中"


def test_reply_falls_back_when_java_unavailable(monkeypatch) -> None:
    service = make_service()

    def boom(_conversation_id):
        raise JavaHumanHandoffClientError("java down", status_code=502, code="UPSTREAM_DOWN")

    monkeypatch.setattr(service.handoff_client, "get_active_handoff", boom)

    def fake_run_ticket_agent_in_thread(_graph, _message, **kwargs):
        return {"intent": "query_order", "final_answer": "订单正在配送中"}

    monkeypatch.setattr(
        "app.services.console_agent_service.run_ticket_agent_in_thread",
        fake_run_ticket_agent_in_thread,
    )

    response = service.reply(actor=ACTOR, conversation_id="conv-t", message="查下订单")

    assert response.reply == "订单正在配送中"


def test_stream_reply_intercepts_when_takeover_active(monkeypatch) -> None:
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)

    events = list(
        service.stream_reply(
            actor=ACTOR,
            conversation_id="conv-t",
            message="你好，还在吗",
            trace_id="trace-1",
        )
    )

    names = [event["event"] for event in events]
    assert names == ["start", "reply", "done"]
    reply_event = next(event for event in events if event["event"] == "reply")
    assert reply_event["data"]["content"] == HUMAN_TAKEOVER_REPLY
    service.conversation_store.append_exchange.assert_called_once()
