from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.core.exceptions import AppException
from app.services.console_agent_service import ConsoleAgentActor, ConsoleAgentService

ACTIVE = {
    "id": 1,
    "conversation_id": "conv-s",
    "user_id": "U1001",
    "status": "in_progress",
    "assigned_agent": "A1001",
}
ACTOR_AGENT = ConsoleAgentActor(user_id="A1001", tenant_id="default", roles=("staff",))
ACTOR_CUSTOMER = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))


def make_service() -> ConsoleAgentService:
    return ConsoleAgentService(Settings(_env_file=None), graph=MagicMock(), conversation_store=MagicMock())


def _message(role: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content, created_at=None)


def test_summary_uses_rule_source_for_rule_mode(monkeypatch):
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)
    service.conversation_store.get.return_value = SimpleNamespace(
        messages=[
            _message("user", "我的订单 202501010001 还不发货"),
            _message("assistant", "正在为您查询"),
            _message("human_agent", "您好，订单已加急"),
        ]
    )

    result = service.conversation_summary(actor=ACTOR_AGENT, conversation_id="conv-s")

    assert result.source == "rule"
    assert "202501010001" in result.summary
    assert "人工" in result.summary


def test_summary_rejects_when_not_active(monkeypatch):
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: None)

    with pytest.raises(AppException) as exc:
        service.conversation_summary(actor=ACTOR_AGENT, conversation_id="conv-s")
    assert exc.value.status_code == 409


def test_summary_rejects_customer(monkeypatch):
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)

    with pytest.raises(AppException) as exc:
        service.conversation_summary(actor=ACTOR_CUSTOMER, conversation_id="conv-s")
    assert exc.value.code == "HANDOFF_NOT_ASSIGNED"


def test_summary_empty_conversation_returns_placeholder(monkeypatch):
    service = make_service()
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)
    service.conversation_store.get.return_value = SimpleNamespace(messages=[])

    result = service.conversation_summary(actor=ACTOR_AGENT, conversation_id="conv-s")

    assert result.summary == "暂无会话消息"
    assert result.source == "rule"


def test_summary_uses_llm_when_real_llm_mode(monkeypatch):
    service = make_service()
    service.settings = Settings(_env_file=None, ticket_agent_model_mode="real_llm")
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)
    service.conversation_store.get.return_value = SimpleNamespace(
        messages=[_message("user", "我的订单 202501010001 还不发货")]
    )
    monkeypatch.setattr(service, "_llm_conversation_summary", lambda messages: "LLM 生成的摘要")

    result = service.conversation_summary(actor=ACTOR_AGENT, conversation_id="conv-s")

    assert result.source == "llm"
    assert result.summary == "LLM 生成的摘要"


def test_summary_falls_back_to_rule_when_llm_fails(monkeypatch):
    from app.core.exceptions import AppException as AE

    service = make_service()
    service.settings = Settings(_env_file=None, ticket_agent_model_mode="real_llm")
    monkeypatch.setattr(service, "_active_human_handoff", lambda _cid: ACTIVE)
    service.conversation_store.get.return_value = SimpleNamespace(
        messages=[_message("user", "我的订单 202501010001 还不发货")]
    )

    def boom(_messages):
        raise AE(code="LLM_DOWN", message="down", status_code=502)

    monkeypatch.setattr(service, "_llm_conversation_summary", boom)

    result = service.conversation_summary(actor=ACTOR_AGENT, conversation_id="conv-s")

    assert result.source == "rule"
    assert "202501010001" in result.summary
