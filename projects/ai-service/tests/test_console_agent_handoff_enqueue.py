from app.schemas.console_agent import (
    ConsoleAgentHumanHandoff,
    ConsoleAgentResponse,
)
from app.services.console_agent_service import ConsoleAgentActor, ConsoleAgentService


class _FakeHandoffClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail_next = False

    def create_handoff(self, **kwargs) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("java down")
        self.calls.append(kwargs)


def _service(fake_client: _FakeHandoffClient) -> ConsoleAgentService:
    service = ConsoleAgentService.__new__(ConsoleAgentService)
    service._handoff_client = fake_client
    return service


def _actor() -> ConsoleAgentActor:
    return ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))


def _response(handoff: ConsoleAgentHumanHandoff | None, emotion: str | None = None) -> ConsoleAgentResponse:
    return ConsoleAgentResponse(
        reply="ok",
        conversation_id="conv-1",
        trace_id="t-1",
        route="order_query",
        human_handoff=handoff,
        emotion=emotion,
    )


def test_enqueues_handoff_with_actor_and_response():
    fake = _FakeHandoffClient()
    service = _service(fake)
    handoff = ConsoleAgentHumanHandoff(
        reason="检测到强烈情绪（angry），建议由人工客服继续跟进。",
        related_order_id="202501010001",
    )
    service._maybe_enqueue_handoff(
        actor=_actor(),
        conversation_id="conv-1",
        response=_response(handoff, emotion="angry"),
    )
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["conversation_id"] == "conv-1"
    assert call["user_id"] == "U1001"
    assert call["tenant_id"] == "default"
    assert call["related_order_id"] == "202501010001"
    assert call["emotion"] == "angry"
    assert "人工" in call["reason"]


def test_skips_when_no_handoff():
    fake = _FakeHandoffClient()
    service = _service(fake)
    service._maybe_enqueue_handoff(actor=_actor(), conversation_id="conv-1", response=_response(None))
    assert fake.calls == []


def test_enqueue_failure_does_not_raise():
    fake = _FakeHandoffClient()
    fake.fail_next = True
    service = _service(fake)
    handoff = ConsoleAgentHumanHandoff(reason="需要人工跟进", related_order_id=None)
    # 失败仅告警，不抛出
    service._maybe_enqueue_handoff(actor=_actor(), conversation_id="conv-1", response=_response(handoff))
