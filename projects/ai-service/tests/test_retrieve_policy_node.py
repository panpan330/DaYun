from app.agents.ticket_agent import build_ticket_agent_input, retrieve_policy_node
from app.rag.generator import RagAnswer, RagAnswerStatus


class RecordingRagService:
    def __init__(self) -> None:
        self.scope = None
        self.calls = 0

    def answer_policy_question(self, query, *, access_scope=None, memory_context=None):
        self.calls += 1
        self.scope = access_scope
        return RagAnswer(answer="ok", status=RagAnswerStatus.ANSWERED)


def test_build_ticket_agent_input_carries_actor_fields():
    state = build_ticket_agent_input("退款政策", actor_roles=("staff",), actor_tenant_id="t1")
    assert state["actor_roles"] == ("staff",)
    assert state["actor_tenant_id"] == "t1"


def test_retrieve_policy_node_passes_access_scope_from_state():
    service = RecordingRagService()
    state = build_ticket_agent_input(
        "退款政策",
        actor_roles=("staff",),
        actor_tenant_id="t1",
    )
    state["ticket_actor_id"] = "U1001"
    retrieve_policy_node(state, service=service)  # type: ignore[arg-type]

    assert service.scope is not None
    assert service.scope.permission_groups == ["customer_service", "public", "internal_staff"]
    assert service.scope.tenant_id == "t1"
    assert service.scope.user_id == "U1001"


def test_retrieve_policy_node_without_actor_fields_passes_none():
    service = RecordingRagService()
    state = build_ticket_agent_input("退款政策")
    retrieve_policy_node(state, service=service)  # type: ignore[arg-type]

    assert service.calls == 1
    assert service.scope is None
