from operator import add as op_add
from typing import Annotated, get_origin

from app.agents.multi_agent_states import (
    KnowledgeWorkerState,
    OrderWorkerState,
    SupervisorState,
    TicketWorkerState,
    SUPERVISOR_OUTPUT_KEYS,
)
from app.agents.ticket_agent import TicketAgentState


def test_supervisor_state_holds_core_fields() -> None:
    state: SupervisorState = {
        "user_message": "查订单",
        "normalized_message": "查订单",
        "intent": "order_query",
        "final_answer": "订单已发货",
        "node_history": ["normalize_user_input"],
    }
    assert state["intent"] == "order_query"
    assert state["final_answer"] == "订单已发货"


def test_worker_states_hold_their_own_fields() -> None:
    knowledge: KnowledgeWorkerState = {
        "normalized_message": "退货政策",
        "rag_answer_status": "answered",
        "final_answer": "退货政策是...",
    }
    order: OrderWorkerState = {
        "normalized_message": "查订单 A1001",
        "order_query_status": "succeeded",
        "order_query_result": {"order_id": "A1001"},
        "final_answer": "订单已发货",
    }
    ticket: TicketWorkerState = {
        "normalized_message": "申请退款",
        "ticket_confirmation_approved": True,
        "pending_ticket_confirmation": {"confirmation_id": "a" * 32},
        "final_answer": "工单已创建",
    }
    assert knowledge["rag_answer_status"] == "answered"
    assert order["order_query_result"]["order_id"] == "A1001"
    assert ticket["ticket_confirmation_approved"] is True


def test_supervisor_output_keys_contains_final_answer() -> None:
    assert "final_answer" in SUPERVISOR_OUTPUT_KEYS


def test_worker_state_keys_are_subset_of_ticket_agent_state() -> None:
    """子图 state 不得引入现有 TicketAgentState 没有的 key（防字段漂移）。"""
    ticket_agent_keys = set(TicketAgentState.__annotations__)
    for worker_state in (KnowledgeWorkerState, OrderWorkerState, TicketWorkerState):
        extra_keys = set(worker_state.__annotations__) - ticket_agent_keys
        assert not extra_keys, (
            f"{worker_state.__name__} 引入了 TicketAgentState 之外的字段: {sorted(extra_keys)}"
        )


def test_node_history_uses_add_reducer() -> None:
    """所有 state 的 node_history 必须使用 add reducer（追加语义）。"""
    for state in (
        SupervisorState,
        KnowledgeWorkerState,
        OrderWorkerState,
        TicketWorkerState,
    ):
        annotation = state.__annotations__["node_history"]
        assert get_origin(annotation) is Annotated, state.__name__
        assert op_add in annotation.__metadata__, state.__name__
