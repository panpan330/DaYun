"""Ticket worker subgraph: ticket creation, refund and cancel worker paths.

Task 8：工单 worker 子图同时承载三条 worker 链——
1. 工单创建链（intent=ticket_request / 非退款非取消）：extract_ticket_fields → 缺字段追问/
   确认 → create_ticket；
2. 退款执行链（intent=refund_request 或 active-refund-collection）：handle_refund_request
   → 缺字段追问/确认 → execute_refund_request；
3. 取消执行链（intent=cancel_request 或 active-cancel-collection）：handle_cancel_request
   → 缺字段追问/确认 → execute_cancel_request。

退款/取消执行链复用 ticket_agent 主图的 handle_refund_request_node / handle_cancel_request_node /
execute_refund_request_node / execute_cancel_request_node，使多 Agent supervisor 下
refund_request/cancel_request 的行为与单 Agent 主图一致（收集 order_id+reason → 确认 → 执行
refund_order/cancel_order），而非退化为仅创建对应工单（规格 4.4：工单 worker 子图原不支持
退款/取消意图，故新增对应节点）。
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.emotion_classifier import CustomerEmotionClassifier
from app.agents.multi_agent_states import TicketWorkerState
from app.agents.workers.emotion_node import make_emotion_recognition_node
from app.agents.ticket_agent import (
    TICKET_AGENT_CONFIRMATION_ROUTES,
    TICKET_AGENT_FIELD_COMPLETION_ROUTES,
    CancelExecutor,
    RefundExecutor,
    TicketCreator,
    ask_missing_ticket_fields_node,
    create_ticket_node,
    execute_cancel_request_node,
    execute_refund_request_node,
    extract_ticket_fields_node,
    handle_cancel_request_node,
    handle_refund_request_node,
    has_active_cancel_collection,
    has_active_refund_collection,
    request_ticket_confirmation_interrupt_node,
    request_ticket_confirmation_node,
    route_by_cancel_fields,
    route_by_refund_fields,
    route_by_ticket_confirmation,
)


def route_ticket_worker_entry(state: TicketWorkerState) -> str:
    """入口分流：取消/退款请求走对应 handle 链，其余走工单字段提取链。

    双判别点（intent 或 active-collection 任一命中即对应链）：
    - 第一轮：supervisor 已分类 intent=cancel_request / refund_request；
    - 后续轮：supervisor_route_node 的 active-collection 强制对应 intent，同时顶层
      cancel_request_active / refund_request_active=True + 仍缺字段兜底，防止
      LLM 路由误判偏离取消/退款流程。
    第二判别刻意要求"仍缺字段"：字段完整、待确认（missing 为空）时 active 标志仍为
    True，若仅凭裸标志判断会把用户改口的工单诉求（如"帮我建个报修工单"）劫持进
    取消/退款链；与 supervisor 各处 active-collection 守卫语义一致。
    """
    if state.get("intent") == "cancel_request" or has_active_cancel_collection(state):
        return "handle_cancel_request"
    if (
        state.get("intent") == "refund_request"
        or has_active_refund_collection(state)
    ):
        return "handle_refund_request"
    return "extract_ticket_fields"


def _extract_ticket_fields_reset_flags(
    state: TicketWorkerState,
) -> TicketWorkerState:
    """进入工单创建链前的字段抽取：清除残留的退款/取消活动标志。

    退款/取消流程停在待确认时 refund_request_active / cancel_request_active 仍为
    True（handle_*_request_node 写入，工单链不清除）。若用户随后改口建工单，抽取
    出的工单字段走 request_ticket_confirmation → route_by_ticket_confirmation 时会
    因该残留标志被误路由到 execute_refund_request / execute_cancel_request（真实
    执行退款/取消，用工单 description 当原因）。入口路由已保证走到本节点的都是非
    退款/取消链，因此在此显式归零标志，使确认路由正确分流到 execute_create_ticket。
    退款/取消链走 handle_*_request_node，不受影响。
    """
    update = extract_ticket_fields_node(state)
    update["refund_request_active"] = False
    update["cancel_request_active"] = False
    return update


TICKET_WORKER_ENTRY_ROUTES = {
    "handle_cancel_request": "handle_cancel_request",
    "handle_refund_request": "handle_refund_request",
    "extract_ticket_fields": "extract_ticket_fields",
}

# 退款执行目标节点在本子图中存在后，确认路由表与 ticket_agent 主图的
# TICKET_AGENT_CONFIRMATION_ROUTES 完全一致，直接复用共享表。


def build_ticket_worker_graph(
    ticket_creator: TicketCreator | None = None,
    *,
    refund_executor: RefundExecutor | None = None,
    cancel_executor: CancelExecutor | None = None,
    checkpointer: Any | None = None,
    interrupt_confirmation: bool = False,
    emotion_classifier: CustomerEmotionClassifier | None = None,
) -> Any:
    def _route_after_emotion(state: TicketWorkerState) -> str:
        if state.get("emotion_handoff_requested"):
            return "handoff"
        return route_ticket_worker_entry(state)

    builder = StateGraph(TicketWorkerState)
    builder.add_edge(START, "emotion_recognition")
    builder.add_node(
        "emotion_recognition",
        make_emotion_recognition_node(emotion_classifier),
    )
    builder.add_conditional_edges(
        "emotion_recognition",
        _route_after_emotion,
        {"handoff": END, **TICKET_WORKER_ENTRY_ROUTES},
    )
    builder.add_node(
        "extract_ticket_fields",
        _extract_ticket_fields_reset_flags,
    )
    builder.add_node(
        "handle_refund_request",
        lambda state: handle_refund_request_node(state),
    )
    builder.add_node(
        "handle_cancel_request",
        lambda state: handle_cancel_request_node(state),
    )
    builder.add_node("ask_missing_ticket_fields", ask_missing_ticket_fields_node)
    builder.add_node(
        "request_ticket_confirmation",
        (
            request_ticket_confirmation_interrupt_node
            if interrupt_confirmation
            else request_ticket_confirmation_node
        ),
    )
    builder.add_node(
        "create_ticket",
        lambda state: create_ticket_node(state, creator=ticket_creator),
    )
    builder.add_node(
        "execute_refund_request",
        lambda state: execute_refund_request_node(
            state,
            refund_executor=refund_executor,
        ),
    )
    builder.add_node(
        "execute_cancel_request",
        lambda state: execute_cancel_request_node(
            state,
            cancel_executor=cancel_executor,
        ),
    )
    builder.add_conditional_edges(
        "extract_ticket_fields",
        lambda state: (
            "ask_missing_fields"
            if (state.get("missing_ticket_fields") or [])
            else "request_confirmation"
        ),
        TICKET_AGENT_FIELD_COMPLETION_ROUTES,
    )
    builder.add_conditional_edges(
        "handle_refund_request",
        route_by_refund_fields,
        TICKET_AGENT_FIELD_COMPLETION_ROUTES,
    )
    builder.add_conditional_edges(
        "handle_cancel_request",
        route_by_cancel_fields,
        TICKET_AGENT_FIELD_COMPLETION_ROUTES,
    )
    builder.add_edge("ask_missing_ticket_fields", END)
    builder.add_edge("create_ticket", END)
    builder.add_edge("execute_refund_request", END)
    builder.add_edge("execute_cancel_request", END)
    builder.add_conditional_edges(
        "request_ticket_confirmation",
        route_by_ticket_confirmation,
        TICKET_AGENT_CONFIRMATION_ROUTES,
    )
    return builder.compile(checkpointer=checkpointer)
