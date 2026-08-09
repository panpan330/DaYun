"""State definitions for the supervisor-worker multi-agent system."""

from operator import add
from typing import Annotated, TypedDict


class SupervisorState(TypedDict, total=False):
    user_message: str
    emotion: str
    emotion_reason: str
    emotion_handoff_requested: bool
    normalized_message: str
    agent_trace_id: str
    # 监督层写入 6 值意图（policy_question/order_query/ticket_request/smalltalk/
    # unsupported/unclear），与 TicketIntent 枚举（退款类）值域不同，故标注为 str
    intent: str
    intent_reason: str
    # 跨 Agent 协作字段（knowledge 子图输出，监督层读取）
    rag_answer_status: str
    rag_citations: list[dict]
    needs_ticket: bool
    ticket_need_source: str
    # ticket worker 子图字段（顶层传入确认/字段、读取创建结果，供监督层编排）
    ticket_fields: dict | None
    missing_ticket_fields: list[str]
    ticket_confirmation_approved: bool | None
    ticket_creation_status: str | None
    created_ticket: dict | None
    ticket_actor_id: str
    # refund 执行字段（ticket worker 退款链输出，顶层读取供监督层编排与
    # console 确认分流 is_refund_execution；active 标志用于多轮 active-collection）
    refund_request_active: bool | None
    refund_status: str | None
    refund_error_code: str | None
    refund_error_message: str | None
    refund_result: dict | None
    # cancel 执行字段（ticket worker 取消链输出，顶层读取供监督层编排与
    # console 确认分流 is_cancel_execution；active 标志用于多轮 active-collection）
    cancel_request_active: bool | None
    cancel_status: str | None
    cancel_error_code: str | None
    cancel_error_message: str | None
    cancel_result: dict | None
    # order worker 子图输出（顶层读取失败状态，供转人工/错误处理编排）
    order_query_order_id: str | None
    order_query_status: str
    order_query_result: dict | None
    order_query_error_code: str | None
    order_query_error_message: str | None
    order_query_error_kind: str | None
    order_query_error_action: str | None
    order_query_retryable: bool | None
    order_query_error_status_code: int | None
    final_answer: str | None
    node_history: Annotated[list[str], add]
    agent_error_code: str | None
    agent_error_message: str | None


class KnowledgeWorkerState(TypedDict, total=False):
    normalized_message: str
    emotion: str
    emotion_reason: str
    emotion_handoff_requested: bool
    agent_trace_id: str
    # 子图读取监督层传入的意图（str，见 SupervisorState.intent 说明）
    intent: str
    rag_query: str
    rag_answer_status: str
    rag_citations: list[dict]
    rag_no_context_reason: str | None
    rag_suggestions: list[str]
    needs_ticket: bool
    ticket_need_reason: str
    ticket_need_source: str
    final_answer: str | None
    node_history: Annotated[list[str], add]


class OrderWorkerState(TypedDict, total=False):
    normalized_message: str
    emotion: str
    emotion_reason: str
    emotion_handoff_requested: bool
    agent_trace_id: str
    order_query_order_id: str | None
    order_query_status: str
    order_query_result: dict | None
    order_query_error_code: str | None
    order_query_error_message: str | None
    order_query_error_kind: str | None
    order_query_error_action: str | None
    order_query_retryable: bool | None
    order_query_error_status_code: int | None
    final_answer: str | None
    node_history: Annotated[list[str], add]


class TicketWorkerState(TypedDict, total=False):
    normalized_message: str
    emotion: str
    emotion_reason: str
    emotion_handoff_requested: bool
    agent_trace_id: str
    # 监督层传入的意图（str，见 SupervisorState.intent 说明；子图入口据此分流退款链）
    intent: str
    rag_answer_status: str
    needs_ticket: bool
    ticket_fields: dict | None
    ticket_actor_id: str
    ticket_field_extraction_source: str
    missing_ticket_fields: list[str]
    missing_ticket_field_question: str
    missing_ticket_field_question_fields: list[str]
    ticket_fields_complete: bool | None
    ticket_need_source: str | None
    ticket_tool_name: str
    ticket_confirmation_required: bool | None
    ticket_confirmation_approved: bool | None
    ticket_confirmation_correction_requested: bool | None
    ticket_confirmation_message: str | None
    pending_ticket_confirmation: dict | None
    ticket_write_safety_status: str | None
    ticket_creation_status: str | None
    ticket_creation_error_code: str | None
    ticket_creation_error_message: str | None
    ticket_creation_idempotency_key: str | None
    created_ticket: dict | None
    refund_request_active: bool | None
    refund_status: str | None
    refund_error_code: str | None
    refund_error_message: str | None
    refund_result: dict | None
    cancel_request_active: bool | None
    cancel_status: str | None
    cancel_error_code: str | None
    cancel_error_message: str | None
    cancel_result: dict | None
    final_answer: str | None
    node_history: Annotated[list[str], add]


# Fields a worker subgraph writes back to the supervisor top-level state.
SUPERVISOR_OUTPUT_KEYS = frozenset(
    {
        "final_answer",
        "needs_ticket",
        "ticket_need_source",
        "rag_answer_status",
        "rag_citations",
        "ticket_creation_status",
        "created_ticket",
        "missing_ticket_fields",
        "refund_request_active",
        "refund_status",
        "refund_error_code",
        "refund_error_message",
        "refund_result",
        "cancel_request_active",
        "cancel_status",
        "cancel_error_code",
        "cancel_error_message",
        "cancel_result",
        "agent_error_code",
        "agent_error_message",
        "order_query_order_id",
        "order_query_status",
        "order_query_result",
        "order_query_error_code",
        "order_query_error_message",
        "order_query_error_kind",
        "order_query_error_action",
        "order_query_retryable",
        "order_query_error_status_code",
    }
)
