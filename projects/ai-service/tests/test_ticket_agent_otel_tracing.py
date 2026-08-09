import re

import pytest

from app.agents.otel_tracing import (
    OTEL_SPAN_ID_HEX_LENGTH,
    OTEL_TRACE_ID_HEX_LENGTH,
    TRACEPARENT_HEADER,
    build_otel_trace_context,
    build_ticket_agent_otel_resource_attributes,
    build_ticket_agent_otel_span_attributes,
    build_ticket_agent_otel_span_plan,
    build_traceparent,
    normalize_otel_span_id,
    normalize_otel_trace_id,
    parse_traceparent,
)
from app.agents.thread_lifecycle import TICKET_AGENT_THREAD_ID_UNSAFE_MESSAGE
from app.core.trace import reset_trace_id, set_trace_id


def test_normalize_otel_trace_and_span_ids_accept_only_non_zero_hex() -> None:
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    span_id = "00f067aa0ba902b7"

    assert normalize_otel_trace_id(trace_id.upper()) == trace_id
    assert normalize_otel_span_id(span_id.upper()) == span_id
    assert normalize_otel_trace_id("0" * OTEL_TRACE_ID_HEX_LENGTH) is None
    assert normalize_otel_span_id("0" * OTEL_SPAN_ID_HEX_LENGTH) is None
    assert normalize_otel_trace_id("client-trace-001") is None
    assert normalize_otel_span_id("bad-span") is None


def test_parse_traceparent_reads_w3c_trace_context() -> None:
    traceparent = (
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    )

    parsed = parse_traceparent(traceparent)

    assert parsed is not None
    assert parsed.version == "00"
    assert parsed.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert parsed.parent_id == "00f067aa0ba902b7"
    assert parsed.trace_flags == "01"
    assert parsed.sampled is True
    assert parsed.format() == traceparent


def test_parse_traceparent_rejects_invalid_values() -> None:
    assert parse_traceparent(None) is None
    assert parse_traceparent("bad") is None
    assert (
        parse_traceparent(
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        )
        is None
    )
    assert (
        parse_traceparent(
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01"
        )
        is None
    )
    assert (
        parse_traceparent(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01"
        )
        is None
    )


def test_build_traceparent_formats_current_span_context() -> None:
    assert build_traceparent(
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        span_id="00f067aa0ba902b7",
        sampled=False,
    ) == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"

    with pytest.raises(ValueError, match="trace_id"):
        build_traceparent(trace_id="bad", span_id="00f067aa0ba902b7")

    with pytest.raises(ValueError, match="span_id"):
        build_traceparent(
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            span_id="bad",
        )


def test_build_otel_trace_context_reuses_incoming_trace_and_parent() -> None:
    context = build_otel_trace_context(
        incoming_traceparent=(
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
        ),
        span_id="5fb397be34d26b51",
        sampled=True,
    )

    assert context.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert context.parent_span_id == "00f067aa0ba902b7"
    assert context.span_id == "5fb397be34d26b51"
    assert context.sampled is False
    assert context.to_headers() == {
        TRACEPARENT_HEADER: (
            "00-4bf92f3577b34da6a3ce929d0e0e4736-5fb397be34d26b51-00"
        )
    }


def test_build_otel_trace_context_falls_back_to_project_trace_id_or_generates() -> None:
    context = build_otel_trace_context(
        project_trace_id="8b0e715c76c8423e9dc95b6c8db8409a",
        span_id="5fb397be34d26b51",
    )

    assert context.trace_id == "8b0e715c76c8423e9dc95b6c8db8409a"
    assert context.parent_span_id is None
    assert context.sampled is True

    generated = build_otel_trace_context(project_trace_id="client-trace-001")

    assert re.fullmatch(r"[0-9a-f]{32}", generated.trace_id)
    assert re.fullmatch(r"[0-9a-f]{16}", generated.span_id)


def test_build_otel_trace_context_can_use_current_request_trace_id() -> None:
    token = set_trace_id("8b0e715c76c8423e9dc95b6c8db8409a")
    try:
        context = build_otel_trace_context(span_id="5fb397be34d26b51")
    finally:
        reset_trace_id(token)

    assert context.trace_id == "8b0e715c76c8423e9dc95b6c8db8409a"


def test_build_ticket_agent_otel_resource_attributes_uses_semantic_keys() -> None:
    attributes = build_ticket_agent_otel_resource_attributes(
        service_version="0.1.0",
        environment="test",
    )

    assert attributes == {
        "service.name": "ai-service",
        "service.namespace": "java-python-ai",
        "deployment.environment.name": "test",
        "service.version": "0.1.0",
    }


def test_build_ticket_agent_otel_span_attributes_keeps_safe_agent_summary() -> None:
    state = {
        "agent_trace_id": "trace-001",
        "intent": "ticket_request",
        "ticket_need_source": "explicit_user_request",
        "order_query_status": "succeeded",
        "rag_answer_status": "answered",
        "ticket_field_extraction_source": "llm",
        "ticket_fields_complete": True,
        "ticket_confirmation_required": True,
        "ticket_confirmation_approved": False,
        "ticket_tool_name": "create_ticket",
        "ticket_tool_access_level": "write",
        "ticket_tool_requires_confirmation": True,
        "ticket_write_safety_status": "confirmation_required",
        "ticket_creation_status": "blocked",
        "fallback_used": False,
        "node_history": ["normalize_user_input", "extract_ticket_fields"],
        "rag_citations": [{"chunk_id": "faq-001"}, {"chunk_id": "faq-002"}],
        "missing_ticket_fields": ["order_id"],
    }

    attributes = build_ticket_agent_otel_span_attributes(
        state,
        operation="invoke_thread",
        thread_id=" ticket-thread-001 ",
        actor_id="demo_user_001",
        elapsed_ms=12.3456,
    )

    assert attributes["otel.scope.name"] == "app.agents.ticket_agent"
    assert attributes["app.component"] == "ticket_agent"
    assert attributes["app.operation"] == "invoke_thread"
    assert attributes["app.trace_id"] == "trace-001"
    assert attributes["app.thread_id"] == "ticket-thread-001"
    assert attributes["app.session_id"] == "ticket-thread-001"
    assert attributes["app.actor_id"] == "demo_user_001"
    assert attributes["agent.intent"] == "ticket_request"
    assert attributes["agent.node.count"] == 2
    assert attributes["agent.node.last"] == "extract_ticket_fields"
    assert attributes["rag.citation.count"] == 2
    assert attributes["ticket.missing_fields.count"] == 1
    assert attributes["app.elapsed_ms"] == 12.35


def test_build_ticket_agent_otel_span_attributes_omits_sensitive_payloads() -> None:
    state = {
        "agent_trace_id": "trace-001",
        "user_message": "my phone is 13800000000",
        "normalized_message": "my phone is 13800000000",
        "rag_query": "refund policy",
        "rag_citations": [{"content": "full chunk text"}],
        "rag_suggestions": ["ask human"],
        "final_answer": "full answer",
        "ticket_fields": {"description": "private complaint"},
        "ticket_creation_args": {"description": "private complaint"},
        "created_ticket": {"id": "T1001", "description": "private complaint"},
        "order_query_result": {"receiver_phone": "13800000000"},
        "pending_ticket_confirmation": {"summary": "private complaint"},
    }

    attributes = build_ticket_agent_otel_span_attributes(
        state,
        operation="invoke_thread",
        extra_attributes={
            "user_message": "should be skipped",
            "raw_payload": {"too": "large"},
        },
    )

    for sensitive_key in (
        "user_message",
        "normalized_message",
        "rag_query",
        "rag_citations",
        "rag_suggestions",
        "final_answer",
        "ticket_fields",
        "ticket_creation_args",
        "created_ticket",
        "order_query_result",
        "pending_ticket_confirmation",
        "raw_payload",
    ):
        assert sensitive_key not in attributes
    assert attributes["rag.citation.count"] == 1


def test_extra_attributes_cannot_override_core_observability_fields() -> None:
    attributes = build_ticket_agent_otel_span_attributes(
        {"agent_trace_id": "trace-001"},
        operation="invoke",
        thread_id="ticket-thread-001",
        actor_id="demo_user_001",
        extra_attributes={
            "app.trace_id": "wrong",
            "app.thread_id": "wrong",
            "deployment.environment.name": "wrong",
            "Custom.Eval.Case_ID": "case-001",
            "custom.retry_count": 2,
            "custom.canary_enabled": True,
            "empty": "   ",
            "complex": {"nested": True},
            "1bad": "skipped",
        },
    )

    assert attributes["app.trace_id"] == "trace-001"
    assert attributes["app.thread_id"] == "ticket-thread-001"
    assert "deployment.environment.name" not in attributes
    assert attributes["custom.eval.case_id"] == "case-001"
    assert attributes["custom.retry_count"] == 2
    assert attributes["custom.canary_enabled"] is True
    assert "empty" not in attributes
    assert "complex" not in attributes
    assert "1bad" not in attributes


def test_ticket_agent_otel_span_plan_summarizes_span_context_and_status() -> None:
    plan = build_ticket_agent_otel_span_plan(
        {
            "agent_trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
            "intent": "order_query",
            "order_query_status": "failed",
            "agent_error_code": "ORDER_QUERY_TIMEOUT",
            "node_history": ["normalize_user_input", "query_order"],
        },
        operation="invoke_thread",
        thread_id="ticket-thread-001",
        actor_id="demo_user_001",
        span_id="5fb397be34d26b51",
        elapsed_ms=305.129,
    )

    assert plan.span_name == "ticket_agent.invoke_thread"
    assert plan.span_kind == "INTERNAL"
    assert plan.status == "ERROR"
    assert plan.status_description == "ORDER_QUERY_TIMEOUT"
    assert plan.trace_context.trace_id == "8b0e715c76c8423e9dc95b6c8db8409a"
    assert plan.trace_context.span_id == "5fb397be34d26b51"
    assert plan.attributes["agent.intent"] == "order_query"
    assert plan.attributes["app.elapsed_ms"] == 305.13
    assert plan.to_start_span_kwargs() == {
        "name": "ticket_agent.invoke_thread",
        "kind": "INTERNAL",
        "attributes": plan.attributes,
    }


def test_ticket_agent_otel_span_plan_reuses_thread_id_validation() -> None:
    with pytest.raises(ValueError, match=TICKET_AGENT_THREAD_ID_UNSAFE_MESSAGE):
        build_ticket_agent_otel_span_plan(
            {"agent_trace_id": "trace-001"},
            operation="invoke_thread",
            thread_id="../bad-thread",
        )
