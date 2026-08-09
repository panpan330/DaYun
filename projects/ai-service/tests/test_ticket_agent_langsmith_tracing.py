import pytest

from app.agents.langsmith_tracing import (
    LANGSMITH_METADATA_TEXT_MAX_LENGTH,
    TICKET_AGENT_LANGSMITH_PROJECT_NAME,
    build_langsmith_trace_tags,
    build_ticket_agent_langsmith_metadata,
    build_ticket_agent_langsmith_trace_context,
    normalize_langsmith_tag,
)
from app.agents.thread_lifecycle import TICKET_AGENT_THREAD_ID_UNSAFE_MESSAGE
from app.core.trace import reset_trace_id, set_trace_id


def test_normalize_langsmith_tag_trims_lowercases_and_removes_unsafe_chars() -> None:
    assert normalize_langsmith_tag("  Local Dev  ") == "local-dev"
    assert normalize_langsmith_tag("operation:Invoke Thread") == (
        "operation:invoke-thread"
    )
    assert normalize_langsmith_tag("   ") is None


def test_build_langsmith_trace_tags_adds_stable_context_and_deduplicates() -> None:
    tags = build_langsmith_trace_tags(
        environment="Local Dev",
        operation="invoke_thread",
        intent="ticket_request",
        extra_tags=["ticket-agent", "manual smoke", None],
    )

    assert tags == [
        "ai-service",
        "ticket-agent",
        "langgraph",
        "env:local-dev",
        "operation:invoke_thread",
        "intent:ticket_request",
        "manual-smoke",
    ]


def test_build_ticket_agent_langsmith_metadata_keeps_observable_safe_fields() -> None:
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

    metadata = build_ticket_agent_langsmith_metadata(
        state,
        operation="invoke_thread",
        thread_id=" ticket-thread-001 ",
        actor_id="demo_user_001",
        elapsed_ms=12.3456,
    )

    assert metadata["component"] == "ticket_agent"
    assert metadata["operation"] == "invoke_thread"
    assert metadata["trace_id"] == "trace-001"
    assert metadata["thread_id"] == "ticket-thread-001"
    assert metadata["session_id"] == "ticket-thread-001"
    assert metadata["actor_id"] == "demo_user_001"
    assert metadata["intent"] == "ticket_request"
    assert metadata["node_count"] == 2
    assert metadata["last_node"] == "extract_ticket_fields"
    assert metadata["rag_citation_count"] == 2
    assert metadata["missing_ticket_fields_count"] == 1
    assert metadata["elapsed_ms"] == 12.35


def test_build_ticket_agent_langsmith_metadata_uses_current_trace_id_when_state_lacks_one() -> None:
    token = set_trace_id("trace-from-context")
    try:
        metadata = build_ticket_agent_langsmith_metadata(
            {},
            operation="invoke_safe",
        )
    finally:
        reset_trace_id(token)

    assert metadata["trace_id"] == "trace-from-context"
    assert metadata["node_count"] == 0
    assert "thread_id" not in metadata
    assert "session_id" not in metadata


def test_build_ticket_agent_langsmith_metadata_omits_sensitive_agent_payloads() -> None:
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

    metadata = build_ticket_agent_langsmith_metadata(
        state,
        operation="invoke_thread",
        extra_metadata={
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
        assert sensitive_key not in metadata
    assert metadata["rag_citation_count"] == 1


def test_extra_metadata_keeps_simple_values_and_cannot_override_core_fields() -> None:
    metadata = build_ticket_agent_langsmith_metadata(
        {"agent_trace_id": "trace-001"},
        operation="invoke",
        thread_id="ticket-thread-001",
        actor_id="demo_user_001",
        extra_metadata={
            "experiment_name": "baseline",
            "retry_count": 2,
            "canary_enabled": True,
            "trace_id": "wrong",
            "thread_id": "wrong",
            "empty": "   ",
            "complex": {"nested": True},
            "long_text": "x" * (LANGSMITH_METADATA_TEXT_MAX_LENGTH + 20),
        },
    )

    assert metadata["trace_id"] == "trace-001"
    assert metadata["thread_id"] == "ticket-thread-001"
    assert metadata["experiment_name"] == "baseline"
    assert metadata["retry_count"] == 2
    assert metadata["canary_enabled"] is True
    assert "empty" not in metadata
    assert "complex" not in metadata
    assert str(metadata["long_text"]).endswith("...")
    assert len(str(metadata["long_text"])) == LANGSMITH_METADATA_TEXT_MAX_LENGTH


def test_langsmith_trace_context_exports_langgraph_config_and_context_kwargs() -> None:
    context = build_ticket_agent_langsmith_trace_context(
        {
            "agent_trace_id": "trace-001",
            "intent": "order_query",
            "node_history": ["normalize_user_input", "query_order"],
        },
        operation="invoke_thread",
        thread_id="ticket-thread-001",
        actor_id="demo_user_001",
        environment="test",
        extra_tags=["regression"],
        extra_metadata={"case_id": "case-001"},
    )

    assert context.project_name == TICKET_AGENT_LANGSMITH_PROJECT_NAME
    assert context.run_name == "ticket_agent.invoke_thread"
    assert context.thread_id == "ticket-thread-001"
    assert context.tags == [
        "ai-service",
        "ticket-agent",
        "langgraph",
        "env:test",
        "operation:invoke_thread",
        "intent:order_query",
        "regression",
    ]
    assert context.metadata["case_id"] == "case-001"

    langgraph_config = context.to_langgraph_config()
    assert langgraph_config["run_name"] == "ticket_agent.invoke_thread"
    assert langgraph_config["configurable"] == {"thread_id": "ticket-thread-001"}
    assert langgraph_config["tags"] == context.tags
    assert langgraph_config["metadata"] == context.metadata

    tracing_kwargs = context.to_tracing_context_kwargs(enabled=False)
    assert tracing_kwargs == {
        "project_name": TICKET_AGENT_LANGSMITH_PROJECT_NAME,
        "enabled": False,
        "tags": context.tags,
        "metadata": context.metadata,
    }


def test_langsmith_trace_context_reuses_thread_id_validation() -> None:
    with pytest.raises(ValueError, match=TICKET_AGENT_THREAD_ID_UNSAFE_MESSAGE):
        build_ticket_agent_langsmith_trace_context(
            {"agent_trace_id": "trace-001"},
            operation="invoke_thread",
            thread_id="../bad-thread",
        )
