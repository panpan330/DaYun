from app.core.ai_service_tracing import (
    AI_SERVICE_HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS,
    build_ai_service_span_attributes,
    build_python_ai_service_tracing_plan,
)
from app.core.trace import reset_trace_id, set_trace_id


def test_tool_chat_tracing_plan_covers_model_tool_and_java_boundaries() -> None:
    plan = build_python_ai_service_tracing_plan(
        flow="tool_chat",
        trace_id="trace-tool-001",
        model="qwen3.7-plus",
        provider="openai-compatible",
        tool_name="query_order",
    )

    assert plan.trace_id == "trace-tool-001"
    assert plan.span_names() == [
        "http.request",
        "request.validation",
        "llm.tool_decision",
        "tool.validation",
        "tool.execution",
        "java.orders.get",
        "llm.final_answer",
    ]
    assert "tool_requested" in plan.event_names()
    assert "permission_denied" in plan.event_names()
    assert "ai_service.tool.calls" in plan.metric_names()
    assert "ai_service.java.client.duration" in plan.metric_names()

    java_span = next(span for span in plan.spans if span.name == "java.orders.get")
    assert java_span.kind == "CLIENT"
    assert java_span.parent_name == "tool.execution"
    assert java_span.attributes["tool.name"] == "query_order"


def test_rag_tracing_plan_covers_retrieval_rerank_and_final_answer() -> None:
    plan = build_python_ai_service_tracing_plan(
        flow="rag_answer",
        trace_id="trace-rag-001",
        vector_store="qdrant",
        collection="learning_rag_chunks",
        top_k=8,
    )

    assert plan.span_names() == [
        "http.request",
        "request.validation",
        "rag.query_rewrite",
        "embedding.call",
        "vector.search",
        "rerank.call",
        "context.compression",
        "llm.final_answer",
    ]
    assert "prompt_injection_detected" in plan.event_names()
    assert "rag_no_relevant_context" in plan.event_names()
    assert "ai_service.rag.retrieval.duration" in plan.metric_names()

    vector_span = next(span for span in plan.spans if span.name == "vector.search")
    assert vector_span.attributes["vector.store"] == "qdrant"
    assert vector_span.attributes["vector.collection"] == "learning_rag_chunks"
    assert vector_span.attributes["rag.top_k"] == 8


def test_stream_chat_plan_tracks_llm_stream_and_sse_stream() -> None:
    plan = build_python_ai_service_tracing_plan(
        flow="stream_chat",
        trace_id="trace-stream-001",
        model="qwen3.7-plus",
    )

    assert "llm.stream" in plan.span_names()
    assert "sse.stream" in plan.span_names()
    assert "sse_client_disconnected" in plan.event_names()
    assert "stream_error_sent" in plan.event_names()


def test_span_attributes_keep_safe_summary_and_omit_sensitive_payloads() -> None:
    attributes = build_ai_service_span_attributes(
        flow="chat",
        operation="llm.call",
        trace_id="trace-001",
        route="/chat",
        method="POST",
        model="qwen3.7-plus",
        provider="openai-compatible",
        extra_attributes={
            "prompt": "full prompt should not be recorded",
            "user_message": "private user message",
            "Authorization": "Bearer secret",
            "custom.retry_count": 2,
            "custom.fallback_used": True,
            "app.trace_id": "wrong-trace",
            "raw_payload": {"too": "large"},
            "empty": "   ",
        },
    )

    assert attributes["app.trace_id"] == "trace-001"
    assert attributes["app.flow"] == "chat"
    assert attributes["app.operation"] == "llm.call"
    assert attributes["http.route"] == "/chat"
    assert attributes["llm.model"] == "qwen3.7-plus"
    assert attributes["custom.retry_count"] == 2
    assert attributes["custom.fallback_used"] is True
    assert "prompt" not in attributes
    assert "user_message" not in attributes
    assert "authorization" not in attributes
    assert "raw_payload" not in attributes
    assert "empty" not in attributes


def test_metrics_use_low_cardinality_attributes_only() -> None:
    plan = build_python_ai_service_tracing_plan(
        flow="tool_chat",
        trace_id="trace-001",
        route="/tool-chat",
        model="qwen3.7-plus",
        provider="openai-compatible",
    )

    for metric in plan.metrics:
        assert metric.attributes["app.flow"] == "tool_chat"
        assert metric.attributes["http.route"] == "/tool-chat"
        assert metric.attributes["llm.model"] == "qwen3.7-plus"
        assert not AI_SERVICE_HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS.intersection(
            metric.attributes
        )


def test_tracing_plan_can_reuse_current_request_trace_id() -> None:
    token = set_trace_id("current-request-trace")
    try:
        plan = build_python_ai_service_tracing_plan(flow="chat")
    finally:
        reset_trace_id(token)

    assert plan.trace_id == "current-request-trace"
    assert all(
        span.attributes["app.trace_id"] == "current-request-trace"
        for span in plan.spans
    )
