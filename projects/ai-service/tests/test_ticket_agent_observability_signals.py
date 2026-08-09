from app.agents.observability_signals import (
    HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS,
    TICKET_AGENT_DURATION_METRIC_NAME,
    TICKET_AGENT_ERROR_METRIC_NAME,
    TICKET_AGENT_INVOCATION_METRIC_NAME,
    TICKET_AGENT_NODE_COUNT_METRIC_NAME,
    build_ticket_agent_investigation_steps,
    build_ticket_agent_observability_signals,
)


def test_observability_signals_correlate_trace_span_logs_and_metrics() -> None:
    signals = build_ticket_agent_observability_signals(
        {
            "agent_trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
            "intent": "ticket_request",
            "ticket_creation_status": "blocked",
            "ticket_write_safety_status": "confirmation_required",
            "node_history": ["normalize_user_input", "request_ticket_confirmation"],
        },
        operation="invoke_thread",
        thread_id="ticket-thread-001",
        actor_id="demo_user_001",
        span_id="5fb397be34d26b51",
        elapsed_ms=42.128,
    )

    assert signals.correlation.app_trace_id == "8b0e715c76c8423e9dc95b6c8db8409a"
    assert signals.correlation.otel_trace_id == "8b0e715c76c8423e9dc95b6c8db8409a"
    assert signals.correlation.span_id == "5fb397be34d26b51"
    assert signals.correlation.thread_id == "ticket-thread-001"
    assert signals.correlation.actor_id == "demo_user_001"
    assert signals.trace.trace_id == signals.correlation.otel_trace_id
    assert signals.trace.app_trace_id == signals.correlation.app_trace_id
    assert signals.trace.span_count == 1
    assert signals.trace.status == "UNSET"
    assert signals.span.name == "ticket_agent.invoke_thread"
    assert signals.span.status == "UNSET"
    assert signals.span.attributes["agent.intent"] == "ticket_request"
    assert signals.log_event_names() == [
        "ticket_agent_started",
        "ticket_agent_finished",
    ]
    assert signals.metric_names() == [
        TICKET_AGENT_INVOCATION_METRIC_NAME,
        TICKET_AGENT_DURATION_METRIC_NAME,
        TICKET_AGENT_NODE_COUNT_METRIC_NAME,
    ]


def test_logs_keep_correlation_ids_but_do_not_store_sensitive_payloads() -> None:
    signals = build_ticket_agent_observability_signals(
        {
            "agent_trace_id": "trace-001",
            "intent": "order_query",
            "user_message": "my phone is 13800000000",
            "final_answer": "full answer",
            "order_query_result": {"receiver_phone": "13800000000"},
            "node_history": ["normalize_user_input", "query_order"],
        },
        operation="invoke_thread",
        thread_id="ticket-thread-001",
        actor_id="demo_user_001",
        span_id="5fb397be34d26b51",
        elapsed_ms=20,
    )

    for log in signals.logs:
        assert log.fields["trace_id"] == "trace-001"
        assert log.fields["otel_trace_id"] == signals.correlation.otel_trace_id
        assert log.fields["span_id"] == "5fb397be34d26b51"
        assert log.fields["thread_id"] == "ticket-thread-001"
        assert "user_message" not in log.fields
        assert "final_answer" not in log.fields
        assert "order_query_result" not in log.fields


def test_metrics_use_low_cardinality_attributes_without_correlation_ids() -> None:
    signals = build_ticket_agent_observability_signals(
        {
            "agent_trace_id": "trace-001",
            "intent": "ticket_request",
            "ticket_creation_status": "blocked",
            "ticket_write_safety_status": "confirmation_required",
            "node_history": ["normalize_user_input"],
        },
        operation="invoke_thread",
        thread_id="ticket-thread-001",
        actor_id="demo_user_001",
        span_id="5fb397be34d26b51",
        elapsed_ms=100,
    )

    for metric in signals.metrics:
        assert metric.attributes["operation"] == "invoke_thread"
        assert metric.attributes["status"] == "ok"
        assert metric.attributes["intent"] == "ticket_request"
        assert metric.attributes["ticket_creation_status"] == "blocked"
        assert metric.attributes["ticket_write_safety_status"] == (
            "confirmation_required"
        )
        assert not HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS.intersection(
            metric.attributes
        )


def test_error_state_adds_failed_log_and_error_counter() -> None:
    signals = build_ticket_agent_observability_signals(
        {
            "agent_trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
            "intent": "order_query",
            "order_query_status": "failed",
            "agent_error_code": "ORDER_QUERY_TIMEOUT",
            "agent_error_node": "query_order",
            "fallback_used": True,
            "node_history": ["normalize_user_input", "query_order"],
        },
        operation="invoke_thread",
        thread_id="ticket-thread-001",
        span_id="5fb397be34d26b51",
        elapsed_ms=305.129,
    )

    failed_log = signals.logs[-1]
    error_metric = next(
        metric
        for metric in signals.metrics
        if metric.name == TICKET_AGENT_ERROR_METRIC_NAME
    )

    assert signals.trace.status == "ERROR"
    assert signals.span.status == "ERROR"
    assert failed_log.event_name == "ticket_agent_failed"
    assert failed_log.severity == "ERROR"
    assert failed_log.fields["error_code"] == "ORDER_QUERY_TIMEOUT"
    assert failed_log.fields["fallback_used"] is True
    assert error_metric.kind == "counter"
    assert error_metric.value == 1
    assert error_metric.attributes["status"] == "error"
    assert error_metric.attributes["error_code"] == "ORDER_QUERY_TIMEOUT"


def test_latency_investigation_starts_from_metrics() -> None:
    steps = build_ticket_agent_investigation_steps("latency_regression")

    assert [step.signal_type for step in steps] == [
        "metric",
        "trace",
        "span",
        "log",
    ]
    assert steps[0].fields == ["ticket_agent.duration", "operation", "intent"]


def test_one_user_failure_investigation_starts_from_logs_then_trace() -> None:
    steps = build_ticket_agent_investigation_steps("one_user_failed")

    assert [step.signal_type for step in steps] == [
        "log",
        "trace",
        "span",
        "metric",
    ]
    assert "trace_id" in steps[0].fields
    assert "thread_id" in steps[0].fields


def test_agent_decision_debug_prioritizes_trace_and_span() -> None:
    steps = build_ticket_agent_investigation_steps("agent_decision_debug")

    assert [step.signal_type for step in steps] == [
        "trace",
        "span",
        "log",
        "metric",
    ]
    assert "agent.intent" in steps[1].fields
    assert "ticket.write_safety.status" in steps[1].fields
