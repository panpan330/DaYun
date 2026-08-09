import pytest

from app.agents.production_logging import (
    TICKET_AGENT_LOG_SEVERITY_NUMBERS,
    build_ticket_agent_log_field_specs,
    build_ticket_agent_production_log_record,
    find_forbidden_ticket_agent_log_fields,
    normalize_ticket_agent_log_severity,
    validate_ticket_agent_event_name,
)


def test_production_log_record_uses_otel_trace_context_as_top_level_fields() -> None:
    record = build_ticket_agent_production_log_record(
        {
            "agent_trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
            "intent": "ticket_request",
            "ticket_creation_status": "succeeded",
            "ticket_write_safety_status": "safe",
            "node_history": ["normalize_user_input", "create_ticket"],
        },
        event_name="ticket_agent.workflow.succeeded",
        operation="invoke_thread",
        body="Ticket agent workflow succeeded.",
        severity_text="INFO",
        thread_id="ticket-thread-001",
        actor_id="demo_user_001",
        span_id="5fb397be34d26b51",
        elapsed_ms=42.128,
        timestamp="2026-07-26T10:20:30.123Z",
        service_version="0.1.0",
        environment="test",
    )

    output = record.to_otel_log_record()

    assert output["timestamp"] == "2026-07-26T10:20:30.123Z"
    assert output["trace_id"] == "8b0e715c76c8423e9dc95b6c8db8409a"
    assert output["span_id"] == "5fb397be34d26b51"
    assert output["trace_flags"] == "01"
    assert output["severity_text"] == "INFO"
    assert output["severity_number"] == TICKET_AGENT_LOG_SEVERITY_NUMBERS["INFO"]
    assert output["event_name"] == "ticket_agent.workflow.succeeded"
    assert output["body"] == "Ticket agent workflow succeeded."
    assert output["attributes"]["operation"] == "invoke_thread"
    assert output["attributes"]["status"] == "succeeded"
    assert output["attributes"]["app_trace_id"] == (
        "8b0e715c76c8423e9dc95b6c8db8409a"
    )
    assert output["attributes"]["thread_id"] == "ticket-thread-001"
    assert output["attributes"]["actor_id"] == "demo_user_001"
    assert output["attributes"]["agent.intent"] == "ticket_request"
    assert output["attributes"]["ticket.creation_status"] == "succeeded"
    assert output["attributes"]["ticket.write_safety_status"] == "safe"
    assert output["attributes"]["agent.node_last"] == "create_ticket"
    assert output["attributes"]["elapsed_ms"] == 42.13


def test_log_record_keeps_resource_fields_separate_from_event_attributes() -> None:
    record = build_ticket_agent_production_log_record(
        {"agent_trace_id": "8b0e715c76c8423e9dc95b6c8db8409a"},
        event_name="ticket_agent.workflow.started",
        operation="invoke_thread",
        body="Ticket agent workflow started.",
        thread_id="ticket-thread-001",
        span_id="5fb397be34d26b51",
        service_version="0.2.0",
        environment="local",
    )

    output = record.to_otel_log_record()

    assert output["resource"] == {
        "service.name": "ai-service",
        "service.namespace": "java-python-ai",
        "deployment.environment.name": "local",
        "service.version": "0.2.0",
    }
    assert "operation" not in output["resource"]
    assert output["attributes"]["operation"] == "invoke_thread"


def test_event_name_validation_rejects_dynamic_or_unqualified_names() -> None:
    assert (
        validate_ticket_agent_event_name("ticket_agent.workflow.failed")
        == "ticket_agent.workflow.failed"
    )

    invalid_event_names = [
        "ticket_agent",
        "ticket_agent.workflow.ORDER123",
        "ticket_agent.workflow.{order_id}",
        "ticket_agent.workflow failed",
        "ticket_agent.workflow/failed",
    ]
    for event_name in invalid_event_names:
        with pytest.raises(ValueError, match="event_name"):
            validate_ticket_agent_event_name(event_name)


def test_extra_attributes_reject_sensitive_payload_fields() -> None:
    with pytest.raises(ValueError, match="api_key"):
        build_ticket_agent_production_log_record(
            {"agent_trace_id": "8b0e715c76c8423e9dc95b6c8db8409a"},
            event_name="ticket_agent.workflow.started",
            operation="invoke_thread",
            body="Ticket agent workflow started.",
            span_id="5fb397be34d26b51",
            extra_attributes={"api_key": "redacted"},
        )

    with pytest.raises(ValueError, match="user_message"):
        build_ticket_agent_production_log_record(
            {"agent_trace_id": "8b0e715c76c8423e9dc95b6c8db8409a"},
            event_name="ticket_agent.workflow.started",
            operation="invoke_thread",
            body="Ticket agent workflow started.",
            span_id="5fb397be34d26b51",
            extra_attributes={"user_message": "包含手机号的用户原文"},
        )


def test_state_payload_fields_are_not_copied_to_production_log_record() -> None:
    record = build_ticket_agent_production_log_record(
        {
            "agent_trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
            "intent": "order_query",
            "user_message": "我的手机号是 13800000000",
            "final_answer": "完整模型回答",
            "order_query_result": {"receiver_phone": "13800000000"},
            "ticket_fields": {"description": "原始问题"},
            "node_history": ["normalize_user_input", "query_order"],
        },
        event_name="ticket_agent.workflow.succeeded",
        operation="invoke_thread",
        body="Ticket agent workflow succeeded.",
        thread_id="ticket-thread-001",
        span_id="5fb397be34d26b51",
    )

    output = record.to_otel_log_record()
    serialized_values = str(output)

    assert output["attributes"]["agent.intent"] == "order_query"
    assert find_forbidden_ticket_agent_log_fields(output) == []
    assert "13800000000" not in serialized_values
    assert "完整模型回答" not in serialized_values
    assert "原始问题" not in serialized_values


def test_error_record_uses_error_severity_and_stable_error_fields() -> None:
    record = build_ticket_agent_production_log_record(
        {
            "agent_trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
            "intent": "order_query",
            "agent_error_code": "ORDER_QUERY_TIMEOUT",
            "agent_error_node": "query_order",
            "fallback_used": True,
            "node_history": ["normalize_user_input", "query_order"],
        },
        event_name="ticket_agent.workflow.failed",
        operation="invoke_thread",
        body="Ticket agent workflow failed.",
        thread_id="ticket-thread-001",
        span_id="5fb397be34d26b51",
        elapsed_ms=305.129,
    )

    output = record.to_otel_log_record()

    assert output["severity_text"] == "ERROR"
    assert output["severity_number"] == TICKET_AGENT_LOG_SEVERITY_NUMBERS["ERROR"]
    assert output["attributes"]["status"] == "failed"
    assert output["attributes"]["error_code"] == "ORDER_QUERY_TIMEOUT"
    assert output["attributes"]["error_node"] == "query_order"
    assert output["attributes"]["agent.fallback_used"] is True
    assert output["attributes"]["elapsed_ms"] == 305.13


def test_log_field_specs_document_required_correlation_and_forbidden_fields() -> None:
    specs = build_ticket_agent_log_field_specs()
    specs_by_name = {spec.name: spec for spec in specs}

    assert specs_by_name["trace_id"].placement == "top_level"
    assert specs_by_name["trace_id"].required is True
    assert specs_by_name["span_id"].category == "correlation"
    assert specs_by_name["thread_id"].placement == "attribute"
    assert specs_by_name["service.name"].placement == "resource"
    assert specs_by_name["event_name"].category == "event_identity"
    assert specs_by_name["user_message"].placement == "forbidden"
    assert specs_by_name["final_answer"].placement == "forbidden"
    assert specs_by_name["order_query_result"].placement == "forbidden"


def test_find_forbidden_log_fields_scans_nested_dicts() -> None:
    fields = {
        "trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
        "attributes": {
            "operation": "invoke_thread",
            "password": "redacted",
            "nested": {"refresh_token": "redacted"},
        },
        "resource": {"service.name": "ai-service"},
    }

    assert find_forbidden_ticket_agent_log_fields(fields) == [
        "attributes.nested.refresh_token",
        "attributes.password",
    ]


def test_log_severity_normalization_accepts_warn_alias() -> None:
    assert normalize_ticket_agent_log_severity("WARN") == "WARNING"
    assert normalize_ticket_agent_log_severity(None, has_error=False) == "INFO"
    assert normalize_ticket_agent_log_severity(None, has_error=True) == "ERROR"

    with pytest.raises(ValueError, match="Unsupported"):
        normalize_ticket_agent_log_severity("LOUD")
