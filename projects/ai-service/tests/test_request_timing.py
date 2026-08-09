import pytest

from app.core.request_timing import (
    build_request_timing_breakdown,
    build_request_timing_stage,
    build_total_http_request_timing_breakdown,
    infer_request_timing_flow,
)
from app.core.trace import reset_trace_id, set_trace_id


def test_request_timing_breakdown_calculates_measured_unaccounted_and_bottleneck() -> None:
    breakdown = build_request_timing_breakdown(
        flow="rag_answer",
        route="/rag/query",
        method="post",
        total_elapsed_ms=1000.0,
        trace_id="trace-timing-001",
        stages=[
            build_request_timing_stage(
                "request.validation",
                elapsed_ms=30.0,
            ),
            build_request_timing_stage(
                "vector.search",
                kind="client",
                elapsed_ms=220.0,
                attributes={"vector.store": "qdrant"},
            ),
            build_request_timing_stage(
                "llm.final_answer",
                kind="client",
                elapsed_ms=600.0,
                attributes={"llm.model": "qwen3.7-plus"},
            ),
        ],
    )

    assert breakdown.trace_id == "trace-timing-001"
    assert breakdown.method == "POST"
    assert breakdown.measured_elapsed_ms == 850.0
    assert breakdown.unaccounted_elapsed_ms == 150.0
    assert breakdown.bottleneck_stage is not None
    assert breakdown.bottleneck_stage.name == "llm.final_answer"
    assert breakdown.stage_percent(breakdown.bottleneck_stage) == 60.0

    fields = breakdown.to_log_fields()
    assert fields["app.flow"] == "rag_answer"
    assert fields["http.route"] == "/rag/query"
    assert fields["request.stage_count"] == 3
    assert fields["request.bottleneck_stage"] == "llm.final_answer"
    assert fields["request.bottleneck_elapsed_ms"] == 600.0
    assert fields["request.bottleneck_percent"] == 60.0


def test_request_timing_filters_sensitive_and_high_cardinality_attributes() -> None:
    stage = build_request_timing_stage(
        "llm.call",
        kind="client",
        elapsed_ms=123.456,
        attributes={
            "llm.model": "qwen3.7-plus",
            "user_message": "private question",
            "Authorization": "Bearer secret",
            "trace_id": "trace-high-cardinality",
            "prompt": "full prompt",
            "retry.count": 2,
            "app.flow": "wrong-flow",
            "request.total_elapsed_ms": 999,
            "empty": "   ",
            "complex": {"not": "safe"},
        },
    )

    assert stage.elapsed_ms == 123.46
    assert stage.attributes == {
        "llm.model": "qwen3.7-plus",
        "retry.count": 2,
    }


def test_total_http_request_timing_breakdown_uses_route_flow_and_status() -> None:
    breakdown = build_total_http_request_timing_breakdown(
        route="/tool-chat",
        method="POST",
        total_elapsed_ms=42.345,
        status_code=200,
        trace_id="trace-http-total",
    )

    fields = breakdown.to_log_fields()

    assert breakdown.flow == "tool_chat"
    assert breakdown.total_elapsed_ms == 42.35
    assert breakdown.status == "ok"
    assert fields["request.bottleneck_stage"] == "http.request"
    assert fields["request.bottleneck_elapsed_ms"] == 42.35
    assert fields["http.status_code"] == 200


def test_total_http_request_timing_breakdown_marks_server_error() -> None:
    breakdown = build_total_http_request_timing_breakdown(
        route="/chat",
        method="POST",
        total_elapsed_ms=10,
        status_code=502,
    )

    assert breakdown.status == "error"
    assert breakdown.stages[0].status == "error"


def test_request_timing_reuses_current_trace_id() -> None:
    token = set_trace_id("current-request-timing-trace")
    try:
        breakdown = build_request_timing_breakdown(
            flow="chat",
            route="/chat",
            method="POST",
            total_elapsed_ms=10,
            stages=[],
        )
    finally:
        reset_trace_id(token)

    assert breakdown.trace_id == "current-request-timing-trace"


def test_request_timing_rejects_invalid_stage_name_or_elapsed() -> None:
    with pytest.raises(ValueError, match="stage name"):
        build_request_timing_stage("  ", elapsed_ms=1)

    with pytest.raises(ValueError, match="elapsed_ms"):
        build_request_timing_stage("llm.call", elapsed_ms=-1)

    with pytest.raises(ValueError, match="total_elapsed_ms"):
        build_request_timing_breakdown(
            flow="chat",
            route="/chat",
            method="POST",
            total_elapsed_ms=float("inf"),
            stages=[],
        )


@pytest.mark.parametrize(
    ("route", "expected_flow"),
    [
        ("/chat", "chat"),
        ("/stream-chat", "stream_chat"),
        ("/rag/query", "rag_answer"),
        ("/tool-chat", "tool_chat"),
        ("/health", "health"),
        ("/unknown", "unknown"),
    ],
)
def test_infer_request_timing_flow(route: str, expected_flow: str) -> None:
    assert infer_request_timing_flow(route) == expected_flow
