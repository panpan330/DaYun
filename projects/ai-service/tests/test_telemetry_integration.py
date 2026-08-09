"""Integration test: span parent/child chains across the span helpers.

Uses an in-memory TracerProvider + exporter (no real Collector, no OTLP
network export) per project constraints.  Verifies the span tree shape that
the production wiring (Task 4) relies on: agent.invoke -> tool.call ->
java.call, and http.request as the outermost span.
"""

from typing import Generator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.agents.tracing_spans import (
    start_agent_span,
    start_http_span,
    start_java_span,
    start_tool_span,
)


def _reset_tracer_provider() -> None:
    # trace.set_tracer_provider is global AND guarded by a once-lock
    # (_TRACER_PROVIDER_SET_ONCE): after the first call it no-ops with a warning,
    # so resetting only _TRACER_PROVIDER is not enough. The public API cannot
    # clear the provider (set_tracer_provider(None) raises), so touch privates.
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    once = getattr(trace, "_TRACER_PROVIDER_SET_ONCE", None)
    if once is not None:
        once._done = False  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_tracer_provider() -> Generator[None, None, None]:
    # Reset the global provider state before and after each test so tests are
    # order-independent and each can install a fresh SDK TracerProvider.
    _reset_tracer_provider()
    yield
    _reset_tracer_provider()


def test_span_parent_child_chain() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    with start_agent_span(intent="order_query", thread_id="t-1", conversation_id="c-1"):
        with start_tool_span(tool_name="query_order"):
            with start_java_span(path="/internal/orders/A1001", method="GET"):
                pass

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    assert "agent.invoke" in by_name
    assert "tool.call" in by_name
    assert "java.call" in by_name

    tool_span = by_name["tool.call"]
    agent_span = by_name["agent.invoke"]
    assert tool_span.parent.span_id == agent_span.context.span_id

    java_span = by_name["java.call"]
    assert java_span.parent.span_id == tool_span.context.span_id


def test_http_span_is_outermost_parent_of_agent_span() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    with start_http_span(method="POST", path="/api/v1/console-agent/reply"):
        with start_agent_span(intent=None, thread_id="t-1", conversation_id="c-1"):
            pass

    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    assert "http.request" in by_name
    assert "agent.invoke" in by_name
    http_span = by_name["http.request"]
    assert dict(http_span.attributes)["method"] == "POST"
    assert dict(http_span.attributes)["path"] == "/api/v1/console-agent/reply"
    agent_span = by_name["agent.invoke"]
    assert agent_span.parent.span_id == http_span.context.span_id
