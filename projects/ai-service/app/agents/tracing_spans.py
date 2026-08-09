"""Real OTEL span helpers translating the plan dataclasses into actual spans.

The plan dataclasses in otel_tracing.py / langsmith_tracing.py remain for
learning/tests; this module is the production adapter.
"""

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace

from app.core.telemetry import get_otel_trace_id, get_tracer, is_telemetry_enabled
from app.core.trace import get_trace_id


def _current_span() -> Any:
    return trace.get_current_span()


def _set_common_attributes() -> None:
    """Attach the app's X-Trace-Id to the current span for log/span correlation.

    Mirrors build_trace_headers(): skip when the X-Trace-Id is the default "-".
    The attribute is named ``x_trace_id`` to avoid clashing with the native OTEL
    ``trace_id`` span attribute (the OTEL trace id is stored as ``otel_trace_id``).
    """
    x_trace_id = get_trace_id()
    if x_trace_id and x_trace_id != "-":
        set_span_attributes({"x_trace_id": x_trace_id})
    otel_trace_id = get_otel_trace_id()
    if otel_trace_id:
        set_span_attributes({"otel_trace_id": otel_trace_id})


def set_span_attributes(attributes: dict[str, Any]) -> None:
    span = _current_span()
    if span is None or not span.is_recording():
        return
    span.set_attributes(attributes)


def set_span_status_error() -> None:
    """Mark the current span as errored (used on handled-failure paths).

    Pure instrumentation: does not alter control flow.  Used where a failure is
    caught and converted into a structured error (e.g. SSE error events) so the
    span still surfaces the failure on the trace tree.
    """
    span = _current_span()
    if span is None or not span.is_recording():
        return
    span.set_status(trace.Status(trace.StatusCode.ERROR))


@contextmanager
def start_agent_span(
    *,
    intent: str | None = None,
    thread_id: str | None = None,
    conversation_id: str | None = None,
) -> Iterator[None]:
    if not is_telemetry_enabled():
        yield
        return
    tracer = get_tracer()
    with tracer.start_as_current_span("agent.invoke") as span:
        attrs: dict[str, Any] = {}
        if intent is not None:
            attrs["intent"] = intent
        if thread_id is not None:
            attrs["thread_id"] = thread_id
        if conversation_id is not None:
            attrs["conversation_id"] = conversation_id
        span.set_attributes(attrs)
        _set_common_attributes()
        yield


@contextmanager
def start_llm_span(
    *,
    model: str,
    provider: str,
    prompt_name: str | None = None,
) -> Iterator[None]:
    if not is_telemetry_enabled():
        yield
        return
    tracer = get_tracer()
    with tracer.start_as_current_span("llm.call") as span:
        attrs: dict[str, Any] = {"model": model, "provider": provider}
        if prompt_name is not None:
            attrs["prompt_name"] = prompt_name
        span.set_attributes(attrs)
        _set_common_attributes()
        yield


@contextmanager
def start_tool_span(*, tool_name: str) -> Iterator[None]:
    if not is_telemetry_enabled():
        yield
        return
    tracer = get_tracer()
    with tracer.start_as_current_span("tool.call") as span:
        span.set_attribute("tool_name", tool_name)
        _set_common_attributes()
        yield


@contextmanager
def start_java_span(*, path: str, method: str = "GET") -> Iterator[None]:
    if not is_telemetry_enabled():
        yield
        return
    tracer = get_tracer()
    with tracer.start_as_current_span("java.call") as span:
        span.set_attributes({"path": path, "method": method})
        _set_common_attributes()
        yield


@contextmanager
def start_http_span(*, method: str, path: str) -> Iterator[None]:
    if not is_telemetry_enabled():
        yield
        return
    tracer = get_tracer()
    with tracer.start_as_current_span("http.request") as span:
        span.set_attributes({"method": method, "path": path})
        _set_common_attributes()
        yield
