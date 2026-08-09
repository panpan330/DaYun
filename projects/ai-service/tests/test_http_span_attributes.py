"""Telemetry tests for the http.request span attributes.

Verifies that the tracing middleware attaches ``status_code`` and
``duration_ms`` to the ``http.request`` span after the response is produced,
and marks 4xx/5xx responses as errored (spec 3.1: http.request major
attributes include status_code / duration_ms). Uses an in-memory exporter and
the real TestClient pipeline (middleware → route), no OTLP network.
"""

from typing import Generator

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.telemetry import shutdown_telemetry


def _reset_tracer_provider() -> None:
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    once = getattr(trace, "_TRACER_PROVIDER_SET_ONCE", None)
    if once is not None:
        once._done = False  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_tracer_provider() -> None:
    _reset_tracer_provider()
    yield
    _reset_tracer_provider()


@pytest.fixture()
def otel_exporter() -> Generator[InMemorySpanExporter, None, None]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    shutdown_telemetry()


def _http_span(exporter: InMemorySpanExporter):
    spans = exporter.get_finished_spans()
    http_spans = [s for s in spans if s.name == "http.request"]
    assert len(http_spans) == 1
    return http_spans[0]


def test_http_request_span_has_status_code_and_duration_ms(
    client: TestClient,
    otel_exporter: InMemorySpanExporter,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    span = _http_span(otel_exporter)
    attrs = dict(span.attributes)
    assert attrs["method"] == "GET"
    assert attrs["path"] == "/health"
    assert attrs["status_code"] == 200
    assert "duration_ms" in attrs
    assert attrs["duration_ms"] >= 0
    # 200 不应标记为错误
    assert span.status.status_code == trace.StatusCode.UNSET


def test_http_request_span_marks_4xx_as_errored(
    client: TestClient,
    otel_exporter: InMemorySpanExporter,
) -> None:
    response = client.get("/api/v1/definitely-not-a-real-route")

    assert response.status_code == 404
    span = _http_span(otel_exporter)
    attrs = dict(span.attributes)
    assert attrs["status_code"] == 404
    assert "duration_ms" in attrs
    assert span.status.status_code == trace.StatusCode.ERROR
