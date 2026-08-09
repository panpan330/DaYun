import logging
import os
from typing import Any, Generator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from app.core.config import Settings
from app.core.telemetry import (
    get_otel_trace_id,
    get_tracer,
    is_telemetry_enabled,
    setup_telemetry,
    shutdown_telemetry,
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


def test_telemetry_disabled_when_endpoint_empty() -> None:
    settings = Settings(_env_file=None)  # otel_exporter_otlp_endpoint=""
    assert is_telemetry_enabled() is False


def test_telemetry_enabled_when_endpoint_configured() -> None:
    settings = Settings(
        _env_file=None,
        otel_exporter_otlp_endpoint="http://localhost:4317",
    )
    setup_telemetry(settings)
    assert is_telemetry_enabled() is True
    tracer = get_tracer()
    assert tracer is not None
    shutdown_telemetry()


def test_setup_is_idempotent() -> None:
    settings = Settings(
        _env_file=None,
        otel_exporter_otlp_endpoint="http://localhost:4317",
    )
    setup_telemetry(settings)
    provider1 = trace.get_tracer_provider()
    # Must be a real SDK TracerProvider (not the default ProxyTracerProvider).
    assert isinstance(provider1, TracerProvider)
    processors_before = len(provider1._active_span_processor._span_processors)
    setup_telemetry(settings)
    provider2 = trace.get_tracer_provider()
    # Second setup must not rebuild: same instance, same span processors.
    assert isinstance(provider2, TracerProvider)
    assert provider1 is provider2
    assert len(provider2._active_span_processor._span_processors) == processors_before
    shutdown_telemetry()


def test_get_otel_trace_id_returns_none_without_span() -> None:
    assert get_otel_trace_id() is None


def test_setup_failure_does_not_raise() -> None:
    # endpoint 指向不可达地址，setup 不应抛异常
    settings = Settings(
        _env_file=None,
        otel_exporter_otlp_endpoint="http://127.0.0.1:1",
    )
    setup_telemetry(settings)  # 不应 raise
    shutdown_telemetry()


def test_langsmith_configured_without_otel_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # LangSmith is an independent switch: it must be configured even when the
    # OTEL exporter endpoint is empty (OTEL disabled).
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGCHAIN_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(
        _env_file=None,
        langsmith_tracing=True,
        langsmith_api_key="test-key",
        otel_exporter_otlp_endpoint="",
    )
    setup_telemetry(settings)
    assert os.environ.get("LANGSMITH_TRACING") == "true"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGSMITH_API_KEY") == "test-key"
    assert os.environ.get("LANGCHAIN_PROJECT") == "ai-service"
    # OTEL remains disabled (endpoint empty): provider was never installed.
    assert is_telemetry_enabled() is False


class _FailingExporter:
    """Exporter stub whose construction always raises (real failure path)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("exporter construction failed")


def test_setup_exporter_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # Exporter construction raises: setup must not raise and must log otel_setup_failed.
    monkeypatch.setattr("app.core.telemetry.OTLPSpanExporter", _FailingExporter)
    settings = Settings(
        _env_file=None,
        otel_exporter_otlp_endpoint="http://127.0.0.1:1",
    )
    with caplog.at_level(logging.ERROR, logger="app.core.telemetry"):
        setup_telemetry(settings)  # 不应 raise
    assert any("otel_setup_failed" in record.message for record in caplog.records)
    # Provider was never installed.
    assert is_telemetry_enabled() is False
    shutdown_telemetry()
