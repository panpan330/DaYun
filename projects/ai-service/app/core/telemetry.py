"""Telemetry setup: OpenTelemetry SDK + conditional LangSmith tracing."""

import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import Settings, get_settings


logger = logging.getLogger(__name__)

_DEFAULT_SERVICE_NAME = "ai-service"


def _build_resource(service_name: str) -> Resource:
    return Resource.create({"service.name": service_name})


def _configure_langsmith(settings: Settings) -> None:
    """Set LangSmith env vars when the feature is enabled (idempotent)."""
    if not settings.langsmith_enabled:
        return
    api_key = settings.resolved_langsmith_api_key
    if api_key is None:
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.otel_service_name or _DEFAULT_SERVICE_NAME)
    logger.info(
        "langsmith_tracing_enabled project=%s",
        settings.otel_service_name or _DEFAULT_SERVICE_NAME,
    )


def setup_telemetry(settings: Settings | None = None) -> None:
    """Initialize OTEL SDK and conditional LangSmith tracing.

    Idempotent: calling twice does not rebuild the provider.
    Failures are logged but never raised (service must start regardless).

    LangSmith and OTEL are independent switches: LangSmith tracing is
    configured whenever enabled, regardless of the OTEL exporter endpoint.
    """
    resolved_settings = settings or get_settings()
    service_name = resolved_settings.otel_service_name or _DEFAULT_SERVICE_NAME
    endpoint = resolved_settings.resolved_otel_exporter_otlp_endpoint

    # Configure LangSmith first: it does not depend on the OTEL endpoint.
    try:
        _configure_langsmith(resolved_settings)
    except Exception:
        logger.exception("langsmith_setup_failed")

    if endpoint is None:
        logger.info("otel_export_disabled endpoint not configured")
        return

    try:
        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            # Already set up — keep the existing provider (idempotent).
            logger.info("otel_export_already_enabled endpoint=%s service_name=%s", endpoint, service_name)
        else:
            provider = TracerProvider(resource=_build_resource(service_name))
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            logger.info("otel_export_enabled endpoint=%s service_name=%s", endpoint, service_name)
    except Exception:
        logger.exception("otel_setup_failed endpoint=%s", endpoint)


def get_tracer() -> Any:
    return trace.get_tracer(_DEFAULT_SERVICE_NAME)


def shutdown_telemetry() -> None:
    """Flush and shut down the tracer provider (idempotent, never raises)."""
    try:
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.shutdown()
    except Exception:
        logger.exception("otel_shutdown_failed")


def is_telemetry_enabled() -> bool:
    provider = trace.get_tracer_provider()
    return isinstance(provider, TracerProvider)


def get_otel_trace_id() -> str | None:
    """Return the current span's OTEL trace_id as hex, or None."""
    span = trace.get_current_span()
    if span is None or not span.get_span_context().is_valid:
        return None
    return format(span.get_span_context().trace_id, "032x")
