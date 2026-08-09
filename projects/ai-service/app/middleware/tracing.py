import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from app.core.request_timing import build_total_http_request_timing_breakdown
from app.core.telemetry import is_telemetry_enabled
from app.core.trace import (
    TRACE_ID_HEADER,
    get_or_create_trace_id,
    reset_trace_id,
    set_trace_id,
)


logger = logging.getLogger(__name__)


def register_trace_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def trace_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        trace_id = get_or_create_trace_id(request.headers.get(TRACE_ID_HEADER))
        request.state.trace_id = trace_id
        token = set_trace_id(trace_id)
        start_time = time.perf_counter()

        logger.info(
            "request_started method=%s path=%s",
            request.method,
            request.url.path,
        )

        try:
            if is_telemetry_enabled():
                from app.agents.tracing_spans import (
                    set_span_attributes,
                    set_span_status_error,
                    start_http_span,
                )

                with start_http_span(
                    method=request.method,
                    path=request.url.path,
                ):
                    response = await _run_request_body(
                        request, call_next, trace_id, start_time
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    set_span_attributes(
                        {
                            "status_code": response.status_code,
                            "duration_ms": elapsed_ms,
                        }
                    )
                    if response.status_code >= 400:
                        set_span_status_error()
                    return response
            return await _run_request_body(request, call_next, trace_id, start_time)
        finally:
            reset_trace_id(token)


async def _run_request_body(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
    trace_id: str,
    start_time: float,
) -> Response:
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "request_failed method=%s path=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response.headers[TRACE_ID_HEADER] = trace_id
    timing = build_total_http_request_timing_breakdown(
        route=request.url.path,
        method=request.method,
        total_elapsed_ms=elapsed_ms,
        status_code=response.status_code,
        trace_id=trace_id,
    )
    timing_fields = timing.to_log_fields()
    logger.info(
        (
            "request_finished method=%s path=%s status_code=%s "
            "elapsed_ms=%.2f bottleneck_stage=%s "
            "bottleneck_elapsed_ms=%s"
        ),
        request.method,
        request.url.path,
        response.status_code,
        timing.total_elapsed_ms,
        timing_fields.get("request.bottleneck_stage"),
        timing_fields.get("request.bottleneck_elapsed_ms"),
    )
    return response
