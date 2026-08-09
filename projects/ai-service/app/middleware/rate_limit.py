import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from app.core.config import Settings
from app.core.exception_handlers import build_error_response
from app.core.rate_limit import (
    InMemoryFixedWindowRateLimiter,
    RateLimitDecision,
    build_rate_limit_checks,
)
from app.core.trace import (
    DEFAULT_TRACE_ID,
    TRACE_ID_HEADER,
    get_or_create_trace_id,
    get_trace_id,
    reset_trace_id,
    set_trace_id,
)


logger = logging.getLogger(__name__)
USER_ID_HEADER = "X-User-Id"
FORWARDED_FOR_HEADER = "X-Forwarded-For"


def register_rate_limit_middleware(app: FastAPI, settings: Settings) -> None:
    limiter = InMemoryFixedWindowRateLimiter()
    app.state.rate_limiter = limiter

    @app.middleware("http")
    async def rate_limit_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        trace_id = _resolve_request_trace_id(request)
        client_identity = _resolve_client_identity(request)
        checks = build_rate_limit_checks(
            settings,
            path=request.url.path,
            method=request.method,
            client_identity=client_identity,
        )

        for check in checks:
            decision = limiter.check(check)
            if not decision.allowed:
                token = set_trace_id(trace_id)
                try:
                    logger.warning(
                        (
                            "rate_limited scope=%s key=%s path=%s "
                            "limit=%s retry_after=%s"
                        ),
                        decision.scope,
                        decision.key,
                        request.url.path,
                        decision.limit,
                        decision.retry_after_seconds,
                    )
                    return _build_rate_limited_response(decision, trace_id)
                finally:
                    reset_trace_id(token)

        return await call_next(request)


def _build_rate_limited_response(
    decision: RateLimitDecision,
    trace_id: str,
) -> Response:
    return build_error_response(
        status_code=429,
        code="RATE_LIMITED",
        message="请求过于频繁，请稍后重试。",
        headers=decision.to_headers(),
        trace_id=trace_id,
    )


def _resolve_request_trace_id(request: Request) -> str:
    state_trace_id = getattr(request.state, "trace_id", None)
    if state_trace_id:
        return state_trace_id
    current_trace_id = get_trace_id()
    if current_trace_id != DEFAULT_TRACE_ID:
        return current_trace_id
    return get_or_create_trace_id(request.headers.get(TRACE_ID_HEADER))


def _resolve_client_identity(request: Request) -> str:
    user_id = request.headers.get(USER_ID_HEADER)
    if user_id and user_id.strip():
        return f"user:{user_id.strip()}"

    forwarded_for = request.headers.get(FORWARDED_FOR_HEADER)
    if forwarded_for and forwarded_for.strip():
        first_ip = forwarded_for.split(",", maxsplit=1)[0].strip()
        if first_ip:
            return f"ip:{first_ip}"

    if request.client is not None and request.client.host:
        return f"ip:{request.client.host}"
    return "ip:unknown"
