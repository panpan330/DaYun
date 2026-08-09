from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.rate_limit import (
    InMemoryFixedWindowRateLimiter,
    RateLimitCheck,
    RateLimitRule,
    build_rate_limit_checks,
    is_ai_route,
    is_tool_route,
    parse_rate_limit_excluded_paths,
)
from app.core.trace import TRACE_ID_HEADER
from app.main import create_app


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_fixed_window_rate_limiter_allows_until_limit_then_blocks() -> None:
    clock = FakeClock(now=100.0)
    limiter = InMemoryFixedWindowRateLimiter(now_func=clock)
    check = RateLimitCheck(
        rule=RateLimitRule(scope="client", limit=2, window_seconds=60),
        identity="user:u1001",
    )

    first = limiter.check(check)
    second = limiter.check(check)
    third = limiter.check(check)

    assert first.allowed is True
    assert first.remaining == 1
    assert second.allowed is True
    assert second.remaining == 0
    assert third.allowed is False
    assert third.retry_after_seconds > 0
    assert third.to_headers()["X-RateLimit-Scope"] == "client"


def test_fixed_window_rate_limiter_resets_when_window_changes() -> None:
    clock = FakeClock(now=100.0)
    limiter = InMemoryFixedWindowRateLimiter(now_func=clock)
    check = RateLimitCheck(
        rule=RateLimitRule(scope="route", limit=1, window_seconds=10),
        identity="GET:/chat",
    )

    assert limiter.check(check).allowed is True
    assert limiter.check(check).allowed is False

    clock.now = 110.0

    assert limiter.check(check).allowed is True


def test_build_rate_limit_checks_respects_excluded_paths_and_disabled_limits() -> None:
    settings = Settings(
        rate_limit_client_requests_per_window=0,
        rate_limit_route_requests_per_window=2,
        rate_limit_ai_requests_per_window=3,
        rate_limit_tool_requests_per_window=4,
        rate_limit_excluded_paths="/health,/ready",
        _env_file=None,
    )

    assert build_rate_limit_checks(
        settings,
        path="/health",
        method="GET",
        client_identity="ip:test",
    ) == []

    checks = build_rate_limit_checks(
        settings,
        path="/tool-chat",
        method="POST",
        client_identity="user:u1001",
    )

    assert [check.rule.scope for check in checks] == ["route", "ai", "tool"]


def test_rate_limit_route_classification_helpers() -> None:
    assert is_ai_route("/chat") is True
    assert is_ai_route("/rag/query") is True
    assert is_tool_route("/tool-chat") is True
    assert is_tool_route("/tools/list") is True
    assert is_ai_route("/health") is False
    assert is_tool_route("/health") is False
    assert parse_rate_limit_excluded_paths("/health，/ready") == frozenset(
        {"/health", "/ready"}
    )


def test_rate_limit_middleware_returns_429_with_retry_headers() -> None:
    app = create_app(
        Settings(
            rate_limit_window_seconds=60,
            rate_limit_client_requests_per_window=2,
            rate_limit_route_requests_per_window=0,
            rate_limit_ai_requests_per_window=0,
            rate_limit_tool_requests_per_window=0,
            rate_limit_excluded_paths="",
            _env_file=None,
        )
    )
    client = TestClient(app)

    first = client.get("/health", headers={"X-User-Id": "u1001"})
    second = client.get("/health", headers={"X-User-Id": "u1001"})
    third = client.get("/health", headers={"X-User-Id": "u1001"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["code"] == "RATE_LIMITED"
    assert third.headers["Retry-After"]
    assert third.headers["X-RateLimit-Scope"] == "client"
    assert third.headers[TRACE_ID_HEADER]


def test_rate_limit_middleware_uses_separate_user_identities() -> None:
    app = create_app(
        Settings(
            rate_limit_client_requests_per_window=1,
            rate_limit_route_requests_per_window=0,
            rate_limit_ai_requests_per_window=0,
            rate_limit_tool_requests_per_window=0,
            rate_limit_excluded_paths="",
            _env_file=None,
        )
    )
    client = TestClient(app)

    first_user = client.get("/health", headers={"X-User-Id": "u1001"})
    second_user = client.get("/health", headers={"X-User-Id": "u2002"})
    blocked_first_user = client.get("/health", headers={"X-User-Id": "u1001"})

    assert first_user.status_code == 200
    assert second_user.status_code == 200
    assert blocked_first_user.status_code == 429
