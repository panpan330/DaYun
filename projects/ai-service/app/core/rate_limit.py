from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings


RateLimitScope = Literal["client", "route", "ai", "tool"]


@dataclass(frozen=True)
class RateLimitRule:
    scope: RateLimitScope
    limit: int
    window_seconds: int

    @property
    def enabled(self) -> bool:
        return self.limit > 0 and self.window_seconds > 0


@dataclass(frozen=True)
class RateLimitCheck:
    rule: RateLimitRule
    identity: str

    @property
    def key(self) -> str:
        return f"{self.rule.scope}:{self.identity}"


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    scope: RateLimitScope
    key: str
    limit: int
    remaining: int
    reset_after_seconds: int
    retry_after_seconds: int

    def to_headers(self) -> dict[str, str]:
        return {
            "Retry-After": str(self.retry_after_seconds),
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset-After": str(self.reset_after_seconds),
            "X-RateLimit-Scope": self.scope,
        }


@dataclass
class _FixedWindowCounter:
    window_start: int
    count: int = 0


class InMemoryFixedWindowRateLimiter:
    def __init__(self, now_func: Callable[[], float] | None = None) -> None:
        self._now_func = now_func or time.time
        self._counters: dict[str, _FixedWindowCounter] = {}
        self._lock = threading.Lock()

    def check(self, check: RateLimitCheck) -> RateLimitDecision:
        rule = check.rule
        if not rule.enabled:
            return RateLimitDecision(
                allowed=True,
                scope=rule.scope,
                key=check.key,
                limit=rule.limit,
                remaining=max(rule.limit, 0),
                reset_after_seconds=0,
                retry_after_seconds=0,
            )

        now = self._now_func()
        window_start = int(now // rule.window_seconds) * rule.window_seconds
        reset_after_seconds = max(1, math.ceil(window_start + rule.window_seconds - now))

        with self._lock:
            counter = self._counters.get(check.key)
            if counter is None or counter.window_start != window_start:
                counter = _FixedWindowCounter(window_start=window_start)
                self._counters[check.key] = counter

            if counter.count >= rule.limit:
                return RateLimitDecision(
                    allowed=False,
                    scope=rule.scope,
                    key=check.key,
                    limit=rule.limit,
                    remaining=0,
                    reset_after_seconds=reset_after_seconds,
                    retry_after_seconds=reset_after_seconds,
                )

            counter.count += 1
            return RateLimitDecision(
                allowed=True,
                scope=rule.scope,
                key=check.key,
                limit=rule.limit,
                remaining=max(0, rule.limit - counter.count),
                reset_after_seconds=reset_after_seconds,
                retry_after_seconds=0,
            )

    def clear(self) -> None:
        with self._lock:
            self._counters.clear()


def build_rate_limit_checks(
    settings: Settings,
    *,
    path: str,
    method: str,
    client_identity: str,
) -> list[RateLimitCheck]:
    if not settings.rate_limit_enabled:
        return []
    if path in parse_rate_limit_excluded_paths(settings.rate_limit_excluded_paths):
        return []

    checks = [
        RateLimitCheck(
            rule=RateLimitRule(
                scope="client",
                limit=settings.rate_limit_client_requests_per_window,
                window_seconds=settings.rate_limit_window_seconds,
            ),
            identity=client_identity,
        ),
        RateLimitCheck(
            rule=RateLimitRule(
                scope="route",
                limit=settings.rate_limit_route_requests_per_window,
                window_seconds=settings.rate_limit_window_seconds,
            ),
            identity=f"{method.upper()}:{path}",
        ),
    ]
    if is_ai_route(path):
        checks.append(
            RateLimitCheck(
                rule=RateLimitRule(
                    scope="ai",
                    limit=settings.rate_limit_ai_requests_per_window,
                    window_seconds=settings.rate_limit_window_seconds,
                ),
                identity=client_identity,
            )
        )
    if is_tool_route(path):
        checks.append(
            RateLimitCheck(
                rule=RateLimitRule(
                    scope="tool",
                    limit=settings.rate_limit_tool_requests_per_window,
                    window_seconds=settings.rate_limit_window_seconds,
                ),
                identity=client_identity,
            )
        )
    return [check for check in checks if check.rule.enabled]


def parse_rate_limit_excluded_paths(raw_paths: str) -> frozenset[str]:
    return frozenset(
        path.strip()
        for path in raw_paths.replace("，", ",").split(",")
        if path.strip()
    )


def is_ai_route(path: str) -> bool:
    return path in {
        "/chat",
        "/stream-chat",
        "/langchain-chat",
        "/extract-ticket",
        "/langchain-extract-ticket",
        "/tool-decision",
        "/tool-chat",
    } or path.startswith("/rag")


def is_tool_route(path: str) -> bool:
    return path in {"/tool-decision", "/tool-chat"} or path.startswith("/tools")
