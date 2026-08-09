"""Runtime dependency probes for the dependency health endpoint.

Each probe returns a ``(status, message, latency_ms)`` tuple where ``status``
is one of ``"ok"``, ``"unreachable"`` or ``"not_configured"``. Probes never
raise: failures are reported as ``unreachable`` so the endpoint can always
answer.
"""

from __future__ import annotations

import socket
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import redis

from app.core.config import Settings

_PROBE_TIMEOUT_SECONDS = 2.0

ProbeResult = tuple[str, str, float | None]


def probe_java_business(settings: Settings) -> ProbeResult:
    base_url = settings.resolved_java_business_service_base_url
    if not base_url:
        return "not_configured", "Java business service base URL is empty.", None
    return _probe_http_get(f"{base_url}/health")


def probe_mcp_product(settings: Settings) -> ProbeResult:
    base_url = settings.resolved_mcp_product_base_url
    if not base_url:
        return "not_configured", "MCP product base URL is empty.", None
    host, port = _host_port_from_url(base_url, settings.mcp_product_port)
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT_SECONDS):
            return "ok", "MCP product server accepts connections.", _elapsed_ms(started)
    except OSError as exc:
        return "unreachable", f"Could not connect to MCP product server: {exc.__class__.__name__}.", None


def probe_redis(settings: Settings) -> ProbeResult:
    redis_url = settings.resolved_agent_redis_url
    if not redis_url:
        return "not_configured", "Agent Redis URL is empty.", None
    started = time.perf_counter()
    try:
        client = redis.Redis.from_url(redis_url, socket_connect_timeout=_PROBE_TIMEOUT_SECONDS)
        client.ping()
        return "ok", "Redis responds to PING.", _elapsed_ms(started)
    except (redis.RedisError, OSError) as exc:
        return "unreachable", f"Redis is unreachable: {exc.__class__.__name__}.", None


def probe_qdrant(settings: Settings) -> ProbeResult:
    base_url = settings.resolved_qdrant_base_url
    if not base_url:
        return "not_configured", "Qdrant base URL is empty.", None
    return _probe_http_get(f"{base_url}/collections")


def _probe_http_get(url: str) -> ProbeResult:
    started = time.perf_counter()
    try:
        response = httpx.get(url, timeout=_PROBE_TIMEOUT_SECONDS)
        if response.status_code < 500:
            return "ok", f"HTTP {response.status_code}.", _elapsed_ms(started)
        return "unreachable", f"HTTP {response.status_code}.", None
    except httpx.HTTPError as exc:
        return "unreachable", f"Request failed: {exc.__class__.__name__}.", None


def _host_port_from_url(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port
    return host, port


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def build_dependency_probes(settings: Settings) -> list[dict[str, Any]]:
    """Run all probes in order and return plain dicts for response serialization."""
    results = [
        ("java_business", probe_java_business(settings)),
        ("mcp_product", probe_mcp_product(settings)),
        ("redis", probe_redis(settings)),
        ("qdrant", probe_qdrant(settings)),
    ]
    probes: list[dict[str, Any]] = []
    for name, (status, message, latency_ms) in results:
        probes.append(
            {
                "name": name,
                "status": status,
                "message": message,
                "latency_ms": latency_ms,
            }
        )
    return probes
