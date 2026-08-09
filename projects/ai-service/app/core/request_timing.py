from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

from app.core.trace import DEFAULT_TRACE_ID, get_trace_id


RequestTimingValue = str | int | float | bool
RequestTimingFlow = Literal[
    "chat",
    "stream_chat",
    "rag_answer",
    "tool_chat",
    "health",
    "unknown",
]
RequestStageKind = Literal["server", "internal", "client"]
RequestStageStatus = Literal["ok", "error", "skipped"]
RequestTimingStatus = Literal["ok", "error"]

REQUEST_TIMING_PROTECTED_KEYS = frozenset(
    {
        "app.trace_id",
        "app.flow",
        "http.route",
        "http.method",
        "request.status",
        "request.total_elapsed_ms",
        "request.stage_count",
        "request.measured_elapsed_ms",
        "request.unaccounted_elapsed_ms",
        "request.bottleneck_stage",
        "request.bottleneck_elapsed_ms",
        "request.bottleneck_percent",
    }
)
REQUEST_TIMING_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "set_cookie",
        "password",
        "secret",
        "token",
        "prompt",
        "raw_prompt",
        "messages",
        "history",
        "user_message",
        "query",
        "final_answer",
        "raw_response",
        "tool_result",
        "document_content",
        "chunk_content",
        "retrieved_documents",
    }
)
REQUEST_TIMING_HIGH_CARDINALITY_KEYS = frozenset(
    {
        "trace_id",
        "span_id",
        "parent_span_id",
        "user_id",
        "actor_id",
        "session_id",
        "thread_id",
        "request_id",
        "order_id",
        "ticket_id",
    }
)

_ATTRIBUTE_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_ATTRIBUTE_KEY_UNSAFE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RequestTimingStage:
    name: str
    kind: RequestStageKind
    elapsed_ms: float
    status: RequestStageStatus = "ok"
    attributes: dict[str, RequestTimingValue] | None = None


@dataclass(frozen=True)
class RequestTimingBreakdown:
    trace_id: str
    flow: RequestTimingFlow
    route: str
    method: str
    status: RequestTimingStatus
    total_elapsed_ms: float
    stages: list[RequestTimingStage]
    attributes: dict[str, RequestTimingValue]

    @property
    def measured_elapsed_ms(self) -> float:
        return round(sum(stage.elapsed_ms for stage in self.stages), 2)

    @property
    def unaccounted_elapsed_ms(self) -> float:
        return round(max(0.0, self.total_elapsed_ms - self.measured_elapsed_ms), 2)

    @property
    def bottleneck_stage(self) -> RequestTimingStage | None:
        measured_stages = [stage for stage in self.stages if stage.status != "skipped"]
        if not measured_stages:
            return None
        return max(measured_stages, key=lambda stage: stage.elapsed_ms)

    def stage_percent(self, stage: RequestTimingStage) -> float | None:
        if self.total_elapsed_ms <= 0:
            return None
        return round(stage.elapsed_ms / self.total_elapsed_ms * 100, 2)

    def to_log_fields(self) -> dict[str, RequestTimingValue]:
        fields: dict[str, RequestTimingValue] = {
            "app.trace_id": self.trace_id,
            "app.flow": self.flow,
            "http.route": self.route,
            "http.method": self.method,
            "request.status": self.status,
            "request.total_elapsed_ms": self.total_elapsed_ms,
            "request.stage_count": len(self.stages),
            "request.measured_elapsed_ms": self.measured_elapsed_ms,
            "request.unaccounted_elapsed_ms": self.unaccounted_elapsed_ms,
        }
        bottleneck = self.bottleneck_stage
        if bottleneck is not None:
            fields["request.bottleneck_stage"] = bottleneck.name
            fields["request.bottleneck_elapsed_ms"] = bottleneck.elapsed_ms
            percent = self.stage_percent(bottleneck)
            if percent is not None:
                fields["request.bottleneck_percent"] = percent

        fields.update(self.attributes)
        return fields


def build_request_timing_stage(
    name: str,
    *,
    elapsed_ms: float,
    kind: RequestStageKind = "internal",
    status: RequestStageStatus = "ok",
    attributes: Mapping[str, object] | None = None,
) -> RequestTimingStage:
    normalized_name = _normalize_stage_name(name)
    if normalized_name is None:
        raise ValueError("stage name must be a safe non-blank identifier")
    return RequestTimingStage(
        name=normalized_name,
        kind=kind,
        elapsed_ms=_normalize_elapsed_ms(elapsed_ms, field_name="elapsed_ms"),
        status=status,
        attributes=_sanitize_attributes(attributes or {}),
    )


def build_request_timing_breakdown(
    *,
    flow: RequestTimingFlow,
    route: str,
    method: str,
    total_elapsed_ms: float,
    stages: Sequence[RequestTimingStage],
    status: RequestTimingStatus = "ok",
    trace_id: str | None = None,
    attributes: Mapping[str, object] | None = None,
) -> RequestTimingBreakdown:
    return RequestTimingBreakdown(
        trace_id=_resolve_trace_id(trace_id),
        flow=flow,
        route=route.strip() or "unknown",
        method=method.strip().upper() or "UNKNOWN",
        status=status,
        total_elapsed_ms=_normalize_elapsed_ms(
            total_elapsed_ms,
            field_name="total_elapsed_ms",
        ),
        stages=list(stages),
        attributes=_sanitize_attributes(attributes or {}),
    )


def infer_request_timing_flow(route: str) -> RequestTimingFlow:
    normalized_route = route.strip().lower()
    if normalized_route == "/chat":
        return "chat"
    if normalized_route == "/stream-chat":
        return "stream_chat"
    if normalized_route.startswith("/rag"):
        return "rag_answer"
    if normalized_route.startswith("/tool"):
        return "tool_chat"
    if normalized_route in {"/health", "/ready"}:
        return "health"
    return "unknown"


def build_total_http_request_timing_breakdown(
    *,
    route: str,
    method: str,
    total_elapsed_ms: float,
    status_code: int,
    trace_id: str | None = None,
) -> RequestTimingBreakdown:
    status: RequestTimingStatus = "ok" if status_code < 500 else "error"
    return build_request_timing_breakdown(
        flow=infer_request_timing_flow(route),
        route=route,
        method=method,
        total_elapsed_ms=total_elapsed_ms,
        status=status,
        trace_id=trace_id,
        stages=[
            build_request_timing_stage(
                "http.request",
                kind="server",
                elapsed_ms=total_elapsed_ms,
                status=status,
                attributes={"http.status_code": status_code},
            )
        ],
        attributes={"http.status_code": status_code},
    )


def _resolve_trace_id(trace_id: str | None) -> str:
    if trace_id is not None and trace_id.strip():
        return trace_id.strip()
    current_trace_id = get_trace_id()
    if current_trace_id != DEFAULT_TRACE_ID:
        return current_trace_id
    return DEFAULT_TRACE_ID


def _normalize_stage_name(name: object) -> str | None:
    if name is None:
        return None
    text = str(name).strip().replace(" ", ".")
    if not text:
        return None
    normalized = _ATTRIBUTE_KEY_UNSAFE_PATTERN.sub(".", text).strip(".-")
    if not normalized or not _ATTRIBUTE_KEY_PATTERN.fullmatch(normalized):
        return None
    return normalized.casefold()


def _normalize_attribute_key(key: object) -> str | None:
    if key is None:
        return None
    text = str(key).strip().replace(" ", "_")
    if not text:
        return None
    normalized = _ATTRIBUTE_KEY_UNSAFE_PATTERN.sub("_", text).strip("_.-").casefold()
    if not normalized or not _ATTRIBUTE_KEY_PATTERN.fullmatch(normalized):
        return None
    return normalized


def _normalize_elapsed_ms(value: float, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(rounded)


def _safe_attribute_value(value: object) -> RequestTimingValue | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _sanitize_attributes(
    attributes: Mapping[str, object],
) -> dict[str, RequestTimingValue]:
    safe_attributes: dict[str, RequestTimingValue] = {}
    for key, value in attributes.items():
        normalized_key = _normalize_attribute_key(key)
        if normalized_key is None:
            continue
        if normalized_key in REQUEST_TIMING_PROTECTED_KEYS:
            continue
        if normalized_key in REQUEST_TIMING_SENSITIVE_KEYS:
            continue
        if normalized_key in REQUEST_TIMING_HIGH_CARDINALITY_KEYS:
            continue
        safe_value = _safe_attribute_value(value)
        if safe_value is not None:
            safe_attributes[normalized_key] = safe_value
    return safe_attributes
