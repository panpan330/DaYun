from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Literal

from app.core.trace import DEFAULT_TRACE_ID, get_trace_id


LLMLogValue = str | int | float | bool
LLMLogOperation = Literal[
    "chat",
    "stream_chat",
    "tool_decision",
    "tool_summary",
    "structured_output",
    "rag_final_answer",
]
LLMLogOutcome = Literal["success", "failure"]

LLM_LOG_PROTECTED_KEYS = frozenset(
    {
        "app.trace_id",
        "llm.operation",
        "llm.outcome",
        "llm.provider",
        "llm.model",
        "llm.elapsed_ms",
        "llm.prompt_tokens",
        "llm.completion_tokens",
        "llm.total_tokens",
        "llm.error_code",
        "http.status_code",
    }
)
LLM_LOG_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "openai_api_key",
        "llm_api_key",
        "embedding_api_key",
        "rerank_api_key",
        "authorization",
        "cookie",
        "set_cookie",
        "bearer_token",
        "secret",
        "token",
        "password",
        "prompt",
        "raw_prompt",
        "system_prompt",
        "developer_prompt",
        "messages",
        "history",
        "input",
        "user_input",
        "user_message",
        "query",
        "raw_query",
        "response",
        "raw_response",
        "model_response",
        "reply",
        "final_answer",
        "content",
        "delta",
        "tool_result",
        "tool_message",
        "document_content",
        "chunk_content",
        "retrieved_documents",
    }
)

_ATTRIBUTE_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_ATTRIBUTE_KEY_UNSAFE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def build_safe_llm_log_payload(
    *,
    operation: LLMLogOperation,
    outcome: LLMLogOutcome,
    provider: str,
    model: str,
    elapsed_ms: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    error_code: str | None = None,
    status_code: int | None = None,
    trace_id: str | None = None,
    extra_fields: Mapping[str, object] | None = None,
) -> dict[str, LLMLogValue]:
    payload: dict[str, LLMLogValue] = {}
    _add_field(payload, "app.trace_id", _resolve_trace_id(trace_id))
    _add_field(payload, "llm.operation", operation)
    _add_field(payload, "llm.outcome", outcome)
    _add_field(payload, "llm.provider", provider)
    _add_field(payload, "llm.model", model)
    _add_field(payload, "llm.elapsed_ms", _round_millis(elapsed_ms))
    _add_field(payload, "llm.prompt_tokens", _safe_non_negative_int(prompt_tokens))
    _add_field(payload, "llm.completion_tokens", _safe_non_negative_int(completion_tokens))
    _add_field(payload, "llm.total_tokens", _safe_non_negative_int(total_tokens))
    _add_field(payload, "llm.error_code", error_code)
    _add_field(payload, "http.status_code", _safe_non_negative_int(status_code))

    if extra_fields is not None:
        _merge_extra_fields(payload, extra_fields)

    return payload


def find_forbidden_llm_log_fields(fields: Mapping[str, object]) -> list[str]:
    forbidden: list[str] = []
    for key in fields:
        normalized_key = _normalize_field_key(key)
        if normalized_key is not None and normalized_key in LLM_LOG_SENSITIVE_KEYS:
            forbidden.append(normalized_key)
    return sorted(set(forbidden))


def _resolve_trace_id(trace_id: str | None) -> str:
    if trace_id is not None and trace_id.strip():
        return trace_id.strip()
    current_trace_id = get_trace_id()
    if current_trace_id != DEFAULT_TRACE_ID:
        return current_trace_id
    return DEFAULT_TRACE_ID


def _normalize_field_key(key: object) -> str | None:
    if key is None:
        return None
    text = str(key).strip().replace(" ", "_")
    if not text:
        return None
    normalized = _ATTRIBUTE_KEY_UNSAFE_PATTERN.sub("_", text).strip("_.-").casefold()
    if not normalized or not _ATTRIBUTE_KEY_PATTERN.fullmatch(normalized):
        return None
    return normalized


def _safe_log_value(value: object) -> LLMLogValue | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _round_millis(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value < 0:
        return None
    return round(value, 2)


def _safe_non_negative_int(value: int | None) -> int | None:
    if isinstance(value, bool) or value is None or value < 0:
        return None
    return value


def _add_field(
    payload: dict[str, LLMLogValue],
    key: object,
    value: object,
) -> None:
    normalized_key = _normalize_field_key(key)
    if normalized_key is None or normalized_key in LLM_LOG_SENSITIVE_KEYS:
        return
    safe_value = _safe_log_value(value)
    if safe_value is not None:
        payload[normalized_key] = safe_value


def _merge_extra_fields(
    payload: dict[str, LLMLogValue],
    extra_fields: Mapping[str, object],
) -> None:
    for key, value in extra_fields.items():
        normalized_key = _normalize_field_key(key)
        if normalized_key is None:
            continue
        if normalized_key in LLM_LOG_PROTECTED_KEYS:
            continue
        if normalized_key in LLM_LOG_SENSITIVE_KEYS:
            continue
        if normalized_key in payload:
            continue
        safe_value = _safe_log_value(value)
        if safe_value is not None:
            payload[normalized_key] = safe_value
