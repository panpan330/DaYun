from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings


LLMRetryDecisionReason = Literal[
    "retry_allowed",
    "retry_after_allowed",
    "retry_disabled",
    "error_not_retryable",
    "max_retries_exhausted",
]

RETRYABLE_LLM_ERROR_CODES = frozenset(
    {
        "LLM_TIMEOUT",
        "LLM_RATE_LIMITED",
        "LLM_PROVIDER_ERROR",
        "LLM_CONNECTION_ERROR",
        "LLM_PROVIDER_STATUS_ERROR",
        "LLM_CALL_FAILED",
    }
)

DEFAULT_LLM_RETRY_INITIAL_DELAY_SECONDS = 0.2
DEFAULT_LLM_RETRY_MULTIPLIER = 2.0
DEFAULT_LLM_RETRY_MAX_DELAY_SECONDS = 2.0


@dataclass(frozen=True)
class LLMRetryDecision:
    should_retry: bool
    reason: LLMRetryDecisionReason
    error_code: str
    attempt_number: int
    max_attempts: int
    next_attempt_number: int | None = None
    next_delay_seconds: float | None = None

    def to_log_fields(self) -> dict[str, str | int | float | bool]:
        fields: dict[str, str | int | float | bool] = {
            "llm.retry_should_retry": self.should_retry,
            "llm.retry_reason": self.reason,
            "llm.retry_error_code": self.error_code,
            "llm.retry_attempt_number": self.attempt_number,
            "llm.retry_max_attempts": self.max_attempts,
        }
        if self.next_attempt_number is not None:
            fields["llm.retry_next_attempt_number"] = self.next_attempt_number
        if self.next_delay_seconds is not None:
            fields["llm.retry_next_delay_seconds"] = self.next_delay_seconds
        return fields


def build_llm_retry_decision(
    settings: Settings,
    *,
    error_code: str,
    attempt_number: int,
    retry_after_seconds: float | None = None,
) -> LLMRetryDecision:
    if attempt_number < 1:
        raise ValueError("attempt_number must be greater than or equal to 1")

    normalized_error_code = error_code.strip().upper() or "UNKNOWN"
    max_attempts = settings.llm_max_retries + 1

    if settings.llm_max_retries == 0:
        return _blocked_decision(
            error_code=normalized_error_code,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            reason="retry_disabled",
        )

    if normalized_error_code not in RETRYABLE_LLM_ERROR_CODES:
        return _blocked_decision(
            error_code=normalized_error_code,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            reason="error_not_retryable",
        )

    if attempt_number >= max_attempts:
        return _blocked_decision(
            error_code=normalized_error_code,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            reason="max_retries_exhausted",
        )

    if normalized_error_code == "LLM_RATE_LIMITED" and retry_after_seconds is not None:
        next_delay_seconds = min(
            _normalize_retry_after_seconds(retry_after_seconds),
            DEFAULT_LLM_RETRY_MAX_DELAY_SECONDS,
        )
        reason: LLMRetryDecisionReason = "retry_after_allowed"
    else:
        next_delay_seconds = calculate_llm_retry_delay(attempt_number)
        reason = "retry_allowed"

    return LLMRetryDecision(
        should_retry=True,
        reason=reason,
        error_code=normalized_error_code,
        attempt_number=attempt_number,
        max_attempts=max_attempts,
        next_attempt_number=attempt_number + 1,
        next_delay_seconds=next_delay_seconds,
    )


def calculate_llm_retry_delay(attempt_number: int) -> float:
    if attempt_number < 1:
        raise ValueError("attempt_number must be greater than or equal to 1")

    delay = DEFAULT_LLM_RETRY_INITIAL_DELAY_SECONDS * (
        DEFAULT_LLM_RETRY_MULTIPLIER ** (attempt_number - 1)
    )
    return _round_seconds(min(delay, DEFAULT_LLM_RETRY_MAX_DELAY_SECONDS))


def extract_retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw_value = headers.get("retry-after")
    if raw_value is None:
        return None

    try:
        parsed = float(str(raw_value).strip())
    except ValueError:
        return None

    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return _round_seconds(parsed)


def _blocked_decision(
    *,
    error_code: str,
    attempt_number: int,
    max_attempts: int,
    reason: LLMRetryDecisionReason,
) -> LLMRetryDecision:
    return LLMRetryDecision(
        should_retry=False,
        reason=reason,
        error_code=error_code,
        attempt_number=attempt_number,
        max_attempts=max_attempts,
    )


def _normalize_retry_after_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("retry_after_seconds must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("retry_after_seconds must be greater than 0")
    return _round_seconds(value)


def _round_seconds(value: float) -> float:
    return max(0.001, round(value, 3))
