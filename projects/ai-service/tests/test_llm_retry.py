from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.llm_retry import (
    build_llm_retry_decision,
    calculate_llm_retry_delay,
    extract_retry_after_seconds,
)


def test_llm_retry_allows_retryable_error_before_attempts_exhausted() -> None:
    decision = build_llm_retry_decision(
        Settings(llm_max_retries=2, _env_file=None),
        error_code="LLM_TIMEOUT",
        attempt_number=1,
    )

    assert decision.should_retry is True
    assert decision.reason == "retry_allowed"
    assert decision.max_attempts == 3
    assert decision.next_attempt_number == 2
    assert decision.next_delay_seconds == 0.2


def test_llm_retry_uses_retry_after_for_rate_limit_with_cap() -> None:
    decision = build_llm_retry_decision(
        Settings(llm_max_retries=2, _env_file=None),
        error_code="LLM_RATE_LIMITED",
        attempt_number=1,
        retry_after_seconds=30,
    )

    assert decision.should_retry is True
    assert decision.reason == "retry_after_allowed"
    assert decision.next_delay_seconds == 2.0


def test_llm_retry_blocks_when_disabled_or_exhausted() -> None:
    disabled = build_llm_retry_decision(
        Settings(llm_max_retries=0, _env_file=None),
        error_code="LLM_TIMEOUT",
        attempt_number=1,
    )
    exhausted = build_llm_retry_decision(
        Settings(llm_max_retries=2, _env_file=None),
        error_code="LLM_TIMEOUT",
        attempt_number=3,
    )

    assert disabled.should_retry is False
    assert disabled.reason == "retry_disabled"
    assert exhausted.should_retry is False
    assert exhausted.reason == "max_retries_exhausted"


def test_llm_retry_blocks_non_retryable_error() -> None:
    decision = build_llm_retry_decision(
        Settings(llm_max_retries=2, _env_file=None),
        error_code="LLM_AUTHENTICATION_FAILED",
        attempt_number=1,
    )

    assert decision.should_retry is False
    assert decision.reason == "error_not_retryable"
    assert decision.next_attempt_number is None


def test_llm_retry_delay_is_exponential_and_capped() -> None:
    assert calculate_llm_retry_delay(1) == 0.2
    assert calculate_llm_retry_delay(2) == 0.4
    assert calculate_llm_retry_delay(8) == 2.0

    with pytest.raises(ValueError, match="attempt_number"):
        calculate_llm_retry_delay(0)


def test_extract_retry_after_seconds_reads_numeric_header() -> None:
    exc = SimpleNamespace(
        response=SimpleNamespace(headers={"retry-after": "1.25"}),
    )

    assert extract_retry_after_seconds(exc) == 1.25
    assert extract_retry_after_seconds(SimpleNamespace()) is None


def test_llm_retry_log_fields_do_not_include_prompt_text() -> None:
    decision = build_llm_retry_decision(
        Settings(llm_max_retries=2, _env_file=None),
        error_code="LLM_CONNECTION_ERROR",
        attempt_number=1,
    )

    serialized = str(decision.to_log_fields())

    assert "LLM_CONNECTION_ERROR" in serialized
    assert "user_message" not in serialized
    assert "prompt" not in serialized
