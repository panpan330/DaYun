from app.core.config import Settings
from app.core.llm_timeout import build_llm_timeout_budget_decision


def test_timeout_budget_allows_retry_when_next_attempt_can_fit() -> None:
    decision = build_llm_timeout_budget_decision(
        Settings(
            request_timeout_seconds=3,
            llm_total_timeout_seconds=10,
            _env_file=None,
        ),
        phase="retry",
        elapsed_seconds=2,
        next_delay_seconds=0.5,
    )

    assert decision.allowed is True
    assert decision.reason == "within_budget"
    assert decision.remaining_seconds == 8
    assert decision.required_seconds == 3.5


def test_timeout_budget_blocks_retry_when_next_attempt_cannot_fit() -> None:
    decision = build_llm_timeout_budget_decision(
        Settings(
            request_timeout_seconds=30,
            llm_total_timeout_seconds=45,
            _env_file=None,
        ),
        phase="retry",
        elapsed_seconds=30,
        next_delay_seconds=0.2,
    )

    assert decision.allowed is False
    assert decision.reason == "retry_budget_exceeded"
    assert decision.remaining_seconds == 15
    assert decision.required_seconds == 30.2


def test_timeout_budget_blocks_fallback_when_fallback_call_cannot_fit() -> None:
    decision = build_llm_timeout_budget_decision(
        Settings(
            request_timeout_seconds=30,
            llm_total_timeout_seconds=45,
            _env_file=None,
        ),
        phase="fallback",
        elapsed_seconds=44,
    )

    assert decision.allowed is False
    assert decision.reason == "fallback_budget_exceeded"
    assert decision.remaining_seconds == 1
    assert decision.required_seconds == 30


def test_timeout_budget_log_fields_do_not_contain_prompt_or_secret() -> None:
    decision = build_llm_timeout_budget_decision(
        Settings(
            request_timeout_seconds=3,
            llm_total_timeout_seconds=10,
            llm_api_key="secret-key",
            _env_file=None,
        ),
        phase="fallback",
        elapsed_seconds=1,
    )

    fields = decision.to_log_fields()

    assert "secret-key" not in str(fields)
    assert "prompt" not in str(fields).lower()
