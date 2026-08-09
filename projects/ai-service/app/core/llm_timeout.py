from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings


LLMTimeoutPhase = Literal["retry", "fallback"]
LLMTimeoutBudgetReason = Literal[
    "within_budget",
    "retry_budget_exceeded",
    "fallback_budget_exceeded",
]


@dataclass(frozen=True)
class LLMTimeoutBudgetDecision:
    allowed: bool
    reason: LLMTimeoutBudgetReason
    phase: LLMTimeoutPhase
    elapsed_seconds: float
    total_timeout_seconds: float
    remaining_seconds: float
    required_seconds: float
    next_delay_seconds: float

    def to_log_fields(self) -> dict[str, float | str | bool]:
        return {
            "llm.timeout.allowed": self.allowed,
            "llm.timeout.reason": self.reason,
            "llm.timeout.phase": self.phase,
            "llm.timeout.elapsed_seconds": self.elapsed_seconds,
            "llm.timeout.total_timeout_seconds": self.total_timeout_seconds,
            "llm.timeout.remaining_seconds": self.remaining_seconds,
            "llm.timeout.required_seconds": self.required_seconds,
            "llm.timeout.next_delay_seconds": self.next_delay_seconds,
        }


def build_llm_timeout_budget_decision(
    settings: Settings,
    *,
    phase: LLMTimeoutPhase,
    elapsed_seconds: float,
    next_delay_seconds: float = 0.0,
) -> LLMTimeoutBudgetDecision:
    total_timeout_seconds = settings.llm_total_timeout_seconds
    remaining_seconds = max(0.0, total_timeout_seconds - elapsed_seconds)
    required_seconds = next_delay_seconds + settings.request_timeout_seconds
    allowed = remaining_seconds >= required_seconds
    reason = "within_budget"
    if not allowed:
        reason = (
            "retry_budget_exceeded"
            if phase == "retry"
            else "fallback_budget_exceeded"
        )

    return LLMTimeoutBudgetDecision(
        allowed=allowed,
        reason=reason,
        phase=phase,
        elapsed_seconds=elapsed_seconds,
        total_timeout_seconds=total_timeout_seconds,
        remaining_seconds=remaining_seconds,
        required_seconds=required_seconds,
        next_delay_seconds=next_delay_seconds,
    )
