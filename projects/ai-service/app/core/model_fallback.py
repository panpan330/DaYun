from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings
from app.core.model_routing import (
    LLMModelRouteDecision,
    LLMRouteOperation,
)


LLMFallbackReason = Literal[
    "retryable_error",
    "disabled",
    "non_retryable_error",
    "same_model",
    "cost_control",
]

_ERROR_CODE_SPLIT_PATTERN = re.compile(r"[,，;；\n\r\t]+")


@dataclass(frozen=True)
class LLMFallbackDecision:
    should_attempt: bool
    reason: LLMFallbackReason
    primary_error_code: str
    primary_model: str
    fallback_route: LLMModelRouteDecision | None = None

    def to_log_fields(self) -> dict[str, str | int | bool]:
        fields: dict[str, str | int | bool] = {
            "llm.fallback_attempted": self.should_attempt,
            "llm.fallback_reason": self.reason,
            "llm.primary_error_code": self.primary_error_code,
            "llm.primary_model": self.primary_model,
        }
        if self.fallback_route is not None:
            fields.update(
                {
                    "llm.fallback_model": self.fallback_route.model,
                    "llm.fallback_tier": self.fallback_route.tier,
                }
            )
        return fields


def build_llm_fallback_decision(
    settings: Settings,
    *,
    primary_route: LLMModelRouteDecision,
    error_code: str,
) -> LLMFallbackDecision:
    normalized_error_code = error_code.strip().upper()

    if not settings.llm_enable_fallback:
        return _build_not_attempted_decision(
            primary_route=primary_route,
            error_code=normalized_error_code,
            reason="disabled",
        )

    if normalized_error_code not in parse_fallback_error_codes(
        settings.llm_fallback_error_codes
    ):
        return _build_not_attempted_decision(
            primary_route=primary_route,
            error_code=normalized_error_code,
            reason="non_retryable_error",
        )

    fallback_route = build_fallback_route_decision(
        settings,
        primary_route=primary_route,
    )
    if (
        fallback_route.provider == primary_route.provider
        and fallback_route.model == primary_route.model
    ):
        return LLMFallbackDecision(
            should_attempt=False,
            reason="same_model",
            primary_error_code=normalized_error_code,
            primary_model=primary_route.model,
            fallback_route=fallback_route,
        )

    return LLMFallbackDecision(
        should_attempt=True,
        reason="retryable_error",
        primary_error_code=normalized_error_code,
        primary_model=primary_route.model,
        fallback_route=fallback_route,
    )


def build_fallback_route_decision(
    settings: Settings,
    *,
    primary_route: LLMModelRouteDecision,
) -> LLMModelRouteDecision:
    return LLMModelRouteDecision(
        provider=settings.llm_provider.strip() or "openai-compatible",
        model=settings.resolved_llm_fallback_model,
        tier=settings.llm_fallback_tier,
        operation=_coerce_operation(primary_route.operation),
        reason="preferred_tier",
        input_chars=primary_route.input_chars,
    )


def disable_fallback_by_cost_control(
    decision: LLMFallbackDecision,
) -> LLMFallbackDecision:
    return LLMFallbackDecision(
        should_attempt=False,
        reason="cost_control",
        primary_error_code=decision.primary_error_code,
        primary_model=decision.primary_model,
        fallback_route=decision.fallback_route,
    )


def parse_fallback_error_codes(raw_error_codes: str) -> frozenset[str]:
    codes: list[str] = []
    for code in _ERROR_CODE_SPLIT_PATTERN.split(raw_error_codes):
        normalized = code.strip().upper()
        if normalized:
            codes.append(normalized)
    return frozenset(dict.fromkeys(codes))


def _build_not_attempted_decision(
    *,
    primary_route: LLMModelRouteDecision,
    error_code: str,
    reason: LLMFallbackReason,
) -> LLMFallbackDecision:
    return LLMFallbackDecision(
        should_attempt=False,
        reason=reason,
        primary_error_code=error_code,
        primary_model=primary_route.model,
    )


def _coerce_operation(operation: str) -> LLMRouteOperation:
    if operation in {
        "chat",
        "stream_chat",
        "rag_answer",
        "tool_call",
        "structured_output",
        "unknown",
    }:
        return operation  # type: ignore[return-value]
    return "unknown"
