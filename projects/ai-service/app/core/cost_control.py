from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings
from app.core.token_usage import (
    SafeTokenCostLogValue,
    TokenCostEstimate,
    TokenPricing,
    TokenUsageSnapshot,
    estimate_text_tokens_roughly,
    estimate_token_cost,
)


LLMCostControlAction = Literal["allow", "cap_output", "block"]
LLMCostControlReason = Literal[
    "disabled",
    "within_budget",
    "input_tokens_exceeded",
    "total_tokens_exceeded",
    "estimated_cost_exceeded",
]


@dataclass(frozen=True)
class LLMCostControlDecision:
    action: LLMCostControlAction
    reason: LLMCostControlReason
    estimated_input_tokens: int
    requested_max_output_tokens: int
    effective_max_output_tokens: int
    reserved_total_tokens: int
    fallback_allowed: bool
    estimated_cost: TokenCostEstimate
    max_estimated_cost: float | None = None

    @property
    def should_block(self) -> bool:
        return self.action == "block"

    @property
    def output_was_capped(self) -> bool:
        return self.action == "cap_output"

    def to_log_fields(self) -> dict[str, SafeTokenCostLogValue]:
        fields: dict[str, SafeTokenCostLogValue] = {
            "llm.cost_control_action": self.action,
            "llm.cost_control_reason": self.reason,
            "llm.estimated_input_tokens": self.estimated_input_tokens,
            "llm.requested_max_output_tokens": self.requested_max_output_tokens,
            "llm.effective_max_output_tokens": self.effective_max_output_tokens,
            "llm.reserved_total_tokens": self.reserved_total_tokens,
            "llm.fallback_allowed_by_cost_control": self.fallback_allowed,
            "llm.preflight_cost_status": self.estimated_cost.status,
        }
        _add_optional_field(fields, "llm.preflight_cost_currency", self.estimated_cost.currency)
        _add_optional_field(
            fields,
            "llm.preflight_estimated_total_cost",
            self.estimated_cost.total_cost,
        )
        _add_optional_field(
            fields,
            "llm.max_estimated_cost_per_request",
            self.max_estimated_cost,
        )
        return fields


def build_llm_cost_control_decision(
    settings: Settings,
    *,
    serialized_messages: Sequence[Mapping[str, str]],
    requested_max_output_tokens: int,
    pricing: TokenPricing | None = None,
) -> LLMCostControlDecision:
    if requested_max_output_tokens <= 0:
        raise ValueError("requested_max_output_tokens must be greater than 0")

    estimated_input_tokens = estimate_messages_tokens_roughly(serialized_messages)
    if not settings.llm_enable_cost_control:
        return _build_decision(
            settings=settings,
            action="allow",
            reason="disabled",
            estimated_input_tokens=estimated_input_tokens,
            requested_max_output_tokens=requested_max_output_tokens,
            effective_max_output_tokens=requested_max_output_tokens,
            pricing=pricing,
        )

    if estimated_input_tokens > settings.llm_max_input_tokens_per_request:
        return _build_decision(
            settings=settings,
            action="block",
            reason="input_tokens_exceeded",
            estimated_input_tokens=estimated_input_tokens,
            requested_max_output_tokens=requested_max_output_tokens,
            effective_max_output_tokens=0,
            pricing=pricing,
        )

    effective_max_output_tokens = requested_max_output_tokens
    action: LLMCostControlAction = "allow"
    reason: LLMCostControlReason = "within_budget"

    reserved_total_tokens = estimated_input_tokens + requested_max_output_tokens
    if reserved_total_tokens > settings.llm_max_total_tokens_per_request:
        effective_max_output_tokens = max(
            0,
            settings.llm_max_total_tokens_per_request - estimated_input_tokens,
        )
        if effective_max_output_tokens < settings.llm_min_output_tokens:
            return _build_decision(
                settings=settings,
                action="block",
                reason="total_tokens_exceeded",
                estimated_input_tokens=estimated_input_tokens,
                requested_max_output_tokens=requested_max_output_tokens,
                effective_max_output_tokens=effective_max_output_tokens,
                pricing=pricing,
            )
        action = "cap_output"
        reason = "total_tokens_exceeded"

    decision = _build_decision(
        settings=settings,
        action=action,
        reason=reason,
        estimated_input_tokens=estimated_input_tokens,
        requested_max_output_tokens=requested_max_output_tokens,
        effective_max_output_tokens=effective_max_output_tokens,
        pricing=pricing,
    )
    if _cost_exceeds_request_limit(settings, decision.estimated_cost):
        return _build_decision(
            settings=settings,
            action="block",
            reason="estimated_cost_exceeded",
            estimated_input_tokens=estimated_input_tokens,
            requested_max_output_tokens=requested_max_output_tokens,
            effective_max_output_tokens=effective_max_output_tokens,
            pricing=pricing,
        )

    return decision


def estimate_messages_tokens_roughly(
    serialized_messages: Sequence[Mapping[str, str]],
) -> int:
    text_parts: list[str] = []
    for message in serialized_messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role or content:
            text_parts.append(f"{role}\n{content}")
    return estimate_text_tokens_roughly("\n\n".join(text_parts))


def _build_decision(
    *,
    settings: Settings,
    action: LLMCostControlAction,
    reason: LLMCostControlReason,
    estimated_input_tokens: int,
    requested_max_output_tokens: int,
    effective_max_output_tokens: int,
    pricing: TokenPricing | None,
) -> LLMCostControlDecision:
    reserved_total_tokens = estimated_input_tokens + max(effective_max_output_tokens, 0)
    return LLMCostControlDecision(
        action=action,
        reason=reason,
        estimated_input_tokens=estimated_input_tokens,
        requested_max_output_tokens=requested_max_output_tokens,
        effective_max_output_tokens=effective_max_output_tokens,
        reserved_total_tokens=reserved_total_tokens,
        fallback_allowed=_is_fallback_allowed(settings, reserved_total_tokens, action),
        estimated_cost=estimate_token_cost(
            TokenUsageSnapshot(
                prompt_tokens=estimated_input_tokens,
                completion_tokens=max(effective_max_output_tokens, 0),
                total_tokens=reserved_total_tokens,
            ),
            pricing,
        ),
        max_estimated_cost=settings.llm_max_estimated_cost_per_request,
    )


def _is_fallback_allowed(
    settings: Settings,
    reserved_total_tokens: int,
    action: LLMCostControlAction,
) -> bool:
    if action == "block":
        return False
    if settings.llm_disable_fallback_above_total_tokens is None:
        return True
    return reserved_total_tokens <= settings.llm_disable_fallback_above_total_tokens


def _cost_exceeds_request_limit(
    settings: Settings,
    estimated_cost: TokenCostEstimate,
) -> bool:
    if settings.llm_max_estimated_cost_per_request is None:
        return False
    if estimated_cost.total_cost is None:
        return False
    return estimated_cost.total_cost > settings.llm_max_estimated_cost_per_request


def _add_optional_field(
    fields: dict[str, SafeTokenCostLogValue],
    key: str,
    value: object,
) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        fields[key] = value
    elif isinstance(value, int):
        fields[key] = value
    elif isinstance(value, float):
        fields[key] = value
    elif isinstance(value, str) and value.strip():
        fields[key] = value.strip()
