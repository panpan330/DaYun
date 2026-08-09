from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from app.core.config import Settings


LLMModelTier = Literal["fast", "balanced", "strong"]
LLMRouteOperation = Literal[
    "chat",
    "stream_chat",
    "rag_answer",
    "tool_call",
    "structured_output",
    "unknown",
]
LLMRouteReason = Literal[
    "preferred_tier",
    "operation_requires_quality",
    "strong_keyword",
    "long_input",
    "fast_keyword",
    "default_tier",
]

_KEYWORD_SPLIT_PATTERN = re.compile(r"[,，;；\n\r\t]+")


@dataclass(frozen=True)
class LLMModelRouteDecision:
    provider: str
    model: str
    tier: LLMModelTier
    operation: LLMRouteOperation
    reason: LLMRouteReason
    input_chars: int

    def to_log_fields(self) -> dict[str, str | int]:
        return {
            "llm.route_tier": self.tier,
            "llm.route_operation": self.operation,
            "llm.route_reason": self.reason,
            "llm.input_chars": self.input_chars,
        }


def route_llm_model(
    settings: Settings,
    *,
    operation: LLMRouteOperation = "unknown",
    input_text: str = "",
    preferred_tier: LLMModelTier | None = None,
) -> LLMModelRouteDecision:
    input_chars = len(input_text or "")

    if preferred_tier is not None:
        return _build_decision(
            settings,
            tier=preferred_tier,
            operation=operation,
            reason="preferred_tier",
            input_chars=input_chars,
        )

    if operation in {"rag_answer", "tool_call", "structured_output"}:
        return _build_decision(
            settings,
            tier="balanced",
            operation=operation,
            reason="operation_requires_quality",
            input_chars=input_chars,
        )

    if _contains_keyword(input_text, parse_route_keywords(settings.llm_route_strong_keywords)):
        return _build_decision(
            settings,
            tier="strong",
            operation=operation,
            reason="strong_keyword",
            input_chars=input_chars,
        )

    if input_chars >= settings.llm_route_long_input_chars:
        return _build_decision(
            settings,
            tier="strong",
            operation=operation,
            reason="long_input",
            input_chars=input_chars,
        )

    if _contains_keyword(input_text, parse_route_keywords(settings.llm_route_fast_keywords)):
        return _build_decision(
            settings,
            tier="fast",
            operation=operation,
            reason="fast_keyword",
            input_chars=input_chars,
        )

    return _build_decision(
        settings,
        tier=settings.llm_default_route_tier,
        operation=operation,
        reason="default_tier",
        input_chars=input_chars,
    )


def parse_route_keywords(raw_keywords: str) -> tuple[str, ...]:
    keywords = []
    for keyword in _KEYWORD_SPLIT_PATTERN.split(raw_keywords):
        normalized = keyword.strip().casefold()
        if normalized:
            keywords.append(normalized)
    return tuple(dict.fromkeys(keywords))


def _build_decision(
    settings: Settings,
    *,
    tier: LLMModelTier,
    operation: LLMRouteOperation,
    reason: LLMRouteReason,
    input_chars: int,
) -> LLMModelRouteDecision:
    return LLMModelRouteDecision(
        provider=settings.llm_provider.strip() or "openai-compatible",
        model=_resolve_model_for_tier(settings, tier),
        tier=tier,
        operation=operation,
        reason=reason,
        input_chars=input_chars,
    )


def _resolve_model_for_tier(settings: Settings, tier: LLMModelTier) -> str:
    if tier == "fast":
        return settings.resolved_llm_fast_model
    if tier == "strong":
        return settings.resolved_llm_strong_model
    return settings.resolved_llm_balanced_model


def _contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    normalized_text = (text or "").casefold()
    return any(keyword in normalized_text for keyword in keywords)
