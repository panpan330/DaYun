from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from typing import Protocol

from pydantic import BaseModel, Field

from app.rag.query_rewrite import (
    build_query_rewrite_warnings,
    extract_business_entities,
    normalize_query_text,
)


DEFAULT_MULTI_QUERY_LIMIT = 4


class MultiQueryCandidate(BaseModel):
    query: str = Field(min_length=1)
    query_type: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class MultiQueryExpansion(BaseModel):
    original_query: str = Field(min_length=1)
    queries: list[MultiQueryCandidate] = Field(min_length=1)
    expanded: bool
    preserved_entities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MultiQueryGenerator(Protocol):
    def generate(
        self,
        query: str,
        *,
        max_queries: int = DEFAULT_MULTI_QUERY_LIMIT,
    ) -> MultiQueryExpansion:
        """Generate multiple retrieval-oriented queries for one user question."""


@dataclass(frozen=True)
class _MultiQueryRule:
    name: str
    term_groups: tuple[tuple[str, ...], ...]
    variants: tuple[tuple[str, str, str], ...]


_MULTI_QUERY_RULES: tuple[_MultiQueryRule, ...] = (
    _MultiQueryRule(
        name="quality_return_freight",
        term_groups=(
            ("质量", "坏了", "破损", "损坏", "商品质量"),
            ("退货", "退款", "售后"),
            ("运费", "邮费", "快递费", "物流费", "费用"),
        ),
        variants=(
            (
                "质量问题售后退货物流费用由谁承担？",
                "semantic_variant",
                "expand_quality_return_freight_synonyms",
            ),
            (
                "商品破损退货商家是否承担运费？",
                "scenario_variant",
                "expand_quality_return_freight_scenario",
            ),
            (
                "退货运费由商家还是用户承担的规则是什么？",
                "keyword_variant",
                "expand_return_freight_responsibility_terms",
            ),
        ),
    ),
    _MultiQueryRule(
        name="refund_arrival_time",
        term_groups=(
            ("退款", "退钱"),
            ("到账", "多久到账", "几天", "时效"),
        ),
        variants=(
            (
                "退款处理完成后多久到账？",
                "semantic_variant",
                "expand_refund_arrival_time_synonyms",
            ),
            (
                "退款原路退回时效是多久？",
                "policy_variant",
                "expand_refund_original_route_timing",
            ),
            (
                "退货审核通过后退款到账时间规则是什么？",
                "scenario_variant",
                "expand_refund_after_review_timing",
            ),
        ),
    ),
    _MultiQueryRule(
        name="shipping_time",
        term_groups=(
            ("订单", "商品"),
            ("发货", "发出", "寄出"),
            ("时效", "多久", "几天", "什么时候"),
        ),
        variants=(
            (
                "订单付款后发货时效规则是什么？",
                "policy_variant",
                "expand_shipping_time_policy_terms",
            ),
            (
                "商品通常多久会发出？",
                "semantic_variant",
                "expand_shipping_time_customer_expression",
            ),
            (
                "订单延迟发货如何处理？",
                "scenario_variant",
                "expand_delayed_shipping_scenario",
            ),
        ),
    ),
    _MultiQueryRule(
        name="return_after_deadline",
        term_groups=(
            ("超过", "7天", "七天", "期限", "好久"),
            ("退货", "退款", "还能退", "能不能退"),
        ),
        variants=(
            (
                "签收后超过退货期限是否还能退货？",
                "policy_variant",
                "expand_return_after_deadline_policy_terms",
            ),
            (
                "超过七天申请退货的处理规则是什么？",
                "semantic_variant",
                "expand_return_deadline_customer_expression",
            ),
            (
                "退货期限过后售后如何处理？",
                "scenario_variant",
                "expand_after_deadline_after_sales_scenario",
            ),
        ),
    ),
)


class RuleBasedMultiQueryGenerator:
    def generate(
        self,
        query: str,
        *,
        max_queries: int = DEFAULT_MULTI_QUERY_LIMIT,
    ) -> MultiQueryExpansion:
        normalized_query = normalize_query_text(query)
        if not normalized_query:
            raise ValueError("query must not be blank")
        _validate_max_queries(max_queries)

        preserved_entities = extract_business_entities(normalized_query)
        warnings = build_query_rewrite_warnings(
            normalized_query,
            preserved_entities=preserved_entities,
        )
        candidates = [
            MultiQueryCandidate(
                query=normalized_query,
                query_type="original",
                reason="preserve_original_query",
            )
        ]

        if warnings:
            return MultiQueryExpansion(
                original_query=normalized_query,
                queries=candidates[:max_queries],
                expanded=False,
                preserved_entities=preserved_entities,
                warnings=warnings,
            )

        for rule in _MULTI_QUERY_RULES:
            if not _matches_rule(normalized_query, rule):
                continue
            for query_variant, query_type, reason in rule.variants:
                candidates.append(
                    MultiQueryCandidate(
                        query=query_variant,
                        query_type=query_type,
                        reason=reason,
                    )
                )
            break

        unique_candidates = _deduplicate_candidates(candidates)[:max_queries]
        return MultiQueryExpansion(
            original_query=normalized_query,
            queries=unique_candidates,
            expanded=len(unique_candidates) > 1,
            preserved_entities=preserved_entities,
            warnings=warnings,
        )


def generate_multi_queries(
    query: str,
    *,
    generator: MultiQueryGenerator | None = None,
    max_queries: int = DEFAULT_MULTI_QUERY_LIMIT,
) -> MultiQueryExpansion:
    selected_generator = generator or RuleBasedMultiQueryGenerator()
    return selected_generator.generate(query, max_queries=max_queries)


def retrieval_queries_from_expansion(expansion: MultiQueryExpansion) -> list[str]:
    return [candidate.query for candidate in expansion.queries]


def format_multi_queries_for_debug(expansion: MultiQueryExpansion) -> list[str]:
    return [
        (
            f"{index}. type={candidate.query_type} "
            f"reason={candidate.reason} query={candidate.query}"
        )
        for index, candidate in enumerate(expansion.queries, start=1)
    ]


def _matches_rule(query: str, rule: _MultiQueryRule) -> bool:
    lowered_query = query.lower()
    return all(
        any(term.lower() in lowered_query for term in term_group)
        for term_group in rule.term_groups
    )


def _deduplicate_candidates(
    candidates: Sequence[MultiQueryCandidate],
) -> list[MultiQueryCandidate]:
    seen: set[str] = set()
    unique: list[MultiQueryCandidate] = []
    for candidate in candidates:
        dedupe_key = normalize_query_text(candidate.query)
        if dedupe_key in seen:
            continue
        unique.append(candidate)
        seen.add(dedupe_key)
    return unique


def _validate_max_queries(max_queries: int) -> None:
    if not isinstance(max_queries, int) or isinstance(max_queries, bool) or max_queries <= 0:
        raise ValueError("max_queries must be a positive integer")


class LLMMultiQueryGenerator:
    """LLM 驱动多查询扩展：生成多条检索变体。LLM 失败回退规则实现。"""

    def __init__(self, settings, *, client=None) -> None:
        self._settings = settings
        self._client = client
        self._fallback = RuleBasedMultiQueryGenerator()

    def generate(
        self,
        query: str,
        *,
        max_queries: int = DEFAULT_MULTI_QUERY_LIMIT,
    ) -> MultiQueryExpansion:
        if not getattr(self._settings, "has_llm_api_key", False):
            return self._fallback.generate(query, max_queries=max_queries)
        try:
            reply = self._call_llm(query, max_queries=max_queries)
        except Exception:
            return self._fallback.generate(query, max_queries=max_queries)
        try:
            variants = json.loads(reply)
        except (ValueError, TypeError):
            return self._fallback.generate(query, max_queries=max_queries)
        if not isinstance(variants, list) or not variants:
            return self._fallback.generate(query, max_queries=max_queries)
        normalized_query = query.strip()
        filtered = [
            str(v)[:200]
            for v in variants[:max_queries]
            if isinstance(v, str) and v.strip()
        ]
        if not filtered:
            return self._fallback.generate(query, max_queries=max_queries)
        return MultiQueryExpansion(
            original_query=normalized_query,
            queries=[
                MultiQueryCandidate(
                    query=q,
                    query_type="llm",
                    reason="llm expansion",
                )
                for q in filtered
            ],
            expanded=len(filtered) > 1,
        )

    def _call_llm(self, query: str, *, max_queries: int) -> str:
        from app.services.llm_service import extract_first_reply

        client = self._client
        if client is None:
            from app.services.llm_client import create_openai_compatible_client

            client = create_openai_compatible_client(self._settings)
        completion = client.chat.completions.create(
            model=self._settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个电商客服检索查询扩展助手。把用户问题从不同角度（语义/政策/"
                        "场景/关键词）扩展为多条检索问法。输出 JSON 字符串数组，每项一条问法，"
                        f"最多 {max_queries} 条。不要输出其他内容。"
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        return extract_first_reply(completion)


def create_multi_query_generator(settings) -> MultiQueryGenerator:
    if getattr(settings, "rag_advanced_mode", "rule") == "llm":
        return LLMMultiQueryGenerator(settings)
    return RuleBasedMultiQueryGenerator()
