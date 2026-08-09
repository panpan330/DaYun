from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import re
from typing import Protocol

from pydantic import BaseModel, Field


class QueryRewriteResult(BaseModel):
    original_query: str = Field(min_length=1)
    rewritten_query: str = Field(min_length=1)
    changed: bool
    rewrite_reasons: list[str] = Field(default_factory=list)
    preserved_entities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> QueryRewriteResult:
        """Rewrite a user question into a retrieval-oriented query."""


@dataclass(frozen=True)
class _RewriteRule:
    name: str
    term_groups: tuple[tuple[str, ...], ...]
    rewritten_query: str
    reason: str


_REWRITE_RULES: tuple[_RewriteRule, ...] = (
    _RewriteRule(
        name="quality_return_freight",
        term_groups=(
            ("质量", "坏了", "破损", "损坏", "坏"),
            ("退", "退货", "退款", "售后"),
            ("运费", "邮费", "快递费", "物流费", "邮寄费"),
        ),
        rewritten_query="商品质量问题退货运费承担规则是什么？",
        reason="map_colloquial_quality_return_freight_to_policy_query",
    ),
    _RewriteRule(
        name="no_reason_return_freight",
        term_groups=(
            ("无理由", "不想要", "买错", "拍错"),
            ("退", "退货", "退款"),
            ("运费", "邮费", "快递费", "物流费", "邮寄费"),
        ),
        rewritten_query="无理由退货运费承担规则是什么？",
        reason="map_no_reason_return_freight_to_policy_query",
    ),
    _RewriteRule(
        name="refund_arrival_time",
        term_groups=(
            ("退款", "退钱", "钱"),
            ("多久到账", "几天到账", "什么时候到账", "啥时候到账", "多久到"),
        ),
        rewritten_query="退款到账时间规则是什么？",
        reason="map_refund_arrival_question_to_policy_query",
    ),
    _RewriteRule(
        name="shipping_time",
        term_groups=(
            ("发货", "发出", "寄出"),
            ("多久", "什么时候", "啥时候", "几天", "时效"),
        ),
        rewritten_query="订单发货时效规则是什么？",
        reason="map_shipping_time_question_to_policy_query",
    ),
    _RewriteRule(
        name="return_after_deadline",
        term_groups=(
            ("7天", "七天", "好久", "超过"),
            ("退", "退货", "退款", "还能退", "能不能退"),
        ),
        rewritten_query="签收后超过退货期限是否还能退货的规则是什么？",
        reason="map_return_deadline_question_to_policy_query",
    ),
    _RewriteRule(
        name="exchange_process",
        term_groups=(
            ("换货", "换一个", "更换"),
            ("流程", "怎么弄", "咋办", "怎么办", "规则"),
        ),
        rewritten_query="商品换货处理流程是什么？",
        reason="map_exchange_question_to_process_query",
    ),
)

_BUSINESS_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]*\d{3,}\b", re.IGNORECASE)
_INSTRUCTION_LIKE_TERMS = (
    "忽略",
    "无视",
    "override",
    "ignore",
    "system prompt",
    "系统提示词",
    "开发者指令",
    "管理员规则",
)


class RuleBasedQueryRewriter:
    def rewrite(self, query: str) -> QueryRewriteResult:
        normalized_query = normalize_query_text(query)
        if not normalized_query:
            raise ValueError("query must not be blank")

        preserved_entities = extract_business_entities(normalized_query)
        warnings = build_query_rewrite_warnings(
            normalized_query,
            preserved_entities=preserved_entities,
        )
        reasons: list[str] = []

        rewritten_query = normalized_query
        for rule in _REWRITE_RULES:
            if _matches_rule(normalized_query, rule):
                rewritten_query = rule.rewritten_query
                reasons.append(rule.reason)
                break

        return QueryRewriteResult(
            original_query=normalized_query,
            rewritten_query=rewritten_query,
            changed=rewritten_query != normalized_query,
            rewrite_reasons=reasons,
            preserved_entities=preserved_entities,
            warnings=warnings,
        )


def rewrite_query_for_retrieval(
    query: str,
    *,
    rewriter: QueryRewriter | None = None,
) -> QueryRewriteResult:
    selected_rewriter = rewriter or RuleBasedQueryRewriter()
    return selected_rewriter.rewrite(query)


def normalize_query_text(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def extract_business_entities(query: str) -> list[str]:
    return _unique_strings(
        match.group(0).upper()
        for match in _BUSINESS_ENTITY_PATTERN.finditer(query)
    )


def build_query_rewrite_warnings(
    query: str,
    *,
    preserved_entities: Sequence[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    if preserved_entities:
        warnings.append("query_contains_business_entity_may_need_tool_calling")
    lowered = query.lower()
    if any(term.lower() in lowered for term in _INSTRUCTION_LIKE_TERMS):
        warnings.append("query_contains_instruction_like_text")
    return warnings


def _matches_rule(query: str, rule: _RewriteRule) -> bool:
    return all(
        any(term.lower() in query.lower() for term in term_group)
        for term_group in rule.term_groups
    )


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return unique


class LLMQueryRewriter:
    """LLM 驱动查询改写：把口语提问规范化为检索问法。LLM 失败回退规则实现。"""

    def __init__(self, settings, *, client=None) -> None:
        self._settings = settings
        self._client = client
        self._fallback = RuleBasedQueryRewriter()

    def rewrite(self, query: str) -> QueryRewriteResult:
        if not getattr(self._settings, "has_llm_api_key", False):
            return self._fallback.rewrite(query)
        try:
            reply = self._call_llm(query)
        except Exception:
            return self._fallback.rewrite(query)
        rewritten = (reply or "").strip()
        if not rewritten:
            return self._fallback.rewrite(query)
        return QueryRewriteResult(
            original_query=query.strip(),
            rewritten_query=rewritten,
            changed=rewritten != query.strip(),
            rewrite_reasons=["llm_rewrite"],
        )

    def _call_llm(self, query: str) -> str:
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
                        "你是一个电商客服检索查询改写助手。把用户的口语化提问改写为"
                        "适合知识库检索的规范问法（保留订单号、金额等关键实体）。"
                        "只输出改写后的问题本身，不要解释。"
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        return extract_first_reply(completion)


def create_query_rewriter(settings) -> QueryRewriter:
    if getattr(settings, "rag_advanced_mode", "rule") == "llm":
        return LLMQueryRewriter(settings)
    return RuleBasedQueryRewriter()
