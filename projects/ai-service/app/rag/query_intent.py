from collections.abc import Iterable
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from app.rag.query_rewrite import (
    build_query_rewrite_warnings,
    extract_business_entities,
    normalize_query_text,
)


QueryIntent = Literal[
    "policy_lookup",
    "order_lookup",
    "ticket_creation",
    "process_lookup",
    "smalltalk",
    "unsafe",
    "unclear",
]
QueryIntentRoute = Literal[
    "rag_policy_retrieval",
    "order_tool_calling",
    "ticket_agent_write_flow",
    "rag_process_retrieval",
    "direct_answer",
    "safety_guard",
    "ask_clarifying_question",
]
QueryIntentConfidence = Literal["high", "medium", "low"]


class QueryIntentClassification(BaseModel):
    normalized_query: str
    intent: QueryIntent
    route: QueryIntentRoute
    confidence: QueryIntentConfidence
    should_use_rag: bool
    should_rewrite_query: bool
    should_expand_multi_query: bool
    preserved_entities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class QueryIntentClassifier(Protocol):
    def classify(self, query: str) -> QueryIntentClassification:
        """Classify a user question before RAG retrieval."""


class RuleBasedQueryIntentClassifier:
    def classify(self, query: str) -> QueryIntentClassification:
        normalized_query = normalize_query_text(query)
        if not normalized_query:
            return _build_classification(
                normalized_query,
                intent="unclear",
                confidence="high",
                reasons=["query_is_blank"],
            )

        preserved_entities = extract_business_entities(normalized_query)
        warnings = build_query_rewrite_warnings(
            normalized_query,
            preserved_entities=preserved_entities,
        )
        lowered_query = normalized_query.lower()

        if "query_contains_instruction_like_text" in warnings:
            return _build_classification(
                normalized_query,
                intent="unsafe",
                confidence="high",
                preserved_entities=preserved_entities,
                warnings=warnings,
                reasons=["query_contains_instruction_like_text"],
            )

        if _contains_any(lowered_query, _SMALLTALK_TERMS):
            return _build_classification(
                normalized_query,
                intent="smalltalk",
                confidence="high",
                preserved_entities=preserved_entities,
                warnings=warnings,
                reasons=["query_matches_smalltalk_terms"],
            )

        if _contains_any(lowered_query, _TICKET_CREATION_TERMS):
            return _build_classification(
                normalized_query,
                intent="ticket_creation",
                confidence="high",
                preserved_entities=preserved_entities,
                warnings=warnings,
                reasons=["query_requests_ticket_or_human_handling"],
            )

        if preserved_entities or _contains_any(lowered_query, _ORDER_LOOKUP_TERMS):
            return _build_classification(
                normalized_query,
                intent="order_lookup",
                confidence="high" if preserved_entities else "medium",
                preserved_entities=preserved_entities,
                warnings=warnings,
                reasons=[
                    "query_contains_business_entity"
                    if preserved_entities
                    else "query_matches_order_lookup_terms"
                ],
            )

        if _is_unclear_query(normalized_query, lowered_query):
            return _build_classification(
                normalized_query,
                intent="unclear",
                confidence="medium",
                preserved_entities=preserved_entities,
                warnings=warnings,
                reasons=["query_is_too_short_or_generic"],
            )

        if _contains_any(lowered_query, _PROCESS_LOOKUP_TERMS):
            return _build_classification(
                normalized_query,
                intent="process_lookup",
                confidence="high",
                preserved_entities=preserved_entities,
                warnings=warnings,
                reasons=["query_asks_customer_service_process"],
            )

        if _contains_any(lowered_query, _POLICY_LOOKUP_TERMS):
            return _build_classification(
                normalized_query,
                intent="policy_lookup",
                confidence="high",
                preserved_entities=preserved_entities,
                warnings=warnings,
                reasons=["query_asks_policy_or_faq"],
            )

        return _build_classification(
            normalized_query,
            intent="unclear",
            confidence="low",
            preserved_entities=preserved_entities,
            warnings=warnings,
            reasons=["query_does_not_match_supported_intent_rules"],
        )


def classify_query_intent(
    query: str,
    *,
    classifier: QueryIntentClassifier | None = None,
) -> QueryIntentClassification:
    selected_classifier = classifier or RuleBasedQueryIntentClassifier()
    return selected_classifier.classify(query)


def is_rag_intent(intent: QueryIntent) -> bool:
    return intent in {"policy_lookup", "process_lookup"}


def route_for_query_intent(intent: QueryIntent) -> QueryIntentRoute:
    return _INTENT_ROUTES[intent]


def format_query_intent_for_debug(
    classification: QueryIntentClassification,
) -> str:
    reasons = ",".join(classification.reasons) or "-"
    warnings = ",".join(classification.warnings) or "-"
    entities = ",".join(classification.preserved_entities) or "-"
    return (
        f"intent={classification.intent} route={classification.route} "
        f"confidence={classification.confidence} "
        f"should_use_rag={classification.should_use_rag} "
        f"should_rewrite_query={classification.should_rewrite_query} "
        f"should_expand_multi_query={classification.should_expand_multi_query} "
        f"entities={entities} warnings={warnings} reasons={reasons}"
    )


_INTENT_ROUTES: dict[QueryIntent, QueryIntentRoute] = {
    "policy_lookup": "rag_policy_retrieval",
    "order_lookup": "order_tool_calling",
    "ticket_creation": "ticket_agent_write_flow",
    "process_lookup": "rag_process_retrieval",
    "smalltalk": "direct_answer",
    "unsafe": "safety_guard",
    "unclear": "ask_clarifying_question",
}

_RAG_INTENTS: set[QueryIntent] = {"policy_lookup", "process_lookup"}

_SMALLTALK_TERMS = (
    "你好",
    "您好",
    "hello",
    "hi",
    "你是谁",
    "你能做什么",
    "你会什么",
)
_TICKET_CREATION_TERMS = (
    "创建工单",
    "建工单",
    "提交工单",
    "售后工单",
    "工单",
    "我要投诉",
    "投诉",
    "人工处理",
    "转人工",
    "客服介入",
    "帮我处理",
)
_ORDER_LOOKUP_TERMS = (
    "我的订单",
    "订单到哪里",
    "订单到哪",
    "物流到哪里",
    "物流到哪",
    "物流信息",
    "物流轨迹",
    "查订单",
    "查询订单",
    "发货了吗",
    "签收了吗",
    "到货了吗",
)
_PROCESS_LOOKUP_TERMS = (
    "流程",
    "怎么申请",
    "如何申请",
    "怎么操作",
    "怎么走",
    "步骤",
    "需要哪些材料",
    "换货怎么弄",
    "售后怎么走",
)
_POLICY_LOOKUP_TERMS = (
    "规则",
    "政策",
    "退款",
    "退货",
    "运费",
    "邮费",
    "期限",
    "超过七天",
    "超过7天",
    "质量问题",
    "无理由",
    "会员",
    "积分",
    "账号安全",
    "faq",
)
_UNCLEAR_EXACT_TERMS = {
    "有问题",
    "帮我看看",
    "这个怎么办",
    "处理一下",
    "怎么办",
    "咋办",
}


def _build_classification(
    normalized_query: str,
    *,
    intent: QueryIntent,
    confidence: QueryIntentConfidence,
    preserved_entities: list[str] | None = None,
    warnings: list[str] | None = None,
    reasons: list[str] | None = None,
) -> QueryIntentClassification:
    should_use_rag = intent in _RAG_INTENTS
    return QueryIntentClassification(
        normalized_query=normalized_query,
        intent=intent,
        route=route_for_query_intent(intent),
        confidence=confidence,
        should_use_rag=should_use_rag,
        should_rewrite_query=should_use_rag,
        should_expand_multi_query=should_use_rag,
        preserved_entities=preserved_entities or [],
        warnings=warnings or [],
        reasons=reasons or [],
    )


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _is_unclear_query(normalized_query: str, lowered_query: str) -> bool:
    return normalized_query in _UNCLEAR_EXACT_TERMS or len(lowered_query) <= 2
