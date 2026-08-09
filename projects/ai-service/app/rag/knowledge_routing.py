from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from app.rag.filters import PayloadFilter, RagAccessScope, build_access_scope_filter
from app.rag.query_intent import (
    QueryIntent,
    QueryIntentClassification,
    classify_query_intent,
)
from app.rag.query_rewrite import normalize_query_text


RagKnowledgeRouteWarning = Literal[
    "RAG_ROUTE_QUERY_INTENT_NOT_RAG",
    "RAG_ROUTE_NO_ACCESSIBLE_KNOWLEDGE_BASE",
    "RAG_ROUTE_CANDIDATE_FILTERED_BY_ACCESS_SCOPE",
    "RAG_ROUTE_USED_FALLBACK_KNOWLEDGE_BASE",
    "RAG_ROUTE_MULTIPLE_KNOWLEDGE_BASES_SELECTED",
    "RAG_ROUTE_CANDIDATES_TRUNCATED",
]


class RagKnowledgeBaseDefinition(BaseModel):
    knowledge_base_id: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    supported_intents: list[QueryIntent] = Field(default_factory=list)
    business_domains: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    permission_groups: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0)
    is_fallback: bool = False


class RagKnowledgeRoute(BaseModel):
    knowledge_base_id: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    route_score: float = Field(ge=0, le=1)
    matched_keywords: list[str] = Field(default_factory=list)
    business_domains: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    permission_groups: list[str] = Field(default_factory=list)
    payload_filter: PayloadFilter | None = None
    reasons: list[str] = Field(default_factory=list)


class RagKnowledgeRouteDecision(BaseModel):
    normalized_query: str
    intent: QueryIntent
    should_use_rag: bool
    selected_route_count: int = Field(ge=0)
    fallback_used: bool = False
    no_route_reason: str | None = None
    routes: list[RagKnowledgeRoute] = Field(default_factory=list)
    warnings: list[RagKnowledgeRouteWarning] = Field(default_factory=list)
    debug_lines: list[str] = Field(default_factory=list)


class RagKnowledgeRouter(Protocol):
    def route(
        self,
        query: str,
        *,
        access_scope: RagAccessScope | None = None,
        classification: QueryIntentClassification | None = None,
        max_routes: int = 2,
    ) -> RagKnowledgeRouteDecision:
        """Select one or more knowledge bases for a RAG query."""


class RuleBasedRagKnowledgeRouter:
    def __init__(
        self,
        knowledge_bases: Sequence[RagKnowledgeBaseDefinition] | None = None,
    ) -> None:
        self.knowledge_bases = list(knowledge_bases or default_rag_knowledge_bases())

    def route(
        self,
        query: str,
        *,
        access_scope: RagAccessScope | None = None,
        classification: QueryIntentClassification | None = None,
        max_routes: int = 2,
    ) -> RagKnowledgeRouteDecision:
        if max_routes <= 0:
            raise ValueError("max_routes must be greater than 0")

        selected_classification = classification or classify_query_intent(query)
        normalized_query = selected_classification.normalized_query or normalize_query_text(query)
        if not selected_classification.should_use_rag:
            return _build_decision(
                normalized_query=normalized_query,
                intent=selected_classification.intent,
                should_use_rag=False,
                no_route_reason="query intent does not use RAG retrieval",
                warnings=["RAG_ROUTE_QUERY_INTENT_NOT_RAG"],
            )

        candidates: list[tuple[RagKnowledgeRoute, bool, int]] = []
        filtered_by_access = False
        for index, definition in enumerate(self.knowledge_bases):
            if selected_classification.intent not in definition.supported_intents:
                continue
            matched_keywords = _matched_keywords(normalized_query, definition.keywords)
            if not matched_keywords and not definition.is_fallback:
                continue
            route_scope = _build_route_scope(definition, access_scope)
            if route_scope is None:
                filtered_by_access = True
                continue
            score = _route_score(definition, matched_keywords)
            route = RagKnowledgeRoute(
                knowledge_base_id=definition.knowledge_base_id,
                collection_name=definition.collection_name,
                display_name=definition.display_name,
                route_score=score,
                matched_keywords=matched_keywords,
                business_domains=route_scope.business_domains,
                doc_types=route_scope.doc_types,
                permission_groups=route_scope.permission_groups,
                payload_filter=build_access_scope_filter(route_scope),
                reasons=_route_reasons(definition, matched_keywords),
            )
            candidates.append((route, definition.is_fallback, index))

        if not candidates:
            warnings: list[RagKnowledgeRouteWarning] = ["RAG_ROUTE_NO_ACCESSIBLE_KNOWLEDGE_BASE"]
            if filtered_by_access:
                warnings.append("RAG_ROUTE_CANDIDATE_FILTERED_BY_ACCESS_SCOPE")
            return _build_decision(
                normalized_query=normalized_query,
                intent=selected_classification.intent,
                should_use_rag=True,
                no_route_reason="no accessible knowledge base matched this RAG query",
                warnings=warnings,
            )

        candidates.sort(
            key=lambda item: (
                -item[0].route_score,
                -_definition_priority(self.knowledge_bases, item[0].knowledge_base_id),
                item[2],
            )
        )
        selected_candidates = candidates[:max_routes]
        selected_routes = [route for route, _, _ in selected_candidates]
        warnings = _route_warnings(
            candidates=candidates,
            selected_candidates=selected_candidates,
            filtered_by_access=filtered_by_access,
            max_routes=max_routes,
        )
        return _build_decision(
            normalized_query=normalized_query,
            intent=selected_classification.intent,
            should_use_rag=True,
            routes=selected_routes,
            warnings=warnings,
        )


def route_rag_knowledge_bases(
    query: str,
    *,
    router: RagKnowledgeRouter | None = None,
    access_scope: RagAccessScope | None = None,
    classification: QueryIntentClassification | None = None,
    max_routes: int = 2,
) -> RagKnowledgeRouteDecision:
    selected_router = router or RuleBasedRagKnowledgeRouter()
    return selected_router.route(
        query,
        access_scope=access_scope,
        classification=classification,
        max_routes=max_routes,
    )


def default_rag_knowledge_bases() -> list[RagKnowledgeBaseDefinition]:
    return [
        RagKnowledgeBaseDefinition(
            knowledge_base_id="customer_policy_refund",
            collection_name="kb_customer_policy",
            display_name="Customer refund and return policy",
            supported_intents=["policy_lookup"],
            business_domains=["refund"],
            doc_types=["policy", "faq"],
            permission_groups=["customer_service", "public"],
            keywords=["退款", "退货", "无理由", "质量问题", "运费", "邮费", "超过七天", "超过7天"],
            priority=90,
        ),
        RagKnowledgeBaseDefinition(
            knowledge_base_id="customer_policy_logistics",
            collection_name="kb_customer_policy",
            display_name="Customer order and logistics policy",
            supported_intents=["policy_lookup"],
            business_domains=["order", "logistics"],
            doc_types=["policy", "faq"],
            permission_groups=["customer_service", "public"],
            keywords=["发货", "物流", "签收", "到货", "运单", "配送", "时效"],
            priority=80,
        ),
        RagKnowledgeBaseDefinition(
            knowledge_base_id="account_security_faq",
            collection_name="kb_account_security",
            display_name="Account security FAQ",
            supported_intents=["policy_lookup"],
            business_domains=["account", "security"],
            doc_types=["faq", "policy"],
            permission_groups=["customer_service", "public"],
            keywords=["账号", "账户", "登录", "密码", "验证码", "安全", "冻结", "绑定", "邮箱"],
            priority=80,
        ),
        RagKnowledgeBaseDefinition(
            knowledge_base_id="customer_service_process",
            collection_name="kb_customer_process",
            display_name="Customer service process SOP",
            supported_intents=["process_lookup"],
            business_domains=["ticket", "after_sale", "process"],
            doc_types=["sop", "process"],
            permission_groups=["customer_service", "internal_staff"],
            keywords=["流程", "步骤", "怎么申请", "如何申请", "怎么操作", "售后", "换货", "材料"],
            priority=85,
        ),
        RagKnowledgeBaseDefinition(
            knowledge_base_id="internal_escalation_process",
            collection_name="kb_internal_process",
            display_name="Internal escalation process",
            supported_intents=["process_lookup"],
            business_domains=["internal", "process"],
            doc_types=["sop"],
            permission_groups=["internal_staff", "admin"],
            keywords=["内部", "升级", "审核", "管理员", "二线", "风控"],
            priority=95,
        ),
        RagKnowledgeBaseDefinition(
            knowledge_base_id="customer_policy_general",
            collection_name="kb_customer_policy",
            display_name="General customer policy and FAQ",
            supported_intents=["policy_lookup"],
            business_domains=["refund", "order", "logistics", "account", "security"],
            doc_types=["policy", "faq"],
            permission_groups=["customer_service", "public"],
            priority=10,
            is_fallback=True,
        ),
    ]


def format_rag_knowledge_route_decision(
    decision: RagKnowledgeRouteDecision,
) -> list[str]:
    lines = [
        (
            f"intent={decision.intent} should_use_rag={decision.should_use_rag} "
            f"routes={decision.selected_route_count} fallback={decision.fallback_used} "
            f"warnings={','.join(decision.warnings) or '-'}"
        )
    ]
    if decision.no_route_reason:
        lines.append(f"no_route_reason={decision.no_route_reason}")
    for route in decision.routes:
        lines.append(
            (
                f"- kb={route.knowledge_base_id} collection={route.collection_name} "
                f"score={route.route_score:.4f} domains={','.join(route.business_domains) or '-'} "
                f"doc_types={','.join(route.doc_types) or '-'} "
                f"permissions={','.join(route.permission_groups) or '-'} "
                f"keywords={','.join(route.matched_keywords) or '-'} "
                f"reasons={','.join(route.reasons) or '-'}"
            )
        )
    return lines


def _build_decision(
    *,
    normalized_query: str,
    intent: QueryIntent,
    should_use_rag: bool,
    routes: Sequence[RagKnowledgeRoute] = (),
    no_route_reason: str | None = None,
    warnings: Sequence[RagKnowledgeRouteWarning] = (),
) -> RagKnowledgeRouteDecision:
    deduped_warnings = _dedupe(list(warnings))
    decision = RagKnowledgeRouteDecision(
        normalized_query=normalized_query,
        intent=intent,
        should_use_rag=should_use_rag,
        selected_route_count=len(routes),
        fallback_used=any(
            "fallback_knowledge_base" in route.reasons
            for route in routes
        ),
        no_route_reason=no_route_reason,
        routes=list(routes),
        warnings=deduped_warnings,
    )
    return decision.model_copy(
        update={"debug_lines": format_rag_knowledge_route_decision(decision)}
    )


def _build_route_scope(
    definition: RagKnowledgeBaseDefinition,
    access_scope: RagAccessScope | None,
) -> RagAccessScope | None:
    permission_groups = _intersect_if_restricted(
        definition.permission_groups,
        access_scope.permission_groups if access_scope else [],
    )
    business_domains = _intersect_if_restricted(
        definition.business_domains,
        access_scope.business_domains if access_scope else [],
    )
    doc_types = _intersect_if_restricted(
        definition.doc_types,
        access_scope.doc_types if access_scope else [],
    )
    if definition.permission_groups and not permission_groups:
        return None
    if definition.business_domains and not business_domains:
        return None
    if definition.doc_types and not doc_types:
        return None
    return RagAccessScope(
        user_id=access_scope.user_id if access_scope else None,
        tenant_id=access_scope.tenant_id if access_scope else None,
        owner_user_id=access_scope.owner_user_id if access_scope else None,
        permission_groups=permission_groups,
        business_domains=business_domains,
        doc_types=doc_types,
        sources=access_scope.sources if access_scope else [],
        visibilities=access_scope.visibilities if access_scope else [],
        statuses=access_scope.statuses if access_scope else [],
        excluded_statuses=access_scope.excluded_statuses if access_scope else [],
    )


def _intersect_if_restricted(
    route_values: Sequence[str],
    restricted_values: Sequence[str],
) -> list[str]:
    normalized_route_values = _dedupe([value.strip() for value in route_values if value.strip()])
    normalized_restricted_values = _dedupe([
        value.strip()
        for value in restricted_values
        if value.strip()
    ])
    if not normalized_restricted_values:
        return normalized_route_values
    return [
        value
        for value in normalized_route_values
        if value in normalized_restricted_values
    ]


def _matched_keywords(query: str, keywords: Sequence[str]) -> list[str]:
    lowered_query = query.lower()
    return [
        keyword
        for keyword in keywords
        if keyword.lower() in lowered_query
    ]


def _route_score(
    definition: RagKnowledgeBaseDefinition,
    matched_keywords: Sequence[str],
) -> float:
    score = 0.4
    score += min(len(matched_keywords) * 0.12, 0.48)
    score += min(definition.priority / 1000, 0.1)
    if definition.is_fallback:
        score -= 0.08
    return round(min(max(score, 0), 1), 6)


def _route_reasons(
    definition: RagKnowledgeBaseDefinition,
    matched_keywords: Sequence[str],
) -> list[str]:
    reasons = ["intent_matches_knowledge_base"]
    if matched_keywords:
        reasons.append("query_matches_knowledge_base_keywords")
    if definition.is_fallback:
        reasons.append("fallback_knowledge_base")
    return reasons


def _route_warnings(
    *,
    candidates: Sequence[tuple[RagKnowledgeRoute, bool, int]],
    selected_candidates: Sequence[tuple[RagKnowledgeRoute, bool, int]],
    filtered_by_access: bool,
    max_routes: int,
) -> list[RagKnowledgeRouteWarning]:
    warnings: list[RagKnowledgeRouteWarning] = []
    if filtered_by_access:
        warnings.append("RAG_ROUTE_CANDIDATE_FILTERED_BY_ACCESS_SCOPE")
    if any(is_fallback for _, is_fallback, _ in selected_candidates):
        warnings.append("RAG_ROUTE_USED_FALLBACK_KNOWLEDGE_BASE")
    if len(selected_candidates) > 1:
        warnings.append("RAG_ROUTE_MULTIPLE_KNOWLEDGE_BASES_SELECTED")
    if len(candidates) > max_routes:
        warnings.append("RAG_ROUTE_CANDIDATES_TRUNCATED")
    return warnings


def _definition_priority(
    definitions: Sequence[RagKnowledgeBaseDefinition],
    knowledge_base_id: str,
) -> int:
    for definition in definitions:
        if definition.knowledge_base_id == knowledge_base_id:
            return definition.priority
    return 0


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


class LLMRagKnowledgeRouter:
    """LLM 驱动知识库路由：按意图选择知识库（映射到 collection）。LLM 失败回退规则实现。"""

    def __init__(
        self,
        settings,
        *,
        client=None,
        knowledge_bases: Sequence[RagKnowledgeBaseDefinition] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self.knowledge_bases = list(knowledge_bases or default_rag_knowledge_bases())
        self._fallback = RuleBasedRagKnowledgeRouter(knowledge_bases=self.knowledge_bases)

    def route(
        self,
        query: str,
        *,
        access_scope: RagAccessScope | None = None,
        classification: QueryIntentClassification | None = None,
        max_routes: int = 2,
    ) -> RagKnowledgeRouteDecision:
        if not getattr(self._settings, "has_llm_api_key", False):
            return self._fallback.route(
                query,
                access_scope=access_scope,
                classification=classification,
                max_routes=max_routes,
            )
        try:
            reply = self._call_llm(query, max_routes=max_routes)
            return self._build_decision_from_reply(query, reply, max_routes=max_routes)
        except Exception:
            return self._fallback.route(
                query,
                access_scope=access_scope,
                classification=classification,
                max_routes=max_routes,
            )

    def _call_llm(self, query: str, *, max_routes: int) -> str:
        from app.services.llm_service import extract_first_reply

        client = self._client
        if client is None:
            from app.services.llm_client import create_openai_compatible_client

            client = create_openai_compatible_client(self._settings)
        candidate_ids = ", ".join(kb.knowledge_base_id for kb in self.knowledge_bases)
        completion = client.chat.completions.create(
            model=self._settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是电商客服知识库路由助手。根据用户问题选择最合适的知识库 id。"
                        f"候选知识库：{candidate_ids}。只输出知识库 id（最多 {max_routes} 个，"
                        "逗号分隔），不要解释。"
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        return extract_first_reply(completion)

    def _build_decision_from_reply(
        self,
        query: str,
        reply: str,
        *,
        max_routes: int,
    ) -> RagKnowledgeRouteDecision:
        selected_ids = [part.strip() for part in (reply or "").split(",") if part.strip()][:max_routes]
        by_id = {kb.knowledge_base_id: kb for kb in self.knowledge_bases}
        routes: list[RagKnowledgeRoute] = []
        for kb_id in selected_ids:
            kb = by_id.get(kb_id)
            if kb is None:
                continue
            routes.append(
                RagKnowledgeRoute(
                    knowledge_base_id=kb.knowledge_base_id,
                    collection_name=kb.collection_name,
                    display_name=kb.display_name,
                    route_score=1.0,
                    matched_keywords=[],
                    business_domains=kb.business_domains,
                    doc_types=kb.doc_types,
                    permission_groups=kb.permission_groups,
                    payload_filter=None,
                    reasons=["llm route"],
                )
            )
        if not routes:
            raise ValueError("LLM returned no valid knowledge base ids")
        return _build_decision(
            normalized_query=query.strip(),
            intent="policy_lookup",
            should_use_rag=True,
            routes=routes,
        )


def create_knowledge_router(settings) -> RagKnowledgeRouter:
    if getattr(settings, "rag_advanced_mode", "rule") == "llm":
        return LLMRagKnowledgeRouter(settings)
    return RuleBasedRagKnowledgeRouter()
