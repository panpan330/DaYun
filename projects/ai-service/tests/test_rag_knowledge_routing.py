from app.rag.filters import RagAccessScope
from app.rag.knowledge_routing import (
    RuleBasedRagKnowledgeRouter,
    default_rag_knowledge_bases,
    format_rag_knowledge_route_decision,
    route_rag_knowledge_bases,
)
from app.rag.query_intent import classify_query_intent


def test_route_rag_knowledge_bases_selects_refund_policy_kb() -> None:
    decision = route_rag_knowledge_bases(
        "质量问题退货运费谁承担？",
        access_scope=RagAccessScope(
            tenant_id="default",
            permission_groups=["customer_service"],
            excluded_statuses=["archived"],
        ),
    )

    assert decision.should_use_rag is True
    assert decision.intent == "policy_lookup"
    assert decision.routes[0].knowledge_base_id == "customer_policy_refund"
    assert decision.routes[0].collection_name == "kb_customer_policy"
    assert decision.routes[0].business_domains == ["refund"]
    assert decision.routes[0].permission_groups == ["customer_service"]
    assert decision.routes[0].payload_filter == {
        "must": [
            {"key": "tenant_id", "match": {"value": "default"}},
            {"key": "permission_group", "match": {"any": ["customer_service"]}},
            {"key": "business_domain", "match": {"any": ["refund"]}},
            {"key": "doc_type", "match": {"any": ["policy", "faq"]}},
        ],
        "must_not": [
            {"key": "status", "match": {"any": ["archived"]}},
        ],
    }


def test_route_rag_knowledge_bases_selects_process_kb_for_process_intent() -> None:
    classification = classify_query_intent("售后换货流程怎么走？")

    decision = route_rag_knowledge_bases(
        "售后换货流程怎么走？",
        classification=classification,
        access_scope=RagAccessScope(permission_groups=["customer_service"]),
    )

    assert decision.intent == "process_lookup"
    assert decision.routes[0].knowledge_base_id == "customer_service_process"
    assert decision.routes[0].collection_name == "kb_customer_process"
    assert decision.routes[0].doc_types == ["sop", "process"]


def test_route_rag_knowledge_bases_skips_non_rag_intent() -> None:
    decision = route_rag_knowledge_bases("订单 A1001 到哪里了？")

    assert decision.should_use_rag is False
    assert decision.selected_route_count == 0
    assert decision.no_route_reason == "query intent does not use RAG retrieval"
    assert decision.warnings == ["RAG_ROUTE_QUERY_INTENT_NOT_RAG"]


def test_route_rag_knowledge_bases_filters_inaccessible_internal_kb() -> None:
    decision = route_rag_knowledge_bases(
        "内部升级审核流程怎么操作？",
        access_scope=RagAccessScope(permission_groups=["customer_service"]),
        max_routes=3,
    )

    assert "RAG_ROUTE_CANDIDATE_FILTERED_BY_ACCESS_SCOPE" in decision.warnings
    assert all(
        route.knowledge_base_id != "internal_escalation_process"
        for route in decision.routes
    )
    assert decision.routes[0].knowledge_base_id == "customer_service_process"


def test_route_rag_knowledge_bases_uses_general_policy_fallback() -> None:
    decision = route_rag_knowledge_bases(
        "你们有哪些政策文档？",
        access_scope=RagAccessScope(permission_groups=["customer_service"]),
    )

    assert decision.routes[0].knowledge_base_id == "customer_policy_general"
    assert decision.fallback_used is True
    assert "RAG_ROUTE_USED_FALLBACK_KNOWLEDGE_BASE" in decision.warnings


def test_route_rag_knowledge_bases_reports_multiple_routes_and_debug_lines() -> None:
    decision = route_rag_knowledge_bases(
        "账号安全和退款规则分别是什么？",
        access_scope=RagAccessScope(permission_groups=["customer_service"]),
        max_routes=2,
    )
    lines = format_rag_knowledge_route_decision(decision)

    assert decision.selected_route_count == 2
    assert "RAG_ROUTE_MULTIPLE_KNOWLEDGE_BASES_SELECTED" in decision.warnings
    assert {route.knowledge_base_id for route in decision.routes} == {
        "customer_policy_refund",
        "account_security_faq",
    }
    assert any("kb=customer_policy_refund" in line for line in lines)
    assert any("kb=account_security_faq" in line for line in lines)


def test_rule_based_router_accepts_custom_kb_catalog() -> None:
    catalog = default_rag_knowledge_bases()[:1]
    router = RuleBasedRagKnowledgeRouter(catalog)

    decision = route_rag_knowledge_bases(
        "退款多久到账？",
        router=router,
        access_scope=RagAccessScope(permission_groups=["customer_service"]),
    )

    assert [route.knowledge_base_id for route in decision.routes] == [
        "customer_policy_refund"
    ]
