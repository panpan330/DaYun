import pytest

from app.rag.filters import (
    RagAccessScope,
    build_access_scope_filter,
    build_match_any_condition,
    build_match_condition,
    build_metadata_filter_report,
    build_payload_filter,
    combine_payload_filters,
    metadata_matches_access_scope,
    normalize_payload_filter,
)


def test_build_match_condition_uses_qdrant_match_value_shape() -> None:
    condition = build_match_condition("permission_group", " customer_service ")

    assert condition == {
        "key": "permission_group",
        "match": {
            "value": "customer_service",
        },
    }


def test_build_match_condition_returns_none_for_missing_value() -> None:
    assert build_match_condition("permission_group", None) is None


def test_build_match_any_condition_uses_qdrant_match_any_shape() -> None:
    condition = build_match_any_condition(
        "permission_group",
        [" customer_service ", "internal_staff", "customer_service"],
    )

    assert condition == {
        "key": "permission_group",
        "match": {
            "any": ["customer_service", "internal_staff"],
        },
    }


def test_build_match_condition_rejects_blank_value() -> None:
    with pytest.raises(ValueError, match="permission_group"):
        build_match_condition("permission_group", "   ")


def test_build_payload_filter_returns_none_when_no_filters_requested() -> None:
    assert build_payload_filter() is None


def test_build_payload_filter_combines_conditions_with_must() -> None:
    payload_filter = build_payload_filter(
        permission_group="customer_service",
        business_domain="order",
        doc_type="policy",
        source="order-shipping-policy.md",
    )

    assert payload_filter == {
        "must": [
            {
                "key": "permission_group",
                "match": {"value": "customer_service"},
            },
            {
                "key": "business_domain",
                "match": {"value": "order"},
            },
            {
                "key": "doc_type",
                "match": {"value": "policy"},
            },
            {
                "key": "source",
                "match": {"value": "order-shipping-policy.md"},
            },
        ]
    }


def test_rag_access_scope_normalizes_current_user_and_allowed_ranges() -> None:
    scope = RagAccessScope(
        user_id=" U1001 ",
        tenant_id=" default ",
        permission_groups=[" customer_service ", "customer_service"],
        business_domains="refund",
        excluded_statuses=[" archived ", "deleted"],
    )

    assert scope.user_id == "U1001"
    assert scope.tenant_id == "default"
    assert scope.permission_groups == ["customer_service"]
    assert scope.business_domains == ["refund"]
    assert scope.excluded_statuses == ["archived", "deleted"]


def test_build_access_scope_filter_combines_tenant_permissions_and_exclusions() -> None:
    scope = RagAccessScope(
        user_id="U1001",
        tenant_id="default",
        permission_groups=["customer_service", "public"],
        business_domains=["refund", "order"],
        doc_types=["policy"],
        visibilities=["tenant", "public"],
        excluded_statuses=["archived", "deleted"],
    )

    payload_filter = build_access_scope_filter(scope)

    assert payload_filter == {
        "must": [
            {"key": "tenant_id", "match": {"value": "default"}},
            {
                "key": "permission_group",
                "match": {"any": ["customer_service", "public"]},
            },
            {"key": "business_domain", "match": {"any": ["refund", "order"]}},
            {"key": "doc_type", "match": {"any": ["policy"]}},
            {"key": "visibility", "match": {"any": ["tenant", "public"]}},
        ],
        "must_not": [
            {"key": "status", "match": {"any": ["archived", "deleted"]}},
        ],
    }


def test_build_payload_filter_combines_access_scope_with_direct_filters() -> None:
    payload_filter = build_payload_filter(
        access_scope=RagAccessScope(
            tenant_id="default",
            permission_groups=["customer_service"],
        ),
        business_domain="refund",
        source="refund-return-policy.md",
    )

    assert payload_filter == {
        "must": [
            {"key": "tenant_id", "match": {"value": "default"}},
            {"key": "permission_group", "match": {"any": ["customer_service"]}},
            {"key": "business_domain", "match": {"value": "refund"}},
            {"key": "source", "match": {"value": "refund-return-policy.md"}},
        ]
    }


def test_combine_payload_filters_preserves_filter_groups() -> None:
    combined = combine_payload_filters(
        {"must": [{"key": "tenant_id", "match": {"value": "default"}}]},
        {"must_not": [{"key": "status", "match": {"value": "archived"}}]},
    )

    assert combined == {
        "must": [{"key": "tenant_id", "match": {"value": "default"}}],
        "must_not": [{"key": "status", "match": {"value": "archived"}}],
    }


def test_metadata_matches_access_scope_checks_tenant_permission_and_status() -> None:
    scope = RagAccessScope(
        tenant_id="default",
        permission_groups=["customer_service"],
        business_domains=["refund"],
        excluded_statuses=["archived"],
    )

    assert metadata_matches_access_scope(
        {
            "tenant_id": "default",
            "permission_group": "customer_service",
            "business_domain": "refund",
            "status": "published",
        },
        scope,
    )
    assert not metadata_matches_access_scope(
        {
            "tenant_id": "default",
            "permission_group": "internal_staff",
            "business_domain": "refund",
            "status": "published",
        },
        scope,
    )
    assert not metadata_matches_access_scope(
        {
            "tenant_id": "default",
            "permission_group": "customer_service",
            "business_domain": "refund",
            "status": "archived",
        },
        scope,
    )


def test_build_metadata_filter_report_outputs_debug_context() -> None:
    report = build_metadata_filter_report(
        RagAccessScope(
            user_id="U1001",
            tenant_id="default",
            permission_groups=["customer_service"],
        )
    )

    assert report.user_id == "U1001"
    assert report.tenant_id == "default"
    assert report.applied_fields == ["tenant_id", "permission_group"]
    assert report.payload_filter is not None
    assert report.debug_lines == [
        "user_id=U1001 tenant_id=default applied_fields=tenant_id,permission_group has_filter=True",
        "filter_groups=must",
    ]


def test_normalize_payload_filter_rejects_empty_filter() -> None:
    with pytest.raises(ValueError, match="payload_filter"):
        normalize_payload_filter({})


def test_rag_access_scope_rejects_blank_list_item() -> None:
    with pytest.raises(ValueError, match="non-blank"):
        RagAccessScope(permission_groups=["customer_service", " "])
