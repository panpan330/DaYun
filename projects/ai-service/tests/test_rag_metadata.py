import pytest

from app.rag.metadata import (
    MetadataValidationError,
    build_qdrant_payload,
    normalize_metadata,
    validate_chunk_metadata,
    validate_document_metadata,
)


def make_document_metadata(**overrides):
    metadata = {
        "source": "order-shipping-policy.md",
        "title": "订单发货规则",
        "file_name": "order-shipping-policy.md",
        "file_extension": ".md",
        "doc_type": "policy",
        "business_domain": "order",
        "permission_group": "customer_service",
    }
    metadata.update(overrides)
    return metadata


def make_chunk_metadata(**overrides):
    metadata = make_document_metadata(
        chunk_id="order_shipping_policy_chunk_0001",
        chunk_index=1,
        chunk_count=3,
        chunk_size_chars=180,
        section="正常发货时效",
    )
    metadata.update(overrides)
    return metadata


def test_normalize_metadata_trims_strings_and_list_values() -> None:
    metadata = normalize_metadata(
        {
            " source ": " order-shipping-policy.md ",
            "tags": [" order ", " ", " shipping "],
            "chunk_index": 1,
            "visible": True,
        }
    )

    assert metadata == {
        "source": "order-shipping-policy.md",
        "tags": ["order", "shipping"],
        "chunk_index": 1,
        "visible": True,
    }


def test_normalize_metadata_rejects_blank_key() -> None:
    with pytest.raises(MetadataValidationError, match="key"):
        normalize_metadata({"   ": "value"})


def test_validate_document_metadata_accepts_required_document_fields() -> None:
    metadata = validate_document_metadata(make_document_metadata())

    assert metadata["source"] == "order-shipping-policy.md"
    assert metadata["permission_group"] == "customer_service"


def test_validate_document_metadata_requires_permission_group() -> None:
    metadata = make_document_metadata()
    metadata.pop("permission_group")

    with pytest.raises(MetadataValidationError, match="permission_group"):
        validate_document_metadata(metadata)


def test_validate_document_metadata_rejects_unsupported_file_extension() -> None:
    metadata = make_document_metadata(file_extension=".pdf")

    with pytest.raises(MetadataValidationError, match="file_extension"):
        validate_document_metadata(metadata)


def test_validate_chunk_metadata_accepts_chunk_fields() -> None:
    metadata = validate_chunk_metadata(make_chunk_metadata())

    assert metadata["chunk_id"] == "order_shipping_policy_chunk_0001"
    assert metadata["chunk_index"] == 1
    assert metadata["section"] == "正常发货时效"


def test_validate_chunk_metadata_requires_positive_chunk_index() -> None:
    metadata = make_chunk_metadata(chunk_index=0)

    with pytest.raises(MetadataValidationError, match="chunk_index"):
        validate_chunk_metadata(metadata)


def test_build_qdrant_payload_keeps_only_public_payload_fields() -> None:
    metadata = make_chunk_metadata(
        internal_note="do not store",
        tags=["order"],
        tenant_id="default",
        owner_user_id="U1001",
        visibility="tenant",
        status="published",
    )

    payload = build_qdrant_payload(
        chunk_id="order_shipping_policy_chunk_0001",
        content="订单付款后 24 小时内发货。",
        metadata=metadata,
    )

    assert payload["content"] == "订单付款后 24 小时内发货。"
    assert payload["chunk_id"] == "order_shipping_policy_chunk_0001"
    assert payload["source"] == "order-shipping-policy.md"
    assert payload["tags"] == ["order"]
    assert payload["tenant_id"] == "default"
    assert payload["owner_user_id"] == "U1001"
    assert payload["visibility"] == "tenant"
    assert payload["status"] == "published"
    assert "internal_note" not in payload


def test_build_qdrant_payload_rejects_chunk_id_mismatch() -> None:
    metadata = make_chunk_metadata(chunk_id="different_chunk")

    with pytest.raises(MetadataValidationError, match="match"):
        build_qdrant_payload(
            chunk_id="order_shipping_policy_chunk_0001",
            content="订单付款后 24 小时内发货。",
            metadata=metadata,
        )


def test_build_qdrant_payload_rejects_blank_content() -> None:
    with pytest.raises(MetadataValidationError, match="content"):
        build_qdrant_payload(
            chunk_id="order_shipping_policy_chunk_0001",
            content="   ",
            metadata=make_chunk_metadata(),
        )
