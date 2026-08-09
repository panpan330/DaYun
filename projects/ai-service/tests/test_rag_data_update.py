import pytest

from app.rag.data_update import (
    build_document_manifest,
    build_rag_data_update_plan,
    detect_document_changes,
    format_rag_data_update_plan,
)
from app.rag.documents import RagDocument


def make_document(
    source: str,
    content: str = "订单付款后 24 小时内发货。",
    *,
    title: str = "订单发货规则",
    doc_type: str = "policy",
    business_domain: str = "order",
    permission_group: str = "customer_service",
) -> RagDocument:
    return RagDocument(
        content=content,
        metadata={
            "source": source,
            "title": title,
            "file_name": source,
            "file_extension": ".md",
            "doc_type": doc_type,
            "business_domain": business_domain,
            "permission_group": permission_group,
        },
    )


def test_build_document_manifest_hash_changes_when_content_or_metadata_changes() -> None:
    original = build_document_manifest([make_document("shipping.md")])
    content_changed = build_document_manifest(
        [make_document("shipping.md", content="订单付款后 48 小时内发货。")]
    )
    metadata_changed = build_document_manifest(
        [
            make_document(
                "shipping.md",
                business_domain="logistics",
            )
        ]
    )

    assert original.document_count == 1
    assert original.sources == ["shipping.md"]
    assert original.manifest_hash != content_changed.manifest_hash
    assert original.manifest_hash != metadata_changed.manifest_hash


def test_detect_document_changes_classifies_new_modified_deleted_and_unchanged() -> None:
    previous = build_document_manifest(
        [
            make_document("unchanged.md"),
            make_document("modified.md", content="旧内容"),
            make_document("deleted.md"),
        ]
    )
    current = build_document_manifest(
        [
            make_document("unchanged.md"),
            make_document("modified.md", content="新内容"),
            make_document("new.md"),
        ]
    )

    changes = detect_document_changes(previous, current, include_unchanged=True)

    assert [(change.source, change.change_type) for change in changes] == [
        ("deleted.md", "deleted"),
        ("modified.md", "modified"),
        ("new.md", "new"),
        ("unchanged.md", "unchanged"),
    ]
    modified = next(change for change in changes if change.source == "modified.md")
    assert modified.old_content_hash != modified.new_content_hash
    assert modified.reason == "content hash changed"


def test_build_rag_data_update_plan_maps_changes_to_incremental_actions() -> None:
    previous = build_document_manifest(
        [
            make_document("modified.md", content="旧内容"),
            make_document("deleted.md"),
        ]
    )
    current = build_document_manifest(
        [
            make_document("modified.md", content="新内容"),
            make_document("new.md"),
        ]
    )

    plan = build_rag_data_update_plan(previous, current)
    lines = format_rag_data_update_plan(plan)

    assert plan.mode == "incremental"
    assert plan.changed_source_count == 3
    assert plan.action_counts == {
        "delete_source": 1,
        "ingest_new": 1,
        "refresh_source": 1,
    }
    assert plan.affected_sources == ["deleted.md", "modified.md", "new.md"]
    assert plan.cache_invalidation_sources == ["deleted.md", "modified.md", "new.md"]
    assert plan.should_rerun_evaluation is True
    assert any("refresh_source source=modified.md" in line for line in lines)


def test_build_rag_data_update_plan_can_include_unchanged_skip_items() -> None:
    previous = build_document_manifest([make_document("same.md")])
    current = build_document_manifest([make_document("same.md")])

    plan = build_rag_data_update_plan(
        previous,
        current,
        include_unchanged=True,
    )

    assert plan.changed_source_count == 0
    assert plan.action_counts == {"skip": 1}
    assert plan.items[0].action == "skip"
    assert plan.should_rerun_evaluation is False


def test_build_rag_data_update_plan_force_reindex_uses_collection_action() -> None:
    previous = build_document_manifest([make_document("old.md")])
    current = build_document_manifest([make_document("new.md")])

    plan = build_rag_data_update_plan(
        previous,
        current,
        force_reindex=True,
    )

    assert plan.mode == "full_reindex"
    assert plan.action_counts == {"reindex_collection": 1}
    assert plan.affected_sources == ["new.md", "old.md"]
    assert plan.items[0].source == "__collection__"
    assert plan.items[0].should_delete_before_upsert is True
    assert plan.should_rerun_evaluation is True


def test_build_document_manifest_rejects_duplicate_or_blank_sources() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_document_manifest(
            [
                make_document("same.md"),
                make_document("same.md", content="另一份内容"),
            ]
        )

    with pytest.raises(ValueError, match="source"):
        build_document_manifest(
            [
                RagDocument(
                    content="内容",
                    metadata={"source": " "},
                )
            ]
        )
