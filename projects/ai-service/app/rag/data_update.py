from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rag.documents import RagDocument


RagDocumentChangeType = Literal["new", "modified", "deleted", "unchanged"]
RagDataUpdateAction = Literal[
    "ingest_new",
    "refresh_source",
    "delete_source",
    "skip",
    "reindex_collection",
]
RagDataUpdateMode = Literal["incremental", "full_reindex"]


class RagDocumentManifestEntry(BaseModel):
    source: str = Field(min_length=1)
    content_hash: str = Field(min_length=16)
    metadata_hash: str = Field(min_length=16)
    content_chars: int = Field(ge=0)
    metadata_summary: dict[str, str] = Field(default_factory=dict)


class RagDocumentManifest(BaseModel):
    document_count: int = Field(ge=0)
    manifest_hash: str = Field(min_length=16)
    sources: list[str] = Field(default_factory=list)
    entries: dict[str, RagDocumentManifestEntry] = Field(default_factory=dict)


class RagDocumentChange(BaseModel):
    source: str = Field(min_length=1)
    change_type: RagDocumentChangeType
    reason: str = Field(min_length=1)
    old_content_hash: str | None = None
    new_content_hash: str | None = None
    old_metadata_hash: str | None = None
    new_metadata_hash: str | None = None


class RagDataUpdatePlanItem(BaseModel):
    source: str = Field(min_length=1)
    action: RagDataUpdateAction
    change_type: RagDocumentChangeType | None = None
    reason: str = Field(min_length=1)
    should_delete_before_upsert: bool = False
    should_invalidate_cache: bool = False
    should_rerun_evaluation: bool = False


class RagDataUpdatePlan(BaseModel):
    mode: RagDataUpdateMode
    previous_manifest_hash: str | None = None
    current_manifest_hash: str = Field(min_length=16)
    item_count: int = Field(ge=0)
    changed_source_count: int = Field(ge=0)
    action_counts: dict[str, int] = Field(default_factory=dict)
    affected_sources: list[str] = Field(default_factory=list)
    cache_invalidation_sources: list[str] = Field(default_factory=list)
    should_rerun_evaluation: bool = False
    items: list[RagDataUpdatePlanItem] = Field(default_factory=list)


def build_document_manifest(
    documents: Sequence[RagDocument],
) -> RagDocumentManifest:
    entries: dict[str, RagDocumentManifestEntry] = {}
    for document in documents:
        entry = _build_manifest_entry(document)
        if entry.source in entries:
            raise ValueError(f"duplicate document source: {entry.source}")
        entries[entry.source] = entry

    sources = sorted(entries)
    manifest_payload = [
        entries[source].model_dump(mode="json")
        for source in sources
    ]
    return RagDocumentManifest(
        document_count=len(entries),
        manifest_hash=_hash_json(manifest_payload),
        sources=sources,
        entries={source: entries[source] for source in sources},
    )


def detect_document_changes(
    previous: RagDocumentManifest,
    current: RagDocumentManifest,
    *,
    include_unchanged: bool = False,
) -> list[RagDocumentChange]:
    changes: list[RagDocumentChange] = []
    all_sources = sorted(set(previous.sources) | set(current.sources))
    for source in all_sources:
        old_entry = previous.entries.get(source)
        new_entry = current.entries.get(source)
        if old_entry is None and new_entry is not None:
            changes.append(
                RagDocumentChange(
                    source=source,
                    change_type="new",
                    reason="source exists only in the current manifest",
                    new_content_hash=new_entry.content_hash,
                    new_metadata_hash=new_entry.metadata_hash,
                )
            )
        elif old_entry is not None and new_entry is None:
            changes.append(
                RagDocumentChange(
                    source=source,
                    change_type="deleted",
                    reason="source exists only in the previous manifest",
                    old_content_hash=old_entry.content_hash,
                    old_metadata_hash=old_entry.metadata_hash,
                )
            )
        elif old_entry is not None and new_entry is not None:
            if _entry_changed(old_entry, new_entry):
                changes.append(
                    RagDocumentChange(
                        source=source,
                        change_type="modified",
                        reason=_modified_reason(old_entry, new_entry),
                        old_content_hash=old_entry.content_hash,
                        new_content_hash=new_entry.content_hash,
                        old_metadata_hash=old_entry.metadata_hash,
                        new_metadata_hash=new_entry.metadata_hash,
                    )
                )
            elif include_unchanged:
                changes.append(
                    RagDocumentChange(
                        source=source,
                        change_type="unchanged",
                        reason="content and metadata hashes are unchanged",
                        old_content_hash=old_entry.content_hash,
                        new_content_hash=new_entry.content_hash,
                        old_metadata_hash=old_entry.metadata_hash,
                        new_metadata_hash=new_entry.metadata_hash,
                    )
                )
    return changes


def build_rag_data_update_plan(
    previous: RagDocumentManifest,
    current: RagDocumentManifest,
    *,
    force_reindex: bool = False,
    include_unchanged: bool = False,
) -> RagDataUpdatePlan:
    if force_reindex:
        affected_sources = sorted(set(previous.sources) | set(current.sources))
        item = RagDataUpdatePlanItem(
            source="__collection__",
            action="reindex_collection",
            reason=(
                "force_reindex=True; rebuild the collection because chunking, "
                "embedding, metadata schema, or index settings changed"
            ),
            should_delete_before_upsert=True,
            should_invalidate_cache=True,
            should_rerun_evaluation=True,
        )
        return RagDataUpdatePlan(
            mode="full_reindex",
            previous_manifest_hash=previous.manifest_hash,
            current_manifest_hash=current.manifest_hash,
            item_count=1,
            changed_source_count=len(affected_sources),
            action_counts={"reindex_collection": 1},
            affected_sources=affected_sources,
            cache_invalidation_sources=affected_sources,
            should_rerun_evaluation=True,
            items=[item],
        )

    changes = detect_document_changes(
        previous,
        current,
        include_unchanged=include_unchanged,
    )
    items = [_plan_item_from_change(change) for change in changes]
    affected_sources = [
        item.source
        for item in items
        if item.action != "skip"
    ]
    cache_invalidation_sources = [
        item.source
        for item in items
        if item.should_invalidate_cache and item.action != "skip"
    ]
    action_counts = Counter(item.action for item in items)
    return RagDataUpdatePlan(
        mode="incremental",
        previous_manifest_hash=previous.manifest_hash,
        current_manifest_hash=current.manifest_hash,
        item_count=len(items),
        changed_source_count=len(affected_sources),
        action_counts=dict(sorted(action_counts.items())),
        affected_sources=affected_sources,
        cache_invalidation_sources=cache_invalidation_sources,
        should_rerun_evaluation=any(item.should_rerun_evaluation for item in items),
        items=items,
    )


def format_rag_data_update_plan(plan: RagDataUpdatePlan) -> list[str]:
    lines = [
        "RAG data update plan",
        f"mode: {plan.mode}",
        f"items: {plan.item_count}",
        f"changed_sources: {plan.changed_source_count}",
        f"actions: {plan.action_counts}",
        f"affected_sources: {_format_sources(plan.affected_sources)}",
        f"cache_invalidation_sources: {_format_sources(plan.cache_invalidation_sources)}",
        f"should_rerun_evaluation: {plan.should_rerun_evaluation}",
    ]
    for item in plan.items:
        lines.append(
            (
                f"- {item.action} source={item.source} "
                f"change={item.change_type or '-'} "
                f"delete_before_upsert={item.should_delete_before_upsert} "
                f"invalidate_cache={item.should_invalidate_cache} "
                f"rerun_eval={item.should_rerun_evaluation} "
                f"reason={item.reason}"
            )
        )
    return lines


def _build_manifest_entry(document: RagDocument) -> RagDocumentManifestEntry:
    source = _metadata_text(document.metadata.get("source"), field_name="source")
    metadata_summary = _metadata_summary(document.metadata)
    return RagDocumentManifestEntry(
        source=source,
        content_hash=_hash_text(document.content),
        metadata_hash=_hash_json(metadata_summary),
        content_chars=len(document.content),
        metadata_summary=metadata_summary,
    )


def _plan_item_from_change(change: RagDocumentChange) -> RagDataUpdatePlanItem:
    if change.change_type == "new":
        return RagDataUpdatePlanItem(
            source=change.source,
            action="ingest_new",
            change_type=change.change_type,
            reason="new source should be embedded and upserted",
            should_invalidate_cache=True,
            should_rerun_evaluation=True,
        )
    if change.change_type == "modified":
        return RagDataUpdatePlanItem(
            source=change.source,
            action="refresh_source",
            change_type=change.change_type,
            reason="modified source should delete old chunks before upserting new chunks",
            should_delete_before_upsert=True,
            should_invalidate_cache=True,
            should_rerun_evaluation=True,
        )
    if change.change_type == "deleted":
        return RagDataUpdatePlanItem(
            source=change.source,
            action="delete_source",
            change_type=change.change_type,
            reason="deleted source should remove its vector-store points",
            should_invalidate_cache=True,
            should_rerun_evaluation=True,
        )
    return RagDataUpdatePlanItem(
        source=change.source,
        action="skip",
        change_type=change.change_type,
        reason="source is unchanged",
    )


def _entry_changed(
    old_entry: RagDocumentManifestEntry,
    new_entry: RagDocumentManifestEntry,
) -> bool:
    return (
        old_entry.content_hash != new_entry.content_hash
        or old_entry.metadata_hash != new_entry.metadata_hash
    )


def _modified_reason(
    old_entry: RagDocumentManifestEntry,
    new_entry: RagDocumentManifestEntry,
) -> str:
    content_changed = old_entry.content_hash != new_entry.content_hash
    metadata_changed = old_entry.metadata_hash != new_entry.metadata_hash
    if content_changed and metadata_changed:
        return "content and metadata hashes changed"
    if content_changed:
        return "content hash changed"
    return "metadata hash changed"


def _metadata_summary(metadata: Mapping[str, Any]) -> dict[str, str]:
    summary: dict[str, str] = {}
    for key in (
        "source",
        "title",
        "doc_type",
        "business_domain",
        "permission_group",
        "visibility",
        "status",
        "tenant_id",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            summary[key] = ",".join(str(item).strip() for item in value if str(item).strip())
        else:
            summary[key] = str(value).strip()
    return summary


def _metadata_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} metadata must not be blank")
    return value.strip()


def _hash_json(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _hash_text(serialized)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _format_sources(sources: Sequence[str]) -> str:
    if not sources:
        return "-"
    return ", ".join(sources)
