from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from pydantic import BaseModel, Field, field_validator

from app.rag.documents import Metadata


PayloadFilter: TypeAlias = dict[str, list[dict[str, Any]]]

FILTERABLE_METADATA_KEYS = (
    "tenant_id",
    "owner_user_id",
    "permission_group",
    "business_domain",
    "doc_type",
    "source",
    "visibility",
    "status",
)


class RagAccessScope(BaseModel):
    user_id: str | None = None
    tenant_id: str | None = None
    owner_user_id: str | None = None
    permission_groups: list[str] = Field(default_factory=list)
    business_domains: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    visibilities: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    excluded_statuses: list[str] = Field(default_factory=list)

    @field_validator("user_id", "tenant_id", "owner_user_id", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("access scope text fields must be strings")
        normalized = value.strip()
        return normalized or None

    @field_validator(
        "permission_groups",
        "business_domains",
        "doc_types",
        "sources",
        "visibilities",
        "statuses",
        "excluded_statuses",
        mode="before",
    )
    @classmethod
    def normalize_string_list(cls, value: object) -> object:
        if value is None:
            return []
        values: Sequence[object]
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, Sequence):
            values = value
        else:
            raise ValueError("access scope list fields must be lists of strings")

        normalized: list[str] = []
        for item in values:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("access scope list fields must contain non-blank strings")
            normalized_item = item.strip()
            if normalized_item not in normalized:
                normalized.append(normalized_item)
        return normalized


class MetadataFilterReport(BaseModel):
    user_id: str | None = None
    tenant_id: str | None = None
    applied_fields: list[str] = Field(default_factory=list)
    payload_filter: PayloadFilter | None = None
    debug_lines: list[str] = Field(default_factory=list)


def build_payload_filter(
    *,
    access_scope: RagAccessScope | None = None,
    tenant_id: str | None = None,
    owner_user_id: str | None = None,
    permission_group: str | None = None,
    permission_groups: Sequence[str] | None = None,
    business_domain: str | None = None,
    business_domains: Sequence[str] | None = None,
    doc_type: str | None = None,
    doc_types: Sequence[str] | None = None,
    source: str | None = None,
    sources: Sequence[str] | None = None,
    visibility: str | None = None,
    visibilities: Sequence[str] | None = None,
    status: str | None = None,
    statuses: Sequence[str] | None = None,
    excluded_statuses: Sequence[str] | None = None,
) -> PayloadFilter | None:
    conditions: list[dict[str, Any]] = []
    must_not_conditions: list[dict[str, Any]] = []

    if access_scope is not None:
        scoped_filter = build_access_scope_filter(access_scope)
        if scoped_filter is not None:
            conditions.extend(scoped_filter.get("must", []))
            must_not_conditions.extend(scoped_filter.get("must_not", []))

    for key, value in (
        ("tenant_id", tenant_id),
        ("owner_user_id", owner_user_id),
        ("permission_group", permission_group),
        ("business_domain", business_domain),
        ("doc_type", doc_type),
        ("source", source),
        ("visibility", visibility),
        ("status", status),
    ):
        condition = build_match_condition(key, value)
        if condition is not None:
            conditions.append(condition)

    for key, values in (
        ("permission_group", permission_groups),
        ("business_domain", business_domains),
        ("doc_type", doc_types),
        ("source", sources),
        ("visibility", visibilities),
        ("status", statuses),
    ):
        condition = build_match_any_condition(key, values)
        if condition is not None:
            conditions.append(condition)

    excluded_status_condition = build_match_any_condition("status", excluded_statuses)
    if excluded_status_condition is not None:
        must_not_conditions.append(excluded_status_condition)

    if not conditions and not must_not_conditions:
        return None

    payload_filter: PayloadFilter = {}
    if conditions:
        payload_filter["must"] = conditions
    if must_not_conditions:
        payload_filter["must_not"] = must_not_conditions
    return payload_filter


def build_access_scope_filter(scope: RagAccessScope) -> PayloadFilter | None:
    return build_payload_filter(
        tenant_id=scope.tenant_id,
        owner_user_id=scope.owner_user_id,
        permission_groups=scope.permission_groups,
        business_domains=scope.business_domains,
        doc_types=scope.doc_types,
        sources=scope.sources,
        visibilities=scope.visibilities,
        statuses=scope.statuses,
        excluded_statuses=scope.excluded_statuses,
    )


def build_metadata_filter_report(
    access_scope: RagAccessScope | None = None,
    *,
    payload_filter: Mapping[str, Any] | None = None,
) -> MetadataFilterReport:
    normalized_filter = normalize_payload_filter(payload_filter)
    if access_scope is not None:
        scoped_filter = build_access_scope_filter(access_scope)
        normalized_filter = combine_payload_filters(scoped_filter, normalized_filter)

    report = MetadataFilterReport(
        user_id=access_scope.user_id if access_scope else None,
        tenant_id=access_scope.tenant_id if access_scope else None,
        applied_fields=_extract_applied_fields(normalized_filter),
        payload_filter=normalized_filter,
    )
    return report.model_copy(
        update={"debug_lines": format_metadata_filter_report_for_debug(report)}
    )


def build_match_condition(
    key: str,
    value: str | None,
) -> dict[str, Any] | None:
    if value is None:
        return None

    normalized_key = key.strip()
    normalized_value = value.strip()
    if not normalized_key:
        raise ValueError("filter key must not be blank")
    if not normalized_value:
        raise ValueError(f"{normalized_key} filter value must not be blank")

    return {
        "key": normalized_key,
        "match": {
            "value": normalized_value,
        },
    }


def build_match_any_condition(
    key: str,
    values: Sequence[str] | None,
) -> dict[str, Any] | None:
    normalized_values = _normalize_string_values(key, values)
    if not normalized_values:
        return None

    normalized_key = key.strip()
    if not normalized_key:
        raise ValueError("filter key must not be blank")

    return {
        "key": normalized_key,
        "match": {
            "any": normalized_values,
        },
    }


def combine_payload_filters(
    *payload_filters: Mapping[str, Any] | None,
) -> PayloadFilter | None:
    combined: PayloadFilter = {}
    for payload_filter in payload_filters:
        normalized_filter = normalize_payload_filter(payload_filter)
        if normalized_filter is None:
            continue
        for group_name in ("must", "should", "must_not"):
            group = normalized_filter.get(group_name)
            if group is None:
                continue
            if not isinstance(group, list):
                raise ValueError(f"payload_filter {group_name} must be a list")
            combined.setdefault(group_name, []).extend(group)

    return combined or None


def metadata_matches_access_scope(
    metadata: Metadata,
    scope: RagAccessScope,
) -> bool:
    if scope.tenant_id and _metadata_text(metadata, "tenant_id") != scope.tenant_id:
        return False
    if (
        scope.owner_user_id
        and _metadata_text(metadata, "owner_user_id") != scope.owner_user_id
    ):
        return False
    if scope.permission_groups and (
        _metadata_text(metadata, "permission_group") not in scope.permission_groups
    ):
        return False
    if scope.business_domains and (
        _metadata_text(metadata, "business_domain") not in scope.business_domains
    ):
        return False
    if scope.doc_types and _metadata_text(metadata, "doc_type") not in scope.doc_types:
        return False
    if scope.sources and _metadata_text(metadata, "source") not in scope.sources:
        return False
    if scope.visibilities and (
        _metadata_text(metadata, "visibility") not in scope.visibilities
    ):
        return False
    if scope.statuses and _metadata_text(metadata, "status") not in scope.statuses:
        return False
    if (
        scope.excluded_statuses
        and _metadata_text(metadata, "status") in scope.excluded_statuses
    ):
        return False
    return True


def format_metadata_filter_report_for_debug(
    report: MetadataFilterReport,
) -> list[str]:
    fields = ",".join(report.applied_fields) or "-"
    lines = [
        (
            f"user_id={report.user_id or '-'} tenant_id={report.tenant_id or '-'} "
            f"applied_fields={fields} has_filter={report.payload_filter is not None}"
        )
    ]
    if report.payload_filter is not None:
        lines.append(f"filter_groups={','.join(report.payload_filter.keys())}")
    return lines


def normalize_payload_filter(
    payload_filter: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if payload_filter is None:
        return None
    if not payload_filter:
        raise ValueError("payload_filter must not be empty")
    return dict(payload_filter)


def _normalize_string_values(
    key: str,
    values: Sequence[str] | None,
) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]

    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{key} filter values must be strings")
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError(f"{key} filter value must not be blank")
        if normalized_value not in normalized:
            normalized.append(normalized_value)
    return normalized


def _extract_applied_fields(
    payload_filter: Mapping[str, Any] | None,
) -> list[str]:
    if payload_filter is None:
        return []
    fields: list[str] = []
    for group_name in ("must", "should", "must_not"):
        group = payload_filter.get(group_name)
        if not isinstance(group, list):
            continue
        for condition in group:
            if not isinstance(condition, Mapping):
                continue
            key = condition.get("key")
            if isinstance(key, str) and key not in fields:
                fields.append(key)
    return fields


def _metadata_text(metadata: Metadata, key: str) -> str | None:
    value = metadata.get(key)
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
