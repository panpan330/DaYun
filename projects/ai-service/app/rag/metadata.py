from collections.abc import Mapping

from app.rag.documents import Metadata, MetadataValue


DOCUMENT_METADATA_KEYS = (
    "tenant_id",
    "owner_user_id",
    "source",
    "title",
    "file_name",
    "file_extension",
    "doc_type",
    "business_domain",
    "permission_group",
    "visibility",
    "status",
)
CHUNK_METADATA_KEYS = (
    *DOCUMENT_METADATA_KEYS,
    "chunk_id",
    "chunk_index",
    "chunk_count",
    "chunk_size_chars",
    "section",
)
QDRANT_PAYLOAD_METADATA_KEYS = (
    *CHUNK_METADATA_KEYS,
    "tags",
)
REQUIRED_DOCUMENT_METADATA_KEYS = (
    "source",
    "title",
    "file_name",
    "file_extension",
    "doc_type",
    "business_domain",
    "permission_group",
)
REQUIRED_CHUNK_METADATA_KEYS = (
    *DOCUMENT_METADATA_KEYS,
    "chunk_id",
    "chunk_index",
    "chunk_count",
    "chunk_size_chars",
)


class MetadataValidationError(ValueError):
    pass


def normalize_metadata(metadata: Mapping[str, MetadataValue]) -> Metadata:
    normalized: Metadata = {}
    for key, value in metadata.items():
        normalized_key = key.strip()
        if not normalized_key:
            raise MetadataValidationError("metadata key must not be blank")
        normalized[normalized_key] = normalize_metadata_value(value)
    return normalized


def normalize_metadata_value(value: MetadataValue) -> MetadataValue:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, list):
        return [item.strip() for item in value if item.strip()]
    raise MetadataValidationError("unsupported metadata value type")


def validate_document_metadata(metadata: Mapping[str, MetadataValue]) -> Metadata:
    normalized = normalize_metadata(metadata)
    for key in REQUIRED_DOCUMENT_METADATA_KEYS:
        _require_non_blank_string(normalized, key)
    if normalized["file_extension"] not in {".md", ".txt"}:
        raise MetadataValidationError("file_extension must be .md or .txt")
    return normalized


def validate_chunk_metadata(metadata: Mapping[str, MetadataValue]) -> Metadata:
    normalized = validate_document_metadata(metadata)
    _require_non_blank_string(normalized, "chunk_id")
    _require_positive_int(normalized, "chunk_index")
    _require_positive_int(normalized, "chunk_count")
    _require_positive_int(normalized, "chunk_size_chars")
    if "section" in normalized:
        _require_non_blank_string(normalized, "section")
    return normalized


def build_qdrant_payload(
    *,
    chunk_id: str,
    content: str,
    metadata: Mapping[str, MetadataValue],
) -> Metadata:
    if not chunk_id.strip():
        raise MetadataValidationError("chunk_id must not be blank")
    if not content.strip():
        raise MetadataValidationError("content must not be blank")

    payload_metadata = normalize_metadata(metadata)
    existing_chunk_id = payload_metadata.get("chunk_id")
    if existing_chunk_id is not None and existing_chunk_id != chunk_id:
        raise MetadataValidationError("metadata chunk_id must match chunk_id")
    payload_metadata["chunk_id"] = chunk_id.strip()

    validated = validate_chunk_metadata(payload_metadata)
    payload = {
        key: validated[key]
        for key in QDRANT_PAYLOAD_METADATA_KEYS
        if key in validated
    }
    payload["content"] = content
    return payload


def _require_non_blank_string(metadata: Metadata, key: str) -> None:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MetadataValidationError(f"{key} must be a non-blank string")


def _require_positive_int(metadata: Metadata, key: str) -> None:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise MetadataValidationError(f"{key} must be a positive integer")
