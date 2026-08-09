from collections.abc import Callable, Mapping, Sequence
import logging
from time import perf_counter
from typing import Any

from app.core.config import Settings
from app.rag.documents import Metadata, RetrievedChunk
from app.rag.embeddings import EmbeddedChunk, Vector
from app.rag.filters import normalize_payload_filter
from app.rag.metadata import build_qdrant_payload
from app.rag.score_interpretation import (
    describe_milvus_score,
    is_score_passing_threshold,
)


logger = logging.getLogger(__name__)

MILVUS_PRIMARY_FIELD = "chunk_id"
MILVUS_VECTOR_FIELD = "embedding"
MILVUS_CONTENT_FIELD = "content"
MILVUS_STRING_FIELD_MAX_LENGTHS = {
    MILVUS_PRIMARY_FIELD: 256,
    MILVUS_CONTENT_FIELD: 8192,
    "source": 512,
    "tenant_id": 128,
    "owner_user_id": 128,
    "title": 512,
    "file_name": 512,
    "file_extension": 32,
    "doc_type": 128,
    "business_domain": 128,
    "permission_group": 128,
    "visibility": 128,
    "status": 128,
    "section": 512,
}
MILVUS_INT_FIELDS = (
    "chunk_index",
    "chunk_count",
    "chunk_size_chars",
)
MILVUS_SCALAR_INDEX_TYPE = "INVERTED"
MILVUS_SCALAR_INDEX_FIELDS = (
    "permission_group",
    "business_domain",
    "doc_type",
    "source",
    "tenant_id",
    "owner_user_id",
    "visibility",
    "status",
)
MILVUS_OUTPUT_FIELDS = (
    MILVUS_PRIMARY_FIELD,
    MILVUS_CONTENT_FIELD,
    "source",
    "tenant_id",
    "owner_user_id",
    "title",
    "file_name",
    "file_extension",
    "doc_type",
    "business_domain",
    "permission_group",
    "visibility",
    "status",
    "chunk_index",
    "chunk_count",
    "chunk_size_chars",
    "section",
)
SUPPORTED_MILVUS_METRICS = {"COSINE", "IP", "L2"}
MILVUS_DISTANCE_ALIASES = {
    "cosine": "COSINE",
    "dot": "IP",
    "ip": "IP",
    "euclid": "L2",
    "l2": "L2",
}
MILVUS_FILTERABLE_FIELDS = {
    "permission_group",
    "business_domain",
    "doc_type",
    "source",
    "tenant_id",
    "owner_user_id",
    "visibility",
    "status",
    "file_name",
    "file_extension",
    "section",
    "chunk_index",
    "chunk_count",
    "chunk_size_chars",
}


class MilvusVectorStoreError(RuntimeError):
    pass


class MilvusCollectionConfigError(MilvusVectorStoreError):
    pass


def normalize_milvus_metric(distance: str) -> str:
    normalized = MILVUS_DISTANCE_ALIASES.get(distance.strip().lower())
    if normalized is None:
        raise ValueError("unsupported Milvus metric type")
    return normalized


def build_milvus_entity(embedded_chunk: EmbeddedChunk) -> dict[str, Any]:
    payload = build_qdrant_payload(
        chunk_id=embedded_chunk.chunk_id,
        content=embedded_chunk.content,
        metadata=embedded_chunk.metadata,
    )
    vector = _normalize_vector(embedded_chunk.vector)

    entity: dict[str, Any] = {
        MILVUS_PRIMARY_FIELD: _require_varchar(
            payload[MILVUS_PRIMARY_FIELD],
            field_name=MILVUS_PRIMARY_FIELD,
        ),
        MILVUS_VECTOR_FIELD: vector,
        MILVUS_CONTENT_FIELD: _require_varchar(
            payload[MILVUS_CONTENT_FIELD],
            field_name=MILVUS_CONTENT_FIELD,
        ),
    }

    for field_name in MILVUS_STRING_FIELD_MAX_LENGTHS:
        if field_name in {MILVUS_PRIMARY_FIELD, MILVUS_CONTENT_FIELD}:
            continue
        value = payload.get(field_name, "")
        entity[field_name] = _require_varchar(value, field_name=field_name)

    for field_name in MILVUS_INT_FIELDS:
        entity[field_name] = _require_int(payload.get(field_name), field_name=field_name)

    return entity


def build_milvus_filter_expression(
    payload_filter: Mapping[str, Any] | None,
) -> str:
    normalized_filter = normalize_payload_filter(payload_filter)
    if normalized_filter is None:
        return ""

    group_expressions: list[str] = []
    must_expression = _build_milvus_condition_group(
        normalized_filter.get("must"),
        group_name="must",
        joiner=" and ",
    )
    if must_expression:
        group_expressions.append(must_expression)

    should_expression = _build_milvus_condition_group(
        normalized_filter.get("should"),
        group_name="should",
        joiner=" or ",
    )
    if should_expression:
        group_expressions.append(_wrap_milvus_expression(should_expression))

    must_not_expression = _build_milvus_condition_group(
        normalized_filter.get("must_not"),
        group_name="must_not",
        joiner=" and ",
        negate=True,
    )
    if must_not_expression:
        group_expressions.append(must_not_expression)

    if not group_expressions:
        raise ValueError("Milvus filter must contain at least one condition")
    return " and ".join(group_expressions)


def build_milvus_scalar_index_name(field_name: str) -> str:
    normalized_field = _normalize_scalar_index_field(field_name)
    return f"idx_{normalized_field}"


def build_milvus_scalar_index_specs(
    fields: Sequence[str] = MILVUS_SCALAR_INDEX_FIELDS,
    *,
    index_type: str = MILVUS_SCALAR_INDEX_TYPE,
) -> list[dict[str, str]]:
    normalized_fields = _normalize_scalar_index_fields(fields)
    if not isinstance(index_type, str) or not index_type.strip():
        raise ValueError("Milvus scalar index_type must be a non-blank string")
    normalized_index_type = index_type.strip().upper()
    return [
        {
            "field_name": field_name,
            "index_type": normalized_index_type,
            "index_name": build_milvus_scalar_index_name(field_name),
        }
        for field_name in normalized_fields
    ]


class MilvusVectorStore:
    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        timeout_seconds: float,
        schema_factory: Callable[..., Any] | None = None,
        data_type: Any | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be blank")
        if "/" in collection_name:
            raise ValueError("collection_name must not contain '/'")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        self.client = client
        self.collection_name = collection_name.strip()
        self.timeout_seconds = timeout_seconds
        self.schema_factory = schema_factory
        self.data_type = data_type
        self.metric_type = "COSINE"

    @classmethod
    def from_settings(cls, settings: Settings) -> "MilvusVectorStore":
        from pymilvus import DataType, MilvusClient

        token = settings.milvus_token.strip() if settings.milvus_token else ""
        client = MilvusClient(
            uri=settings.resolved_milvus_uri,
            token=token,
            timeout=settings.milvus_timeout_seconds,
        )
        return cls(
            client=client,
            collection_name=settings.milvus_collection_name,
            timeout_seconds=settings.milvus_timeout_seconds,
            schema_factory=MilvusClient.create_schema,
            data_type=DataType,
        )

    def ensure_collection(self, *, vector_size: int, distance: str = "Cosine") -> None:
        _validate_vector_size(vector_size)
        self.metric_type = normalize_milvus_metric(distance)

        start_time = perf_counter()
        try:
            collection_exists = self.client.has_collection(
                collection_name=self.collection_name,
                timeout=self.timeout_seconds,
            )
            if collection_exists:
                self._assert_existing_collection_matches(vector_size=vector_size)
                self.ensure_scalar_indexes()
                self.client.load_collection(
                    collection_name=self.collection_name,
                    timeout=self.timeout_seconds,
                )
                self._log_finished("milvus_collection_ready", start_time)
                return

            self._create_collection(vector_size=vector_size)
            self.client.load_collection(
                collection_name=self.collection_name,
                timeout=self.timeout_seconds,
            )
            self._log_finished("milvus_collection_created", start_time)
        except MilvusVectorStoreError:
            raise
        except Exception as exc:
            raise MilvusVectorStoreError("failed to operate Milvus collection") from exc

    def upsert_embedded_chunks(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
        *,
        wait: bool = True,
    ) -> int:
        if not embedded_chunks:
            return 0

        _validate_same_vector_size(embedded_chunks)
        entities = [build_milvus_entity(chunk) for chunk in embedded_chunks]
        start_time = perf_counter()

        try:
            result = self.client.upsert(
                collection_name=self.collection_name,
                data=entities,
                timeout=self.timeout_seconds,
            )
            if wait and hasattr(self.client, "flush"):
                self.client.flush(
                    collection_name=self.collection_name,
                    timeout=self.timeout_seconds,
                )
        except Exception as exc:
            raise MilvusVectorStoreError("failed to upsert Milvus entities") from exc

        entity_count = _extract_write_count(result, expected_count=len(entities))
        self._log_finished(
            "milvus_entities_upserted",
            start_time,
            entity_count=entity_count,
        )
        return entity_count

    def query_similar(
        self,
        query_vector: Vector,
        *,
        top_k: int,
        payload_filter: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> list[RetrievedChunk]:
        del with_payload
        _validate_query_vector(query_vector)
        _validate_top_k(top_k)
        _validate_score_threshold(score_threshold)
        filter_expression = build_milvus_filter_expression(payload_filter)
        output_fields = list(MILVUS_OUTPUT_FIELDS)
        if with_vector:
            output_fields.append(MILVUS_VECTOR_FIELD)

        start_time = perf_counter()
        try:
            result = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field=MILVUS_VECTOR_FIELD,
                filter=filter_expression,
                limit=top_k,
                output_fields=output_fields,
                search_params={"metric_type": self.metric_type, "params": {}},
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise MilvusVectorStoreError("failed to search Milvus entities") from exc

        hits = _flatten_search_result(result)
        retrieved_chunks = [
            _build_retrieved_chunk(hit)
            for hit in hits
        ]
        filtered_chunks = _apply_score_threshold(
            retrieved_chunks,
            score_threshold=score_threshold,
            metric_type=self.metric_type,
        )
        self._log_finished(
            "milvus_entities_searched",
            start_time,
            entity_count=len(filtered_chunks),
        )
        return filtered_chunks

    def ensure_scalar_indexes(
        self,
        *,
        fields: Sequence[str] = MILVUS_SCALAR_INDEX_FIELDS,
        index_type: str = MILVUS_SCALAR_INDEX_TYPE,
        sync: bool = True,
    ) -> list[str]:
        scalar_fields = _normalize_scalar_index_fields(fields)
        if not scalar_fields:
            return []

        start_time = perf_counter()
        try:
            existing_indexes = set(self.list_indexes())
            missing_fields = [
                field_name
                for field_name in scalar_fields
                if not _field_has_milvus_index(field_name, existing_indexes)
            ]
            if not missing_fields:
                return []

            index_params = self.client.prepare_index_params()
            _add_milvus_scalar_indexes(
                index_params,
                fields=missing_fields,
                index_type=index_type,
            )
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
                timeout=self.timeout_seconds,
                sync=sync,
            )
        except Exception as exc:
            raise MilvusVectorStoreError(
                "failed to ensure Milvus scalar indexes"
            ) from exc

        self._log_finished(
            "milvus_scalar_indexes_ready",
            start_time,
            entity_count=len(missing_fields),
        )
        return missing_fields

    def list_indexes(self) -> list[str]:
        try:
            indexes = self.client.list_indexes(collection_name=self.collection_name)
        except Exception as exc:
            raise MilvusVectorStoreError("failed to list Milvus indexes") from exc
        if not isinstance(indexes, Sequence) or isinstance(indexes, str):
            raise MilvusVectorStoreError("Milvus list_indexes returned invalid result")
        return [str(index_name) for index_name in indexes]

    def _create_collection(self, *, vector_size: int) -> None:
        if self.schema_factory is None or self.data_type is None:
            raise MilvusCollectionConfigError(
                "Milvus schema_factory and data_type are required to create collection"
            )

        schema = self.schema_factory(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name=MILVUS_PRIMARY_FIELD,
            datatype=self.data_type.VARCHAR,
            is_primary=True,
            max_length=MILVUS_STRING_FIELD_MAX_LENGTHS[MILVUS_PRIMARY_FIELD],
        )
        schema.add_field(
            field_name=MILVUS_VECTOR_FIELD,
            datatype=self.data_type.FLOAT_VECTOR,
            dim=vector_size,
        )
        for field_name, max_length in MILVUS_STRING_FIELD_MAX_LENGTHS.items():
            if field_name == MILVUS_PRIMARY_FIELD:
                continue
            schema.add_field(
                field_name=field_name,
                datatype=self.data_type.VARCHAR,
                max_length=max_length,
            )
        for field_name in MILVUS_INT_FIELDS:
            schema.add_field(
                field_name=field_name,
                datatype=self.data_type.INT64,
            )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=MILVUS_VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
        )
        _add_milvus_scalar_indexes(index_params)
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            timeout=self.timeout_seconds,
        )

    def _assert_existing_collection_matches(self, *, vector_size: int) -> None:
        description = self.client.describe_collection(
            collection_name=self.collection_name,
            timeout=self.timeout_seconds,
        )
        vector_field = _find_field(description, MILVUS_VECTOR_FIELD)
        if vector_field is None:
            raise MilvusCollectionConfigError(
                "Milvus collection is missing embedding vector field"
            )

        existing_dim = _extract_field_dimension(vector_field)
        if existing_dim is None:
            raise MilvusCollectionConfigError(
                "Milvus collection vector field is missing dimension"
            )
        if existing_dim != vector_size:
            raise MilvusCollectionConfigError(
                "Milvus collection vector config does not match current embedding model"
            )

    def _log_finished(
        self,
        event: str,
        start_time: float,
        *,
        entity_count: int | None = None,
    ) -> None:
        elapsed_ms = (perf_counter() - start_time) * 1000
        logger.info(
            "%s collection=%s entity_count=%s elapsed_ms=%.2f",
            event,
            self.collection_name,
            entity_count,
            elapsed_ms,
        )


def _add_milvus_scalar_indexes(
    index_params: Any,
    *,
    fields: Sequence[str] = MILVUS_SCALAR_INDEX_FIELDS,
    index_type: str = MILVUS_SCALAR_INDEX_TYPE,
) -> None:
    for index_spec in build_milvus_scalar_index_specs(
        fields,
        index_type=index_type,
    ):
        index_params.add_index(**index_spec)


def _build_milvus_condition_group(
    conditions: Any,
    *,
    group_name: str,
    joiner: str,
    negate: bool = False,
) -> str:
    if conditions is None:
        return ""
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"Milvus filter {group_name} must be a non-empty list")

    expressions: list[str] = []
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise ValueError("Milvus filter condition must be an object")
        expression = _build_milvus_condition_expression(condition)
        if negate:
            expression = f"not ({expression})"
        expressions.append(expression)
    return joiner.join(expressions)


def _build_milvus_condition_expression(condition: Mapping[str, Any]) -> str:
    if "match" in condition:
        return _build_milvus_match_expression(condition)
    if "range" in condition:
        return _build_milvus_range_expression(condition)
    raise ValueError("Milvus filter condition must contain match or range")


def _build_milvus_match_expression(condition: Mapping[str, Any]) -> str:
    key = condition.get("key")
    match = condition.get("match")
    normalized_key = _normalize_filter_field(key)
    if not isinstance(match, Mapping):
        raise ValueError("Milvus match filter must be an object")
    if "value" in match:
        return f"{normalized_key} == {_format_milvus_filter_value(match['value'])}"
    if "any" in match:
        return f"{normalized_key} in {_format_milvus_filter_values(match['any'])}"
    raise ValueError("Milvus match filter must contain value or any")


def _build_milvus_range_expression(condition: Mapping[str, Any]) -> str:
    key = condition.get("key")
    range_filter = condition.get("range")
    normalized_key = _normalize_filter_field(key)
    if normalized_key not in MILVUS_INT_FIELDS:
        raise ValueError("Milvus range filter supports integer metadata fields only")
    if not isinstance(range_filter, Mapping):
        raise ValueError("Milvus range filter must be an object")

    operator_map = (
        ("gt", ">"),
        ("gte", ">="),
        ("lt", "<"),
        ("lte", "<="),
    )
    expressions: list[str] = []
    for option_name, operator in operator_map:
        if option_name not in range_filter:
            continue
        value = range_filter[option_name]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError("Milvus range filter values must be numbers")
        expressions.append(f"{normalized_key} {operator} {value}")
    if not expressions:
        raise ValueError("Milvus range filter must contain gt, gte, lt, or lte")
    return " and ".join(expressions)


def _normalize_filter_field(key: Any) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Milvus filter key must be a non-blank string")
    normalized_key = key.strip()
    if normalized_key not in MILVUS_FILTERABLE_FIELDS:
        raise ValueError("unsupported Milvus filter field")
    return normalized_key


def _normalize_scalar_index_field(field_name: str) -> str:
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("Milvus scalar index field must be a non-blank string")
    normalized_field = field_name.strip()
    if normalized_field not in MILVUS_FILTERABLE_FIELDS:
        raise ValueError("unsupported Milvus scalar index field")
    if normalized_field == MILVUS_VECTOR_FIELD:
        raise ValueError("Milvus vector field is not a scalar index field")
    return normalized_field


def _normalize_scalar_index_fields(fields: Sequence[str]) -> list[str]:
    if isinstance(fields, str):
        raise ValueError("Milvus scalar index fields must be a sequence")
    normalized_fields: list[str] = []
    for field_name in fields:
        normalized_field = _normalize_scalar_index_field(field_name)
        if normalized_field not in normalized_fields:
            normalized_fields.append(normalized_field)
    return normalized_fields


def _field_has_milvus_index(field_name: str, existing_indexes: set[str]) -> bool:
    return (
        field_name in existing_indexes
        or build_milvus_scalar_index_name(field_name) in existing_indexes
    )


def _format_milvus_filter_value(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("Milvus filter string value must not be blank")
        escaped = normalized.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    raise ValueError("unsupported Milvus filter value type")


def _format_milvus_filter_values(values: Any) -> str:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError("Milvus match any value must be a non-empty sequence")
    formatted_values = [_format_milvus_filter_value(value) for value in values]
    if not formatted_values:
        raise ValueError("Milvus match any value must not be empty")
    return f"[{', '.join(formatted_values)}]"


def _wrap_milvus_expression(expression: str) -> str:
    if " or " not in expression:
        return expression
    return f"({expression})"


def _normalize_vector(vector: Sequence[float]) -> list[float]:
    if not vector:
        raise ValueError("vector must not be empty")
    normalized: list[float] = []
    for value in vector:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError("vector must contain only numbers")
        normalized.append(float(value))
    return normalized


def _require_varchar(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    max_length = MILVUS_STRING_FIELD_MAX_LENGTHS[field_name]
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} exceeds Milvus max_length")
    return normalized


def _require_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _validate_vector_size(vector_size: int) -> None:
    if vector_size <= 0:
        raise ValueError("vector_size must be greater than 0")


def _validate_same_vector_size(embedded_chunks: Sequence[EmbeddedChunk]) -> None:
    first_size = len(embedded_chunks[0].vector)
    for embedded_chunk in embedded_chunks:
        if len(embedded_chunk.vector) != first_size:
            raise ValueError("all embedding vectors must have the same size")


def _validate_query_vector(query_vector: Sequence[float]) -> None:
    if not query_vector:
        raise ValueError("query_vector must not be empty")
    for value in query_vector:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError("query_vector must contain only numbers")


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _validate_score_threshold(score_threshold: float | None) -> None:
    if score_threshold is None:
        return
    if not isinstance(score_threshold, int | float) or isinstance(score_threshold, bool):
        raise ValueError("score_threshold must be a number")


def _extract_write_count(result: Any, *, expected_count: int) -> int:
    if isinstance(result, Mapping):
        for key in ("upsert_count", "insert_count"):
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        primary_keys = result.get("primary_keys") or result.get("ids")
        if isinstance(primary_keys, Sequence) and not isinstance(primary_keys, str):
            return len(primary_keys)
    return expected_count


def _flatten_search_result(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, list):
        raise MilvusVectorStoreError("Milvus search returned invalid result")
    if not result:
        return []
    first = result[0]
    if isinstance(first, list):
        hits = first
    else:
        hits = result
    if not all(isinstance(hit, dict) for hit in hits):
        raise MilvusVectorStoreError("Milvus search hits must be objects")
    return hits


def _build_retrieved_chunk(hit: dict[str, Any]) -> RetrievedChunk:
    entity = hit.get("entity")
    if not isinstance(entity, Mapping):
        raise MilvusVectorStoreError("Milvus search hit is missing entity")

    point_id = hit.get("id") or entity.get(MILVUS_PRIMARY_FIELD)
    if point_id is None:
        raise MilvusVectorStoreError("Milvus search hit is missing id")

    chunk_id = entity.get(MILVUS_PRIMARY_FIELD) or str(point_id)
    content = entity.get(MILVUS_CONTENT_FIELD)
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise MilvusVectorStoreError("Milvus search entity is missing chunk_id")
    if not isinstance(content, str) or not content.strip():
        raise MilvusVectorStoreError("Milvus search entity is missing content")

    raw_score = hit.get("distance", hit.get("score"))
    if not isinstance(raw_score, int | float) or isinstance(raw_score, bool):
        raise MilvusVectorStoreError("Milvus search hit is missing score")

    metadata: Metadata = {
        key: value
        for key, value in entity.items()
        if key not in {MILVUS_CONTENT_FIELD, MILVUS_VECTOR_FIELD}
    }
    metadata[MILVUS_PRIMARY_FIELD] = chunk_id
    return RetrievedChunk(
        point_id=str(point_id),
        chunk_id=chunk_id,
        content=content,
        metadata=metadata,
        score=float(raw_score),
    )


def _apply_score_threshold(
    chunks: Sequence[RetrievedChunk],
    *,
    score_threshold: float | None,
    metric_type: str,
) -> list[RetrievedChunk]:
    if score_threshold is None:
        return list(chunks)
    meaning = describe_milvus_score(metric_type)
    return [
        chunk
        for chunk in chunks
        if is_score_passing_threshold(chunk.score, score_threshold, meaning)
    ]


def _find_field(description: Any, field_name: str) -> Mapping[str, Any] | None:
    if not isinstance(description, Mapping):
        return None

    fields = description.get("fields")
    if fields is None:
        schema = description.get("schema")
        if isinstance(schema, Mapping):
            fields = schema.get("fields")
    if not isinstance(fields, list):
        return None

    for field in fields:
        if not isinstance(field, Mapping):
            continue
        current_name = (
            field.get("name")
            or field.get("field_name")
            or field.get("fieldName")
        )
        if current_name == field_name:
            return field
    return None


def _extract_field_dimension(field: Mapping[str, Any]) -> int | None:
    candidate_values = [
        field.get("dim"),
        field.get("dimension"),
    ]
    for params_key in (
        "params",
        "type_params",
        "typeParams",
        "element_type_params",
        "elementTypeParams",
    ):
        params = field.get(params_key)
        if isinstance(params, Mapping):
            candidate_values.extend([params.get("dim"), params.get("dimension")])

    for value in candidate_values:
        parsed = _parse_int(value)
        if parsed is not None:
            return parsed
    return None


def _parse_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None
