import pytest

from app.core.config import Settings
from app.rag.embeddings import EmbeddedChunk
from app.rag.milvus_store import (
    MILVUS_PRIMARY_FIELD,
    MILVUS_SCALAR_INDEX_FIELDS,
    MILVUS_SCALAR_INDEX_TYPE,
    MILVUS_VECTOR_FIELD,
    MilvusCollectionConfigError,
    MilvusVectorStore,
    MilvusVectorStoreError,
    build_milvus_scalar_index_name,
    build_milvus_scalar_index_specs,
    build_milvus_entity,
    build_milvus_filter_expression,
    normalize_milvus_metric,
)


class FakeDataType:
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    INT64 = "INT64"


class FakeSchema:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.fields: list[dict] = []

    def add_field(self, **kwargs) -> None:
        self.fields.append(kwargs)


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict] = []

    def add_index(self, **kwargs) -> None:
        self.indexes.append(kwargs)


class FakeMilvusClient:
    def __init__(
        self,
        *,
        has_collection: bool = False,
        description: dict | None = None,
        upsert_result: dict | None = None,
        search_result: list | None = None,
        index_names: list[str] | None = None,
        error_on: str | None = None,
    ) -> None:
        self.has_collection_value = has_collection
        self.description = description or {
            "fields": [
                {
                    "name": MILVUS_VECTOR_FIELD,
                    "params": {"dim": "4"},
                }
            ]
        }
        self.upsert_result = upsert_result or {"upsert_count": 1}
        self.search_result = search_result or []
        self.index_names = index_names or [MILVUS_VECTOR_FIELD]
        self.error_on = error_on
        self.created_collections: list[dict] = []
        self.created_indexes: list[dict] = []
        self.loaded_collections: list[dict] = []
        self.upsert_calls: list[dict] = []
        self.flush_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self.list_index_calls: list[dict] = []
        self.prepared_index_params: list[FakeIndexParams] = []

    def has_collection(self, **kwargs) -> bool:
        self._raise_if_needed("has_collection")
        return self.has_collection_value

    def describe_collection(self, **kwargs) -> dict:
        self._raise_if_needed("describe_collection")
        return self.description

    def prepare_index_params(self) -> FakeIndexParams:
        self._raise_if_needed("prepare_index_params")
        index_params = FakeIndexParams()
        self.prepared_index_params.append(index_params)
        return index_params

    def create_collection(self, **kwargs) -> None:
        self._raise_if_needed("create_collection")
        self.created_collections.append(kwargs)
        index_params = kwargs.get("index_params")
        if index_params is not None:
            for index in index_params.indexes:
                self.index_names.append(index.get("index_name") or index["field_name"])

    def list_indexes(self, **kwargs) -> list[str]:
        self._raise_if_needed("list_indexes")
        self.list_index_calls.append(kwargs)
        return list(self.index_names)

    def create_index(self, **kwargs) -> None:
        self._raise_if_needed("create_index")
        self.created_indexes.append(kwargs)
        index_params = kwargs.get("index_params")
        if index_params is not None:
            for index in index_params.indexes:
                self.index_names.append(index.get("index_name") or index["field_name"])

    def load_collection(self, **kwargs) -> None:
        self._raise_if_needed("load_collection")
        self.loaded_collections.append(kwargs)

    def upsert(self, **kwargs) -> dict:
        self._raise_if_needed("upsert")
        self.upsert_calls.append(kwargs)
        return self.upsert_result

    def flush(self, **kwargs) -> None:
        self._raise_if_needed("flush")
        self.flush_calls.append(kwargs)

    def search(self, **kwargs) -> list:
        self._raise_if_needed("search")
        self.search_calls.append(kwargs)
        return self.search_result

    def _raise_if_needed(self, action: str) -> None:
        if self.error_on == action:
            raise RuntimeError(f"{action} failed")


def make_embedded_chunk(**overrides) -> EmbeddedChunk:
    payload = {
        "chunk_id": "refund_return_policy_chunk_0001",
        "content": "用户个人原因退货时，退货运费通常由用户承担。",
        "metadata": {
            "source": "refund-return-policy.md",
            "title": "退款退货规则",
            "file_name": "refund-return-policy.md",
            "file_extension": ".md",
            "doc_type": "policy",
            "business_domain": "refund",
            "permission_group": "customer_service",
            "chunk_id": "refund_return_policy_chunk_0001",
            "chunk_index": 1,
            "chunk_count": 1,
            "chunk_size_chars": 24,
            "section": "运费处理",
        },
        "vector": [0.1, 0.2, 0.3, 0.4],
    }
    payload.update(overrides)
    return EmbeddedChunk(**payload)


def make_store(client: FakeMilvusClient) -> MilvusVectorStore:
    return MilvusVectorStore(
        client=client,
        collection_name="learning_milvus_chunks",
        timeout_seconds=1.0,
        schema_factory=FakeSchema,
        data_type=FakeDataType,
    )


def test_normalize_milvus_metric_maps_project_distance_names() -> None:
    assert normalize_milvus_metric("Cosine") == "COSINE"
    assert normalize_milvus_metric("Dot") == "IP"
    assert normalize_milvus_metric("Euclid") == "L2"

    with pytest.raises(ValueError, match="unsupported"):
        normalize_milvus_metric("Manhattan")


def test_build_milvus_entity_keeps_vector_content_and_scalar_fields() -> None:
    entity = build_milvus_entity(
        make_embedded_chunk(
            metadata={
                **make_embedded_chunk().metadata,
                "tenant_id": "default",
                "owner_user_id": "U1001",
                "visibility": "tenant",
                "status": "published",
            }
        )
    )

    assert entity[MILVUS_PRIMARY_FIELD] == "refund_return_policy_chunk_0001"
    assert entity[MILVUS_VECTOR_FIELD] == [0.1, 0.2, 0.3, 0.4]
    assert entity["content"] == "用户个人原因退货时，退货运费通常由用户承担。"
    assert entity["source"] == "refund-return-policy.md"
    assert entity["business_domain"] == "refund"
    assert entity["permission_group"] == "customer_service"
    assert entity["tenant_id"] == "default"
    assert entity["owner_user_id"] == "U1001"
    assert entity["visibility"] == "tenant"
    assert entity["status"] == "published"
    assert entity["chunk_index"] == 1


def test_build_milvus_filter_expression_converts_qdrant_must_matches() -> None:
    expression = build_milvus_filter_expression(
        {
            "must": [
                {"key": "permission_group", "match": {"value": "customer_service"}},
                {"key": "tenant_id", "match": {"value": "default"}},
                {"key": "business_domain", "match": {"value": "refund"}},
            ]
        }
    )

    assert (
        expression
        == 'permission_group == "customer_service" and tenant_id == "default" and business_domain == "refund"'
    )


def test_build_milvus_filter_expression_supports_any_range_should_and_must_not() -> None:
    expression = build_milvus_filter_expression(
        {
            "must": [
                {
                    "key": "source",
                    "match": {"any": ["refund-return-policy.md", "order.md"]},
                },
                {"key": "chunk_index", "range": {"gte": 2, "lte": 5}},
            ],
            "should": [
                {"key": "doc_type", "match": {"value": "policy"}},
                {"key": "doc_type", "match": {"value": "faq"}},
            ],
            "must_not": [
                {"key": "permission_group", "match": {"value": "internal_only"}},
            ],
        }
    )

    assert expression == (
        'source in ["refund-return-policy.md", "order.md"] '
        "and chunk_index >= 2 and chunk_index <= 5 "
        'and (doc_type == "policy" or doc_type == "faq") '
        'and not (permission_group == "internal_only")'
    )


def test_build_milvus_filter_expression_rejects_invalid_advanced_filters() -> None:
    with pytest.raises(ValueError, match="integer"):
        build_milvus_filter_expression(
            {"must": [{"key": "source", "range": {"gte": 1}}]}
        )

    with pytest.raises(ValueError, match="any"):
        build_milvus_filter_expression(
            {"must": [{"key": "source", "match": {"any": []}}]}
        )


def test_build_milvus_filter_expression_rejects_unsupported_shape() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        build_milvus_filter_expression(
            {"must": [{"key": "unknown", "match": {"value": "x"}}]}
        )

    with pytest.raises(ValueError, match="must"):
        build_milvus_filter_expression({"should": []})


def test_build_milvus_scalar_index_specs_names_filterable_indexes() -> None:
    assert build_milvus_scalar_index_name("permission_group") == "idx_permission_group"

    specs = build_milvus_scalar_index_specs(["permission_group", "source"])

    assert specs == [
        {
            "field_name": "permission_group",
            "index_type": MILVUS_SCALAR_INDEX_TYPE,
            "index_name": "idx_permission_group",
        },
        {
            "field_name": "source",
            "index_type": MILVUS_SCALAR_INDEX_TYPE,
            "index_name": "idx_source",
        },
    ]


def test_milvus_store_creates_schema_indexes_and_loads_collection_when_missing() -> None:
    client = FakeMilvusClient(has_collection=False)
    store = make_store(client)

    store.ensure_collection(vector_size=4, distance="Cosine")

    assert len(client.created_collections) == 1
    created = client.created_collections[0]
    assert created["collection_name"] == "learning_milvus_chunks"
    assert created["timeout"] == 1.0
    schema = created["schema"]
    field_names = [field["field_name"] for field in schema.fields]
    assert MILVUS_PRIMARY_FIELD in field_names
    assert MILVUS_VECTOR_FIELD in field_names
    assert "content" in field_names
    primary_field = next(
        field for field in schema.fields if field["field_name"] == MILVUS_PRIMARY_FIELD
    )
    vector_field = next(
        field for field in schema.fields if field["field_name"] == MILVUS_VECTOR_FIELD
    )
    assert primary_field["is_primary"] is True
    assert vector_field["dim"] == 4
    created_indexes = created["index_params"].indexes
    assert created_indexes[0] == {
            "field_name": MILVUS_VECTOR_FIELD,
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
    }
    scalar_indexes = created_indexes[1:]
    assert [index["field_name"] for index in scalar_indexes] == list(
        MILVUS_SCALAR_INDEX_FIELDS
    )
    assert {index["index_type"] for index in scalar_indexes} == {
        MILVUS_SCALAR_INDEX_TYPE
    }
    assert client.loaded_collections == [
        {
            "collection_name": "learning_milvus_chunks",
            "timeout": 1.0,
        }
    ]


def test_milvus_store_accepts_existing_collection_with_matching_dimension() -> None:
    client = FakeMilvusClient(has_collection=True)
    store = make_store(client)

    store.ensure_collection(vector_size=4)

    assert client.created_collections == []
    assert len(client.created_indexes) == 1
    created_index_params = client.created_indexes[0]["index_params"]
    assert [index["field_name"] for index in created_index_params.indexes] == list(
        MILVUS_SCALAR_INDEX_FIELDS
    )
    assert len(client.loaded_collections) == 1


def test_milvus_store_skips_scalar_indexes_when_existing_collection_has_them() -> None:
    existing_indexes = [
        MILVUS_VECTOR_FIELD,
        *[
            build_milvus_scalar_index_name(field_name)
            for field_name in MILVUS_SCALAR_INDEX_FIELDS
        ],
    ]
    client = FakeMilvusClient(has_collection=True, index_names=existing_indexes)
    store = make_store(client)

    store.ensure_collection(vector_size=4)

    assert client.list_index_calls == [
        {"collection_name": "learning_milvus_chunks"}
    ]
    assert client.created_indexes == []
    assert len(client.loaded_collections) == 1


def test_milvus_store_can_ensure_missing_scalar_indexes_directly() -> None:
    client = FakeMilvusClient(has_collection=True, index_names=[MILVUS_VECTOR_FIELD])
    store = make_store(client)

    missing_fields = store.ensure_scalar_indexes(fields=["permission_group", "source"])

    assert missing_fields == ["permission_group", "source"]
    assert len(client.created_indexes) == 1
    call = client.created_indexes[0]
    assert call["collection_name"] == "learning_milvus_chunks"
    assert call["timeout"] == 1.0
    assert call["sync"] is True
    assert call["index_params"].indexes == [
        {
            "field_name": "permission_group",
            "index_type": MILVUS_SCALAR_INDEX_TYPE,
            "index_name": "idx_permission_group",
        },
        {
            "field_name": "source",
            "index_type": MILVUS_SCALAR_INDEX_TYPE,
            "index_name": "idx_source",
        },
    ]


def test_milvus_store_rejects_existing_collection_with_different_dimension() -> None:
    client = FakeMilvusClient(
        has_collection=True,
        description={
            "fields": [
                {
                    "name": MILVUS_VECTOR_FIELD,
                    "params": {"dim": "8"},
                }
            ]
        },
    )
    store = make_store(client)

    with pytest.raises(MilvusCollectionConfigError, match="does not match"):
        store.ensure_collection(vector_size=4)


def test_milvus_store_upserts_embedded_chunks() -> None:
    client = FakeMilvusClient(upsert_result={"upsert_count": 1})
    store = make_store(client)

    count = store.upsert_embedded_chunks([make_embedded_chunk()])

    assert count == 1
    assert len(client.upsert_calls) == 1
    call = client.upsert_calls[0]
    assert call["collection_name"] == "learning_milvus_chunks"
    assert call["timeout"] == 1.0
    assert call["data"][0][MILVUS_PRIMARY_FIELD] == "refund_return_policy_chunk_0001"
    assert call["data"][0]["content"] == "用户个人原因退货时，退货运费通常由用户承担。"
    assert client.flush_calls == [
        {
            "collection_name": "learning_milvus_chunks",
            "timeout": 1.0,
        }
    ]


def test_milvus_store_queries_similar_entities() -> None:
    client = FakeMilvusClient(
        search_result=[
            [
                {
                    "id": "refund_return_policy_chunk_0001",
                    "distance": 0.91,
                    "entity": {
                        "chunk_id": "refund_return_policy_chunk_0001",
                        "content": "用户个人原因退货时，退货运费通常由用户承担。",
                        "source": "refund-return-policy.md",
                        "section": "运费处理",
                        "business_domain": "refund",
                    },
                }
            ]
        ]
    )
    store = make_store(client)

    chunks = store.query_similar(
        [0.1, 0.2, 0.3, 0.4],
        top_k=3,
        payload_filter={
            "must": [
                {"key": "permission_group", "match": {"value": "customer_service"}},
                {"key": "business_domain", "match": {"value": "refund"}},
            ]
        },
        score_threshold=0.8,
    )

    assert len(chunks) == 1
    assert chunks[0].point_id == "refund_return_policy_chunk_0001"
    assert chunks[0].chunk_id == "refund_return_policy_chunk_0001"
    assert chunks[0].content == "用户个人原因退货时，退货运费通常由用户承担。"
    assert chunks[0].metadata["source"] == "refund-return-policy.md"
    assert chunks[0].score == 0.91
    call = client.search_calls[0]
    assert call["anns_field"] == MILVUS_VECTOR_FIELD
    assert call["data"] == [[0.1, 0.2, 0.3, 0.4]]
    assert call["limit"] == 3
    assert (
        call["filter"]
        == 'permission_group == "customer_service" and business_domain == "refund"'
    )
    assert call["search_params"] == {"metric_type": "COSINE", "params": {}}


def test_milvus_store_filters_score_threshold_for_l2_distance() -> None:
    client = FakeMilvusClient(
        search_result=[
            [
                {
                    "id": "near",
                    "distance": 0.2,
                    "entity": {
                        "chunk_id": "near",
                        "content": "近距离结果",
                    },
                },
                {
                    "id": "far",
                    "distance": 1.5,
                    "entity": {
                        "chunk_id": "far",
                        "content": "远距离结果",
                    },
                },
            ]
        ]
    )
    store = make_store(client)
    store.metric_type = "L2"

    chunks = store.query_similar(
        [0.1, 0.2, 0.3, 0.4],
        top_k=2,
        score_threshold=0.5,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["near"]


def test_milvus_store_rejects_invalid_inputs_before_client_call() -> None:
    client = FakeMilvusClient()
    store = make_store(client)

    with pytest.raises(ValueError, match="query_vector"):
        store.query_similar([], top_k=3)

    with pytest.raises(ValueError, match="top_k"):
        store.query_similar([0.1, 0.2], top_k=0)

    with pytest.raises(ValueError, match="score_threshold"):
        store.query_similar([0.1, 0.2], top_k=3, score_threshold=True)

    first = make_embedded_chunk(chunk_id="chunk_0001", vector=[0.1, 0.2])
    second = make_embedded_chunk(chunk_id="chunk_0002", vector=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="same size"):
        store.upsert_embedded_chunks([first, second])

    assert client.search_calls == []
    assert client.upsert_calls == []


def test_milvus_store_maps_client_errors() -> None:
    client = FakeMilvusClient(error_on="search")
    store = make_store(client)

    with pytest.raises(MilvusVectorStoreError, match="search"):
        store.query_similar([0.1, 0.2], top_k=1)


def test_milvus_store_constructor_uses_milvus_config_values() -> None:
    settings = Settings(
        milvus_uri=" http://localhost:19530/ ",
        milvus_collection_name="demo_milvus_chunks",
        milvus_timeout_seconds=2.5,
        milvus_token="fake-milvus-token-for-test",
        _env_file=None,
    )

    # Import patching is intentionally not used here because from_settings imports
    # PyMilvus lazily. The behavior is covered by configuration tests and the
    # adapter tests above.
    store = MilvusVectorStore(
        client=FakeMilvusClient(),
        collection_name=settings.milvus_collection_name,
        timeout_seconds=settings.milvus_timeout_seconds,
        schema_factory=FakeSchema,
        data_type=FakeDataType,
    )

    assert settings.resolved_milvus_uri == "http://localhost:19530"
    assert store.collection_name == "demo_milvus_chunks"
    assert store.timeout_seconds == 2.5
