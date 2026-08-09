import json
from uuid import UUID

import httpx
import pytest

from app.core.config import Settings
from app.rag.embeddings import EmbeddedChunk
from app.rag.vector_store import (
    QdrantCollectionConfigError,
    QdrantVectorStore,
    QdrantVectorStoreError,
    build_qdrant_point,
    build_qdrant_point_id,
)


def make_embedded_chunk(**overrides) -> EmbeddedChunk:
    payload = {
        "chunk_id": "shipping_chunk_0001",
        "content": "Orders ship within 24 hours.",
        "metadata": {
            "source": "shipping.md",
            "title": "Shipping Policy",
            "file_name": "shipping.md",
            "file_extension": ".md",
            "doc_type": "policy",
            "business_domain": "order",
            "permission_group": "customer_service",
            "chunk_id": "shipping_chunk_0001",
            "chunk_index": 1,
            "chunk_count": 1,
            "chunk_size_chars": 29,
        },
        "vector": [0.1, 0.2, 0.3, 0.4],
    }
    payload.update(overrides)
    return EmbeddedChunk(**payload)


def make_store(handler) -> QdrantVectorStore:
    return QdrantVectorStore(
        base_url="http://qdrant.test",
        collection_name="learning_chunks",
        timeout_seconds=1.0,
        api_key="test-api-key",
        transport=httpx.MockTransport(handler),
    )


def test_build_qdrant_point_id_returns_stable_uuid() -> None:
    first = build_qdrant_point_id("shipping_chunk_0001")
    second = build_qdrant_point_id("shipping_chunk_0001")

    assert first == second
    assert UUID(first)


def test_build_qdrant_point_keeps_content_and_metadata_in_payload() -> None:
    embedded = make_embedded_chunk()

    point = build_qdrant_point(embedded)

    assert point["id"] == build_qdrant_point_id("shipping_chunk_0001")
    assert point["vector"] == [0.1, 0.2, 0.3, 0.4]
    assert point["payload"]["chunk_id"] == "shipping_chunk_0001"
    assert point["payload"]["content"] == "Orders ship within 24 hours."
    assert point["payload"]["source"] == "shipping.md"
    assert point["payload"]["permission_group"] == "customer_service"


def test_qdrant_store_creates_collection_when_missing() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(404, json={"status": "error"}, request=request)
        body = json.loads(request.content.decode("utf-8"))
        assert request.method == "PUT"
        assert body == {"vectors": {"size": 4, "distance": "Cosine"}}
        assert request.headers["api-key"] == "test-api-key"
        return httpx.Response(200, json={"status": "ok", "result": True}, request=request)

    store = make_store(handler)

    store.ensure_collection(vector_size=4)

    assert [request.method for request in requests] == ["GET", "PUT"]
    assert requests[0].url.path == "/collections/learning_chunks"


def test_qdrant_store_accepts_existing_matching_collection() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {
                    "config": {
                        "params": {
                            "vectors": {"size": 4, "distance": "Cosine"},
                        }
                    }
                },
            },
            request=request,
        )

    store = make_store(handler)

    store.ensure_collection(vector_size=4)

    assert [request.method for request in requests] == ["GET"]


def test_qdrant_store_rejects_existing_collection_with_different_vector_size() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {
                    "config": {
                        "params": {
                            "vectors": {"size": 8, "distance": "Cosine"},
                        }
                    }
                },
            },
            request=request,
        )

    store = make_store(handler)

    with pytest.raises(QdrantCollectionConfigError, match="does not match"):
        store.ensure_collection(vector_size=4)


def test_qdrant_store_upserts_embedded_chunks() -> None:
    captured_body: dict | None = None
    captured_url: httpx.URL | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body, captured_url
        captured_url = request.url
        captured_body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {"status": "acknowledged", "operation_id": 1},
            },
            request=request,
        )

    store = make_store(handler)

    count = store.upsert_embedded_chunks([make_embedded_chunk()], wait=True)

    assert count == 1
    assert captured_url is not None
    assert captured_url.path == "/collections/learning_chunks/points"
    assert captured_url.params["wait"] == "true"
    assert captured_body is not None
    assert captured_body["points"][0]["payload"]["chunk_id"] == "shipping_chunk_0001"
    assert captured_body["points"][0]["payload"]["content"] == "Orders ship within 24 hours."


def test_qdrant_store_deletes_points_by_payload_filter() -> None:
    captured_body: dict | None = None
    captured_url: httpx.URL | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body, captured_url
        captured_url = request.url
        captured_body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {"status": "acknowledged", "operation_id": 2},
            },
            request=request,
        )

    store = make_store(handler)

    store.delete_points_by_filter(
        {"must": [{"key": "source", "match": {"value": "shipping.md"}}]},
        wait=False,
    )

    assert captured_url is not None
    assert captured_url.path == "/collections/learning_chunks/points/delete"
    assert captured_url.params["wait"] == "false"
    assert captured_body == {
        "filter": {
            "must": [{"key": "source", "match": {"value": "shipping.md"}}]
        }
    }


def test_qdrant_store_queries_similar_points() -> None:
    captured_body: dict | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        assert request.method == "POST"
        assert request.url.path == "/collections/learning_chunks/points/query"
        captured_body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {
                    "points": [
                        {
                            "id": "point-1",
                            "score": 0.87,
                            "payload": {
                                "chunk_id": "shipping_chunk_0001",
                                "content": "Orders ship within 24 hours.",
                                "source": "shipping.md",
                                "section": "Shipping",
                            },
                        }
                    ]
                },
            },
            request=request,
        )

    store = make_store(handler)

    chunks = store.query_similar([0.1, 0.2, 0.3, 0.4], top_k=3)

    assert captured_body == {
        "query": [0.1, 0.2, 0.3, 0.4],
        "limit": 3,
        "with_payload": True,
        "with_vector": False,
    }
    assert len(chunks) == 1
    assert chunks[0].point_id == "point-1"
    assert chunks[0].chunk_id == "shipping_chunk_0001"
    assert chunks[0].content == "Orders ship within 24 hours."
    assert chunks[0].metadata["source"] == "shipping.md"
    assert chunks[0].score == 0.87


def test_qdrant_store_queries_similar_points_with_payload_filter() -> None:
    captured_body: dict | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {
                    "points": [
                        {
                            "id": "point-1",
                            "score": 0.87,
                            "payload": {
                                "chunk_id": "shipping_chunk_0001",
                                "content": "Orders ship within 24 hours.",
                                "source": "shipping.md",
                            },
                        }
                    ]
                },
            },
            request=request,
        )

    store = make_store(handler)

    store.query_similar(
        [0.1, 0.2, 0.3, 0.4],
        top_k=3,
        payload_filter={
            "must": [
                {"key": "permission_group", "match": {"value": "customer_service"}},
                {"key": "business_domain", "match": {"value": "order"}},
            ]
        },
    )

    assert captured_body is not None
    assert captured_body["filter"] == {
        "must": [
            {"key": "permission_group", "match": {"value": "customer_service"}},
            {"key": "business_domain", "match": {"value": "order"}},
        ]
    }


def test_qdrant_store_queries_similar_points_with_score_threshold() -> None:
    captured_body: dict | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_body
        captured_body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {
                    "points": [
                        {
                            "id": "point-1",
                            "score": 0.92,
                            "payload": {
                                "chunk_id": "shipping_chunk_0001",
                                "content": "Orders ship within 24 hours.",
                            },
                        }
                    ]
                },
            },
            request=request,
        )

    store = make_store(handler)

    chunks = store.query_similar(
        [0.1, 0.2, 0.3, 0.4],
        top_k=3,
        score_threshold=0.8,
    )

    assert captured_body is not None
    assert captured_body["score_threshold"] == 0.8
    assert chunks[0].score == 0.92


def test_qdrant_store_accepts_legacy_query_result_list_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": [
                    {
                        "id": "point-1",
                        "score": 0.87,
                        "payload": {
                            "chunk_id": "shipping_chunk_0001",
                            "content": "Orders ship within 24 hours.",
                        },
                    }
                ],
            },
            request=request,
        )

    store = make_store(handler)

    chunks = store.query_similar([0.1, 0.2, 0.3, 0.4], top_k=1)

    assert chunks[0].chunk_id == "shipping_chunk_0001"


def test_qdrant_store_rejects_invalid_query_options() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid query should not call Qdrant")

    store = make_store(handler)

    with pytest.raises(ValueError, match="query_vector"):
        store.query_similar([], top_k=3)

    with pytest.raises(ValueError, match="top_k"):
        store.query_similar([0.1, 0.2], top_k=0)

    with pytest.raises(ValueError, match="score_threshold"):
        store.query_similar([0.1, 0.2], top_k=3, score_threshold=True)


def test_qdrant_store_rejects_query_result_without_payload_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {
                    "points": [
                        {
                            "id": "point-1",
                            "score": 0.87,
                            "payload": {
                                "chunk_id": "shipping_chunk_0001",
                            },
                        }
                    ]
                },
            },
            request=request,
        )

    store = make_store(handler)

    with pytest.raises(QdrantVectorStoreError, match="content"):
        store.query_similar([0.1, 0.2], top_k=1)


def test_qdrant_store_returns_zero_when_no_chunks_need_upsert() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("empty upsert should not call Qdrant")

    store = make_store(handler)

    assert store.upsert_embedded_chunks([]) == 0


def test_qdrant_store_rejects_empty_delete_filter_before_http_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid delete should not call Qdrant")

    store = make_store(handler)

    with pytest.raises(ValueError, match="payload_filter"):
        store.delete_points_by_filter({})


def test_qdrant_store_rejects_mismatched_vector_sizes_before_upsert() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid vectors should not call Qdrant")

    store = make_store(handler)
    first = make_embedded_chunk(chunk_id="chunk_0001", vector=[0.1, 0.2])
    second = make_embedded_chunk(chunk_id="chunk_0002", vector=[0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="same size"):
        store.upsert_embedded_chunks([first, second])


def test_qdrant_store_maps_http_error_to_vector_store_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error"}, request=request)

    store = make_store(handler)

    with pytest.raises(QdrantVectorStoreError, match="status 500"):
        store.upsert_embedded_chunks([make_embedded_chunk()])


def test_qdrant_store_validates_payload_metadata_before_upsert() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid payload should not call Qdrant")

    store = make_store(handler)
    embedded_chunk = make_embedded_chunk(
        metadata={
            "source": "shipping.md",
            "title": "Shipping Policy",
        }
    )

    with pytest.raises(ValueError, match="file_name"):
        store.upsert_embedded_chunks([embedded_chunk])


def test_qdrant_store_from_settings_uses_qdrant_config() -> None:
    settings = Settings(
        qdrant_base_url=" http://localhost:6333/ ",
        qdrant_collection_name="demo_chunks",
        qdrant_timeout_seconds=2.5,
        qdrant_api_key="secret",
        _env_file=None,
    )

    store = QdrantVectorStore.from_settings(settings)

    assert store.base_url == "http://localhost:6333"
    assert store.collection_name == "demo_chunks"
    assert store.timeout_seconds == 2.5
    assert store.api_key == "secret"


def test_scroll_all_handles_records_without_score_and_paginates() -> None:
    """Scroll API 的 Record 无 score 字段，且支持 next_page_offset 分页。"""
    import httpx

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        body = {
            "result": {
                "points": [
                    {
                        "id": "p1",
                        "payload": {
                            "content": "退货运费由商家承担",
                            "chunk_id": "refund_chunk_0001",
                            "source": "refund-return-policy.md",
                        },
                    }
                ],
                "next_page_offset": None,
            }
        }
        return httpx.Response(200, json=body)

    store = make_store(handler)
    chunks = list(store.scroll_all(batch_size=10))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "refund_chunk_0001"
    assert chunks[0].score == 0.0  # Scroll Record 无 score，默认 0.0
    assert calls == ["/collections/learning_chunks/points/scroll"]


def test_scroll_all_follows_next_page_offset() -> None:
    """scroll 分页：第一页带 next_page_offset，第二页返回空则终止。"""
    import httpx

    page = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if page["count"] == 0:
            page["count"] = 1
            body = {
                "result": {
                    "points": [
                        {
                            "id": "p1",
                            "payload": {
                                "content": "退款政策七天无理由",
                                "chunk_id": "refund_chunk_0001",
                                "source": "refund-return-policy.md",
                            },
                        }
                    ],
                    "next_page_offset": 42,
                }
            }
        else:
            body = {"result": {"points": [], "next_page_offset": None}}
        return httpx.Response(200, json=body)

    store = make_store(handler)
    chunks = list(store.scroll_all(batch_size=10))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "refund_chunk_0001"


def test_scroll_all_passes_offset_to_second_page() -> None:
    """第二页请求必须携带上一页的 next_page_offset。"""
    import httpx

    requests: list[dict] = []
    page = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body_req = _json.loads(request.content or b"{}")
        requests.append(body_req)
        if page["count"] == 0:
            page["count"] = 1
            body = {
                "result": {
                    "points": [
                        {
                            "id": "p1",
                            "payload": {
                                "content": "退款政策七天无理由",
                                "chunk_id": "refund_chunk_0001",
                                "source": "refund-return-policy.md",
                            },
                        }
                    ],
                    "next_page_offset": 42,
                }
            }
        else:
            body = {"result": {"points": [], "next_page_offset": None}}
        return httpx.Response(200, json=body)

    store = make_store(handler)
    list(store.scroll_all(batch_size=10))
    assert len(requests) == 2
    assert requests[0].get("offset") is None
    assert requests[1].get("offset") == 42


def make_settings(**overrides) -> Settings:
    base = {
        "qdrant_base_url": "http://qdrant.test",
        "qdrant_collection_name": "learning_rag_chunks_v4_1024",
        "qdrant_timeout_seconds": 1.0,
        "qdrant_api_key": None,
    }
    base.update(overrides)
    return Settings(**base)


def test_from_settings_accepts_collection_name_override() -> None:
    store = QdrantVectorStore.from_settings(
        make_settings(), collection_name="kb_customer_policy"
    )
    assert store.collection_name == "kb_customer_policy"


def test_from_settings_defaults_to_settings_collection() -> None:
    store = QdrantVectorStore.from_settings(make_settings())
    assert store.collection_name == "learning_rag_chunks_v4_1024"


def test_list_collections_returns_names() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/collections"
        body = {
            "result": {
                "collections": [
                    {"name": "kb_customer_policy"},
                    {"name": "kb_account_security"},
                ]
            }
        }
        return httpx.Response(200, json=body)

    store = make_store(handler)
    assert store.list_collections() == ["kb_customer_policy", "kb_account_security"]


def test_count_points_returns_zero_when_collection_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"status": {"error": "Not found"}})

    store = make_store(handler)
    assert store.count_points() == 0
