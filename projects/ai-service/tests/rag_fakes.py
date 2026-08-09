from collections.abc import Mapping, Sequence
from typing import Any

from app.rag.documents import RetrievedChunk
from app.rag.embeddings import EmbeddedChunk, Vector


def make_retrieved_chunk(**overrides: Any) -> RetrievedChunk:
    payload = {
        "point_id": "point-1",
        "chunk_id": "order_shipping_policy_chunk_0001",
        "content": "订单付款后 24 小时内发货。",
        "metadata": {
            "source": "order-shipping-policy.md",
            "title": "订单发货规则",
            "section": "正常发货时效",
            "chunk_id": "order_shipping_policy_chunk_0001",
        },
        "score": 0.91,
    }
    payload.update(overrides)
    return RetrievedChunk(**payload)


class FakeEmbeddingModel:
    def __init__(
        self,
        *,
        dimension: int = 4,
        vectors: Sequence[Vector] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.dimension = dimension
        self.vectors = [list(vector) for vector in vectors] if vectors is not None else None
        self.error = error
        self.calls: list[list[str]] = []

    @property
    def last_texts(self) -> list[str]:
        if not self.calls:
            raise AssertionError("FakeEmbeddingModel was not called")
        return self.calls[-1]

    def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        if self.vectors is not None:
            return [list(vector) for vector in self.vectors]
        return [
            [round((text_index + 1) / (index + 2), 6) for index in range(self.dimension)]
            for text_index, _ in enumerate(texts)
        ]


class FakeVectorStoreReader:
    def __init__(
        self,
        chunks: Sequence[RetrievedChunk] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.chunks = list(chunks) if chunks is not None else [make_retrieved_chunk()]
        self.error = error
        self.calls: list[dict[str, Any]] = []

    @property
    def last_call(self) -> dict[str, Any]:
        if not self.calls:
            raise AssertionError("FakeVectorStoreReader was not called")
        return self.calls[-1]

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
        self.calls.append(
            {
                "query_vector": list(query_vector),
                "top_k": top_k,
                "payload_filter": payload_filter,
                "score_threshold": score_threshold,
                "with_payload": with_payload,
                "with_vector": with_vector,
            }
        )
        if self.error is not None:
            raise self.error
        chunks = sorted(self.chunks, key=lambda chunk: chunk.score, reverse=True)
        if score_threshold is not None:
            chunks = [
                chunk
                for chunk in chunks
                if chunk.score >= score_threshold
            ]
        return chunks[:top_k]


class FakeVectorStoreWriter:
    collection_name = "fake_chunks"

    def __init__(
        self,
        *,
        ensure_error: Exception | None = None,
        upsert_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.ensure_error = ensure_error
        self.upsert_error = upsert_error
        self.delete_error = delete_error
        self.ensure_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.embedded_chunks: list[EmbeddedChunk] = []

    @property
    def last_ensure_call(self) -> dict[str, Any]:
        if not self.ensure_calls:
            raise AssertionError("FakeVectorStoreWriter.ensure_collection was not called")
        return self.ensure_calls[-1]

    @property
    def last_upsert_call(self) -> dict[str, Any]:
        if not self.upsert_calls:
            raise AssertionError(
                "FakeVectorStoreWriter.upsert_embedded_chunks was not called"
            )
        return self.upsert_calls[-1]

    @property
    def last_delete_call(self) -> dict[str, Any]:
        if not self.delete_calls:
            raise AssertionError("FakeVectorStoreWriter.delete_points_by_filter was not called")
        return self.delete_calls[-1]

    def ensure_collection(self, *, vector_size: int, distance: str = "Cosine") -> None:
        self.ensure_calls.append(
            {
                "vector_size": vector_size,
                "distance": distance,
            }
        )
        if self.ensure_error is not None:
            raise self.ensure_error

    def upsert_embedded_chunks(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
        *,
        wait: bool = True,
    ) -> int:
        self.embedded_chunks = list(embedded_chunks)
        self.upsert_calls.append(
            {
                "embedded_chunks": self.embedded_chunks,
                "wait": wait,
            }
        )
        if self.upsert_error is not None:
            raise self.upsert_error
        return len(self.embedded_chunks)

    def delete_points_by_filter(
        self,
        payload_filter: Mapping[str, Any],
        *,
        wait: bool = True,
    ) -> None:
        self.delete_calls.append(
            {
                "payload_filter": dict(payload_filter),
                "wait": wait,
            }
        )
        if self.delete_error is not None:
            raise self.delete_error
