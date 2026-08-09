from collections.abc import Iterable, Mapping, Sequence
import logging
from time import perf_counter
from typing import Any
from uuid import UUID, uuid5

import httpx

from app.core.config import Settings
from app.rag.documents import Metadata, RetrievedChunk
from app.rag.embeddings import EmbeddedChunk
from app.rag.filters import normalize_payload_filter
from app.rag.metadata import build_qdrant_payload


logger = logging.getLogger(__name__)

QDRANT_POINT_NAMESPACE = UUID("f8bdf7bb-92f5-42c7-8a69-4e50e1cd3150")
SUPPORTED_QDRANT_DISTANCES = {"Cosine", "Dot", "Euclid", "Manhattan"}


class QdrantVectorStoreError(RuntimeError):
    pass


class QdrantCollectionConfigError(QdrantVectorStoreError):
    pass


def build_qdrant_point_id(chunk_id: str) -> str:
    if not chunk_id.strip():
        raise ValueError("chunk_id must not be blank")
    return str(uuid5(QDRANT_POINT_NAMESPACE, chunk_id))


def build_qdrant_point(embedded_chunk: EmbeddedChunk) -> dict[str, Any]:
    payload = build_qdrant_payload(
        chunk_id=embedded_chunk.chunk_id,
        content=embedded_chunk.content,
        metadata=embedded_chunk.metadata,
    )

    return {
        "id": build_qdrant_point_id(embedded_chunk.chunk_id),
        "vector": embedded_chunk.vector,
        "payload": payload,
    }


class QdrantVectorStore:
    def __init__(
        self,
        *,
        base_url: str,
        collection_name: str,
        timeout_seconds: float,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be blank")
        if "/" in collection_name:
            raise ValueError("collection_name must not contain '/'")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        self.base_url = base_url.strip().rstrip("/")
        self.collection_name = collection_name.strip()
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key.strip() if api_key and api_key.strip() else None
        self.transport = transport

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        collection_name: str | None = None,
    ) -> "QdrantVectorStore":
        return cls(
            base_url=settings.resolved_qdrant_base_url,
            collection_name=collection_name or settings.qdrant_collection_name,
            timeout_seconds=settings.qdrant_timeout_seconds,
            api_key=settings.qdrant_api_key,
        )

    def list_collections(self) -> list[str]:
        path = "/collections"
        try:
            with self._client() as client:
                response = client.get(path)
            if response.status_code == 404:
                return []
            response.raise_for_status()
            body = response.json()
            return [
                item["name"]
                for item in body.get("result", {}).get("collections", [])
            ]
        except QdrantVectorStoreError:
            raise
        except Exception as exc:
            raise QdrantVectorStoreError("failed to list collections") from exc

    def count_points(self) -> int:
        path = f"/collections/{self.collection_name}"
        try:
            with self._client() as client:
                response = client.get(path)
            if response.status_code == 404:
                return 0
            response.raise_for_status()
            body = response.json()
            return int(body.get("result", {}).get("points_count", 0))
        except QdrantVectorStoreError:
            raise
        except Exception as exc:
            raise QdrantVectorStoreError("failed to count points") from exc

    def ensure_collection(self, *, vector_size: int, distance: str = "Cosine") -> None:
        _validate_vector_size(vector_size)
        _validate_distance(distance)

        path = f"/collections/{self.collection_name}"
        start_time = perf_counter()
        try:
            with self._client() as client:
                response = client.get(path)
                if response.status_code == 404:
                    create_response = client.put(
                        path,
                        json={
                            "vectors": {
                                "size": vector_size,
                                "distance": distance,
                            }
                        },
                    )
                    self._raise_for_bad_response(create_response, "create collection")
                    self._assert_qdrant_ok(create_response, "create collection")
                    self._log_finished("qdrant_collection_created", start_time)
                    return

                self._raise_for_bad_response(response, "get collection")
                self._assert_existing_collection_matches(
                    response,
                    vector_size=vector_size,
                    distance=distance,
                )
                self._log_finished("qdrant_collection_ready", start_time)
        except httpx.RequestError as exc:
            raise QdrantVectorStoreError("failed to connect to Qdrant") from exc

    def upsert_embedded_chunks(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
        *,
        wait: bool = True,
    ) -> int:
        if not embedded_chunks:
            return 0

        _validate_same_vector_size(embedded_chunks)
        points = [build_qdrant_point(chunk) for chunk in embedded_chunks]
        path = f"/collections/{self.collection_name}/points"
        start_time = perf_counter()

        try:
            with self._client() as client:
                response = client.put(
                    path,
                    params={"wait": str(wait).lower()},
                    json={"points": points},
                )
        except httpx.RequestError as exc:
            raise QdrantVectorStoreError("failed to connect to Qdrant") from exc

        self._raise_for_bad_response(response, "upsert points")
        self._assert_qdrant_ok(response, "upsert points")
        self._log_finished("qdrant_points_upserted", start_time, point_count=len(points))
        return len(points)

    def delete_points_by_filter(
        self,
        payload_filter: Mapping[str, Any],
        *,
        wait: bool = True,
    ) -> None:
        normalized_filter = normalize_payload_filter(payload_filter)
        if normalized_filter is None:
            raise ValueError("payload_filter must not be empty")

        path = f"/collections/{self.collection_name}/points/delete"
        start_time = perf_counter()
        try:
            with self._client() as client:
                response = client.post(
                    path,
                    params={"wait": str(wait).lower()},
                    json={"filter": normalized_filter},
                )
        except httpx.RequestError as exc:
            raise QdrantVectorStoreError("failed to connect to Qdrant") from exc

        self._raise_for_bad_response(response, "delete points")
        self._assert_qdrant_ok(response, "delete points")
        self._log_finished("qdrant_points_deleted", start_time)

    def scroll_all(self, *, batch_size: int = 100) -> Iterable[RetrievedChunk]:
        """Scroll every point in the collection (used to build keyword indexes)."""
        path = f"/collections/{self.collection_name}/points/scroll"
        next_offset: Any = None
        while True:
            request_body: dict[str, Any] = {
                "limit": batch_size,
                "with_payload": True,
                "with_vector": False,
            }
            if next_offset is not None:
                request_body["offset"] = next_offset
            try:
                with self._client() as client:
                    response = client.post(path, json=request_body)
            except httpx.RequestError as exc:
                raise QdrantVectorStoreError("failed to connect to Qdrant") from exc
            self._raise_for_bad_response(response, "scroll points")
            data = response.json().get("result", {})
            points = data.get("points", [])
            for point in points:
                yield _build_retrieved_chunk(point, score=0.0)
            next_offset = data.get("next_page_offset")
            if next_offset is None or not points:
                break

    def query_similar(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        payload_filter: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> list[RetrievedChunk]:
        _validate_query_vector(query_vector)
        _validate_top_k(top_k)
        _validate_score_threshold(score_threshold)
        normalized_filter = normalize_payload_filter(payload_filter)

        path = f"/collections/{self.collection_name}/points/query"
        request_body: dict[str, Any] = {
            "query": query_vector,
            "limit": top_k,
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if normalized_filter is not None:
            request_body["filter"] = normalized_filter
        if score_threshold is not None:
            request_body["score_threshold"] = score_threshold

        start_time = perf_counter()
        try:
            with self._client() as client:
                response = client.post(
                    path,
                    json=request_body,
                )
        except httpx.RequestError as exc:
            raise QdrantVectorStoreError("failed to connect to Qdrant") from exc

        self._raise_for_bad_response(response, "query points")
        points = _extract_query_points(response)
        retrieved_chunks = [_build_retrieved_chunk(point) for point in points]
        self._log_finished(
            "qdrant_points_queried",
            start_time,
            point_count=len(retrieved_chunks),
        )
        return retrieved_chunks

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            headers=self._headers(),
            transport=self.transport,
        )

    def _headers(self) -> dict[str, str]:
        if self.api_key is None:
            return {}
        return {"api-key": self.api_key}

    def _raise_for_bad_response(self, response: httpx.Response, action: str) -> None:
        if response.status_code < 400:
            return
        raise QdrantVectorStoreError(
            f"Qdrant {action} failed with status {response.status_code}"
        )

    def _assert_qdrant_ok(self, response: httpx.Response, action: str) -> None:
        try:
            data = response.json()
        except ValueError as exc:
            raise QdrantVectorStoreError(f"Qdrant {action} returned invalid JSON") from exc

        if data.get("status") != "ok":
            raise QdrantVectorStoreError(f"Qdrant {action} returned non-ok status")

    def _assert_existing_collection_matches(
        self,
        response: httpx.Response,
        *,
        vector_size: int,
        distance: str,
    ) -> None:
        try:
            data = response.json()
            vectors_config = data["result"]["config"]["params"]["vectors"]
        except (ValueError, KeyError, TypeError) as exc:
            raise QdrantCollectionConfigError(
                "Qdrant collection config is missing vector settings"
            ) from exc

        if not isinstance(vectors_config, dict) or "size" not in vectors_config:
            raise QdrantCollectionConfigError(
                "this lesson supports only unnamed dense vector collections"
            )

        existing_size = vectors_config.get("size")
        existing_distance = vectors_config.get("distance")
        if existing_size != vector_size or existing_distance != distance:
            raise QdrantCollectionConfigError(
                "Qdrant collection vector config does not match current embedding model"
            )

    def _log_finished(
        self,
        event: str,
        start_time: float,
        *,
        point_count: int | None = None,
    ) -> None:
        elapsed_ms = (perf_counter() - start_time) * 1000
        logger.info(
            "%s collection=%s point_count=%s elapsed_ms=%.2f",
            event,
            self.collection_name,
            point_count,
            elapsed_ms,
        )


def _validate_vector_size(vector_size: int) -> None:
    if vector_size <= 0:
        raise ValueError("vector_size must be greater than 0")


def _validate_distance(distance: str) -> None:
    if distance not in SUPPORTED_QDRANT_DISTANCES:
        raise ValueError("unsupported Qdrant distance metric")


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


def _extract_query_points(response: httpx.Response) -> list[dict[str, Any]]:
    try:
        data = response.json()
    except ValueError as exc:
        raise QdrantVectorStoreError("Qdrant query points returned invalid JSON") from exc

    if data.get("status") != "ok":
        raise QdrantVectorStoreError("Qdrant query points returned non-ok status")

    result = data.get("result")
    if isinstance(result, dict):
        points = result.get("points")
    else:
        points = result

    if not isinstance(points, list):
        raise QdrantVectorStoreError("Qdrant query points returned invalid result")

    return points


def _build_retrieved_chunk(point: dict[str, Any], *, score: float | None = None) -> RetrievedChunk:
    payload = point.get("payload")
    if not isinstance(payload, dict):
        raise QdrantVectorStoreError("Qdrant query point is missing payload")

    content = payload.get("content")
    chunk_id = payload.get("chunk_id")
    if not isinstance(content, str) or not content.strip():
        raise QdrantVectorStoreError("Qdrant query payload is missing content")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise QdrantVectorStoreError("Qdrant query payload is missing chunk_id")

    # Scroll API 的 Record 无 score 字段（仅 query 的 ScoredPoint 有）；
    # 调用方未提供时默认 0.0（scroll 场景 score 无意义）。
    if score is None:
        raw_score = point.get("score")
        if isinstance(raw_score, int | float) and not isinstance(raw_score, bool):
            score = float(raw_score)
        else:
            score = 0.0
    point_id = point.get("id")
    if point_id is None:
        raise QdrantVectorStoreError("Qdrant query point is missing id")

    metadata: Metadata = {
        key: value
        for key, value in payload.items()
        if key != "content"
    }
    return RetrievedChunk(
        point_id=str(point_id),
        chunk_id=chunk_id,
        content=content,
        metadata=metadata,
        score=score,
    )
