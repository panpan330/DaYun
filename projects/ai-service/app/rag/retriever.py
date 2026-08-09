from collections.abc import Sequence
from typing import Protocol

from app.rag.documents import RetrievedChunk
from app.rag.embeddings import EmbeddingModel, Vector
from app.rag.errors import (
    rag_embedding_bad_response,
    rag_embedding_failed,
    rag_vector_store_failed,
)
from app.rag.filters import PayloadFilter, RagAccessScope, build_payload_filter


DEFAULT_TOP_K = 5


class VectorStoreReader(Protocol):
    def query_similar(
        self,
        query_vector: Vector,
        *,
        top_k: int,
        payload_filter: PayloadFilter | None = None,
        score_threshold: float | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> list[RetrievedChunk]:
        """Return the most similar chunks for a query vector."""


def retrieve_top_k(
    query: str,
    *,
    embedding_model: EmbeddingModel,
    vector_store: VectorStoreReader,
    top_k: int = DEFAULT_TOP_K,
    access_scope: RagAccessScope | None = None,
    permission_group: str | None = None,
    business_domain: str | None = None,
    doc_type: str | None = None,
    source: str | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be blank")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    _validate_score_threshold(score_threshold)

    try:
        vectors = embedding_model.embed_texts([normalized_query])
    except Exception as exc:
        raise rag_embedding_failed(exc) from exc

    if len(vectors) != 1:
        raise rag_embedding_bad_response()

    query_vector = vectors[0]
    if len(query_vector) != embedding_model.dimension:
        raise rag_embedding_bad_response()

    payload_filter = build_payload_filter(
        access_scope=access_scope,
        permission_group=permission_group,
        business_domain=business_domain,
        doc_type=doc_type,
        source=source,
    )

    try:
        return vector_store.query_similar(
            query_vector,
            top_k=top_k,
            payload_filter=payload_filter,
            score_threshold=score_threshold,
            with_payload=True,
            with_vector=False,
        )
    except Exception as exc:
        raise rag_vector_store_failed(exc) from exc


def format_retrieved_chunks_for_debug(chunks: Sequence[RetrievedChunk]) -> list[str]:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown-source")
        section = chunk.metadata.get("section", "unknown-section")
        lines.append(
            f"{index}. score={chunk.score:.4f} source={source} section={section} chunk_id={chunk.chunk_id}"
        )
    return lines


def _validate_score_threshold(score_threshold: float | None) -> None:
    if score_threshold is None:
        return
    if not isinstance(score_threshold, int | float) or isinstance(score_threshold, bool):
        raise ValueError("score_threshold must be a number")
