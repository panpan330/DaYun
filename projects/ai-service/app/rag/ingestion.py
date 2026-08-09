from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.rag.documents import RagDocument
from app.rag.embeddings import EmbeddingModel, embed_chunks
from app.rag.errors import rag_embedding_failed, rag_vector_store_failed
from app.rag.filters import build_payload_filter
from app.rag.loaders import load_document, load_documents_from_directory
from app.rag.splitters import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_documents_into_chunks,
)


class VectorStoreWriter(Protocol):
    collection_name: str

    def ensure_collection(self, *, vector_size: int, distance: str = "Cosine") -> None:
        """Create or validate the target vector collection."""

    def upsert_embedded_chunks(self, embedded_chunks, *, wait: bool = True) -> int:
        """Write embedded chunks to the vector store."""


class VectorStoreUpdater(VectorStoreWriter, Protocol):
    def delete_points_by_filter(
        self,
        payload_filter: Mapping[str, Any],
        *,
        wait: bool = True,
    ) -> None:
        """Delete vector points that match a metadata filter."""


class RagIngestionResult(BaseModel):
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    vector_dimension: int = Field(gt=0)
    collection_name: str = Field(min_length=1)
    replaced_source_count: int = Field(default=0, ge=0)


class RagDeletionResult(BaseModel):
    source: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)


def ingest_directory_to_vector_store(
    directory: Path | str,
    *,
    embedding_model: EmbeddingModel,
    vector_store: VectorStoreWriter,
    include_readme: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    distance: str = "Cosine",
    wait: bool = True,
) -> RagIngestionResult:
    documents = load_documents_from_directory(directory, include_readme=include_readme)
    chunks = split_documents_into_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    try:
        embedded_chunks = embed_chunks(chunks, embedding_model=embedding_model)
    except Exception as exc:
        raise rag_embedding_failed(exc) from exc

    try:
        vector_store.ensure_collection(
            vector_size=embedding_model.dimension,
            distance=distance,
        )
        vector_count = vector_store.upsert_embedded_chunks(embedded_chunks, wait=wait)
    except Exception as exc:
        raise rag_vector_store_failed(exc) from exc

    return RagIngestionResult(
        document_count=len(documents),
        chunk_count=len(chunks),
        vector_count=vector_count,
        vector_dimension=embedding_model.dimension,
        collection_name=vector_store.collection_name,
    )


def refresh_directory_in_vector_store(
    directory: Path | str,
    *,
    embedding_model: EmbeddingModel,
    vector_store: VectorStoreUpdater,
    include_readme: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    distance: str = "Cosine",
    wait: bool = True,
) -> RagIngestionResult:
    documents = load_documents_from_directory(directory, include_readme=include_readme)
    chunks = split_documents_into_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    try:
        embedded_chunks = embed_chunks(chunks, embedding_model=embedding_model)
    except Exception as exc:
        raise rag_embedding_failed(exc) from exc

    sources = _extract_document_sources(documents)

    try:
        vector_store.ensure_collection(
            vector_size=embedding_model.dimension,
            distance=distance,
        )
        for source in sources:
            payload_filter = build_payload_filter(source=source)
            if payload_filter is None:
                raise ValueError("source payload filter must not be empty")
            vector_store.delete_points_by_filter(payload_filter, wait=wait)
        vector_count = vector_store.upsert_embedded_chunks(embedded_chunks, wait=wait)
    except Exception as exc:
        raise rag_vector_store_failed(exc) from exc

    return RagIngestionResult(
        document_count=len(documents),
        chunk_count=len(chunks),
        vector_count=vector_count,
        vector_dimension=embedding_model.dimension,
        collection_name=vector_store.collection_name,
        replaced_source_count=len(sources),
    )


def delete_document_from_vector_store(
    source: str,
    *,
    vector_store: VectorStoreUpdater,
    wait: bool = True,
) -> RagDeletionResult:
    payload_filter = build_payload_filter(source=source)
    if payload_filter is None:
        raise ValueError("source must not be blank")

    try:
        vector_store.delete_points_by_filter(payload_filter, wait=wait)
    except Exception as exc:
        raise rag_vector_store_failed(exc) from exc

    return RagDeletionResult(
        source=source.strip(),
        collection_name=vector_store.collection_name,
    )


def _extract_document_sources(documents: list[RagDocument]) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for document in documents:
        source = document.metadata.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("document source metadata must not be blank")
        normalized_source = source.strip()
        if normalized_source in seen:
            continue
        sources.append(normalized_source)
        seen.add(normalized_source)
    return sources


def ingest_single_document(
    path: Path | str,
    *,
    embedding_model: EmbeddingModel,
    vector_store: VectorStoreWriter,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    wait: bool = True,
) -> RagIngestionResult:
    document = load_document(path)
    chunks = split_documents_into_chunks(
        [document],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    try:
        embedded_chunks = embed_chunks(chunks, embedding_model=embedding_model)
    except Exception as exc:
        raise rag_embedding_failed(exc) from exc

    try:
        vector_store.ensure_collection(
            vector_size=embedding_model.dimension,
            distance="Cosine",
        )
        vector_count = vector_store.upsert_embedded_chunks(embedded_chunks, wait=wait)
    except Exception as exc:
        raise rag_vector_store_failed(exc) from exc

    return RagIngestionResult(
        document_count=1,
        chunk_count=len(chunks),
        vector_count=vector_count,
        vector_dimension=embedding_model.dimension,
        collection_name=vector_store.collection_name,
    )


def update_single_document(
    path: Path | str,
    *,
    embedding_model: EmbeddingModel,
    vector_store: VectorStoreUpdater,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    wait: bool = True,
) -> RagIngestionResult:
    source = Path(path).name
    delete_document_from_vector_store(
        source,
        vector_store=vector_store,
        wait=wait,
    )
    return ingest_single_document(
        path,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        wait=wait,
    )
