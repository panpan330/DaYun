from collections.abc import Sequence
import hashlib
from typing import Any, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.rag.documents import Metadata, RagChunk


Vector = list[float]
DEFAULT_EMBEDDING_BATCH_SIZE = 64
FLOAT32_BYTES = 4


class EmbeddingModel(Protocol):
    @property
    def dimension(self) -> int:
        """Number of float values returned for each text."""

    def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        """Convert each input text into one vector."""


class EmbeddedChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)
    vector: Vector = Field(min_length=1)


class DeterministicHashEmbeddingModel:
    """Deterministic placeholder embedding model used before real embedding APIs."""

    def __init__(self, *, dimension: int = 8) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be greater than 0")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Vector:
        if not text.strip():
            raise ValueError("text to embed must not be blank")

        vector: Vector = []
        for index in range(self.dimension):
            seed = f"{index}:{text}".encode("utf-8")
            digest = hashlib.sha256(seed).digest()
            raw_value = int.from_bytes(digest[:8], byteorder="big", signed=False)
            vector.append(round(raw_value / 2**64, 6))
        return vector


class OpenAICompatibleEmbeddingModel:
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        dimension: int,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        request_dimensions: bool = False,
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model must not be blank")
        if dimension <= 0:
            raise ValueError("embedding dimension must be greater than 0")
        _validate_batch_size(batch_size)

        self.client = client
        self.model = model.strip()
        self._dimension = dimension
        self.batch_size = batch_size
        self.request_dimensions = request_dimensions

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleEmbeddingModel":
        api_key = settings.resolved_embedding_api_key
        if api_key is None:
            raise ValueError("EMBEDDING_API_KEY is not configured")

        client_kwargs: dict[str, object] = {
            "api_key": api_key,
            "max_retries": settings.llm_max_retries,
            "timeout": settings.request_timeout_seconds,
        }
        base_url = settings.resolved_embedding_base_url
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        return cls(
            client=OpenAI(**client_kwargs),
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            request_dimensions=settings.embedding_request_dimensions,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        text_batches = split_texts_into_batches(texts, batch_size=self.batch_size)
        vectors: list[Vector] = []
        for batch in text_batches:
            request_body: dict[str, Any] = {
                "model": self.model,
                "input": batch,
                "encoding_format": "float",
            }
            if self.request_dimensions:
                request_body["dimensions"] = self.dimension

            response = self.client.embeddings.create(**request_body)
            vectors.extend(_extract_embedding_vectors(response, expected_count=len(batch)))

        for vector in vectors:
            _validate_vector_dimension(vector, self.dimension)
        return vectors


def embed_chunks(
    chunks: Sequence[RagChunk],
    *,
    embedding_model: EmbeddingModel,
) -> list[EmbeddedChunk]:
    texts = [chunk.content for chunk in chunks]
    vectors = embedding_model.embed_texts(texts)

    if len(vectors) != len(chunks):
        raise ValueError("embedding result count must match chunk count")

    embedded_chunks: list[EmbeddedChunk] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        if len(vector) != embedding_model.dimension:
            raise ValueError("embedding vector size must match model dimension")
        embedded_chunks.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                vector=vector,
            )
        )
    return embedded_chunks


def split_texts_into_batches(
    texts: Sequence[str],
    *,
    batch_size: int,
) -> list[list[str]]:
    _validate_batch_size(batch_size)
    normalized_texts = [text for text in texts]
    for text in normalized_texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text to embed must not be blank")
    return [
        normalized_texts[index : index + batch_size]
        for index in range(0, len(normalized_texts), batch_size)
    ]


def estimate_dense_vector_storage_bytes(
    *,
    vector_count: int,
    dimension: int,
    bytes_per_value: int = FLOAT32_BYTES,
) -> int:
    if vector_count < 0:
        raise ValueError("vector_count must be greater than or equal to 0")
    if dimension <= 0:
        raise ValueError("dimension must be greater than 0")
    if bytes_per_value <= 0:
        raise ValueError("bytes_per_value must be greater than 0")
    return vector_count * dimension * bytes_per_value


def _validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("embedding batch_size must be greater than 0")


def _extract_embedding_vectors(response: Any, *, expected_count: int) -> list[Vector]:
    data = getattr(response, "data", None)
    if not isinstance(data, Sequence):
        raise ValueError("embedding response data must be a sequence")
    if len(data) != expected_count:
        raise ValueError("embedding result count must match input count")

    vectors: list[Vector] = []
    for item in data:
        embedding = getattr(item, "embedding", None)
        if not isinstance(embedding, Sequence):
            raise ValueError("embedding response item is missing embedding vector")
        vector: Vector = []
        for value in embedding:
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError("embedding vector must contain only numbers")
            vector.append(float(value))
        vectors.append(vector)
    return vectors


def _validate_vector_dimension(vector: Vector, dimension: int) -> None:
    if len(vector) != dimension:
        raise ValueError("embedding vector size must match model dimension")
    for value in vector:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError("embedding vector must contain only numbers")
