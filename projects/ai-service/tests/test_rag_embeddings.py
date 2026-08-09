from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.rag.documents import RagChunk
from app.rag import embeddings
from app.rag.embeddings import (
    DeterministicHashEmbeddingModel,
    OpenAICompatibleEmbeddingModel,
    embed_chunks,
    estimate_dense_vector_storage_bytes,
    split_texts_into_batches,
)


class FakeEmbeddingEndpoint:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[dict] = []
        self.offset = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        input_texts = kwargs["input"]
        vectors = self.vectors[self.offset : self.offset + len(input_texts)]
        self.offset += len(input_texts)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=vector)
                for vector in vectors
            ]
        )


class FakeEmbeddingClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = FakeEmbeddingEndpoint(vectors)


def test_hash_embedding_model_returns_stable_vectors() -> None:
    model = DeterministicHashEmbeddingModel(dimension=4)

    first = model.embed_texts(["shipping policy"])
    second = model.embed_texts(["shipping policy"])

    assert first == second
    assert len(first[0]) == 4
    assert all(0 <= value <= 1 for value in first[0])


def test_hash_embedding_model_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        DeterministicHashEmbeddingModel(dimension=0)


def test_hash_embedding_model_rejects_blank_text() -> None:
    model = DeterministicHashEmbeddingModel(dimension=4)

    with pytest.raises(ValueError, match="must not be blank"):
        model.embed_texts(["   "])


def test_embed_chunks_preserves_chunk_content_metadata_and_vector() -> None:
    model = DeterministicHashEmbeddingModel(dimension=4)
    chunk = RagChunk(
        chunk_id="shipping_chunk_0001",
        content="Orders ship within 24 hours.",
        metadata={"source": "shipping.md", "chunk_index": 1},
    )

    embedded_chunks = embed_chunks([chunk], embedding_model=model)

    assert len(embedded_chunks) == 1
    embedded = embedded_chunks[0]
    assert embedded.chunk_id == "shipping_chunk_0001"
    assert embedded.content == "Orders ship within 24 hours."
    assert embedded.metadata["source"] == "shipping.md"
    assert len(embedded.vector) == 4


def test_embed_chunks_rejects_vector_count_mismatch() -> None:
    class BadCountEmbeddingModel:
        dimension = 4

        def embed_texts(self, texts):
            return []

    chunk = RagChunk(
        chunk_id="chunk_0001",
        content="content",
        metadata={"source": "demo.md"},
    )

    with pytest.raises(ValueError, match="result count"):
        embed_chunks([chunk], embedding_model=BadCountEmbeddingModel())


def test_embed_chunks_rejects_vector_dimension_mismatch() -> None:
    class BadDimensionEmbeddingModel:
        dimension = 4

        def embed_texts(self, texts):
            return [[0.1, 0.2]]

    chunk = RagChunk(
        chunk_id="chunk_0001",
        content="content",
        metadata={"source": "demo.md"},
    )

    with pytest.raises(ValueError, match="vector size"):
        embed_chunks([chunk], embedding_model=BadDimensionEmbeddingModel())


def test_split_texts_into_batches_groups_inputs() -> None:
    batches = split_texts_into_batches(
        ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"],
        batch_size=2,
    )

    assert batches == [
        ["chunk-1", "chunk-2"],
        ["chunk-3", "chunk-4"],
        ["chunk-5"],
    ]


def test_split_texts_into_batches_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        split_texts_into_batches(["chunk"], batch_size=0)

    with pytest.raises(ValueError, match="must not be blank"):
        split_texts_into_batches(["chunk", "   "], batch_size=2)


def test_estimate_dense_vector_storage_bytes_uses_dimension_and_count() -> None:
    assert estimate_dense_vector_storage_bytes(vector_count=1000, dimension=1536) == 6_144_000
    assert estimate_dense_vector_storage_bytes(
        vector_count=1000,
        dimension=1024,
        bytes_per_value=2,
    ) == 2_048_000


def test_openai_compatible_embedding_model_batches_requests() -> None:
    client = FakeEmbeddingClient(
        vectors=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    )
    model = OpenAICompatibleEmbeddingModel(
        client=client,
        model=" text-embedding-v4 ",
        dimension=3,
        batch_size=2,
        request_dimensions=True,
    )

    vectors = model.embed_texts(["订单发货", "退款规则", "账号安全"])

    assert vectors == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    assert client.embeddings.calls == [
        {
            "model": "text-embedding-v4",
            "input": ["订单发货", "退款规则"],
            "encoding_format": "float",
            "dimensions": 3,
        },
        {
            "model": "text-embedding-v4",
            "input": ["账号安全"],
            "encoding_format": "float",
            "dimensions": 3,
        },
    ]


def test_openai_compatible_embedding_model_can_omit_dimensions_parameter() -> None:
    client = FakeEmbeddingClient(vectors=[[0.1, 0.2]])
    model = OpenAICompatibleEmbeddingModel(
        client=client,
        model="text-embedding-3-small",
        dimension=2,
        request_dimensions=False,
    )

    assert model.embed_texts(["shipping"]) == [[0.1, 0.2]]
    assert "dimensions" not in client.embeddings.calls[0]


def test_openai_compatible_embedding_model_validates_provider_response() -> None:
    count_mismatch_client = FakeEmbeddingClient(vectors=[[0.1, 0.2]])
    count_mismatch_model = OpenAICompatibleEmbeddingModel(
        client=count_mismatch_client,
        model="text-embedding-3-small",
        dimension=2,
        batch_size=2,
    )

    with pytest.raises(ValueError, match="result count"):
        count_mismatch_model.embed_texts(["first", "second"])

    dimension_mismatch_client = FakeEmbeddingClient(vectors=[[0.1]])
    dimension_mismatch_model = OpenAICompatibleEmbeddingModel(
        client=dimension_mismatch_client,
        model="text-embedding-3-small",
        dimension=2,
    )

    with pytest.raises(ValueError, match="vector size"):
        dimension_mismatch_model.embed_texts(["first"])

    non_numeric_client = FakeEmbeddingClient(vectors=[[0.1, True]])
    non_numeric_model = OpenAICompatibleEmbeddingModel(
        client=non_numeric_client,
        model="text-embedding-3-small",
        dimension=2,
    )

    with pytest.raises(ValueError, match="only numbers"):
        non_numeric_model.embed_texts(["first"])


def test_openai_compatible_embedding_model_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list[object] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.embeddings = FakeEmbeddingEndpoint(vectors=[])
            created_clients.append(self)

    monkeypatch.setattr(embeddings, "OpenAI", FakeOpenAI)
    settings = Settings(
        embedding_api_key="embedding-test-key",
        embedding_base_url=" https://embedding.example.com/v1/ ",
        embedding_model="text-embedding-v4",
        embedding_dimension=1024,
        embedding_batch_size=10,
        embedding_request_dimensions=True,
        llm_max_retries=3,
        request_timeout_seconds=12.5,
        _env_file=None,
    )

    model = OpenAICompatibleEmbeddingModel.from_settings(settings)

    assert model.model == "text-embedding-v4"
    assert model.dimension == 1024
    assert model.batch_size == 10
    assert model.request_dimensions is True
    assert created_clients[0].kwargs == {
        "api_key": "embedding-test-key",
        "base_url": "https://embedding.example.com/v1",
        "max_retries": 3,
        "timeout": 12.5,
    }
