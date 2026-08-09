import pytest
from pydantic import ValidationError

from app.rag.documents import RagChunk, RagDocument, RetrievedChunk


def test_rag_document_keeps_content_and_metadata() -> None:
    document = RagDocument(
        content="订单超过 72 小时未发货，可以创建投诉工单。",
        metadata={
            "source": "order_policy.md",
            "doc_type": "policy",
            "permission_group": "customer_service",
        },
    )

    assert document.content == "订单超过 72 小时未发货，可以创建投诉工单。"
    assert document.metadata["source"] == "order_policy.md"


def test_rag_chunk_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        RagChunk(chunk_id="order_policy_001_chunk_001", content="")


def test_rag_chunk_uses_chunk_id_as_future_point_id() -> None:
    chunk = RagChunk(
        chunk_id="order_policy_001_chunk_003",
        content="订单超过 72 小时未发货，可以创建投诉工单。",
        metadata={"chunk_index": 3},
    )

    assert chunk.chunk_id == "order_policy_001_chunk_003"
    assert chunk.metadata["chunk_index"] == 3


def test_retrieved_chunk_keeps_score_and_payload_metadata() -> None:
    chunk = RetrievedChunk(
        point_id="point-001",
        chunk_id="order_policy_chunk_0001",
        content="订单付款后 24 小时内发货。",
        metadata={
            "source": "order-shipping-policy.md",
            "section": "正常发货时效",
        },
        score=0.87,
    )

    assert chunk.point_id == "point-001"
    assert chunk.chunk_id == "order_policy_chunk_0001"
    assert chunk.metadata["source"] == "order-shipping-policy.md"
    assert chunk.score == 0.87
