from pathlib import Path

import pytest

from app.rag.documents import RagDocument
from app.rag.loaders import load_document, load_documents_from_directory
from app.rag.splitters import (
    split_document_into_chunks,
    split_documents_into_chunks,
    split_text_into_blocks,
)


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"


def test_split_text_into_blocks_uses_blank_lines_as_boundaries() -> None:
    text = "第一段\n\n第二段\n\n\n第三段"

    assert split_text_into_blocks(text) == ["第一段", "第二段", "第三段"]


def test_split_markdown_document_keeps_metadata_and_stable_chunk_ids() -> None:
    document = load_document(
        KNOWLEDGE_BASE_DIR / "order-shipping-policy.md",
        base_dir=KNOWLEDGE_BASE_DIR,
    )

    chunks = split_document_into_chunks(document, chunk_size=180, chunk_overlap=30)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "order_shipping_policy_chunk_0001"
    assert chunks[0].metadata["source"] == "order-shipping-policy.md"
    assert chunks[0].metadata["chunk_index"] == 1
    assert chunks[0].metadata["chunk_count"] == len(chunks)
    assert "文档类型" not in chunks[0].content


def test_split_markdown_document_tracks_section_headings() -> None:
    document = load_document(
        KNOWLEDGE_BASE_DIR / "refund-return-policy.md",
        base_dir=KNOWLEDGE_BASE_DIR,
    )

    chunks = split_document_into_chunks(document, chunk_size=160, chunk_overlap=20)
    sections = {chunk.metadata.get("section") for chunk in chunks}

    assert "退款退货规则" in sections
    assert "七天无理由退货" in sections
    assert "退款到账时间" in sections


def test_split_document_preserves_paragraph_overlap_when_possible() -> None:
    document = RagDocument(
        content="alpha-001\n\nbeta-0002\n\ngamma-003",
        metadata={"source": "overlap-demo.md"},
    )

    chunks = split_document_into_chunks(document, chunk_size=22, chunk_overlap=10)

    assert len(chunks) == 2
    assert chunks[0].content == "alpha-001\n\nbeta-0002"
    assert chunks[1].content == "beta-0002\n\ngamma-003"


def test_split_document_rejects_invalid_overlap() -> None:
    document = RagDocument(content="短文档", metadata={"source": "demo.md"})

    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        split_document_into_chunks(document, chunk_size=20, chunk_overlap=20)


def test_split_loaded_directory_documents_into_chunks() -> None:
    documents = load_documents_from_directory(KNOWLEDGE_BASE_DIR)

    chunks = split_documents_into_chunks(documents, chunk_size=220, chunk_overlap=40)
    sources = {chunk.metadata["source"] for chunk in chunks}

    assert len(chunks) > len(documents)
    assert "order-shipping-policy.md" in sources
    assert "logistics-tracking-faq.txt" in sources
