from pathlib import Path

import pytest

from app.rag.loaders import (
    clean_document_text,
    load_document,
    load_documents_from_directory,
)


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"


def test_clean_document_text_normalizes_line_endings_and_blank_lines() -> None:
    raw_text = "第一行  \r\n\r\n\r\n第二行\t \r第三行"

    assert clean_document_text(raw_text) == "第一行\n\n第二行\n第三行"


def test_load_markdown_document_extracts_title_and_metadata() -> None:
    document = load_document(
        KNOWLEDGE_BASE_DIR / "order-shipping-policy.md",
        base_dir=KNOWLEDGE_BASE_DIR,
    )

    assert document.metadata["source"] == "order-shipping-policy.md"
    assert document.metadata["title"] == "订单发货规则"
    assert document.metadata["doc_type"] == "policy"
    assert document.metadata["business_domain"] == "order"
    assert document.metadata["permission_group"] == "customer_service"
    assert "超过 72 小时未发货" in document.content


def test_load_txt_document_extracts_plain_text_title_and_metadata() -> None:
    document = load_document(
        KNOWLEDGE_BASE_DIR / "logistics-tracking-faq.txt",
        base_dir=KNOWLEDGE_BASE_DIR,
    )

    assert document.metadata["source"] == "logistics-tracking-faq.txt"
    assert document.metadata["title"] == "物流查询常见问题"
    assert document.metadata["doc_type"] == "faq"
    assert document.metadata["business_domain"] == "logistics"
    assert "问题：物流三天没有更新怎么办？" in document.content


def test_load_documents_from_directory_skips_readme_by_default() -> None:
    documents = load_documents_from_directory(KNOWLEDGE_BASE_DIR)

    sources = {document.metadata["source"] for document in documents}

    assert len(documents) >= 4
    assert "README.md" not in sources
    assert {
        "account-security-faq.md",
        "logistics-tracking-faq.txt",
        "order-shipping-policy.md",
        "refund-return-policy.md",
    } <= sources


def test_load_document_rejects_unsupported_suffix(tmp_path: Path) -> None:
    document_path = tmp_path / "policy.docx"
    document_path.write_text("fake docx text", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported document type"):
        load_document(document_path)
