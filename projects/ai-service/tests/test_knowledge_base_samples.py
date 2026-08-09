from pathlib import Path


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"


def test_sample_knowledge_base_documents_exist() -> None:
    expected_files = {
        "order-shipping-policy.md",
        "refund-return-policy.md",
        "logistics-tracking-faq.txt",
        "account-security-faq.md",
    }

    actual_files = {path.name for path in KNOWLEDGE_BASE_DIR.iterdir() if path.is_file()}

    assert expected_files.issubset(actual_files)


def test_sample_knowledge_base_documents_are_not_empty() -> None:
    sample_files = [
        path
        for path in KNOWLEDGE_BASE_DIR.iterdir()
        if path.suffix in {".md", ".txt"} and path.name != "README.md"
    ]

    assert sample_files
    for path in sample_files:
        assert path.read_text(encoding="utf-8").strip()
