from pathlib import Path
import re

from app.rag.documents import Metadata, RagDocument
from app.rag.metadata import normalize_metadata


SUPPORTED_DOCUMENT_SUFFIXES = {".md", ".txt", ".pdf"}
DEFAULT_IGNORED_FILE_NAMES = {"README.md"}

METADATA_KEY_MAP = {
    "文档类型": "doc_type",
    "业务领域": "business_domain",
    "权限组": "permission_group",
}


def clean_document_text(raw_text: str) -> str:
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def _extract_prefixed_value(line: str, prefix: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith(">"):
        stripped = stripped[1:].strip()
    if not stripped.startswith(prefix):
        return None
    value = stripped.removeprefix(prefix).strip()
    if value.startswith(("：", ":")):
        value = value[1:].strip()
    return value or None


def extract_document_title(text: str, *, suffix: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if suffix == ".md" and stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped.startswith(">"):
            continue
        return stripped
    raise ValueError("document title not found")


def extract_inline_metadata(text: str) -> Metadata:
    metadata: Metadata = {}
    for line in text.splitlines()[:20]:
        for source_key, target_key in METADATA_KEY_MAP.items():
            value = _extract_prefixed_value(line, source_key)
            if value is not None:
                metadata[target_key] = value
    return metadata


def _build_source(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return path.name
    return path.relative_to(base_dir).as_posix()


def load_document(path: Path | str, *, base_dir: Path | str | None = None) -> RagDocument:
    document_path = Path(path)
    if document_path.suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError(f"unsupported document type: {document_path.suffix}")
    if not document_path.is_file():
        raise FileNotFoundError(document_path)

    resolved_base_dir = Path(base_dir) if base_dir is not None else None
    if document_path.suffix == ".pdf":
        from app.rag.pdf_loaders import clean_pdf_pages, extract_pdf_pages

        text = clean_document_text(
            "\n\n".join(clean_pdf_pages(extract_pdf_pages(document_path)))
        )
    else:
        text = clean_document_text(document_path.read_text(encoding="utf-8"))
    if not text:
        raise ValueError(f"document is empty: {document_path}")

    metadata: Metadata = {
        "source": _build_source(document_path, resolved_base_dir),
        "title": extract_document_title(text, suffix=document_path.suffix),
        "file_name": document_path.name,
        "file_extension": document_path.suffix,
    }
    metadata.update(extract_inline_metadata(text))

    return RagDocument(content=text, metadata=normalize_metadata(metadata))


def load_documents_from_directory(
    directory: Path | str,
    *,
    include_readme: bool = False,
) -> list[RagDocument]:
    source_dir = Path(directory)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)

    documents: list[RagDocument] = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            continue
        if not include_readme and path.name in DEFAULT_IGNORED_FILE_NAMES:
            continue
        documents.append(load_document(path, base_dir=source_dir))
    return documents
