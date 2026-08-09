import re

from app.rag.documents import Metadata, RagChunk, RagDocument
from app.rag.metadata import normalize_metadata


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
INLINE_METADATA_KEYS = ("文档类型", "业务领域", "权限组")


def split_text_into_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _join_blocks(blocks: list[str]) -> str:
    return "\n\n".join(blocks).strip()


def _is_inline_metadata_block(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return False

    for line in lines:
        if line.startswith(">"):
            line = line[1:].strip()
        if not any(line.startswith(key) for key in INLINE_METADATA_KEYS):
            return False
    return True


def _extract_markdown_heading(block: str) -> str | None:
    first_line = block.splitlines()[0].strip()
    if not first_line.startswith("#"):
        return None
    heading = first_line.lstrip("#").strip()
    return heading or None


def _split_oversized_block(
    block: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if len(block) <= chunk_size:
        return [block]

    segments: list[str] = []
    start = 0
    while start < len(block):
        end = min(start + chunk_size, len(block))
        segment = block[start:end].strip()
        if segment:
            segments.append(segment)
        if end == len(block):
            break
        start = max(end - chunk_overlap, start + 1)
    return segments


def _select_overlap_blocks(blocks: list[str], max_chars: int) -> list[str]:
    if max_chars <= 0:
        return []

    selected: list[str] = []
    for block in reversed(blocks):
        candidate = [block, *selected]
        if len(_join_blocks(candidate)) > max_chars:
            break
        selected.insert(0, block)
    return selected


def _build_chunk_id(metadata: Metadata, chunk_index: int) -> str:
    raw_source = str(metadata.get("source") or metadata.get("title") or "document")
    base = raw_source.rsplit(".", maxsplit=1)[0]
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()
    if not normalized:
        normalized = "document"
    return f"{normalized}_chunk_{chunk_index:04d}"


def _validate_chunk_options(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def split_document_into_chunks(
    document: RagDocument,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RagChunk]:
    _validate_chunk_options(chunk_size, chunk_overlap)

    records: list[tuple[str, str | None]] = []
    buffer: list[str] = []
    current_section: str | None = None

    def flush(*, keep_overlap: bool) -> None:
        nonlocal buffer
        if not buffer:
            return
        records.append((_join_blocks(buffer), current_section))
        buffer = _select_overlap_blocks(buffer, chunk_overlap) if keep_overlap else []

    for block in split_text_into_blocks(document.content):
        if _is_inline_metadata_block(block):
            continue

        heading = _extract_markdown_heading(block)
        if heading is not None:
            flush(keep_overlap=False)
            current_section = heading

        for segment in _split_oversized_block(
            block,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            candidate = _join_blocks([*buffer, segment])
            if buffer and len(candidate) > chunk_size:
                flush(keep_overlap=True)
                candidate = _join_blocks([*buffer, segment])
                if buffer and len(candidate) > chunk_size:
                    buffer = []
            buffer.append(segment)

    flush(keep_overlap=False)

    chunks: list[RagChunk] = []
    chunk_count = len(records)
    for index, (content, section) in enumerate(records, start=1):
        chunk_id = _build_chunk_id(document.metadata, index)
        metadata: Metadata = normalize_metadata({
            **document.metadata,
            "chunk_id": chunk_id,
            "chunk_index": index,
            "chunk_count": chunk_count,
            "chunk_size_chars": len(content),
        })
        if section is not None:
            metadata["section"] = section
        chunks.append(
            RagChunk(
                chunk_id=chunk_id,
                content=content,
                metadata=metadata,
            )
        )
    return chunks


def split_documents_into_chunks(
    documents: list[RagDocument],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for document in documents:
        chunks.extend(
            split_document_into_chunks(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks
