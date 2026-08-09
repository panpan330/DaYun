from collections.abc import Sequence
from enum import Enum
import re

from pydantic import BaseModel, Field, model_validator

from app.rag.documents import Metadata, RetrievedChunk
from app.rag.hybrid import extract_keyword_terms


DEFAULT_CONTEXT_BUDGET_CHARS = 1800
DEFAULT_MAX_CHUNK_CHARS = 700
DEFAULT_MIN_CHUNK_CHARS = 160
OMISSION_MARKER = "\n...\n"


class ContextCompressionAction(str, Enum):
    KEEP_FULL = "keep_full"
    COMPRESS = "compress"
    DROP = "drop"


class ContextCompressionPolicy(BaseModel):
    max_total_chars: int = Field(default=DEFAULT_CONTEXT_BUDGET_CHARS, gt=0)
    max_chunk_chars: int = Field(default=DEFAULT_MAX_CHUNK_CHARS, gt=0)
    min_chunk_chars: int = Field(default=DEFAULT_MIN_CHUNK_CHARS, gt=0)
    always_keep_top_n: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_budget_shape(self) -> "ContextCompressionPolicy":
        if self.min_chunk_chars > self.max_chunk_chars:
            raise ValueError("min_chunk_chars must not exceed max_chunk_chars")
        if self.min_chunk_chars > self.max_total_chars:
            raise ValueError("min_chunk_chars must not exceed max_total_chars")
        return self


class ContextCompressionItem(BaseModel):
    chunk_id: str = Field(min_length=1)
    original_rank: int = Field(ge=1)
    action: ContextCompressionAction
    original_chars: int = Field(ge=0)
    final_chars: int = Field(ge=0)
    saved_chars: int = Field(ge=0)
    score: float
    source: str | None = None
    section: str | None = None
    query_term_hits: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class ContextCompressionReport(BaseModel):
    query: str = Field(min_length=1)
    budget_chars: int = Field(gt=0)
    original_total_chars: int = Field(ge=0)
    final_total_chars: int = Field(ge=0)
    saved_chars: int = Field(ge=0)
    input_chunk_count: int = Field(ge=0)
    kept_chunk_count: int = Field(ge=0)
    compressed_chunk_count: int = Field(ge=0)
    dropped_chunk_count: int = Field(ge=0)
    compressed_chunks: list[RetrievedChunk] = Field(default_factory=list)
    items: list[ContextCompressionItem] = Field(default_factory=list)
    debug_lines: list[str] = Field(default_factory=list)


def compress_retrieved_context(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    policy: ContextCompressionPolicy | None = None,
) -> ContextCompressionReport:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be blank")

    active_policy = policy or ContextCompressionPolicy()
    query_terms = extract_keyword_terms(normalized_query)
    remaining_chars = active_policy.max_total_chars
    compressed_chunks: list[RetrievedChunk] = []
    items: list[ContextCompressionItem] = []

    for rank, chunk in enumerate(chunks, start=1):
        original_content = chunk.content.strip()
        original_chars = len(original_content)
        query_term_hits = _find_query_term_hits(query_terms, chunk)
        source = _metadata_text(chunk.metadata, "source")
        section = _metadata_text(chunk.metadata, "section")

        if remaining_chars <= 0:
            items.append(
                _build_item(
                    chunk,
                    rank=rank,
                    action=ContextCompressionAction.DROP,
                    original_chars=original_chars,
                    final_chars=0,
                    query_term_hits=query_term_hits,
                    reason="context budget exhausted",
                    source=source,
                    section=section,
                )
            )
            continue

        max_allowed_for_chunk = min(active_policy.max_chunk_chars, remaining_chars)
        if (
            rank > active_policy.always_keep_top_n
            and max_allowed_for_chunk < active_policy.min_chunk_chars
        ):
            items.append(
                _build_item(
                    chunk,
                    rank=rank,
                    action=ContextCompressionAction.DROP,
                    original_chars=original_chars,
                    final_chars=0,
                    query_term_hits=query_term_hits,
                    reason="remaining budget is below min_chunk_chars",
                    source=source,
                    section=section,
                )
            )
            continue

        if original_chars <= max_allowed_for_chunk:
            final_content = original_content
            action = ContextCompressionAction.KEEP_FULL
            reason = "chunk fits budget without compression"
        else:
            final_content = _compress_chunk_text(
                normalized_query,
                original_content,
                max_chars=max_allowed_for_chunk,
            )
            action = ContextCompressionAction.COMPRESS
            reason = _compression_reason(rank, active_policy, query_term_hits)

        final_chars = len(final_content)
        if final_chars <= 0:
            items.append(
                _build_item(
                    chunk,
                    rank=rank,
                    action=ContextCompressionAction.DROP,
                    original_chars=original_chars,
                    final_chars=0,
                    query_term_hits=query_term_hits,
                    reason="compressed content is empty",
                    source=source,
                    section=section,
                )
            )
            continue

        remaining_chars -= final_chars
        compressed_chunk = _copy_chunk_with_compressed_content(
            chunk,
            content=final_content,
            action=action,
            original_rank=rank,
            original_chars=original_chars,
            final_chars=final_chars,
        )
        compressed_chunks.append(compressed_chunk)
        items.append(
            _build_item(
                chunk,
                rank=rank,
                action=action,
                original_chars=original_chars,
                final_chars=final_chars,
                query_term_hits=query_term_hits,
                reason=reason,
                source=source,
                section=section,
            )
        )

    original_total_chars = sum(len(chunk.content.strip()) for chunk in chunks)
    final_total_chars = sum(len(chunk.content.strip()) for chunk in compressed_chunks)
    compressed_count = sum(
        1
        for item in items
        if item.action is ContextCompressionAction.COMPRESS
    )
    dropped_count = sum(
        1
        for item in items
        if item.action is ContextCompressionAction.DROP
    )
    report = ContextCompressionReport(
        query=normalized_query,
        budget_chars=active_policy.max_total_chars,
        original_total_chars=original_total_chars,
        final_total_chars=final_total_chars,
        saved_chars=max(original_total_chars - final_total_chars, 0),
        input_chunk_count=len(chunks),
        kept_chunk_count=len(compressed_chunks),
        compressed_chunk_count=compressed_count,
        dropped_chunk_count=dropped_count,
        compressed_chunks=compressed_chunks,
        items=items,
    )
    return report.model_copy(
        update={"debug_lines": format_context_compression_report_for_debug(report)}
    )


def estimate_context_chars(chunks: Sequence[RetrievedChunk]) -> int:
    return sum(len(chunk.content.strip()) for chunk in chunks)


def format_context_compression_report_for_debug(
    report: ContextCompressionReport,
) -> list[str]:
    lines = [
        (
            f"budget={report.budget_chars} original={report.original_total_chars} "
            f"final={report.final_total_chars} saved={report.saved_chars} "
            f"input_chunks={report.input_chunk_count} kept={report.kept_chunk_count} "
            f"compressed={report.compressed_chunk_count} dropped={report.dropped_chunk_count}"
        )
    ]
    for item in report.items:
        source = item.source or "unknown-source"
        section = item.section or "unknown-section"
        hits = ",".join(item.query_term_hits[:6]) or "-"
        lines.append(
            (
                f"{item.original_rank}. action={item.action.value} "
                f"chars={item.original_chars}->{item.final_chars} "
                f"saved={item.saved_chars} score={item.score:.4f} "
                f"source={source} section={section} "
                f"chunk_id={item.chunk_id} hits={hits} reason={item.reason}"
            )
        )
    return lines


def _compress_chunk_text(
    query: str,
    content: str,
    *,
    max_chars: int,
) -> str:
    if len(content) <= max_chars:
        return content
    if max_chars <= 0:
        return ""

    query_terms = extract_keyword_terms(query)
    units = _split_text_units(content)
    selected = _select_relevant_units(units, query_terms, max_chars=max_chars)
    if selected:
        return _fit_text_to_chars(OMISSION_MARKER.join(selected), max_chars=max_chars)
    return _head_tail_excerpt(content, max_chars=max_chars)


def _select_relevant_units(
    units: Sequence[str],
    query_terms: Sequence[str],
    *,
    max_chars: int,
) -> list[str]:
    if not query_terms:
        return []
    scored_units: list[tuple[int, int, str]] = []
    for index, unit in enumerate(units):
        score = _query_term_score(unit, query_terms)
        if score > 0:
            scored_units.append((score, index, unit))
    selected_indexes: list[int] = []
    used_chars = 0
    for _, index, unit in sorted(scored_units, key=lambda item: (-item[0], item[1])):
        unit_chars = len(unit)
        separator_chars = len(OMISSION_MARKER) if selected_indexes else 0
        if used_chars + separator_chars + unit_chars > max_chars:
            continue
        selected_indexes.append(index)
        used_chars += separator_chars + unit_chars
    return [units[index] for index in sorted(selected_indexes)]


def _split_text_units(content: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", content.strip())
    if not normalized:
        return []
    units = [
        unit.strip()
        for unit in re.split(r"(?<=[。！？.!?；;])\s+", normalized)
        if unit.strip()
    ]
    return units or [normalized]


def _head_tail_excerpt(content: str, *, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    if max_chars <= len(OMISSION_MARKER) + 2:
        return content[:max_chars]
    side_chars = (max_chars - len(OMISSION_MARKER)) // 2
    tail_chars = max_chars - len(OMISSION_MARKER) - side_chars
    return f"{content[:side_chars].rstrip()}{OMISSION_MARKER}{content[-tail_chars:].lstrip()}"


def _fit_text_to_chars(text: str, *, max_chars: int) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return _head_tail_excerpt(stripped, max_chars=max_chars)


def _find_query_term_hits(
    query_terms: Sequence[str],
    chunk: RetrievedChunk,
) -> list[str]:
    search_text = _chunk_search_text(chunk)
    return [
        term
        for term in query_terms
        if term and term in search_text
    ]


def _query_term_score(text: str, query_terms: Sequence[str]) -> int:
    normalized = text.lower()
    return sum(normalized.count(term) for term in query_terms)


def _chunk_search_text(chunk: RetrievedChunk) -> str:
    metadata_values = [
        value
        for key in ("source", "title", "section", "doc_type", "business_domain")
        if isinstance((value := chunk.metadata.get(key)), str)
    ]
    return "\n".join([chunk.content, *metadata_values]).lower()


def _copy_chunk_with_compressed_content(
    chunk: RetrievedChunk,
    *,
    content: str,
    action: ContextCompressionAction,
    original_rank: int,
    original_chars: int,
    final_chars: int,
) -> RetrievedChunk:
    metadata: Metadata = {
        **chunk.metadata,
        "context_compression_action": action.value,
        "context_original_rank": original_rank,
        "context_original_chars": original_chars,
        "context_final_chars": final_chars,
    }
    return chunk.model_copy(
        update={
            "content": content,
            "metadata": metadata,
        }
    )


def _build_item(
    chunk: RetrievedChunk,
    *,
    rank: int,
    action: ContextCompressionAction,
    original_chars: int,
    final_chars: int,
    query_term_hits: Sequence[str],
    reason: str,
    source: str | None,
    section: str | None,
) -> ContextCompressionItem:
    return ContextCompressionItem(
        chunk_id=chunk.chunk_id,
        original_rank=rank,
        action=action,
        original_chars=original_chars,
        final_chars=final_chars,
        saved_chars=max(original_chars - final_chars, 0),
        score=chunk.score,
        source=source,
        section=section,
        query_term_hits=list(query_term_hits),
        reason=reason,
    )


def _compression_reason(
    rank: int,
    policy: ContextCompressionPolicy,
    query_term_hits: Sequence[str],
) -> str:
    if rank <= policy.always_keep_top_n:
        return "top-ranked chunk was compressed to fit per-chunk or total budget"
    if query_term_hits:
        return "chunk was compressed around query-matching text"
    return "chunk was compressed with head-tail excerpt because no query terms matched"


def _metadata_text(metadata: Metadata, key: str) -> str | None:
    value = metadata.get(key)
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
