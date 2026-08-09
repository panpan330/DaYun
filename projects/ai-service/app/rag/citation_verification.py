from collections import Counter
from collections.abc import Sequence
from enum import Enum
import re

from pydantic import BaseModel, Field

from app.rag.documents import RetrievedChunk
from app.rag.generator import RagAnswer, RagAnswerStatus, RagCitation


ASCII_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "will",
    "with",
}


class CitationFindingSeverity(str, Enum):
    WARNING = "warning"
    BLOCKING = "blocking"


class CitationFindingCategory(str, Enum):
    RESPONSE_STATE = "response_state"
    CITATION_REFERENCE = "citation_reference"
    CITATION_METADATA = "citation_metadata"
    ANSWER_SUPPORT = "answer_support"


class CitationVerificationPolicy(BaseModel):
    require_citations_for_answered: bool = True
    min_answer_support_score: float = Field(default=0.12, ge=0, le=1)
    min_citation_support_score: float = Field(default=0.05, ge=0, le=1)
    warn_on_duplicate_citations: bool = True


class CitationVerificationFinding(BaseModel):
    code: str = Field(min_length=1)
    category: CitationFindingCategory
    severity: CitationFindingSeverity
    message: str = Field(min_length=1)
    citation_index: int | None = Field(default=None, ge=1)
    source_index: int | None = Field(default=None, ge=1)
    chunk_id: str | None = None
    source: str | None = None
    evidence: str | None = None


class CitationVerificationReport(BaseModel):
    answer_status: RagAnswerStatus
    is_valid: bool
    retrieved_chunk_count: int = Field(ge=0)
    checked_citation_count: int = Field(ge=0)
    cited_chunk_count: int = Field(ge=0)
    missing_citation_count: int = Field(ge=0)
    answer_support_score: float = Field(ge=0, le=1)
    findings: list[CitationVerificationFinding] = Field(default_factory=list)
    debug_lines: list[str] = Field(default_factory=list)


def verify_rag_answer_sources(
    rag_answer: RagAnswer,
    retrieved_chunks: Sequence[RetrievedChunk],
    *,
    policy: CitationVerificationPolicy | None = None,
) -> CitationVerificationReport:
    active_policy = policy or CitationVerificationPolicy()
    chunks = list(retrieved_chunks)
    findings: list[CitationVerificationFinding] = []
    cited_chunks: dict[str, RetrievedChunk] = {}

    if rag_answer.status is RagAnswerStatus.NO_CONTEXT:
        findings.extend(_inspect_no_context_answer(rag_answer, chunks))
    else:
        findings.extend(_inspect_answered_state(rag_answer, chunks, active_policy))
        cited_chunks = _inspect_citations(
            rag_answer.citations,
            chunks,
            answer=rag_answer.answer,
            findings=findings,
            policy=active_policy,
        )

    answer_support_score = _answer_support_score(
        rag_answer.answer,
        list(cited_chunks.values()),
    )
    if (
        rag_answer.status is RagAnswerStatus.ANSWERED
        and cited_chunks
        and answer_support_score < active_policy.min_answer_support_score
    ):
        findings.append(
            CitationVerificationFinding(
                code="RAG_ANSWER_LOW_TEXT_OVERLAP",
                category=CitationFindingCategory.ANSWER_SUPPORT,
                severity=CitationFindingSeverity.WARNING,
                message=(
                    "The answer has low lexical overlap with its cited chunks. "
                    "This is a heuristic warning, not a semantic judgment."
                ),
                evidence=f"{answer_support_score:.4f}",
            )
        )

    missing_citation_count = sum(
        1
        for finding in findings
        if finding.category is CitationFindingCategory.CITATION_REFERENCE
        and finding.severity is CitationFindingSeverity.BLOCKING
    )
    is_valid = not any(
        finding.severity is CitationFindingSeverity.BLOCKING
        for finding in findings
    )
    report = CitationVerificationReport(
        answer_status=rag_answer.status,
        is_valid=is_valid,
        retrieved_chunk_count=len(chunks),
        checked_citation_count=len(rag_answer.citations),
        cited_chunk_count=len(cited_chunks),
        missing_citation_count=missing_citation_count,
        answer_support_score=answer_support_score,
        findings=findings,
    )
    return report.model_copy(
        update={"debug_lines": format_citation_verification_report_for_debug(report)}
    )


def format_citation_verification_report_for_debug(
    report: CitationVerificationReport,
) -> list[str]:
    lines = [
        (
            f"valid={report.is_valid} status={report.answer_status.value} "
            f"retrieved={report.retrieved_chunk_count} "
            f"citations={report.checked_citation_count} "
            f"cited_chunks={report.cited_chunk_count} "
            f"missing_citations={report.missing_citation_count} "
            f"answer_support={report.answer_support_score:.4f} "
            f"findings={len(report.findings)}"
        )
    ]
    for finding in report.findings:
        citation_index = finding.citation_index or "-"
        source_index = finding.source_index or "-"
        chunk_id = finding.chunk_id or "unknown-chunk"
        source = finding.source or "unknown-source"
        evidence = f" evidence={finding.evidence}" if finding.evidence else ""
        lines.append(
            (
                f"{finding.severity.value} {finding.category.value} "
                f"code={finding.code} citation={citation_index} "
                f"source_index={source_index} source={source} "
                f"chunk_id={chunk_id}{evidence}"
            )
        )
    return lines


def _inspect_no_context_answer(
    rag_answer: RagAnswer,
    chunks: Sequence[RetrievedChunk],
) -> list[CitationVerificationFinding]:
    findings: list[CitationVerificationFinding] = []
    if rag_answer.citations:
        findings.append(
            CitationVerificationFinding(
                code="RAG_NO_CONTEXT_HAS_CITATIONS",
                category=CitationFindingCategory.RESPONSE_STATE,
                severity=CitationFindingSeverity.BLOCKING,
                message="A no-context answer must not return source citations.",
            )
        )
    if chunks:
        findings.append(
            CitationVerificationFinding(
                code="RAG_NO_CONTEXT_WITH_RETRIEVED_CHUNKS",
                category=CitationFindingCategory.RESPONSE_STATE,
                severity=CitationFindingSeverity.WARNING,
                message=(
                    "A no-context answer was verified with retrieved chunks. "
                    "Check whether filtering removed them before generation."
                ),
            )
        )
    return findings


def _inspect_answered_state(
    rag_answer: RagAnswer,
    chunks: Sequence[RetrievedChunk],
    policy: CitationVerificationPolicy,
) -> list[CitationVerificationFinding]:
    findings: list[CitationVerificationFinding] = []
    if not chunks:
        findings.append(
            CitationVerificationFinding(
                code="RAG_ANSWERED_WITHOUT_RETRIEVED_CHUNKS",
                category=CitationFindingCategory.RESPONSE_STATE,
                severity=CitationFindingSeverity.BLOCKING,
                message="An answered RAG response must be verified against retrieved chunks.",
            )
        )
    if policy.require_citations_for_answered and not rag_answer.citations:
        findings.append(
            CitationVerificationFinding(
                code="RAG_ANSWERED_WITHOUT_CITATIONS",
                category=CitationFindingCategory.RESPONSE_STATE,
                severity=CitationFindingSeverity.BLOCKING,
                message="An answered RAG response must include at least one citation.",
            )
        )
    return findings


def _inspect_citations(
    citations: Sequence[RagCitation],
    chunks: Sequence[RetrievedChunk],
    *,
    answer: str,
    findings: list[CitationVerificationFinding],
    policy: CitationVerificationPolicy,
) -> dict[str, RetrievedChunk]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    citation_counts = Counter(citation.chunk_id for citation in citations)
    cited_chunks: dict[str, RetrievedChunk] = {}

    for citation_index, citation in enumerate(citations, start=1):
        before_count = len(findings)
        indexed_chunk = _chunk_at_source_index(citation, chunks)
        chunk_by_id = chunks_by_id.get(citation.chunk_id)

        if indexed_chunk is None:
            findings.append(
                _citation_finding(
                    citation,
                    code="RAG_CITATION_SOURCE_INDEX_OUT_OF_RANGE",
                    category=CitationFindingCategory.CITATION_REFERENCE,
                    severity=CitationFindingSeverity.BLOCKING,
                    message="Citation source_index does not point to a retrieved chunk.",
                    citation_index=citation_index,
                )
            )

        if chunk_by_id is None:
            findings.append(
                _citation_finding(
                    citation,
                    code="RAG_CITATION_CHUNK_NOT_RETRIEVED",
                    category=CitationFindingCategory.CITATION_REFERENCE,
                    severity=CitationFindingSeverity.BLOCKING,
                    message="Citation chunk_id was not found in retrieved chunks.",
                    citation_index=citation_index,
                )
            )

        if indexed_chunk is not None and indexed_chunk.chunk_id != citation.chunk_id:
            findings.append(
                _citation_finding(
                    citation,
                    code="RAG_CITATION_SOURCE_INDEX_MISMATCH",
                    category=CitationFindingCategory.CITATION_REFERENCE,
                    severity=CitationFindingSeverity.BLOCKING,
                    message="Citation source_index points to a different chunk_id.",
                    citation_index=citation_index,
                    evidence=f"indexed_chunk_id={indexed_chunk.chunk_id}",
                )
            )

        target_chunk = chunk_by_id
        if target_chunk is not None:
            findings.extend(
                _inspect_citation_metadata(
                    citation,
                    target_chunk,
                    citation_index=citation_index,
                )
            )
            support_score = _text_overlap_score(answer, target_chunk.content)
            if support_score < policy.min_citation_support_score:
                findings.append(
                    _citation_finding(
                        citation,
                        code="RAG_CITATION_LOW_TEXT_OVERLAP",
                        category=CitationFindingCategory.ANSWER_SUPPORT,
                        severity=CitationFindingSeverity.WARNING,
                        message=(
                            "Citation metadata is traceable, but its text has low "
                            "overlap with the cited chunk."
                        ),
                        citation_index=citation_index,
                        evidence=f"{support_score:.4f}",
                    )
                )

        if policy.warn_on_duplicate_citations and citation_counts[citation.chunk_id] > 1:
            findings.append(
                _citation_finding(
                    citation,
                    code="RAG_CITATION_DUPLICATE_CHUNK",
                    category=CitationFindingCategory.CITATION_REFERENCE,
                    severity=CitationFindingSeverity.WARNING,
                    message="The same chunk_id is cited more than once.",
                    citation_index=citation_index,
                )
            )

        citation_has_blocking_finding = any(
            finding.severity is CitationFindingSeverity.BLOCKING
            for finding in findings[before_count:]
        )
        if target_chunk is not None and not citation_has_blocking_finding:
            cited_chunks[target_chunk.chunk_id] = target_chunk

    return cited_chunks


def _inspect_citation_metadata(
    citation: RagCitation,
    chunk: RetrievedChunk,
    *,
    citation_index: int,
) -> list[CitationVerificationFinding]:
    findings: list[CitationVerificationFinding] = []
    expected_source = _metadata_text(chunk, "source") or "unknown-source"
    if citation.source.strip() != expected_source:
        findings.append(
            _citation_finding(
                citation,
                code="RAG_CITATION_SOURCE_MISMATCH",
                category=CitationFindingCategory.CITATION_METADATA,
                severity=CitationFindingSeverity.BLOCKING,
                message="Citation source does not match retrieved chunk metadata.",
                citation_index=citation_index,
                evidence=f"expected={expected_source}",
            )
        )

    expected_title = _metadata_text(chunk, "title")
    if _optional_text(citation.title) != expected_title:
        findings.append(
            _citation_finding(
                citation,
                code="RAG_CITATION_TITLE_MISMATCH",
                category=CitationFindingCategory.CITATION_METADATA,
                severity=CitationFindingSeverity.WARNING,
                message="Citation title does not match retrieved chunk metadata.",
                citation_index=citation_index,
                evidence=f"expected={expected_title or '-'}",
            )
        )

    expected_section = _metadata_text(chunk, "section")
    if _optional_text(citation.section) != expected_section:
        findings.append(
            _citation_finding(
                citation,
                code="RAG_CITATION_SECTION_MISMATCH",
                category=CitationFindingCategory.CITATION_METADATA,
                severity=CitationFindingSeverity.WARNING,
                message="Citation section does not match retrieved chunk metadata.",
                citation_index=citation_index,
                evidence=f"expected={expected_section or '-'}",
            )
        )
    return findings


def _chunk_at_source_index(
    citation: RagCitation,
    chunks: Sequence[RetrievedChunk],
) -> RetrievedChunk | None:
    index = citation.source_index - 1
    if index < 0 or index >= len(chunks):
        return None
    return chunks[index]


def _citation_finding(
    citation: RagCitation,
    *,
    code: str,
    category: CitationFindingCategory,
    severity: CitationFindingSeverity,
    message: str,
    citation_index: int,
    evidence: str | None = None,
) -> CitationVerificationFinding:
    return CitationVerificationFinding(
        code=code,
        category=category,
        severity=severity,
        message=message,
        citation_index=citation_index,
        source_index=citation.source_index,
        chunk_id=citation.chunk_id,
        source=citation.source,
        evidence=evidence,
    )


def _metadata_text(chunk: RetrievedChunk, key: str) -> str | None:
    value = chunk.metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _answer_support_score(
    answer: str,
    cited_chunks: Sequence[RetrievedChunk],
) -> float:
    if not cited_chunks:
        return 0.0
    evidence = "\n".join(chunk.content for chunk in cited_chunks)
    return _text_overlap_score(answer, evidence)


def _text_overlap_score(left: str, right: str) -> float:
    left_terms = _extract_evidence_terms(left)
    if not left_terms:
        return 0.0
    right_terms = _extract_evidence_terms(right)
    if not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def _extract_evidence_terms(text: str) -> set[str]:
    normalized = text.lower()
    terms = {
        match.group(0)
        for match in ASCII_WORD_RE.finditer(normalized)
        if len(match.group(0)) >= 2
        and match.group(0) not in ENGLISH_STOPWORDS
    }
    cjk_chars = CJK_CHAR_RE.findall(normalized)
    terms.update(
        "".join(pair)
        for pair in zip(cjk_chars, cjk_chars[1:], strict=False)
    )
    return terms
