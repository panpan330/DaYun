from collections import Counter
from collections.abc import Iterable, Sequence
import re
from typing import Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from app.rag.documents import Metadata, RagChunk, RetrievedChunk
from app.rag.embeddings import EmbeddingModel
from app.rag.retriever import DEFAULT_TOP_K, VectorStoreReader, retrieve_top_k


DEFAULT_KEYWORD_TOP_K = 5
DEFAULT_HYBRID_TOP_K = 5
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3


class KeywordSearchResult(BaseModel):
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)
    score: float = Field(ge=0)
    matched_terms: list[str] = Field(default_factory=list)


class KeywordRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_KEYWORD_TOP_K,
        min_score: float = 0.0,
        permission_group: str | None = None,
        business_domain: str | None = None,
        doc_type: str | None = None,
        source: str | None = None,
    ) -> list[KeywordSearchResult]:
        """Return keyword-matched chunks for a query."""


class SimpleKeywordRetriever:
    def __init__(self, chunks: Sequence[RagChunk]) -> None:
        self.chunks = list(chunks)

    @classmethod
    def from_retrieved_chunks(cls, chunks: Sequence[RetrievedChunk]) -> "SimpleKeywordRetriever":
        """从检索结果构建关键词索引（真库 chunk 来源）。"""
        rag_chunks = [
            RagChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]
        return cls(rag_chunks)

    @classmethod
    def from_vector_store(
        cls,
        vector_store: VectorStoreReader,
    ) -> "SimpleKeywordRetriever":
        """从向量库拉取全量 chunk 构建关键词索引（供 hybrid 使用）。

        依赖 VectorStoreReader 提供 scroll_all() 拉全量能力；
        实现不支持时抛 ValueError（调用方应降级为纯向量检索）。
        """
        collector = getattr(vector_store, "scroll_all", None)
        if callable(collector):
            chunks = list(collector())
        else:
            raise ValueError(
                "vector_store must provide scroll_all() to build keyword index; "
                "fall back to vector-only retrieval when unsupported"
            )
        return cls.from_retrieved_chunks(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_KEYWORD_TOP_K,
        min_score: float = 0.0,
        permission_group: str | None = None,
        business_domain: str | None = None,
        doc_type: str | None = None,
        source: str | None = None,
    ) -> list[KeywordSearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        _validate_positive_top_k(top_k)
        _validate_min_score(min_score)

        query_terms = extract_keyword_terms(normalized_query)
        if not query_terms:
            raise ValueError("query must contain searchable terms")

        results: list[KeywordSearchResult] = []
        for chunk in self.chunks:
            if not _metadata_matches(
                chunk.metadata,
                permission_group=permission_group,
                business_domain=business_domain,
                doc_type=doc_type,
                source=source,
            ):
                continue
            result = _score_chunk_by_terms(chunk, query_terms)
            if result is not None and result.score >= min_score:
                results.append(result)

        return sorted(
            results,
            key=lambda result: (-result.score, result.chunk_id),
        )[:top_k]


class HybridSearchResult(BaseModel):
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)
    hybrid_score: float = Field(ge=0)
    vector_score: float | None = None
    keyword_score: float | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)


class HybridFusionReport(BaseModel):
    top_k: int = Field(gt=0)
    vector_weight: float = Field(ge=0)
    keyword_weight: float = Field(ge=0)
    vector_result_count: int = Field(ge=0)
    keyword_result_count: int = Field(ge=0)
    fused_result_count: int = Field(ge=0)
    vector_only_count: int = Field(ge=0)
    keyword_only_count: int = Field(ge=0)
    both_count: int = Field(ge=0)
    overlap_chunk_ids: list[str] = Field(default_factory=list)
    top_chunk_id: str | None = None
    results: list[HybridSearchResult] = Field(default_factory=list)
    debug_lines: list[str] = Field(default_factory=list)


class HybridSearchWeights(BaseModel):
    vector_weight: float = Field(default=DEFAULT_VECTOR_WEIGHT, ge=0)
    keyword_weight: float = Field(default=DEFAULT_KEYWORD_WEIGHT, ge=0)

    @field_validator("vector_weight", "keyword_weight", mode="before")
    @classmethod
    def reject_bool_weights(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("hybrid weights must be numbers")
        return value

    @model_validator(mode="after")
    def validate_at_least_one_weight(self) -> "HybridSearchWeights":
        if self.vector_weight == 0 and self.keyword_weight == 0:
            raise ValueError("at least one hybrid weight must be greater than 0")
        return self


def extract_keyword_terms(text: str) -> list[str]:
    normalized = text.strip().lower()
    if not normalized:
        return []

    terms: list[str] = []
    for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", normalized):
        if re.fullmatch(r"[a-z0-9_]+", token):
            if len(token) >= 2:
                terms.append(token)
            continue
        terms.extend(_build_cjk_ngrams(token))
    return _unique_strings(terms)


def fuse_hybrid_results(
    vector_chunks: Sequence[RetrievedChunk],
    keyword_results: Sequence[KeywordSearchResult],
    *,
    top_k: int = DEFAULT_HYBRID_TOP_K,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> list[HybridSearchResult]:
    _validate_positive_top_k(top_k)
    weights = HybridSearchWeights(
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    max_vector_score = max((chunk.score for chunk in vector_chunks), default=0.0)
    max_keyword_score = max((result.score for result in keyword_results), default=0.0)

    merged: dict[str, HybridSearchResult] = {}
    for chunk in vector_chunks:
        vector_score = _normalize_score(chunk.score, max_vector_score)
        merged[chunk.chunk_id] = HybridSearchResult(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            metadata=chunk.metadata,
            hybrid_score=round(vector_score * weights.vector_weight, 6),
            vector_score=chunk.score,
            retrieval_sources=["vector"],
        )

    for result in keyword_results:
        keyword_score = _normalize_score(result.score, max_keyword_score)
        existing = merged.get(result.chunk_id)
        if existing is None:
            merged[result.chunk_id] = HybridSearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                metadata=result.metadata,
                hybrid_score=round(keyword_score * weights.keyword_weight, 6),
                keyword_score=result.score,
                retrieval_sources=["keyword"],
                matched_terms=result.matched_terms,
            )
            continue

        existing.keyword_score = result.score
        existing.matched_terms = result.matched_terms
        existing.retrieval_sources = _unique_strings(
            [*existing.retrieval_sources, "keyword"]
        )
        existing.hybrid_score = round(
            existing.hybrid_score + keyword_score * weights.keyword_weight,
            6,
        )

    return sorted(
        merged.values(),
        key=lambda result: (
            -result.hybrid_score,
            -len(result.retrieval_sources),
            result.chunk_id,
        ),
    )[:top_k]


def build_hybrid_fusion_report(
    vector_chunks: Sequence[RetrievedChunk],
    keyword_results: Sequence[KeywordSearchResult],
    *,
    top_k: int = DEFAULT_HYBRID_TOP_K,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> HybridFusionReport:
    results = fuse_hybrid_results(
        vector_chunks,
        keyword_results,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    vector_chunk_ids = {chunk.chunk_id for chunk in vector_chunks}
    keyword_chunk_ids = {result.chunk_id for result in keyword_results}
    returned_chunk_ids = {result.chunk_id for result in results}
    overlap_chunk_ids = sorted(vector_chunk_ids & keyword_chunk_ids)

    return HybridFusionReport(
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        vector_result_count=len(vector_chunks),
        keyword_result_count=len(keyword_results),
        fused_result_count=len(results),
        vector_only_count=len(
            (vector_chunk_ids - keyword_chunk_ids) & returned_chunk_ids
        ),
        keyword_only_count=len(
            (keyword_chunk_ids - vector_chunk_ids) & returned_chunk_ids
        ),
        both_count=len((vector_chunk_ids & keyword_chunk_ids) & returned_chunk_ids),
        overlap_chunk_ids=overlap_chunk_ids,
        top_chunk_id=results[0].chunk_id if results else None,
        results=results,
        debug_lines=format_hybrid_results_for_debug(results),
    )


def format_hybrid_results_for_debug(results: Sequence[HybridSearchResult]) -> list[str]:
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        sources = ",".join(result.retrieval_sources) or "unknown"
        matched = ",".join(result.matched_terms[:6]) or "-"
        source = result.metadata.get("source", "unknown-source")
        section = result.metadata.get("section", "unknown-section")
        lines.append(
            (
                f"{index}. hybrid_score={result.hybrid_score:.4f} "
                f"vector_score={_format_optional_score(result.vector_score)} "
                f"keyword_score={_format_optional_score(result.keyword_score)} "
                f"sources={sources} source={source} section={section} "
                f"chunk_id={result.chunk_id} matched={matched}"
            )
        )
    return lines


def hybrid_retrieve(
    query: str,
    *,
    embedding_model: EmbeddingModel,
    vector_store: VectorStoreReader,
    keyword_retriever: KeywordRetriever,
    vector_top_k: int = DEFAULT_TOP_K,
    keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
    final_top_k: int = DEFAULT_HYBRID_TOP_K,
    vector_score_threshold: float | None = None,
    keyword_min_score: float = 0.0,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    permission_group: str | None = None,
    business_domain: str | None = None,
    doc_type: str | None = None,
    source: str | None = None,
) -> list[HybridSearchResult]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be blank")

    vector_chunks = retrieve_top_k(
        normalized_query,
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=vector_top_k,
        permission_group=permission_group,
        business_domain=business_domain,
        doc_type=doc_type,
        source=source,
        score_threshold=vector_score_threshold,
    )
    keyword_results = keyword_retriever.search(
        normalized_query,
        top_k=keyword_top_k,
        min_score=keyword_min_score,
        permission_group=permission_group,
        business_domain=business_domain,
        doc_type=doc_type,
        source=source,
    )
    return fuse_hybrid_results(
        vector_chunks,
        keyword_results,
        top_k=final_top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )


def _build_cjk_ngrams(token: str) -> list[str]:
    if len(token) <= 2:
        return [token]
    bigrams = [token[index : index + 2] for index in range(len(token) - 1)]
    trigrams = [token[index : index + 3] for index in range(len(token) - 2)]
    return [*bigrams, *trigrams]


def _score_chunk_by_terms(
    chunk: RagChunk,
    query_terms: Sequence[str],
) -> KeywordSearchResult | None:
    search_text = _build_keyword_search_text(chunk)
    term_counts = Counter(extract_keyword_terms(search_text))
    matched_terms = [term for term in query_terms if term_counts.get(term, 0) > 0]
    if not matched_terms:
        return None

    query_weight = sum(_term_weight(term) for term in query_terms)
    matched_weight = sum(
        _term_weight(term) * min(term_counts[term], 3)
        for term in matched_terms
    )
    score = round(min(matched_weight / query_weight, 1.0), 6)
    return KeywordSearchResult(
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        metadata=chunk.metadata,
        score=score,
        matched_terms=matched_terms,
    )


def _build_keyword_search_text(chunk: RagChunk) -> str:
    metadata_values = [
        value
        for key in ("source", "title", "section", "doc_type", "business_domain")
        if isinstance((value := chunk.metadata.get(key)), str)
    ]
    return "\n".join([chunk.content, *metadata_values])


def _metadata_matches(
    metadata: Metadata,
    *,
    permission_group: str | None = None,
    business_domain: str | None = None,
    doc_type: str | None = None,
    source: str | None = None,
) -> bool:
    for key, expected in (
        ("permission_group", permission_group),
        ("business_domain", business_domain),
        ("doc_type", doc_type),
        ("source", source),
    ):
        if expected is None:
            continue
        actual = metadata.get(key)
        if not isinstance(actual, str) or actual.strip() != expected.strip():
            return False
    return True


def _term_weight(term: str) -> int:
    return max(len(term), 1)


def _normalize_score(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return score / max_score


def _format_optional_score(score: float | None) -> str:
    if score is None:
        return "none"
    return f"{score:.4f}"


def _validate_positive_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _validate_min_score(min_score: float) -> None:
    if not isinstance(min_score, int | float) or isinstance(min_score, bool):
        raise ValueError("min_score must be a number")
    if min_score < 0:
        raise ValueError("min_score must be greater than or equal to 0")


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return unique
