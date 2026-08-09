from collections import Counter
from collections.abc import Iterable, Sequence
from time import perf_counter
from typing import Any
from urllib.parse import urlparse
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings
from app.rag.documents import Metadata, RetrievedChunk
from app.rag.hybrid import HybridSearchResult, KeywordSearchResult, extract_keyword_terms
from app.rag.score_interpretation import RetrievalScoreMeaning


DEFAULT_RERANK_TOP_K = 3
CONTENT_MATCH_WEIGHT = 0.55
TITLE_SECTION_MATCH_WEIGHT = 0.2
RETRIEVAL_SCORE_WEIGHT = 0.15
SOURCE_AGREEMENT_WEIGHT = 0.1
RERANK_ENDPOINT_PATH = "/rerank"


class RerankModelError(RuntimeError):
    pass


class RerankCandidate(BaseModel):
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)
    retrieval_score: float | None = Field(default=None, ge=0)
    retrieval_sources: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)

    @field_validator("retrieval_score", mode="before")
    @classmethod
    def reject_bool_retrieval_score(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("retrieval_score must be a number")
        return value


class RerankScoreBreakdown(BaseModel):
    content_match_score: float = Field(ge=0, le=1)
    title_section_match_score: float = Field(ge=0, le=1)
    normalized_retrieval_score: float = Field(ge=0, le=1)
    source_agreement_score: float = Field(ge=0, le=1)


class RerankedChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)
    retrieval_score: float | None = None
    rerank_score: float = Field(ge=0, le=1)
    original_rank: int = Field(ge=1)
    rerank_rank: int = Field(ge=1)
    score_breakdown: RerankScoreBreakdown
    retrieval_sources: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)


class RerankReport(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(gt=0)
    candidate_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    top_before_chunk_id: str | None = None
    top_after_chunk_id: str | None = None
    moved_count: int = Field(ge=0)
    promoted_chunk_ids: list[str] = Field(default_factory=list)
    dropped_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_score_direction: str = Field(min_length=1)
    results: list[RerankedChunk] = Field(default_factory=list)
    debug_lines: list[str] = Field(default_factory=list)


class RerankExecutionResult(BaseModel):
    results: list[RerankedChunk] = Field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: str | None = None
    elapsed_ms: float = Field(ge=0)


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int = DEFAULT_RERANK_TOP_K,
        retrieval_score_meaning: RetrievalScoreMeaning | None = None,
    ) -> list[RerankedChunk]:
        """Reorder already-retrieved candidates for a query."""


class RuleBasedReranker:
    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int = DEFAULT_RERANK_TOP_K,
        retrieval_score_meaning: RetrievalScoreMeaning | None = None,
    ) -> list[RerankedChunk]:
        return rerank_candidates(
            query,
            candidates,
            top_k=top_k,
            retrieval_score_meaning=retrieval_score_meaning,
        )


class HttpReranker:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        api_key: str | None = None,
        max_retries: int = 0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("rerank base_url must not be blank")
        if not model.strip():
            raise ValueError("rerank model must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("rerank timeout_seconds must be greater than 0")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool):
            raise ValueError("rerank max_retries must be an integer")
        if max_retries < 0:
            raise ValueError("rerank max_retries must be greater than or equal to 0")

        self.base_url = base_url.strip().rstrip("/")
        self.endpoint_url = _resolve_rerank_endpoint_url(self.base_url)
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key.strip() if api_key and api_key.strip() else None
        self.max_retries = max_retries
        self.transport = transport

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> "HttpReranker":
        base_url = settings.resolved_rerank_base_url
        if base_url is None:
            raise ValueError("RERANK_BASE_URL is not configured")
        return cls(
            base_url=base_url,
            model=settings.rerank_model,
            timeout_seconds=settings.rerank_timeout_seconds,
            api_key=settings.resolved_rerank_api_key,
            max_retries=settings.rerank_max_retries,
            transport=transport,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int = DEFAULT_RERANK_TOP_K,
        retrieval_score_meaning: RetrievalScoreMeaning | None = None,
    ) -> list[RerankedChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")
        _validate_positive_top_k(top_k)
        if not candidates:
            return []

        model_results = self._request_rerank(
            normalized_query,
            candidates,
            top_k=top_k,
        )
        return _build_reranked_chunks_from_model_results(
            normalized_query,
            candidates,
            model_results,
            retrieval_score_meaning=retrieval_score_meaning,
        )

    def _request_rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
    ) -> list[dict[str, float | int]]:
        request_body = self._build_request_body(query, candidates, top_k=top_k)

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(
                    timeout=self.timeout_seconds,
                    headers=self._headers(),
                    transport=self.transport,
                ) as client:
                    response = client.post(self.endpoint_url, json=request_body)
                if response.status_code in {408, 429} or response.status_code >= 500:
                    if attempt < self.max_retries:
                        continue
                _raise_for_bad_rerank_response(response)
                return _extract_model_rerank_results(
                    _parse_rerank_response_json(response),
                    candidate_count=len(candidates),
                )
            except httpx.RequestError as exc:
                if attempt >= self.max_retries:
                    raise RerankModelError("rerank provider request failed") from exc
        raise RerankModelError("rerank provider request failed")

    def _headers(self) -> dict[str, str]:
        if self.api_key is None:
            return {"Content-Type": "application/json"}
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_body(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int,
    ) -> dict[str, object]:
        documents = [candidate.content for candidate in candidates]
        if _uses_nested_dashscope_rerank_body(self.endpoint_url, self.model):
            return {
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": top_k,
                },
            }

        return {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
            "return_documents": False,
        }


def make_rerank_candidates_from_retrieved_chunks(
    chunks: Sequence[RetrievedChunk],
) -> list[RerankCandidate]:
    return [
        RerankCandidate(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            metadata=chunk.metadata,
            retrieval_score=chunk.score,
            retrieval_sources=["vector"],
        )
        for chunk in chunks
    ]


def make_rerank_candidates_from_keyword_results(
    results: Sequence[KeywordSearchResult],
) -> list[RerankCandidate]:
    return [
        RerankCandidate(
            chunk_id=result.chunk_id,
            content=result.content,
            metadata=result.metadata,
            retrieval_score=result.score,
            retrieval_sources=["keyword"],
            matched_terms=result.matched_terms,
        )
        for result in results
    ]


def make_rerank_candidates_from_hybrid_results(
    results: Sequence[HybridSearchResult],
) -> list[RerankCandidate]:
    return [
        RerankCandidate(
            chunk_id=result.chunk_id,
            content=result.content,
            metadata=result.metadata,
            retrieval_score=result.hybrid_score,
            retrieval_sources=result.retrieval_sources,
            matched_terms=result.matched_terms,
        )
        for result in results
    ]


def rerank_candidates(
    query: str,
    candidates: Sequence[RerankCandidate],
    *,
    top_k: int = DEFAULT_RERANK_TOP_K,
    retrieval_score_meaning: RetrievalScoreMeaning | None = None,
) -> list[RerankedChunk]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be blank")
    _validate_positive_top_k(top_k)

    query_terms = extract_keyword_terms(normalized_query)
    if not query_terms:
        raise ValueError("query must contain searchable terms")

    max_retrieval_score = max(
        (
            candidate.retrieval_score
            for candidate in candidates
            if candidate.retrieval_score is not None
        ),
        default=0.0,
    )
    min_retrieval_score = min(
        (
            candidate.retrieval_score
            for candidate in candidates
            if candidate.retrieval_score is not None
        ),
        default=0.0,
    )

    scored_payloads: list[dict[str, object]] = []
    for original_rank, candidate in enumerate(candidates, start=1):
        breakdown = _build_score_breakdown(
            query_terms,
            candidate,
            max_retrieval_score=max_retrieval_score,
            min_retrieval_score=min_retrieval_score,
            retrieval_score_meaning=retrieval_score_meaning,
        )
        matched_terms = _merge_terms(
            candidate.matched_terms,
            _matched_terms(query_terms, extract_keyword_terms(candidate.content)),
            _matched_terms(query_terms, _title_section_terms(candidate.metadata)),
        )
        rerank_score = round(
            breakdown.content_match_score * CONTENT_MATCH_WEIGHT
            + breakdown.title_section_match_score * TITLE_SECTION_MATCH_WEIGHT
            + breakdown.normalized_retrieval_score * RETRIEVAL_SCORE_WEIGHT
            + breakdown.source_agreement_score * SOURCE_AGREEMENT_WEIGHT,
            6,
        )
        scored_payloads.append(
            {
                "chunk_id": candidate.chunk_id,
                "content": candidate.content,
                "metadata": candidate.metadata,
                "retrieval_score": candidate.retrieval_score,
                "rerank_score": rerank_score,
                "original_rank": original_rank,
                "score_breakdown": breakdown,
                "retrieval_sources": candidate.retrieval_sources,
                "matched_terms": matched_terms,
            }
        )

    sorted_payloads = sorted(
        scored_payloads,
        key=lambda payload: (
            -float(payload["rerank_score"]),
            -payload["score_breakdown"].content_match_score,
            -payload["score_breakdown"].title_section_match_score,
            int(payload["original_rank"]),
            str(payload["chunk_id"]),
        ),
    )[:top_k]

    return [
        RerankedChunk(
            **payload,
            rerank_rank=rerank_rank,
        )
        for rerank_rank, payload in enumerate(sorted_payloads, start=1)
    ]


def build_rerank_report(
    query: str,
    candidates: Sequence[RerankCandidate],
    *,
    top_k: int = DEFAULT_RERANK_TOP_K,
    retrieval_score_meaning: RetrievalScoreMeaning | None = None,
) -> RerankReport:
    results = rerank_candidates(
        query,
        candidates,
        top_k=top_k,
        retrieval_score_meaning=retrieval_score_meaning,
    )
    returned_chunk_ids = {chunk.chunk_id for chunk in results}
    promoted_chunk_ids = [
        chunk.chunk_id
        for chunk in results
        if chunk.original_rank > chunk.rerank_rank
    ]
    dropped_chunk_ids = [
        candidate.chunk_id
        for candidate in candidates
        if candidate.chunk_id not in returned_chunk_ids
    ]
    return RerankReport(
        query=query.strip(),
        top_k=top_k,
        candidate_count=len(candidates),
        returned_count=len(results),
        top_before_chunk_id=candidates[0].chunk_id if candidates else None,
        top_after_chunk_id=results[0].chunk_id if results else None,
        moved_count=sum(
            1
            for chunk in results
            if chunk.original_rank != chunk.rerank_rank
        ),
        promoted_chunk_ids=promoted_chunk_ids,
        dropped_chunk_ids=dropped_chunk_ids,
        retrieval_score_direction=(
            retrieval_score_meaning.direction
            if retrieval_score_meaning is not None
            else "higher_is_better"
        ),
        results=results,
        debug_lines=format_reranked_chunks_for_debug(results),
    )


def rerank_with_fallback(
    query: str,
    candidates: Sequence[RerankCandidate],
    *,
    primary_reranker: Reranker,
    fallback_reranker: Reranker | None = None,
    top_k: int = DEFAULT_RERANK_TOP_K,
    retrieval_score_meaning: RetrievalScoreMeaning | None = None,
) -> RerankExecutionResult:
    fallback = fallback_reranker or RuleBasedReranker()
    start_time = perf_counter()
    try:
        results = primary_reranker.rerank(
            query,
            candidates,
            top_k=top_k,
            retrieval_score_meaning=retrieval_score_meaning,
        )
        return RerankExecutionResult(
            results=results,
            used_fallback=False,
            elapsed_ms=_elapsed_ms_since(start_time),
        )
    except Exception as exc:
        results = fallback.rerank(
            query,
            candidates,
            top_k=top_k,
            retrieval_score_meaning=retrieval_score_meaning,
        )
        return RerankExecutionResult(
            results=results,
            used_fallback=True,
            fallback_reason=type(exc).__name__,
            elapsed_ms=_elapsed_ms_since(start_time),
        )


def reranked_chunks_to_retrieved_chunks(
    chunks: Sequence[RerankedChunk],
) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            point_id=str(chunk.metadata.get("point_id", chunk.chunk_id)),
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            metadata=chunk.metadata,
            score=chunk.rerank_score,
        )
        for chunk in chunks
    ]


def format_reranked_chunks_for_debug(chunks: Sequence[RerankedChunk]) -> list[str]:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown-source")
        section = chunk.metadata.get("section", "unknown-section")
        sources = ",".join(chunk.retrieval_sources) or "unknown"
        matched = ",".join(chunk.matched_terms[:6]) or "-"
        lines.append(
            (
                f"{index}. rerank_score={chunk.rerank_score:.4f} "
                f"original_rank={chunk.original_rank} "
                f"retrieval_score={_format_optional_score(chunk.retrieval_score)} "
                f"content_match={chunk.score_breakdown.content_match_score:.4f} "
                f"title_section_match={chunk.score_breakdown.title_section_match_score:.4f} "
                f"sources={sources} source={source} section={section} "
                f"chunk_id={chunk.chunk_id} matched={matched}"
            )
        )
    return lines


def _build_score_breakdown(
    query_terms: Sequence[str],
    candidate: RerankCandidate,
    *,
    max_retrieval_score: float,
    min_retrieval_score: float,
    retrieval_score_meaning: RetrievalScoreMeaning | None,
) -> RerankScoreBreakdown:
    content_score = _weighted_overlap_score(
        query_terms,
        extract_keyword_terms(candidate.content),
    )
    title_section_score = _weighted_overlap_score(
        query_terms,
        _title_section_terms(candidate.metadata),
    )
    normalized_retrieval_score = _normalize_score(
        candidate.retrieval_score,
        max_retrieval_score,
        min_retrieval_score=min_retrieval_score,
        retrieval_score_meaning=retrieval_score_meaning,
    )
    return RerankScoreBreakdown(
        content_match_score=content_score,
        title_section_match_score=title_section_score,
        normalized_retrieval_score=normalized_retrieval_score,
        source_agreement_score=_source_agreement_score(candidate.retrieval_sources),
    )


def _build_reranked_chunks_from_model_results(
    query: str,
    candidates: Sequence[RerankCandidate],
    model_results: Sequence[dict[str, float | int]],
    *,
    retrieval_score_meaning: RetrievalScoreMeaning | None,
) -> list[RerankedChunk]:
    query_terms = extract_keyword_terms(query)
    max_retrieval_score = max(
        (
            candidate.retrieval_score
            for candidate in candidates
            if candidate.retrieval_score is not None
        ),
        default=0.0,
    )
    min_retrieval_score = min(
        (
            candidate.retrieval_score
            for candidate in candidates
            if candidate.retrieval_score is not None
        ),
        default=0.0,
    )
    chunks: list[RerankedChunk] = []
    for rerank_rank, model_result in enumerate(model_results, start=1):
        candidate = candidates[int(model_result["index"])]
        breakdown = _build_score_breakdown(
            query_terms,
            candidate,
            max_retrieval_score=max_retrieval_score,
            min_retrieval_score=min_retrieval_score,
            retrieval_score_meaning=retrieval_score_meaning,
        )
        matched_terms = _merge_terms(
            candidate.matched_terms,
            _matched_terms(query_terms, extract_keyword_terms(candidate.content)),
            _matched_terms(query_terms, _title_section_terms(candidate.metadata)),
        )
        chunks.append(
            RerankedChunk(
                chunk_id=candidate.chunk_id,
                content=candidate.content,
                metadata=candidate.metadata,
                retrieval_score=candidate.retrieval_score,
                rerank_score=round(float(model_result["relevance_score"]), 6),
                original_rank=int(model_result["index"]) + 1,
                rerank_rank=rerank_rank,
                score_breakdown=breakdown,
                retrieval_sources=candidate.retrieval_sources,
                matched_terms=matched_terms,
            )
        )
    return chunks


def _title_section_terms(metadata: Metadata) -> list[str]:
    values = [
        value
        for key in ("title", "section")
        if isinstance((value := metadata.get(key)), str)
    ]
    return extract_keyword_terms("\n".join(values))


def _weighted_overlap_score(
    query_terms: Sequence[str],
    target_terms: Sequence[str],
) -> float:
    if not query_terms:
        return 0.0
    term_counts = Counter(target_terms)
    total_weight = sum(_term_weight(term) for term in query_terms)
    matched_weight = sum(
        _term_weight(term) * min(term_counts[term], 2)
        for term in query_terms
        if term_counts.get(term, 0) > 0
    )
    return round(min(matched_weight / total_weight, 1.0), 6)


def _matched_terms(
    query_terms: Sequence[str],
    target_terms: Sequence[str],
) -> list[str]:
    term_counts = Counter(target_terms)
    return [
        term
        for term in query_terms
        if term_counts.get(term, 0) > 0
    ]


def _normalize_score(
    score: float | None,
    max_score: float,
    *,
    min_retrieval_score: float,
    retrieval_score_meaning: RetrievalScoreMeaning | None,
) -> float:
    if score is None:
        return 0.0
    if (
        retrieval_score_meaning is not None
        and retrieval_score_meaning.direction == "lower_is_better"
    ):
        if max_score <= min_retrieval_score:
            return 1.0
        return round(
            1 - ((score - min_retrieval_score) / (max_score - min_retrieval_score)),
            6,
        )
    if max_score <= 0:
        return 0.0
    return round(min(score / max_score, 1.0), 6)


def _source_agreement_score(retrieval_sources: Sequence[str]) -> float:
    unique_sources = set(retrieval_sources)
    return 1.0 if len(unique_sources) >= 2 else 0.0


def _term_weight(term: str) -> int:
    return max(len(term), 1)


def _validate_positive_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _merge_terms(*term_groups: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in term_groups:
        for term in group:
            if term in seen:
                continue
            merged.append(term)
            seen.add(term)
    return merged


def _format_optional_score(score: float | None) -> str:
    if score is None:
        return "none"
    return f"{score:.4f}"


def _raise_for_bad_rerank_response(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    raise RerankModelError(
        f"rerank provider returned status {response.status_code}"
    )


def _parse_rerank_response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise RerankModelError("rerank provider returned invalid JSON") from exc


def _extract_model_rerank_results(
    data: Any,
    *,
    candidate_count: int,
) -> list[dict[str, float | int]]:
    if not isinstance(data, dict):
        raise RerankModelError("rerank response must be an object")
    results = data.get("results")
    if results is None and isinstance(data.get("output"), dict):
        results = data["output"].get("results")
    if not isinstance(results, list):
        raise RerankModelError("rerank response results must be a list")

    parsed_results: list[dict[str, float | int]] = []
    seen_indexes: set[int] = set()
    for item in results:
        if not isinstance(item, dict):
            raise RerankModelError("rerank response result must be an object")
        index = item.get("index")
        relevance_score = item.get("relevance_score")
        if not isinstance(index, int) or isinstance(index, bool):
            raise RerankModelError("rerank result index must be an integer")
        if index < 0 or index >= candidate_count or index in seen_indexes:
            raise RerankModelError("rerank result index is out of range or duplicated")
        if (
            not isinstance(relevance_score, int | float)
            or isinstance(relevance_score, bool)
        ):
            raise RerankModelError("rerank result relevance_score must be a number")
        parsed_results.append(
            {
                "index": index,
                "relevance_score": float(relevance_score),
            }
        )
        seen_indexes.add(index)
    return sorted(
        parsed_results,
        key=lambda result: (-float(result["relevance_score"]), int(result["index"])),
    )


def _elapsed_ms_since(start_time: float) -> float:
    return round((perf_counter() - start_time) * 1000, 3)


def _resolve_rerank_endpoint_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/rerank") or path.endswith("/reranks"):
        return base_url
    if "/api/v1/services/rerank/" in path:
        return base_url
    return f"{base_url}{RERANK_ENDPOINT_PATH}"


def _uses_nested_dashscope_rerank_body(endpoint_url: str, model: str) -> bool:
    parsed = urlparse(endpoint_url)
    if "/api/v1/services/rerank/" in parsed.path:
        return True
    return model.strip() in {"gte-rerank-v2", "qwen3-vl-rerank"}
