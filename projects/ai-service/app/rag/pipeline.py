"""Enhanced RAG answer pipeline: optional advanced modules wired together.

Follows the feature switches on Settings (RAG_ENABLE_*). When all switches are
off (default), the pipeline degrades to the basic path (retrieve_top_k + rerank
+ generate) so behavior matches the pre-phase-2 ProductionPolicyRagService.
"""

from __future__ import annotations

import logging

from app.rag.context_compression import compress_retrieved_context
from app.rag.citation_verification import verify_rag_answer_sources
from app.rag.documents import RetrievedChunk
from app.rag.generator import RagAnswer, RagAnswerStatus, create_rag_answer_service
from app.rag.knowledge_routing import create_knowledge_router
from app.rag.multi_query import create_multi_query_generator, retrieval_queries_from_expansion
from app.rag.query_rewrite import create_query_rewriter
from app.rag.rerank import (
    make_rerank_candidates_from_retrieved_chunks,
    rerank_with_fallback,
    reranked_chunks_to_retrieved_chunks,
)


logger = logging.getLogger(__name__)


def _embedding_model(settings):
    from app.rag.embeddings import OpenAICompatibleEmbeddingModel

    return OpenAICompatibleEmbeddingModel.from_settings(settings)


def _vector_store(settings, *, collection_name: str | None = None):
    from app.rag.vector_store import QdrantVectorStore

    if collection_name:
        return QdrantVectorStore.from_settings(settings, collection_name=collection_name)
    return QdrantVectorStore.from_settings(settings)


def _rerank(query, chunks: list[RetrievedChunk], settings) -> list[RetrievedChunk]:
    from app.rag.rerank import HttpReranker

    reranker = HttpReranker.from_settings(settings)
    result = rerank_with_fallback(
        query,
        make_rerank_candidates_from_retrieved_chunks(chunks),
        primary_reranker=reranker,
        top_k=settings.rerank_top_n,
    )
    return reranked_chunks_to_retrieved_chunks(result.results)


def _retrieve(query, *, settings, collection_name: str | None = None) -> list[RetrievedChunk]:
    from app.rag.retriever import retrieve_top_k

    return retrieve_top_k(
        query,
        embedding_model=_embedding_model(settings),
        vector_store=_vector_store(settings, collection_name=collection_name),
        top_k=settings.rerank_candidate_count,
    )


def _hybrid_retrieve(query, *, settings, collection_name: str | None = None) -> list[RetrievedChunk]:
    from app.rag.hybrid import SimpleKeywordRetriever, hybrid_retrieve

    store = _vector_store(settings, collection_name=collection_name)
    keyword_retriever = SimpleKeywordRetriever.from_vector_store(store)
    results = hybrid_retrieve(
        query,
        embedding_model=_embedding_model(settings),
        vector_store=store,
        keyword_retriever=keyword_retriever,
        vector_top_k=settings.rerank_candidate_count,
        final_top_k=settings.rerank_candidate_count,
        vector_weight=settings.rag_hybrid_vector_weight,
        keyword_weight=settings.rag_hybrid_keyword_weight,
    )
    # HybridSearchResult -> RetrievedChunk
    return [
        RetrievedChunk(
            point_id=r.chunk_id,
            chunk_id=r.chunk_id,
            content=r.content,
            metadata=r.metadata,
            score=r.hybrid_score,
        )
        for r in results
    ]


def enhanced_rag_answer(
    query: str,
    *,
    settings,
    access_scope=None,
) -> RagAnswer:
    """Run the RAG pipeline with optional advanced modules (feature switches)."""
    from app.core.exceptions import AppException

    try:
        return _enhanced_rag_answer_inner(query, settings=settings, access_scope=access_scope)
    except ValueError as exc:
        message = str(exc)
        if "embedding" in message.lower() or "api key" in message.lower():
            raise AppException(
                code="RAG_EMBEDDING_CONFIG_MISSING",
                message="RAG embedding configuration is incomplete.",
                status_code=500,
            ) from exc
        if "rerank" in message.lower():
            raise AppException(
                code="RAG_RERANK_CONFIG_MISSING",
                message="RAG rerank configuration is incomplete.",
                status_code=500,
            ) from exc
        raise


def _enhanced_rag_answer_inner(query, *, settings, access_scope=None) -> RagAnswer:
    working_query = query.strip()

    # 1. rewrite
    if getattr(settings, "rag_enable_rewrite", False):
        rewrite_result = create_query_rewriter(settings).rewrite(working_query)
        if rewrite_result.rewritten_query:
            working_query = rewrite_result.rewritten_query

    # 2. route (optional collection selection)
    collection_name: str | None = None
    if getattr(settings, "rag_enable_routing", False):
        decision = create_knowledge_router(settings).route(working_query, access_scope=access_scope)
        if decision.should_use_rag and decision.routes:
            collection_name = decision.routes[0].collection_name
        elif decision.should_use_rag and not decision.routes:
            blocked_warnings = {
                "RAG_ROUTE_NO_ACCESSIBLE_KNOWLEDGE_BASE",
                "RAG_ROUTE_CANDIDATE_FILTERED_BY_ACCESS_SCOPE",
            }
            if any(w in blocked_warnings for w in (decision.warnings or [])):
                logger.warning(
                    "rag_permission_blocked query=%r user=%r tenant=%r",
                    working_query[:120],
                    getattr(access_scope, "user_id", None),
                    getattr(access_scope, "tenant_id", None),
                )
                return RagAnswer(
                    answer="该问题涉及的内容需要相应权限才能查看。",
                    status=RagAnswerStatus.PERMISSION_DENIED,
                    permission_denied=True,
                )
            # 非权限原因无匹配：保持现行为（回退默认 collection）

    # 3. multi_query
    retrieval_queries = [working_query]
    if getattr(settings, "rag_enable_multi_query", False):
        expansion = create_multi_query_generator(settings).generate(working_query)
        expanded = retrieval_queries_from_expansion(expansion)
        if expanded:
            retrieval_queries = expanded

    # 4. retrieve (hybrid or vector) per query, dedupe by chunk_id
    seen: set[str] = set()
    merged: list[RetrievedChunk] = []
    for rq in retrieval_queries:
        if getattr(settings, "rag_enable_hybrid", False):
            chunks = _hybrid_retrieve(rq, settings=settings, collection_name=collection_name)
        else:
            chunks = _retrieve(rq, settings=settings, collection_name=collection_name)
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append(chunk)

    # 5. rerank
    reranked = _rerank(working_query, merged, settings) if merged else []

    # 6. context compression
    if getattr(settings, "rag_enable_context_compression", False):
        compression = compress_retrieved_context(working_query, reranked)
        reranked = list(compression.compressed_chunks)

    # 7. generate
    answer = create_rag_answer_service(settings).generate_answer_with_citations(
        query,
        chunks=reranked,
    )

    # 8. citation verification (log-only enhancement)
    if getattr(settings, "rag_enable_citation_verify", False) and answer.citations:
        report = verify_rag_answer_sources(answer, reranked)
        logger.info(
            "citation_verify run_id=%s verified=%s issues=%s",
            getattr(answer, "trace_id", "-"),
            report.all_checks_passed,
            len(report.issues or []),
        )

    return answer
