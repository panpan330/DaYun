import logging

from fastapi import APIRouter, Depends

from app.core.ai_security_boundary import require_prompt_injection_safe
from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.core.trace import get_trace_id
from app.rag.embeddings import EmbeddingModel, OpenAICompatibleEmbeddingModel
from app.rag.generator import RagAnswerService, create_rag_answer_service
from app.rag.rerank import (
    HttpReranker,
    Reranker,
    make_rerank_candidates_from_retrieved_chunks,
    rerank_with_fallback,
    reranked_chunks_to_retrieved_chunks,
)
from app.rag.retriever import VectorStoreReader, retrieve_top_k
from app.rag.vector_store import QdrantVectorStore
from app.schemas.rag import RagAskRequest, RagAskResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai/rag", tags=["rag"])


def get_rag_embedding_model(
    settings: Settings = Depends(get_settings),
) -> EmbeddingModel:
    try:
        return OpenAICompatibleEmbeddingModel.from_settings(settings)
    except ValueError as exc:
        raise AppException(
            code="RAG_EMBEDDING_CONFIG_MISSING",
            message="真实 embedding 配置不完整，无法执行 RAG 问答。",
            status_code=500,
        ) from exc


def get_rag_vector_store(
    settings: Settings = Depends(get_settings),
) -> VectorStoreReader:
    return QdrantVectorStore.from_settings(settings)


def get_rag_reranker(
    settings: Settings = Depends(get_settings),
) -> Reranker:
    try:
        return HttpReranker.from_settings(settings)
    except ValueError as exc:
        raise AppException(
            code="RAG_RERANK_CONFIG_MISSING",
            message="真实 rerank 配置不完整，无法执行 RAG 问答。",
            status_code=500,
        ) from exc


def get_rag_answer_service(
    settings: Settings = Depends(get_settings),
) -> RagAnswerService:
    return create_rag_answer_service(settings)


@router.post("/ask", response_model=RagAskResponse)
def ask_rag(
    request: RagAskRequest,
    settings: Settings = Depends(get_settings),
    embedding_model: EmbeddingModel = Depends(get_rag_embedding_model),
    vector_store: VectorStoreReader = Depends(get_rag_vector_store),
    reranker: Reranker = Depends(get_rag_reranker),
    answer_service: RagAnswerService = Depends(get_rag_answer_service),
) -> RagAskResponse:
    require_prompt_injection_safe(request.query, source="user")
    candidate_count = request.candidate_count or settings.rerank_candidate_count
    top_n = request.top_n or settings.rerank_top_n

    logger.info(
        "rag_ask_requested query_length=%s candidate_count=%s top_n=%s",
        len(request.query),
        candidate_count,
        top_n,
    )

    retrieved_chunks = retrieve_top_k(
        request.query,
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=candidate_count,
        permission_group=request.permission_group,
        business_domain=request.business_domain,
        doc_type=request.doc_type,
        source=request.source,
        score_threshold=request.score_threshold,
    )
    candidates = make_rerank_candidates_from_retrieved_chunks(retrieved_chunks)
    rerank_result = rerank_with_fallback(
        request.query,
        candidates,
        primary_reranker=reranker,
        fallback_reranker=None if request.allow_rerank_fallback else _NoFallbackReranker(),
        top_k=top_n,
    )
    answer_chunks = reranked_chunks_to_retrieved_chunks(rerank_result.results)
    answer = answer_service.generate_answer_with_citations(
        request.query,
        chunks=answer_chunks,
    )

    return RagAskResponse(
        answer=answer.answer,
        status=answer.status,
        citations=answer.citations,
        no_context_reason=answer.no_context_reason,
        suggestions=answer.suggestions,
        retrieved_count=len(retrieved_chunks),
        reranked_count=len(rerank_result.results),
        used_rerank_fallback=rerank_result.used_fallback,
        rerank_elapsed_ms=rerank_result.elapsed_ms,
        collection_name=settings.qdrant_collection_name,
        embedding_model=settings.embedding_model,
        rerank_model=settings.rerank_model,
        llm_model=settings.llm_model,
        trace_id=get_trace_id(),
    )


class _NoFallbackReranker:
    def rerank(self, query, candidates, *, top_k, retrieval_score_meaning=None):
        raise AppException(
            code="RAG_RERANK_FAILED",
            message="真实 rerank 调用失败，RAG 验收未通过。",
            status_code=502,
        )
