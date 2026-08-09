from pydantic import BaseModel, Field

from app.rag.generator import RagAnswerStatus, RagCitation, RagNoContextReason


class RagAskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    candidate_count: int | None = Field(default=None, ge=1, le=100)
    top_n: int | None = Field(default=None, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0)
    allow_rerank_fallback: bool = Field(default=False)
    permission_group: str | None = Field(default=None, min_length=1, max_length=80)
    business_domain: str | None = Field(default=None, min_length=1, max_length=80)
    doc_type: str | None = Field(default=None, min_length=1, max_length=80)
    source: str | None = Field(default=None, min_length=1, max_length=200)


class RagAskResponse(BaseModel):
    answer: str = Field(min_length=1)
    status: RagAnswerStatus
    citations: list[RagCitation] = Field(default_factory=list)
    no_context_reason: RagNoContextReason | None = None
    suggestions: list[str] = Field(default_factory=list)
    retrieved_count: int = Field(ge=0)
    reranked_count: int = Field(ge=0)
    used_rerank_fallback: bool
    rerank_elapsed_ms: float = Field(ge=0)
    collection_name: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    rerank_model: str = Field(min_length=1)
    llm_model: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
