from typing import Literal

from pydantic import BaseModel, Field


KnowledgeBaseEmbeddingMode = Literal["fake", "real"]


class KnowledgeBaseDocumentStatus(BaseModel):
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    file_extension: str = Field(min_length=1)
    doc_type: str | None = None
    business_domain: str | None = None
    permission_group: str | None = None


class KnowledgeBaseStatusResponse(BaseModel):
    documents: list[KnowledgeBaseDocumentStatus]
    document_count: int = Field(ge=0)
    collection_name: str = Field(min_length=1)
    qdrant_base_url: str = Field(min_length=1)
    fake_embedding_dimension: int = Field(gt=0)
    real_embedding_configured: bool
    trace_id: str = Field(min_length=1)


class KnowledgeBaseIngestRequest(BaseModel):
    embedding_mode: KnowledgeBaseEmbeddingMode = Field(default="fake")
    refresh: bool = Field(default=True)
    wait: bool = Field(default=True)
    include_readme: bool = Field(default=False)
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)


class KnowledgeBaseIngestResponse(BaseModel):
    embedding_mode: KnowledgeBaseEmbeddingMode
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    vector_dimension: int = Field(gt=0)
    collection_name: str = Field(min_length=1)
    replaced_source_count: int = Field(ge=0)
    trace_id: str = Field(min_length=1)


class KnowledgeBaseCollectionStatus(BaseModel):
    collection_name: str = Field(min_length=1)
    knowledge_base_ids: list[str] = Field(default_factory=list)
    display_name: str = ""
    point_count: int = Field(default=0, ge=0)
    exists: bool = False
    is_legacy: bool = False


class KnowledgeBaseCollectionsResponse(BaseModel):
    collections: list[KnowledgeBaseCollectionStatus] = Field(default_factory=list)
    legacy_collections: list[KnowledgeBaseCollectionStatus] = Field(default_factory=list)
    trace_id: str = Field(min_length=1)


class KnowledgeBaseDocumentCreateRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    business_domain: str = Field(default="general", max_length=64)
    permission_group: str = Field(default="public", max_length=64)
    doc_type: str = Field(default="policy", max_length=32)
    collection_name: str = Field(min_length=1, max_length=128)
    embedding_mode: KnowledgeBaseEmbeddingMode = Field(default="fake")
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)


class KnowledgeBaseDocumentUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    content: str | None = None
    business_domain: str | None = Field(default=None, max_length=64)
    permission_group: str | None = Field(default=None, max_length=64)
    doc_type: str | None = Field(default=None, max_length=32)
    embedding_mode: KnowledgeBaseEmbeddingMode = Field(default="fake")
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)


class KnowledgeBaseDocumentIngestRequest(BaseModel):
    embedding_mode: KnowledgeBaseEmbeddingMode = Field(default="fake")
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)


class KnowledgeBaseDocumentView(BaseModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    business_domain: str
    permission_group: str
    doc_type: str
    collection_name: str
    chunk_count: int = Field(default=0, ge=0)
    source_file_name: str
    exists_local: bool = False
    status: str = "enabled"
    updated_at: str | None = None


class KnowledgeBaseDocumentListView(BaseModel):
    documents: list[KnowledgeBaseDocumentView] = Field(default_factory=list)
    document_count: int = Field(ge=0)
    trace_id: str = Field(min_length=1)
