from typing import TypeAlias

from pydantic import BaseModel, Field


MetadataValue: TypeAlias = str | int | float | bool | list[str]
Metadata: TypeAlias = dict[str, MetadataValue]


class RagDocument(BaseModel):
    content: str = Field(
        min_length=1,
        description="Cleaned source document text before chunking.",
    )
    metadata: Metadata = Field(
        default_factory=dict,
        description="Document-level metadata such as source, title, doc_type, and permission_group.",
    )


class RagChunk(BaseModel):
    chunk_id: str = Field(
        min_length=1,
        description="Stable project chunk identifier stored in vector-store payloads.",
    )
    content: str = Field(
        min_length=1,
        description="Chunk text that will be embedded and later passed to the model as context.",
    )
    metadata: Metadata = Field(
        default_factory=dict,
        description="Chunk-level metadata that will become Qdrant payload fields.",
    )


class RetrievedChunk(BaseModel):
    point_id: str = Field(
        min_length=1,
        description="Vector-store point id returned by retrieval.",
    )
    chunk_id: str = Field(
        min_length=1,
        description="Project chunk identifier stored in the point payload.",
    )
    content: str = Field(
        min_length=1,
        description="Retrieved chunk text that can later be passed to the model.",
    )
    metadata: Metadata = Field(
        default_factory=dict,
        description="Retrieved chunk metadata copied from the vector-store payload.",
    )
    score: float = Field(
        description="Raw retrieval score or distance returned by the vector store.",
    )
