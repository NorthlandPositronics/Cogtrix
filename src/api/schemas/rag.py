"""RAG (Retrieval-Augmented Generation) document management schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.api.schemas.common import ensure_utc

IngestStatus = Literal["pending", "processing", "indexed", "failed"]


class DocumentOut(BaseModel):
    """An ingested document in the RAG index."""

    id: str = Field(
        ...,
        description="UUID v4 document identifier.",
        examples=["9a3c1b2e-5d4f-11ee-be56-0242ac120002"],
    )
    filename: str = Field(
        ...,
        description="Original uploaded filename.",
        examples=["climate_policy.pdf"],
    )
    content_type: str = Field(
        ...,
        description="MIME type of the uploaded file.",
        examples=["application/pdf"],
    )
    size_bytes: int = Field(
        ...,
        description="File size in bytes.",
        examples=[204800],
    )
    chunk_count: int = Field(
        ...,
        description="Number of text chunks produced during ingestion.",
        examples=[47],
    )
    status: IngestStatus = Field(
        ...,
        description="Ingestion status.",
        examples=["indexed"],
    )
    error: str | None = Field(
        default=None,
        description="Error description when status is 'failed'; null otherwise.",
    )
    ingested_at: datetime = Field(
        ...,
        description="UTC timestamp when ingestion completed.",
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the upload was received.",
    )

    _ensure_utc = field_validator("ingested_at", "created_at", mode="before")(ensure_utc)


class RAGSearchRequest(BaseModel):
    """Request body for POST /api/v1/rag/search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Semantic search query.",
        examples=["What are the main climate mitigation strategies?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of chunks to return.",
        examples=[5],
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict search to specific document IDs; null searches all documents.",
    )


class RAGChunkOut(BaseModel):
    """A retrieved document chunk from a RAG search."""

    document_id: str = Field(..., description="UUID v4 of the source document.")
    document_name: str = Field(..., description="Original filename of the source document.")
    chunk_index: int = Field(..., description="Zero-based chunk position within the document.")
    text: str = Field(..., description="Chunk text content.")
    score: float = Field(
        ...,
        description="Cosine similarity score (0.0–1.0).",
        examples=[0.92],
    )


class RAGSearchResponse(BaseModel):
    """Results of a RAG semantic search."""

    query: str = Field(..., description="The submitted query.")
    chunks: list[RAGChunkOut] = Field(..., description="Retrieved chunks ranked by relevance.")
    total_documents_searched: int = Field(
        ...,
        description="Number of documents searched.",
        examples=[12],
    )
