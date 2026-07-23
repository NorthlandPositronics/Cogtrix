"""Tests for cogtrix_core/api/schemas/rag.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cogtrix_core.api.schemas.rag import (
    DocumentOut,
    RAGChunkOut,
    RAGSearchRequest,
    RAGSearchResponse,
)


class TestDocumentOut:
    """DocumentOut schema construction and validation."""

    def test_document_out_valid(self) -> None:
        """Valid input constructs without error."""
        now = datetime.now(UTC)
        doc = DocumentOut(
            id="doc-123",
            filename="report.pdf",
            content_type="application/pdf",
            size_bytes=204800,
            chunk_count=47,
            status="indexed",
            ingested_at=now,
            created_at=now,
        )
        assert doc.filename == "report.pdf"
        assert doc.chunk_count == 47
        assert doc.status == "indexed"
        assert doc.error is None

    def test_document_out_failed_status(self) -> None:
        """Failed status with error message."""
        now = datetime.now(UTC)
        doc = DocumentOut(
            id="doc-123",
            filename="bad.pdf",
            content_type="application/pdf",
            size_bytes=0,
            chunk_count=0,
            status="failed",
            error="Corrupted PDF",
            ingested_at=now,
            created_at=now,
        )
        assert doc.status == "failed"
        assert doc.error == "Corrupted PDF"

    def test_document_out_naive_datetime(self) -> None:
        """Naive datetime gets UTC tzinfo attached."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        doc = DocumentOut(
            id="doc-123",
            filename="report.pdf",
            content_type="application/pdf",
            size_bytes=100,
            chunk_count=1,
            status="pending",
            ingested_at=naive,
            created_at=naive,
        )
        assert doc.ingested_at.tzinfo is not None
        assert doc.created_at.tzinfo is not None

    def test_document_out_invalid_status(self) -> None:
        """Invalid status raises ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            DocumentOut(
                id="doc-123",
                filename="report.pdf",
                content_type="application/pdf",
                size_bytes=100,
                chunk_count=1,
                status="invalid",
                ingested_at=now,
                created_at=now,
            )

    def test_document_out_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            DocumentOut(
                id="doc-123",
                filename="report.pdf",
                content_type="application/pdf",
                size_bytes=100,
                chunk_count=1,
                # status missing
                ingested_at=now,
                created_at=now,
            )


class TestRAGSearchRequest:
    """RAGSearchRequest schema construction and validation."""

    def test_rag_search_request_valid(self) -> None:
        """Valid input constructs without error."""
        req = RAGSearchRequest(query="climate policy", top_k=5)
        assert req.query == "climate policy"
        assert req.top_k == 5
        assert req.document_ids is None

    def test_rag_search_request_with_document_ids(self) -> None:
        """Document IDs restriction works."""
        req = RAGSearchRequest(query="climate policy", top_k=5, document_ids=["doc-1", "doc-2"])
        assert req.document_ids == ["doc-1", "doc-2"]

    def test_rag_search_request_empty_query(self) -> None:
        """Empty query raises ValidationError."""
        with pytest.raises(ValidationError):
            RAGSearchRequest(query="", top_k=5)

    def test_rag_search_request_query_too_long(self) -> None:
        """Query over 2048 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            RAGSearchRequest(query="x" * 2049, top_k=5)

    def test_rag_search_request_top_k_too_low(self) -> None:
        """top_k below 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            RAGSearchRequest(query="test", top_k=0)

    def test_rag_search_request_top_k_too_high(self) -> None:
        """top_k above 50 raises ValidationError."""
        with pytest.raises(ValidationError):
            RAGSearchRequest(query="test", top_k=51)

    def test_rag_search_request_defaults(self) -> None:
        """Defaults are correct."""
        req = RAGSearchRequest(query="test")
        assert req.top_k == 5
        assert req.document_ids is None


class TestRAGChunkOut:
    """RAGChunkOut schema construction and validation."""

    def test_rag_chunk_out_valid(self) -> None:
        """Valid input constructs without error."""
        chunk = RAGChunkOut(
            document_id="doc-123",
            document_name="report.pdf",
            chunk_index=3,
            text="This is chunk 3.",
            score=0.92,
        )
        assert chunk.chunk_index == 3
        assert chunk.score == 0.92

    def test_rag_chunk_out_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            RAGChunkOut(
                document_id="doc-123",
                document_name="report.pdf",
                chunk_index=3,
                text="This is chunk 3.",
                # score missing
            )


class TestRAGSearchResponse:
    """RAGSearchResponse schema construction and validation."""

    def test_rag_search_response_valid(self) -> None:
        """Valid input with nested chunks constructs without error."""
        response = RAGSearchResponse(
            query="climate policy",
            chunks=[
                RAGChunkOut(
                    document_id="doc-1",
                    document_name="a.pdf",
                    chunk_index=0,
                    text="Chunk 0",
                    score=0.95,
                ),
                RAGChunkOut(
                    document_id="doc-2",
                    document_name="b.pdf",
                    chunk_index=1,
                    text="Chunk 1",
                    score=0.87,
                ),
            ],
            total_documents_searched=12,
        )
        assert len(response.chunks) == 2
        assert response.total_documents_searched == 12

    def test_rag_search_response_empty_chunks(self) -> None:
        """Empty chunks list is valid."""
        response = RAGSearchResponse(
            query="no match",
            chunks=[],
            total_documents_searched=5,
        )
        assert response.chunks == []

    def test_rag_search_response_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            RAGSearchResponse(
                query="test",
                chunks=[],
                # total_documents_searched missing
            )
