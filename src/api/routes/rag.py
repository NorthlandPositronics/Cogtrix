"""RAG (Retrieval-Augmented Generation) document management endpoints.

Endpoints:
    POST   /api/v1/rag/documents             — upload and ingest a document
    GET    /api/v1/rag/documents             — list ingested documents (paginated)
    GET    /api/v1/rag/documents/{id}        — get document details and status
    DELETE /api/v1/rag/documents/{id}        — delete a document from the index
    POST   /api/v1/rag/search               — semantic search over the knowledge base
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import TokenData, get_current_user, require_admin
from src.api.db.engine import get_db
from src.api.db.models import RagDocument
from src.api.pagination import decode_cursor, encode_cursor
from src.api.schemas.common import APIResponse, CursorPage
from src.api.schemas.rag import DocumentOut, RAGChunkOut, RAGSearchRequest, RAGSearchResponse
from src.api.tasks.rag import ingest_document_task

log = logging.getLogger("cogtrix.api.rag")

router = APIRouter(prefix="/rag", tags=["RAG / Documents"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".markdown", ".csv"})


def _get_uploads_dir() -> Path:
    """Resolve uploads directory using COGTRIX_DATA_DIR if set."""
    import os

    data_dir = os.environ.get("COGTRIX_DATA_DIR", "data")
    return Path(data_dir, "api", "uploads").resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _validate_doc_id(document_id: str) -> None:
    """Raise 400 if document_id is not a well-formed UUID v4 string.

    Guards against path traversal attacks where an attacker passes a
    ``document_id`` like ``../../etc`` to escape the uploads directory.
    """
    if not _UUID_RE.match(document_id.lower()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_DOCUMENT_ID", "message": "Invalid document ID format."},
        )


def _content_type_for(filename: str) -> str:
    """Guess MIME type from filename; fall back to octet-stream."""
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"


def _file_size(path: Path) -> int:
    """Return file size in bytes; 0 if file does not exist."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _doc_to_out(doc: RagDocument) -> DocumentOut:
    """Convert an ORM ``RagDocument`` to a ``DocumentOut`` schema object.

    ``content_type`` and ``size_bytes`` are derived at read time from the
    uploaded file on disk — no extra DB columns are required.
    """
    upload_dir = _get_uploads_dir() / doc.id
    # Find the uploaded file
    size_bytes = 0
    if upload_dir.exists():
        for child in upload_dir.iterdir():
            if child.is_file():
                size_bytes = _file_size(child)
                break

    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        content_type=_content_type_for(doc.filename),
        size_bytes=size_bytes,
        chunk_count=doc.chunk_count,
        status=doc.status,  # type: ignore[arg-type]
        error=doc.error,
        ingested_at=doc.indexed_at if doc.indexed_at is not None else doc.created_at,
        created_at=doc.created_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/documents",
    summary="Upload and ingest a document",
    description=(
        "Upload a file for ingestion into the RAG knowledge base. "
        "Supported formats: PDF, TXT, MD, DOCX, HTML. "
        "Ingestion (chunking + embedding) runs asynchronously; poll GET /documents/{id} "
        "until status is 'indexed'. Admin only."
    ),
    response_model=APIResponse[DocumentOut],
    status_code=202,
    responses={
        202: {"description": "Upload accepted; ingestion queued (status: pending)."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        413: {"description": "File too large (max 50 MB)."},
        415: {"description": "Unsupported file type (VALIDATION_ERROR)."},
    },
)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ..., description="Document to ingest (PDF, TXT, MD, DOCX, HTML; max 50 MB)."
    ),
    current_user: TokenData = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[DocumentOut]:
    """Upload a document and queue it for RAG ingestion (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, VALIDATION_ERROR, INGEST_FAILED.
    """
    # Strip any directory components from the uploaded filename to prevent path traversal.
    filename = Path(file.filename or "upload").name or "upload"

    # Validate file extension
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "VALIDATION_ERROR",
                "message": (
                    f"Unsupported file type '{suffix}'. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
                ),
            },
        )

    # Read and size-check the upload
    data = await file.read()
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "VALIDATION_ERROR",
                "message": f"File too large ({len(data)} bytes); maximum allowed is 50 MB.",
            },
        )

    # Persist to disk
    doc_id = str(uuid.uuid4())
    upload_dir = _get_uploads_dir() / doc_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename
    file_path.write_bytes(data)
    log.info("rag_upload: doc_id=%s file=%s size=%d", doc_id, filename, len(data))

    # Create DB record
    doc = RagDocument(
        id=doc_id,
        filename=filename,
        status="pending",
        chunk_count=0,
        created_at=datetime.now(UTC),
    )
    db.add(doc)
    await db.commit()

    # Queue background ingestion — must run after commit so the background
    # task's fresh AsyncSessionLocal sees the persisted row.
    background_tasks.add_task(ingest_document_task, doc_id, file_path)

    return APIResponse(data=_doc_to_out(doc))


@router.get(
    "/documents",
    summary="List ingested documents",
    description="List all documents in the RAG index with their ingestion status.",
    response_model=APIResponse[CursorPage[DocumentOut]],
    responses={
        200: {"description": "Document list returned."},
        401: {"description": "Not authenticated."},
    },
)
async def list_documents(
    cursor: str | None = None,
    limit: int = 50,
    doc_status: str | None = Query(
        default=None,
        alias="status",
        description="Filter by ingestion status: pending, processing, indexed, failed.",
    ),
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[CursorPage[DocumentOut]]:
    """List all RAG documents (paginated).

    Query parameters:
        cursor — pagination cursor.
        limit  — page size (1–200, default 50).
        status — filter by ingestion status: pending, processing, indexed, failed.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, INVALID_CURSOR.
    """
    limit = max(1, min(limit, 200))

    # Decode cursor (it encodes the last-seen doc created_at + id)
    after_id: str | None = None
    if cursor:
        try:
            after_id = decode_cursor(cursor)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_CURSOR", "message": "The pagination cursor is malformed."},
            ) from exc

    # Count total (with optional status filter)
    count_stmt = select(func.count()).select_from(RagDocument)
    if doc_status:
        count_stmt = count_stmt.where(RagDocument.status == doc_status)
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # Fetch page — compound keyset cursor on (created_at, id) to ensure stable ordering.
    stmt = select(RagDocument).order_by(RagDocument.created_at.asc(), RagDocument.id.asc())
    if doc_status:
        stmt = stmt.where(RagDocument.status == doc_status)
    if after_id:
        # Resolve cursor document's (created_at, id) for a correct compound keyset.
        cursor_result = await db.execute(
            select(RagDocument.created_at, RagDocument.id).where(RagDocument.id == after_id)
        )
        cursor_row = cursor_result.one_or_none()
        if cursor_row is not None:
            cursor_ts, cursor_id = cursor_row
            stmt = stmt.where(
                (RagDocument.created_at > cursor_ts)
                | ((RagDocument.created_at == cursor_ts) & (RagDocument.id > cursor_id))
            )
    stmt = stmt.limit(limit + 1)

    result = await db.execute(stmt)
    rows = list(result.scalars().all())

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    next_cursor: str | None = None
    if has_more and rows:
        next_cursor = encode_cursor(rows[-1].id)

    # _doc_to_out performs disk I/O (stat for file size); run in thread to avoid
    # blocking the event loop when paginating over many documents.
    items = await asyncio.to_thread(lambda: [_doc_to_out(row) for row in rows])
    page: CursorPage[DocumentOut] = CursorPage(
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
        total=total,
    )
    return APIResponse(data=page)


@router.get(
    "/documents/{document_id}",
    summary="Get document details",
    description="Return details and current ingestion status for a single document.",
    response_model=APIResponse[DocumentOut],
    responses={
        200: {"description": "Document details returned."},
        401: {"description": "Not authenticated."},
        404: {"description": "Document not found (DOCUMENT_NOT_FOUND)."},
    },
)
async def get_document(
    document_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[DocumentOut]:
    """Return details for a single RAG document.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, DOCUMENT_NOT_FOUND.
    """
    _validate_doc_id(document_id)
    doc = await db.get(RagDocument, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "DOCUMENT_NOT_FOUND",
                "message": "The requested RAG document does not exist.",
            },
        )
    out = await asyncio.to_thread(_doc_to_out, doc)
    return APIResponse(data=out)


@router.delete(
    "/documents/{document_id}",
    summary="Delete a document",
    description=("Remove a document and all its chunks from the RAG index. " "Admin only."),
    response_model=APIResponse[None],
    responses={
        200: {"description": "Document deleted."},
        401: {"description": "Not authenticated."},
        403: {"description": "Admin required (FORBIDDEN)."},
        404: {"description": "Document not found (DOCUMENT_NOT_FOUND)."},
    },
)
async def delete_document(
    document_id: str,
    current_user: TokenData = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[None]:
    """Delete a document from the RAG index (admin only).

    Auth: admin bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, FORBIDDEN, DOCUMENT_NOT_FOUND.
    """
    _validate_doc_id(document_id)
    doc = await db.get(RagDocument, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "DOCUMENT_NOT_FOUND",
                "message": "The requested RAG document does not exist.",
            },
        )

    # Delete uploaded file and vectordb from disk
    upload_dir = _get_uploads_dir() / document_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
        log.info("rag_delete: removed upload dir %s", upload_dir)

    await db.delete(doc)
    await db.commit()
    log.info("rag_delete: doc_id=%s removed from DB", document_id)
    return APIResponse(data=None)


@router.post(
    "/search",
    summary="Semantic search over the knowledge base",
    description=(
        "Perform a vector similarity search over all indexed documents. "
        "Returns the top-k most relevant chunks ranked by cosine similarity."
    ),
    response_model=APIResponse[RAGSearchResponse],
    responses={
        200: {"description": "Search results returned."},
        401: {"description": "Not authenticated."},
        422: {"description": "Validation error (VALIDATION_ERROR)."},
    },
)
async def search_rag(
    body: RAGSearchRequest,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[RAGSearchResponse]:
    """Semantic search over the RAG knowledge base.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, VALIDATION_ERROR.
    """
    # Determine which documents to search
    stmt = select(RagDocument).where(RagDocument.status == "indexed")
    if body.document_ids:
        stmt = stmt.where(RagDocument.id.in_(body.document_ids))
    result = await db.execute(stmt)
    indexed_docs = list(result.scalars().all())

    if not indexed_docs:
        return APIResponse(
            data=RAGSearchResponse(
                query=body.query,
                chunks=[],
                total_documents_searched=0,
            )
        )

    # Attempt FAISS search over per-document vector stores
    chunks: list[RAGChunkOut] = []
    docs_searched = 0

    try:
        chunks, docs_searched = await asyncio.to_thread(
            _search_faiss, body.query, body.top_k, indexed_docs
        )
    except Exception as exc:
        log.warning("rag_search: search failed, returning empty results: %s", exc)

    return APIResponse(
        data=RAGSearchResponse(
            query=body.query,
            chunks=chunks,
            total_documents_searched=docs_searched,
        )
    )


# ---------------------------------------------------------------------------
# FAISS search helper (runs in thread)
# ---------------------------------------------------------------------------


def _search_faiss(
    query: str,
    top_k: int,
    indexed_docs: list[Any],
) -> tuple[list[RAGChunkOut], int]:
    """Search per-document FAISS indexes and return ranked chunks.

    Returns (chunks, docs_searched).  If FAISS or embeddings are unavailable,
    returns ([], 0) rather than raising.
    """
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        return [], 0

    try:
        from src.tools.rag import _get_embeddings

        embeddings = _get_embeddings()
    except Exception as exc:
        log.debug("RAG embeddings unavailable for search: %s", exc)
        return [], 0

    all_chunks: list[tuple[float, RAGChunkOut]] = []
    docs_searched = 0

    for doc in indexed_docs:
        vectordb_dir = _get_uploads_dir() / doc.id / "vectordb" / "faiss_index"
        if not vectordb_dir.exists():
            continue
        try:
            # allow_dangerous_deserialization is safe here: vectordb_dir is
            # written exclusively by the server-side ingest pipeline and is
            # never user-controlled after upload.
            store = FAISS.load_local(
                str(vectordb_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            results = store.similarity_search_with_score(query, k=top_k)
            docs_searched += 1
            for chunk_idx, (langchain_doc, score) in enumerate(results):
                # FAISS returns L2 distance; convert to a 0–1 similarity score
                similarity = float(max(0.0, 1.0 - score / 2.0))
                all_chunks.append(
                    (
                        similarity,
                        RAGChunkOut(
                            document_id=doc.id,
                            document_name=doc.filename,
                            chunk_index=chunk_idx,
                            text=langchain_doc.page_content,
                            score=similarity,
                        ),
                    )
                )
        except Exception as exc:
            log.debug("rag_search: skipping doc %s: %s", doc.id, exc)

    # Sort by score descending and take top_k
    all_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in all_chunks[:top_k]], docs_searched
