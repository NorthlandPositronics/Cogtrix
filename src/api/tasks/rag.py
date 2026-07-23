"""Background task for RAG document ingestion.

Runs document ingestion (chunking + embedding) in a thread pool so it does not
block the FastAPI event loop.  Updates the ``rag_documents`` DB row with the
final status (``indexed`` or ``failed``) and stores chunk count.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import update

from src.api.db.engine import AsyncSessionLocal
from src.api.db.models import RagDocument

log = logging.getLogger("cogtrix.api.tasks.rag")

# Upload storage root — relative to cwd at startup
_UPLOADS_DIR = Path("data/api/uploads")


def _run_ingest(doc_id: str, file_path: Path) -> tuple[bool, int, str | None]:
    """Run document ingestion synchronously (called via asyncio.to_thread).

    Places a single file into a per-document docs directory, ingests it into a
    per-document FAISS index, and returns (success, chunk_count, error_message).
    """
    from src.rag.ingest import IngestConfig, ingest_documents

    docs_dir = file_path.parent
    vectordb_dir = _UPLOADS_DIR / doc_id / "vectordb"

    config = IngestConfig(
        docs_dir=docs_dir,
        vectordb_dir=vectordb_dir,
    )
    try:
        result = ingest_documents(config)
    except Exception as exc:
        return False, 0, str(exc)

    if result.success:
        return True, result.chunks_created, None
    error = "; ".join(result.errors) if result.errors else "Ingestion failed"
    return False, result.chunks_created, error


async def ingest_document_task(doc_id: str, file_path: Path) -> None:
    """Background task: ingest a document and update its DB status.

    1. Marks the document as ``processing``.
    2. Calls the RAG ingest pipeline in a thread.
    3. On success: sets status ``indexed``, ``indexed_at``, ``chunk_count``.
    4. On failure: sets status ``failed``, ``error``.
    """
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(RagDocument).where(RagDocument.id == doc_id).values(status="processing")
        )
        await db.commit()
    log.info("rag_ingest: doc_id=%s status=processing file=%s", doc_id, file_path)

    try:
        success, chunk_count, error_msg = await asyncio.to_thread(_run_ingest, doc_id, file_path)
    except Exception as exc:
        success = False
        chunk_count = 0
        error_msg = f"Unexpected error: {exc}"
        log.exception("rag_ingest: doc_id=%s unexpected error", doc_id)

    async with AsyncSessionLocal() as db:
        if success:
            await db.execute(
                update(RagDocument)
                .where(RagDocument.id == doc_id)
                .values(
                    status="indexed",
                    indexed_at=datetime.now(UTC),
                    chunk_count=chunk_count,
                    error=None,
                )
            )
            log.info("rag_ingest: doc_id=%s status=indexed chunks=%d", doc_id, chunk_count)
        else:
            await db.execute(
                update(RagDocument)
                .where(RagDocument.id == doc_id)
                .values(status="failed", error=error_msg)
            )
            log.warning("rag_ingest: doc_id=%s status=failed error=%s", doc_id, error_msg)
        await db.commit()
