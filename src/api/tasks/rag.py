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
from typing import Any

from sqlalchemy import update

# Import the engine module (cheap) rather than the AsyncSessionLocal
# attribute (lazy — touches the filesystem on first access).  We resolve
# AsyncSessionLocal at call time via _db.AsyncSessionLocal so module import
# stays side-effect free.
from src.api.db import engine as _db
from src.api.db.models import RagDocument

log = logging.getLogger("cogtrix.api.tasks.rag")


def _get_uploads_dir() -> Path:
    """Resolve uploads directory using COGTRIX_DATA_DIR if set."""
    import os

    data_dir = os.environ.get("COGTRIX_DATA_DIR", "data")
    return Path(data_dir, "api", "uploads").resolve()


def _run_ingest(doc_id: str, file_path: Path) -> tuple[bool, int, str | None]:
    """Run document ingestion synchronously (called via asyncio.to_thread).

    Places a single file into a per-document docs directory, ingests it into a
    per-document FAISS index, and returns (success, chunk_count, error_message).
    """
    from src.rag.ingest import IngestConfig, ingest_documents

    docs_dir = file_path.parent
    # ``vectordb_dir`` is the EXACT FAISS index directory (see #1951).
    # ``_collect_faiss_dirs`` in ``src/tools/rag.py`` looks for the
    # per-document index at ``<uploads>/<doc_id>/vectordb/faiss_index/``,
    # so we include that trailing segment explicitly here.
    vectordb_dir = _get_uploads_dir() / doc_id / "vectordb" / "faiss_index"

    # Resolve embedding provider from app config so the API path uses the
    # same provider as the CLI (instead of defaulting to Ollama).
    emb_kwargs: dict[str, Any] = {}
    try:
        from src.config import get_cached_config

        # #2101: reuse the process-wide resolved config instead of re-reading
        # os.environ on every ingest task.
        cfg = get_cached_config()
        emb_type, emb_model, emb_base_url, emb_api_key = cfg.resolve_embedding_config()
        emb_kwargs = {
            "embedding_provider": emb_type,
            "embedding_model": emb_model,
            "base_url": emb_base_url,
            "api_key": emb_api_key,
        }
        # #2071: also honor the operator's chunk settings (previously dropped,
        # so rag.chunk_size / rag.chunk_overlap overrides were ignored on the
        # API ingest path, diverging from the CLI).
        rag_cfg = getattr(cfg, "rag", None)
        if rag_cfg is not None:
            if getattr(rag_cfg, "chunk_size", None) is not None:
                emb_kwargs["chunk_size"] = rag_cfg.chunk_size
            if getattr(rag_cfg, "chunk_overlap", None) is not None:
                emb_kwargs["chunk_overlap"] = rag_cfg.chunk_overlap
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not resolve embedding config; using defaults: %s", exc)

    config = IngestConfig(
        docs_dir=docs_dir,
        vectordb_dir=vectordb_dir,
        **emb_kwargs,
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
    async with _db.AsyncSessionLocal() as db:
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

    async with _db.AsyncSessionLocal() as db:
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
