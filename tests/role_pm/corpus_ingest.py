"""Idempotent corpus ingestion for the PM role-test harness (#1948).

Hashes the corpus directory and stores the hash next to the FAISS
index.  Re-ingests only when the hash changes; otherwise reuses the
existing index.  Keeps the role-test harness fast on repeat runs
without sacrificing correctness when the corpus changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


_HASH_FILENAME = ".corpus_hash.json"


@dataclass(slots=True)
class CorpusIngestResult:
    """Outcome of a corpus ingest attempt."""

    vectordb_dir: Path
    skipped: bool
    documents_loaded: int
    chunks_created: int


# Schema version — bump whenever the ingest config (chunk_size,
# chunk_overlap, embedding model defaults, etc.) changes in a way
# that requires re-ingestion to take effect.  The corpus hash mixes
# this version in so an existing FAISS index is automatically
# invalidated when the schema bumps.
_INGEST_SCHEMA_VERSION = "v2-chunk500-overlap50"


def _hash_corpus(corpus_dir: Path) -> str:
    """Return a stable sha256 over every ``*.md`` file's name + content
    AND the current ingest schema version.

    Sorted alphabetically so the hash is reproducible regardless of
    filesystem iteration order.  Mixes the schema version in so a
    chunk_size / chunk_overlap change forces re-ingestion even when
    the corpus content is identical.
    """
    hasher = hashlib.sha256()
    hasher.update(_INGEST_SCHEMA_VERSION.encode("utf-8"))
    hasher.update(b"\0")
    for path in sorted(corpus_dir.glob("*.md")):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _read_stored_hash(vectordb_dir: Path) -> str | None:
    hash_path = vectordb_dir / _HASH_FILENAME
    if not hash_path.exists():
        return None
    try:
        return json.loads(hash_path.read_text())["sha256"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _write_stored_hash(vectordb_dir: Path, sha256: str) -> None:
    vectordb_dir.mkdir(parents=True, exist_ok=True)
    (vectordb_dir / _HASH_FILENAME).write_text(json.dumps({"sha256": sha256}))


def _index_present(vectordb_dir: Path) -> bool:
    """Mirrors ``src.tools.rag._has_faiss_index`` — index exists when
    the directory holds either ``index.faiss`` or any ``*.faiss`` file."""
    if not vectordb_dir.is_dir():
        return False
    if (vectordb_dir / "index.faiss").exists():
        return True
    return any(vectordb_dir.glob("*.faiss"))


_BM25_SIDECAR_FILENAME = "bm25.pkl"


def _ensure_bm25_sidecar(vectordb_dir: Path) -> None:
    """Build the BM25 sidecar from the existing FAISS docstore if it's absent (#2008).

    Best-effort and embedding-free: the sidecar is derived purely from the stored
    chunk texts (``metadata.json``), so a hash-skip (no re-ingest) can still gain
    hybrid retrieval without re-embedding. A failure never breaks the skip path —
    the query layer falls back to pure-vector when the sidecar is missing.
    """
    if (vectordb_dir / _BM25_SIDECAR_FILENAME).exists():
        return
    meta_path = vectordb_dir / "metadata.json"
    if not meta_path.exists():
        return
    try:
        from langchain_core.documents import Document

        from cogtrix_core.rag.bm25 import build_sidecar, save_sidecar

        raw = json.loads(meta_path.read_text())
        docs = [
            Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
            for d in raw.get("documents", [])
        ]
        sidecar = build_sidecar(docs)
        if sidecar is not None:
            save_sidecar(sidecar, vectordb_dir)
            log.info("BM25 sidecar built from docstore (%d chunks) at %s", len(docs), vectordb_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; pure-vector still works
        log.warning("BM25 sidecar build from docstore skipped: %s", exc)


def ingest_corpus_idempotent(
    corpus_dir: Path,
    vectordb_dir: Path,
    *,
    embedding_provider: str = "openai",
    embedding_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> CorpusIngestResult:
    """Ingest *corpus_dir* into *vectordb_dir* unless the hash matches.

    Post-#1951: *vectordb_dir* follows the ``IngestConfig.vectordb_dir``
    convention as the EXACT directory that holds ``index.faiss``.  The
    corpus-hash sentinel is stored alongside the index file in the same
    directory.

    The hash function covers every ``*.md`` file in ``corpus_dir``;
    docs added, removed, or modified cause re-ingestion.  When the
    hash matches AND a FAISS index is present, this returns
    immediately with ``skipped=True`` and the recorded counts from
    the previous run set to zero (no run happened).
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    expected_hash = _hash_corpus(corpus_dir)
    stored_hash = _read_stored_hash(vectordb_dir)

    if stored_hash == expected_hash and _index_present(vectordb_dir):
        log.info(
            "Corpus hash unchanged (%s...); reusing existing FAISS index at %s",
            expected_hash[:12],
            vectordb_dir,
        )
        # #2008: the BM25 sidecar is derived from the (unchanged) docstore, not
        # the embeddings, so refresh it here without a costly re-embed when it's
        # missing — e.g. a committed index that predates hybrid retrieval. Keeps
        # the hybrid path (use_bm25_hybrid) working after a plain hash-skip.
        _ensure_bm25_sidecar(vectordb_dir)
        return CorpusIngestResult(
            vectordb_dir=vectordb_dir,
            skipped=True,
            documents_loaded=0,
            chunks_created=0,
        )

    # Re-ingest.  Wipe the existing index so the new build is clean
    # (FAISS does not support in-place document removal cleanly).  Keep
    # the corpus-hash sentinel so we can detect stale state if the
    # subsequent ingest fails before writing a new hash.
    if vectordb_dir.exists():
        for child in vectordb_dir.iterdir():
            if child.name == _HASH_FILENAME:
                continue
            if child.is_file():
                child.unlink()
            else:
                _rmtree(child)

    # Import inside the function so the module load does not require
    # the optional ``[rag]`` extra to be installed when the test
    # infrastructure is merely imported (e.g. by pyright during a
    # generic check).
    from cogtrix_core.rag.ingest import IngestConfig, ingest_documents

    config = IngestConfig(
        docs_dir=corpus_dir,
        vectordb_dir=vectordb_dir,
        # Diagnostic probing during the first live run (see #1952)
        # showed qwen3-embedding has weak discriminative retrieval at
        # 1500-char chunks — chunks span multiple topics and the
        # per-chunk semantic vector becomes diffuse.  Smaller chunks
        # give each chunk tighter focus.  Re-tuned 1500/200 → 500/50.
        chunk_size=500,
        chunk_overlap=50,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        base_url=base_url,
        api_key=api_key,
        # #2008: build the BM25 sidecar so hybrid retrieval can lexically surface
        # exact tokens (numbers, IDs, proper nouns) that qwen3-embedding buries —
        # e.g. the $1,106,500 budget chunk sat at embedding rank ~165/296, far
        # below the CE re-rank pool; BM25 puts it at ~#8, into the pool.
        build_bm25_sidecar=True,
    )

    log.info(
        "Ingesting corpus at %s into %s (provider=%s)",
        corpus_dir,
        vectordb_dir,
        embedding_provider,
    )
    result: Any = ingest_documents(config)

    if not result.success:
        raise RuntimeError(f"Corpus ingestion failed: {result.errors}")

    _write_stored_hash(vectordb_dir, expected_hash)

    return CorpusIngestResult(
        vectordb_dir=vectordb_dir,
        skipped=False,
        documents_loaded=result.documents_loaded,
        chunks_created=result.chunks_created,
    )


def _rmtree(path: Path) -> None:
    """Minimal recursive remove — avoids the ``shutil`` import for one
    helper, keeps the cost honest.  Walks the tree depth-first."""
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    for child in path.iterdir():
        _rmtree(child)
    path.rmdir()
