"""Document ingestion for RAG knowledge base.

Loads documents from a directory, splits them into chunks,
creates embeddings, and stores in a FAISS vector database.
"""

import json
import logging
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.api.rag_index import save_faiss_store
from src.utils.atomic_write import atomic_write_json

_entity_index_lock = threading.Lock()
_log = logging.getLogger("cogtrix.ingest")

_INGEST_PREPARE_TIMEOUT = 60.0  # seconds; tests monkey-patch this down

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "in",
        "of",
        "for",
        "with",
        "their",
        "they",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "we",
        "our",
        "he",
        "she",
        "i",
        "my",
        "your",
        "his",
        "her",
        "by",
        "at",
        "on",
        "as",
        "not",
        "but",
        "if",
        "then",
        "than",
        "from",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "so",
        "all",
        "also",
        "into",
        "about",
        "up",
        "which",
        "what",
        "who",
        "when",
        "where",
        "how",
        "any",
        "each",
        "other",
        "more",
        "no",
        "can",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "being",
    }
)


@dataclass
class IngestConfig:
    """Configuration for document ingestion.

    ``vectordb_dir`` is the *exact* directory where the FAISS index files
    (``index.faiss`` / ``index.pkl``) are written.  This matches the
    convention used by ``src.tools.rag.configure_rag`` and
    ``_collect_faiss_dirs`` (which look for ``index.faiss`` directly inside
    the configured directory).  Callers that want the historical layout —
    a ``faiss_index/`` sub-directory under some parent — must include that
    segment in the path they pass (e.g. ``parent / "faiss_index"``).

    Prior to #1951 this field meant the *parent* directory and the ingest
    code silently appended ``/faiss_index``; that asymmetry caused
    ``src/tools/rag.py`` (which already passed ``data/vectordb/faiss_index``)
    to produce a doubled ``data/vectordb/faiss_index/faiss_index/`` layout
    where the query side could never find the index.
    """

    docs_dir: Path
    vectordb_dir: Path
    # #1952 Option C: lowered from 2000/200 → 800/100.  Diagnostic probing
    # of the qwen3-embedding model (see #1952's Regime B / C analysis +
    # tests/role_pm/corpus_ingest.py at 500/50) showed that
    # 2000-character chunks span multiple topics — the per-chunk
    # semantic vector becomes diffuse, and retrieval pulls toward
    # whichever document happens to sit at the corpus centroid.  Smaller
    # chunks give each one tighter focus.  800/100 is the moderate
    # default: meaningfully smaller than 2000 (Regime B/C partial
    # relief per the issue) without being as aggressive as the role_pm
    # harness's 500/50 (which is calibrated for that specific corpus).
    # Operators with explicit ``rag.chunk_size`` in ``~/.cogtrix.yaml``
    # keep their override; operators on defaults get the new value on
    # next ``python cogtrix.py --ingest``.
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_provider: str = "ollama"
    embedding_model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    entity_index_path: Path | None = None
    # #1981: opt-in BM25 sidecar for hybrid retrieval.  Default ``False``
    # keeps every existing ingest pipeline pure-vector.  When ``True``
    # a ``bm25.pkl`` is written alongside ``index.faiss`` and
    # ``src.tools.rag`` can fuse vector + BM25 ranks at query time
    # (gated separately by ``configure_rag({"use_bm25_hybrid": True})``).
    build_bm25_sidecar: bool = False


@dataclass
class IngestResult:
    """Result of document ingestion."""

    success: bool
    documents_loaded: int = 0
    chunks_created: int = 0
    vector_store_path: Path | None = None
    errors: list[str] = field(default_factory=list)


def _get_loader(path: Path):
    """Return appropriate document loader for file type.

    Args:
        path: Path to the document file.

    Returns:
        Document loader instance or None if unsupported.
    """
    from langchain_community.document_loaders import CSVLoader, TextLoader

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError as exc:
            raise ImportError(
                "PDF ingestion requires pypdf. Install it with: uv add 'cogtrix[rag]'"
            ) from exc
        return PyPDFLoader(str(path))

    elif suffix in {".md", ".markdown"}:
        # Try UnstructuredMarkdownLoader if available, else use TextLoader
        try:
            from langchain_community.document_loaders import UnstructuredMarkdownLoader

            return UnstructuredMarkdownLoader(str(path))
        except (ImportError, ModuleNotFoundError):
            # Fallback to TextLoader (reads as plain text)
            return TextLoader(str(path))

    elif suffix == ".csv":
        return CSVLoader(str(path))

    elif suffix == ".txt":
        return TextLoader(str(path))

    return None


def _load_documents(docs_dir: Path) -> tuple[list[Document], list[str]]:
    """Load all documents from directory.

    Args:
        docs_dir: Directory containing documents.

    Returns:
        Tuple of (loaded documents, error messages).
    """
    documents: list[Document] = []
    errors: list[str] = []

    if not docs_dir.exists():
        errors.append(f"Documents directory not found: {docs_dir}")
        return documents, errors

    if not docs_dir.is_dir():
        errors.append(f"Not a directory: {docs_dir}")
        return documents, errors

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue

        loader = _get_loader(path)
        rel = path.relative_to(docs_dir)
        if loader is None:
            errors.append(f"Skipped unsupported file: {rel}")
            continue

        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            errors.append(f"Failed to load {rel}: {e}")

    return documents, errors


def _create_embeddings(config: IngestConfig):
    """Create embeddings instance based on provider config.

    Delegates to the centralized ``src.providers`` registry.

    Args:
        config: Ingestion configuration.

    Returns:
        Embeddings instance.

    Raises:
        ValueError: If provider is not supported.
        NotImplementedError: If provider has no embedding support.
    """
    from src.providers import create_embeddings

    return create_embeddings(
        config.embedding_provider,
        model=config.embedding_model,
        base_url=config.base_url,
        api_key=config.api_key,
    )


def _split_documents(documents: list[Document], config: IngestConfig) -> list[Document]:
    """Split documents into chunks.

    Args:
        documents: List of loaded documents.
        config: Ingestion configuration.

    Returns:
        List of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(documents)


def ingest_documents(config: IngestConfig) -> IngestResult:
    """Ingest documents and build vector store.

    Main entry point for document ingestion. Loads documents from the
    configured directory, splits them into chunks, creates embeddings,
    and saves to a FAISS vector store.

    Args:
        config: Ingestion configuration.

    Returns:
        IngestResult with success status and statistics.
    """
    result = IngestResult(success=False)

    # Load documents
    documents, load_errors = _load_documents(config.docs_dir)
    result.errors.extend(load_errors)

    if not documents:
        result.errors.append(
            f"No documents loaded from {config.docs_dir}. "
            "Add PDF, Markdown, CSV, or TXT files and retry."
        )
        return result

    result.documents_loaded = len(documents)

    # Split into chunks
    try:
        chunks = _split_documents(documents, config)
        if not chunks:
            result.errors.append("No text content found in documents after splitting")
            return result
        result.chunks_created = len(chunks)
    except Exception as e:
        result.errors.append(f"Failed to split documents: {e}")
        return result

    # Create embeddings
    try:
        embeddings = _create_embeddings(config)
    except Exception as e:
        result.errors.append(f"Failed to create embeddings: {e}")
        return result

    # Build and save vector store
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)

        # See ``IngestConfig`` docstring: ``vectordb_dir`` is the exact
        # FAISS index directory; no implicit ``/faiss_index`` append.
        persist_path = config.vectordb_dir
        save_faiss_store(vector_store, persist_path)

        result.vector_store_path = persist_path
        result.success = True

    except Exception as e:
        result.errors.append(f"Failed to build vector store: {e}")
        return result

    # #1981: opt-in BM25 sidecar — built only when explicitly enabled
    # via ``config.build_bm25_sidecar``.  A sidecar-write failure is
    # logged but does NOT fail the ingest: dense retrieval is still
    # authoritative, hybrid is an opt-in enhancement.
    if config.build_bm25_sidecar:
        _maybe_build_bm25_sidecar(chunks, config.vectordb_dir)

    return result


def _prepare_ingest_file(path: Path, config: IngestConfig) -> tuple[str, list[Document]] | None:
    """Load and split a single document file for a later combined ingest."""
    try:
        loader = _get_loader(path)
        if loader is None:
            return None
        docs = loader.load()
        if not docs:
            return None
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            return None
        return str(path), chunks
    except Exception:
        return None


def _ingest_one_file(path: Path, config: IngestConfig) -> bool:
    """Ingest a single file into the vector store.

    Returns True on success, False on any error.
    """
    try:
        prepared = _prepare_ingest_file(path, config)
        if prepared is None:
            return False
        _, chunks = prepared
        embeddings = _create_embeddings(config)
        vector_store = FAISS.from_documents(chunks, embeddings)
        persist_path = config.vectordb_dir
        save_faiss_store(vector_store, persist_path)
        if config.build_bm25_sidecar:
            _maybe_build_bm25_sidecar(chunks, config.vectordb_dir)
        return True
    except Exception:
        return False


def ingest_many(
    paths: list[Any],
    config: IngestConfig,
    workers: int = 8,
) -> dict[str, bool]:
    """Ingest multiple files in parallel.

    Args:
        paths: Sequence of file paths to ingest.
        config: Ingestion configuration shared across all files.
        workers: Maximum worker threads (capped at min(len(paths), workers, 8)).

    Returns:
        Dict mapping ``str(path)`` to True (success) or False (failure).
    """
    if not paths:
        return {}

    actual_workers = min(len(paths), workers, 8)
    results: dict[str, bool] = {}
    prepared_chunks: list[Document] = []
    successful_paths: list[str] = []

    # Use explicit ThreadPoolExecutor (not `with`) so shutdown(wait=False)
    # can be used on timeout — `__exit__` calls shutdown(wait=True) which
    # blocks on hung threads.
    pool = ThreadPoolExecutor(max_workers=actual_workers)
    try:
        future_to_path = {pool.submit(_prepare_ingest_file, Path(p), config): str(p) for p in paths}
        for future, path_str in future_to_path.items():
            try:
                prepared = future.result(timeout=_INGEST_PREPARE_TIMEOUT)
            except TimeoutError:
                future.cancel()
                _log.warning(
                    "Ingest file %s timed out after %.1fs — marking as failed",
                    path_str,
                    _INGEST_PREPARE_TIMEOUT,
                )
                results[path_str] = False
                continue
            except Exception:
                results[path_str] = False
                continue

            if prepared is None:
                results[path_str] = False
                continue

            _, chunks = prepared
            results[path_str] = True
            successful_paths.append(path_str)
            prepared_chunks.extend(chunks)
    finally:
        pool.shutdown(wait=False)

    if not prepared_chunks:
        return results

    try:
        embeddings = _create_embeddings(config)
        vector_store = FAISS.from_documents(prepared_chunks, embeddings)
        persist_path = config.vectordb_dir
        save_faiss_store(vector_store, persist_path)
        if config.build_bm25_sidecar:
            _maybe_build_bm25_sidecar(prepared_chunks, config.vectordb_dir)
    except Exception:
        for path_str in successful_paths:
            results[path_str] = False
        return results

    return results


def _extract_entities(
    chunks: list[Document],
    source_name: str,
) -> dict[str, list[str]]:
    """Extract named entities and key terms from document chunks.

    Extracts:
    - Capitalized multi-word phrases (e.g. "John Smith", "New York")
    - Quoted strings (double-quoted, 3+ chars)
    - Frequent non-stop words (appearing 3+ times across all chunks)

    Args:
        chunks: Document chunks to scan.
        source_name: Base name used to form chunk references.

    Returns:
        Dict mapping entity name → list of chunk references
        (e.g. ``"source_name:chunk_0"``).
    """
    entities: dict[str, list[str]] = {}
    word_counter: Counter[str] = Counter()

    for idx, chunk in enumerate(chunks):
        ref = f"{source_name}:chunk_{idx}"
        text = chunk.page_content

        # Quoted strings (double-quoted, minimum 3 chars)
        for quoted in re.findall(r'"([^"]{3,})"', text):
            key = quoted.lower()
            entities.setdefault(key, [])
            if ref not in entities[key]:
                entities[key].append(ref)

        # Capitalized consecutive-word phrases (2+ words)
        for phrase in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
            entities.setdefault(phrase, [])
            if ref not in entities[phrase]:
                entities[phrase].append(ref)

        # Word frequency for non-stop words
        for word in re.findall(r"\b[a-z]{4,}\b", text.lower()):
            if word not in _STOP_WORDS:
                word_counter[word] += 1

    # Add frequent words that appear 3+ times
    for word, count in word_counter.items():
        if count >= 3:
            for idx, chunk in enumerate(chunks):
                if re.search(r"\b" + re.escape(word) + r"\b", chunk.page_content, re.IGNORECASE):
                    ref = f"{source_name}:chunk_{idx}"
                    entities.setdefault(word, [])
                    if ref not in entities[word]:
                        entities[word].append(ref)

    return entities


def _update_entity_index(
    new_entities: dict[str, list[str]],
    index_path: Path,
) -> None:
    """Merge *new_entities* into the JSON entity index at *index_path*.

    Thread-safe: acquires a module-level lock before reading and writing.
    Uses atomic rename to prevent corrupt JSON on crash.
    """
    with _entity_index_lock:
        existing: dict[str, list[str]] = {}
        if index_path.exists():
            try:
                existing = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        for entity, refs in new_entities.items():
            existing_refs = existing.setdefault(entity, [])
            for ref in refs:
                if ref not in existing_refs:
                    existing_refs.append(ref)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_write_json(index_path) as fh:
            fh.write(json.dumps(existing, indent=2))


def _maybe_build_bm25_sidecar(chunks: list[Document], vectordb_dir: Path) -> None:
    """Build + persist a BM25 sidecar alongside the FAISS index (#1981).

    Best-effort: failures are logged at WARNING and swallowed.  The
    dense index has already been saved by the caller; an unwritable
    sidecar must not corrupt that.  ``IngestConfig.build_bm25_sidecar``
    gates this entire path — when ``False``, callers never reach here.
    """
    try:
        from src.rag.bm25 import build_sidecar, save_sidecar
    except ImportError as exc:
        _log.warning("BM25 sidecar build skipped — module import failed: %s", exc)
        return

    try:
        sidecar = build_sidecar(chunks)
    except Exception as exc:  # noqa: BLE001 — best-effort sidecar
        _log.warning("BM25 sidecar build failed: %s", exc)
        return

    if sidecar is None:
        return  # build_sidecar logged the reason (empty after filtering)

    try:
        out = save_sidecar(sidecar, vectordb_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort sidecar
        _log.warning("BM25 sidecar write failed: %s", exc)
        return

    _log.info(
        "BM25 sidecar written: chunks=%d, path=%s",
        len(sidecar.corpus_tokens),
        out,
    )
