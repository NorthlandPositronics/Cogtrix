"""Document ingestion for RAG knowledge base.

Loads documents from a directory, splits them into chunks,
creates embeddings, and stores in a FAISS vector database.
"""

import json
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

from src.utils.atomic_write import atomic_write_json

_entity_index_lock = threading.Lock()

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
    """Configuration for document ingestion."""

    docs_dir: Path
    vectordb_dir: Path
    chunk_size: int = 2000
    chunk_overlap: int = 200
    embedding_provider: str = "ollama"
    embedding_model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    entity_index_path: Path | None = None


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

        config.vectordb_dir.mkdir(parents=True, exist_ok=True)
        persist_path = config.vectordb_dir / "faiss_index"
        vector_store.save_local(str(persist_path))

        result.vector_store_path = persist_path
        result.success = True

    except Exception as e:
        result.errors.append(f"Failed to build vector store: {e}")
        return result

    return result


def _ingest_one_file(path: Path, config: IngestConfig) -> bool:
    """Ingest a single file into the vector store.

    Returns True on success, False on any error.
    """
    try:
        loader = _get_loader(path)
        if loader is None:
            return False
        docs = loader.load()
        if not docs:
            return False
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            return False
        embeddings = _create_embeddings(config)
        vector_store = FAISS.from_documents(chunks, embeddings)
        config.vectordb_dir.mkdir(parents=True, exist_ok=True)
        persist_path = config.vectordb_dir / "faiss_index"
        vector_store.save_local(str(persist_path))
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

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        future_to_path = {pool.submit(_ingest_one_file, Path(p), config): str(p) for p in paths}
        for future, path_str in future_to_path.items():
            try:
                results[path_str] = future.result()
            except Exception:
                results[path_str] = False

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
