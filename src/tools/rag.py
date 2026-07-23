"""
RAG tool: query_knowledge_base, save_to_knowledge_base
Uses FAISS index stored at data/vectordb/faiss_index.
Supports multiple embedding providers via the ``src.providers`` registry.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.api.rag_index import load_faiss_store_safe, save_faiss_store

# Try to import required modules
try:
    from langchain_community.vectorstores import FAISS

    FAISS_AVAILABLE = True
except ImportError:
    FAISS = None  # type: ignore[misc, assignment]
    FAISS_AVAILABLE = False


VECTOR_DIR = Path("data/vectordb/faiss_index")
_AGENT_NOTES_SUBDIR = "agent_notes"

# Default configuration from environment variables
_DEFAULT_EMBEDDING_PROVIDER = os.getenv("COGTRIX_EMBEDDING_PROVIDER", "ollama")
_DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Runtime configuration (set by configure_rag())
_rag_config: dict[str, Any] = {
    "embedding_provider": _DEFAULT_EMBEDDING_PROVIDER,
    "embedding_model": _DEFAULT_OLLAMA_EMBEDDING_MODEL,
    "base_url": _DEFAULT_OLLAMA_BASE_URL,
    "api_key": None,
    "vectordb_dir": str(VECTOR_DIR),
    "api_uploads_dir": None,
    "entity_index_path": None,
    "score_threshold": 0.0,
}


def configure_rag(config: dict) -> None:
    """
    Configure RAG tool with runtime settings.

    Called from cogtrix.py to pass configuration from .cogtrix.json.

    Args:
        config: Dictionary with keys:
            - embedding_provider: "openai", "ollama", or "google"
            - embedding_model: Model name for embeddings
            - base_url: Ollama server URL (if using Ollama)
            - api_key: Provider API key (for OpenAI/Google; None for Ollama)
            - vectordb_dir: Path to the FAISS index directory
    """
    if "vectordb_dir" in config and config["vectordb_dir"] is not None:
        vdir = Path(str(config["vectordb_dir"]))
        if not vdir.is_absolute():
            resolved = (Path.cwd() / vdir).resolve()
            cwd_resolved = Path.cwd().resolve()
            if not resolved.is_relative_to(cwd_resolved):
                raise ValueError(
                    f"Path traversal detected in vectordb_dir: {config['vectordb_dir']!r}"
                )
    if "api_uploads_dir" in config and config["api_uploads_dir"] is not None:
        uploads = Path(str(config["api_uploads_dir"]))
        if not uploads.is_absolute():
            resolved = (Path.cwd() / uploads).resolve()
            cwd_resolved = Path.cwd().resolve()
            if not resolved.is_relative_to(cwd_resolved):
                raise ValueError(
                    f"Path traversal detected in api_uploads_dir: {config['api_uploads_dir']!r}"
                )
    # Atomic reference swap — safe for concurrent readers without a lock
    global _rag_config
    _rag_config = {**_rag_config, **config}


class KnowledgeQueryInput(BaseModel):
    """Input schema for querying the knowledge base."""

    question: str = Field(description="The question or topic to search for")
    k: int = Field(default=4, description="Number of results to return (1-10)")


class SaveToKnowledgeBaseInput(BaseModel):
    """Input schema for saving a note to the agent knowledge base."""

    content: str = Field(description="The fact, finding, or note to persist")
    source: str = Field(
        default="agent",
        description="Origin label for this entry (e.g. 'agent', 'user', tool name)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional topic tags to attach to the entry",
    )


def _get_embeddings():
    """Get embeddings instance based on configured provider.

    Delegates to the centralized ``src.providers`` registry.
    """
    from src.providers import create_embeddings

    provider = (_rag_config["embedding_provider"] or "ollama").lower()
    model = _rag_config["embedding_model"]
    base_url = _rag_config["base_url"]
    api_key = _rag_config["api_key"]

    return create_embeddings(provider, model=model, base_url=base_url, api_key=api_key)


def save_to_knowledge_base(
    content: str,
    source: str = "agent",
    tags: list[str] | None = None,
) -> str:
    """Persist a fact or note to the agent knowledge base for future retrieval.

    Embeds *content* and appends it to a dedicated agent-notes FAISS sub-index
    (``{vectordb_dir}/../agent_notes/``).  Falls back to a plain JSONL file when
    FAISS is not installed.

    Args:
        content: The fact, finding, or note to persist.
        source: Origin label for this entry.
        tags: Optional topic tags.

    Returns:
        Confirmation string on success, or an error message.
    """
    from src.logging_config import get_logger

    log = get_logger()

    if not content or not content.strip():
        return "Error: content must be non-empty."

    tags = tags or []
    metadata: dict[str, Any] = {
        "source": source,
        "tags": tags,
        "timestamp": datetime.now(UTC).isoformat(),
        "type": "agent_note",
    }

    notes_dir = _agent_notes_faiss_dir()

    if not FAISS_AVAILABLE:
        # Plain-text fallback: JSONL file alongside where the FAISS dir would be
        jsonl_path = notes_dir.parent / f"{_AGENT_NOTES_SUBDIR}.jsonl"
        try:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"content": content.strip(), **metadata}
            with jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.debug("Agent note written to fallback JSONL: %s", jsonl_path)
            return "Saved to knowledge base."
        except OSError as exc:
            return f"Error saving to knowledge base: {exc}"

    try:
        from langchain_core.documents import Document

        embeddings = _get_embeddings()
        doc = Document(page_content=content.strip(), metadata=metadata)

        notes_dir.mkdir(parents=True, exist_ok=True)

        if _has_faiss_index(notes_dir):
            store = load_faiss_store_safe(notes_dir, embeddings)
            if store is None:
                return (
                    "Error: failed to load existing knowledge base index. "
                    "Try rebuilding it with `python cogtrix.py --ingest`."
                )
            store.add_documents([doc])
        else:
            store = FAISS.from_documents([doc], embeddings)

        save_faiss_store(store, notes_dir)
        log.debug("Agent note saved to FAISS index: %s", notes_dir)
        return "Saved to knowledge base."

    except Exception as exc:
        return f"Error saving to knowledge base: {exc}"


def _has_faiss_index(directory: Path) -> bool:
    """Return True if *directory* contains at least one FAISS index file."""
    return directory.is_dir() and (
        (directory / "index.faiss").exists() or any(directory.glob("*.faiss"))
    )


def _agent_notes_faiss_dir() -> Path:
    """Return the FAISS sub-index path for agent notes."""
    vectordb_dir = Path(_rag_config["vectordb_dir"] or str(VECTOR_DIR))
    return vectordb_dir.parent / _AGENT_NOTES_SUBDIR


def _collect_faiss_dirs() -> list[Path]:
    """Return all FAISS index directories to search.

    Checks:
    1. The global CLI-ingest path (``_rag_config["vectordb_dir"]``).
    2. Per-document indexes created by the API ingestion pipeline
       (``_rag_config["api_uploads_dir"]/{doc_id}/vectordb/faiss_index``).
    3. The agent-notes sub-index (``{vectordb_dir}/../agent_notes/``).
    """
    dirs: list[Path] = []

    # Global CLI-ingest index — verify actual index files exist (BUG-200)
    global_dir = Path(_rag_config["vectordb_dir"] or str(VECTOR_DIR))
    if _has_faiss_index(global_dir):
        dirs.append(global_dir)

    # Per-document API indexes
    api_uploads = _rag_config.get("api_uploads_dir")
    if api_uploads:
        uploads_path = Path(api_uploads).resolve()
        if uploads_path.is_dir():
            for doc_dir in uploads_path.iterdir():
                # Skip symlinks to prevent traversal to attacker-controlled dirs
                if doc_dir.is_symlink():
                    continue
                resolved = doc_dir.resolve()
                if not resolved.is_relative_to(uploads_path):
                    continue
                idx = resolved / "vectordb" / "faiss_index"
                # Containment check on the resolved idx path prevents symlink
                # traversal via intermediate components like vectordb (BUG-191)
                idx_resolved = idx.resolve()
                if not idx_resolved.is_relative_to(uploads_path):
                    continue
                if _has_faiss_index(idx):
                    dirs.append(idx)

    # Agent-notes sub-index
    notes_dir = _agent_notes_faiss_dir()
    if _has_faiss_index(notes_dir):
        dirs.append(notes_dir)

    return sorted(dirs)


def query_knowledge_base(
    question: str,
    k: int = 4,
    score_threshold: float | None = None,
) -> str:
    """
    Search the knowledge base for information related to a question.

    Args:
        question: The question or topic to search for
        k: Number of results to return (1-10)

    Returns:
        Relevant document chunks or error message
    """
    if not FAISS_AVAILABLE:
        return "Error: FAISS not available. Run: uv add faiss-cpu"

    faiss_dirs = _collect_faiss_dirs()
    if not faiss_dirs:
        return (
            "No knowledge base found. Please build it first.\n\n"
            "Steps:\n"
            "1. Add documents to the 'docs/' directory (PDF, MD, TXT, CSV)\n"
            "2. Run: python cogtrix.py --ingest\n"
            "3. Try your query again"
        )

    # Clamp k to reasonable range
    k = min(max(1, k), 10)

    try:
        # Get embeddings from environment/config
        embeddings = _get_embeddings()

        # Search all available FAISS indexes and merge results.
        # Use similarity_search_with_score for cross-index relevance
        # ranking (BUG-193) — FAISS L2 distance: lower = more similar.
        scored_docs: list[tuple[Any, float]] = []
        errors: list[str] = []
        for vector_dir in faiss_dirs:
            try:
                store = load_faiss_store_safe(vector_dir, embeddings)
                if store is None:
                    errors.append(f"{vector_dir}: index not loadable")
                    continue
                pairs = store.similarity_search_with_score(question, k=k)
                scored_docs.extend(pairs)
            except Exception as exc:
                errors.append(f"{vector_dir}: {exc}")
                continue

        if not scored_docs:
            if errors:
                return (
                    "Error querying knowledge base. "
                    f"All {len(errors)} index(es) failed:\n" + "\n".join(f"  - {e}" for e in errors)
                )
            return "No relevant documents found for your question."

        # Sort by score ascending (lower L2 distance = more relevant)
        scored_docs.sort(key=lambda x: x[1])

        # Apply score_threshold: convert L2 distance to similarity = 1/(1+d)
        effective_threshold = (
            score_threshold
            if score_threshold is not None
            else float(_rag_config.get("score_threshold") or 0.0)
        )
        if effective_threshold > 0.0:
            scored_docs = [
                (doc, dist)
                for doc, dist in scored_docs
                if 1.0 / (1.0 + dist) >= effective_threshold
            ]
            if not scored_docs:
                return (
                    f"No results met the minimum similarity threshold ({effective_threshold:.2f}). "
                    "Try lowering score_threshold or rephrasing your query."
                )

        # Deduplicate by full content hash and take top k
        seen: set[str] = set()
        unique_docs = []
        for doc, _score in scored_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        all_docs = unique_docs[:k]

        # Format results
        results = []
        results.append(f"Found {len(all_docs)} relevant document(s):\n")

        for i, doc in enumerate(all_docs, 1):
            meta = doc.metadata or {}
            source = meta.get("source", "unknown")

            # Clean up source path for display
            if source != "unknown":
                source = Path(source).name

            # Get page number if available
            page = meta.get("page")
            page_info = f" (page {page})" if page else ""

            # Clean up content
            content = doc.page_content.strip()

            # Truncate very long chunks
            if len(content) > 500:
                content = content[:500] + "..."

            results.append(f"[{i}] Source: {source}{page_info}")
            results.append(f"    {content}")
            results.append("")

        return "\n".join(results)

    except ImportError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error querying knowledge base: {e}"


def get_knowledge_base_info() -> str:
    """
    Get information about the current knowledge base.

    Returns:
        Information about the vector store or status message
    """
    faiss_dirs = _collect_faiss_dirs()
    if not faiss_dirs:
        return "No knowledge base found. Run 'python cogtrix.py --ingest' to create one."

    try:
        # Collect index files across all FAISS directories
        index_files: list[Path] = []
        for d in faiss_dirs:
            index_files.extend(d.glob("*"))
        if not index_files:
            return "Knowledge base directory exists but appears empty."

        info = []
        info.append(f"Knowledge base indexes: {len(faiss_dirs)}")
        for d in faiss_dirs:
            info.append(f"  - {d.absolute()}")
        info.append(f"Index files: {len(index_files)}")

        # Show file sizes
        total_size = sum(f.stat().st_size for f in index_files if f.is_file())
        if total_size < 1024:
            size_str = f"{total_size} bytes"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        else:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        info.append(f"Total size: {size_str}")

        # Show embedding provider info
        provider = _rag_config["embedding_provider"]
        info.append(f"\nEmbedding provider: {provider}")
        if provider == "ollama":
            info.append(f"Ollama URL: {_rag_config['base_url']}")
            info.append(f"Ollama model: {_rag_config['embedding_model']}")

        return "\n".join(info)

    except Exception as e:
        return f"Error getting knowledge base info: {e}"


def knowledge_base_exists() -> bool:
    """Return True if at least one FAISS index is available."""
    return bool(_collect_faiss_dirs())


def knowledge_base_stats() -> tuple[int, int]:
    """Return (index_count, total_size_bytes) across all FAISS indexes."""
    dirs = _collect_faiss_dirs()
    total_size = 0
    for d in dirs:
        try:
            for f in d.iterdir():
                try:
                    if f.is_file():
                        total_size += f.stat().st_size
                except OSError:
                    pass
        except OSError:
            pass
    return len(dirs), total_size


def _build_description() -> str:
    """Build a dynamic tool description based on index state."""
    dirs = _collect_faiss_dirs()
    if not dirs:
        return (
            "Search the knowledge base for information. "
            "Use this to find answers from uploaded documents."
        )
    count = len(dirs)
    total_size = 0
    for d in dirs:
        try:
            for f in d.iterdir():
                try:
                    if f.is_file():
                        total_size += f.stat().st_size
                except OSError:
                    pass
        except OSError:
            pass
    if total_size < 1024 * 1024:
        size_str = f"{total_size / 1024:.0f} KB"
    else:
        size_str = f"{total_size / (1024 * 1024):.1f} MB"
    if count == 1:
        return (
            f"Search your knowledge base ({size_str} indexed). "
            "Use this to find answers from ingested documents before searching the web."
        )
    return (
        f"Search your knowledge base ({count} document indexes, {size_str} total). "
        "Use this to find answers from ingested documents before searching the web."
    )


def rag_find_entity(entity_name: str, max_results: int = 10) -> str:
    """Look up an entity in the RAG entity index and return source chunk references.

    Args:
        entity_name: The entity name to search for (case-insensitive).
        max_results: Maximum number of chunk references to return.

    Returns:
        Chunk reference list or a message when the entity is not found.
    """
    import json as _json

    entity_index_path = _rag_config.get("entity_index_path")
    if not entity_index_path:
        return "Entity index is not configured. Ingest documents first."

    try:
        raw = Path(str(entity_index_path)).read_text(encoding="utf-8")
        index: dict[str, list[str]] = _json.loads(raw)
    except Exception as exc:
        return f"Error loading entity index: {exc}"

    # Case-insensitive lookup
    lower_query = entity_name.lower()
    matched_key = next((k for k in index if k.lower() == lower_query), None)
    if matched_key is None:
        return f"Entity '{entity_name}' not found in the knowledge base."

    refs = index[matched_key]
    shown = refs[:max_results]
    lines = [f"Entity: {matched_key}", f"Found in {len(refs)} chunk(s):"]
    lines.extend(f"  - {r}" for r in shown)
    if len(refs) > max_results:
        lines.append(f"  ... and {len(refs) - max_results} more.")
    return "\n".join(lines)


def rag_ingest(paths: str) -> str:
    """Ingest one or more document files into the knowledge base.

    Args:
        paths: Comma-separated file paths to ingest.

    Returns:
        Summary of ingestion results.
    """
    from src.rag.ingest import IngestConfig, ingest_many

    path_list = [p.strip() for p in paths.split(",") if p.strip()]
    if not path_list:
        return "No file paths provided. Pass comma-separated file paths to ingest."

    vectordb_dir = _rag_config.get("vectordb_dir") or str(VECTOR_DIR)
    ingest_config = IngestConfig(
        docs_dir=Path(path_list[0]).parent,
        vectordb_dir=Path(str(vectordb_dir)),
        embedding_provider=str(_rag_config.get("embedding_provider") or "ollama"),
    )
    results = ingest_many([Path(p) for p in path_list], ingest_config)
    success = sum(1 for v in results.values() if v)
    total = len(results)
    parts = [f"Ingested {success}/{total} file(s) successfully."]
    failed = [p for p, ok in results.items() if not ok]
    if failed:
        parts.append("Failed:")
        parts.extend(f"  - {p}" for p in failed)
    return "\n".join(parts)


TOOL_CONFIGS = [
    {
        "name": "query_knowledge_base",
        "description": (
            "Search the knowledge base for information. "
            "Use this to find answers from uploaded documents."
        ),
        "input_schema": KnowledgeQueryInput,
        "requires_confirmation": False,
        "function": query_knowledge_base,
    },
    {
        "name": "save_to_knowledge_base",
        "description": (
            "Persist a fact, finding, or note to the agent knowledge base so it can be "
            "retrieved in future sessions. Use this when you discover information worth "
            "keeping across conversations: key facts, research results, decisions, or "
            "reusable knowledge. Do not use for transient scratchpad notes."
        ),
        "input_schema": SaveToKnowledgeBaseInput,
        "requires_confirmation": False,
        "function": save_to_knowledge_base,
    },
]

# Backward-compatible alias — callers that import TOOL_CONFIG (e.g. configure.py)
# still get the query tool config dict.
TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "query_knowledge_base",
    "save_to_knowledge_base",
    "rag_find_entity",
    "rag_ingest",
    "get_knowledge_base_info",
    "configure_rag",
    "knowledge_base_exists",
    "knowledge_base_stats",
    "KnowledgeQueryInput",
    "SaveToKnowledgeBaseInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
