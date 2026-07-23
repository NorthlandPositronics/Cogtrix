"""
RAG tool: query_knowledge_base
Uses FAISS index stored at data/vectordb/faiss_index.
Supports multiple embedding providers: OpenAI and Ollama.
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field

# Try to import required modules
try:
    from langchain_community.vectorstores import FAISS

    FAISS_AVAILABLE = True
except ImportError:
    FAISS = None  # type: ignore[misc, assignment]
    FAISS_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings

    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OpenAIEmbeddings = None  # type: ignore[misc, assignment]
    OPENAI_EMBEDDINGS_AVAILABLE = False

try:
    from langchain_ollama import OllamaEmbeddings

    OLLAMA_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OllamaEmbeddings = None  # type: ignore[misc, assignment]
    OLLAMA_EMBEDDINGS_AVAILABLE = False


VECTOR_DIR = Path("data/vectordb/faiss_index")

# Default configuration from environment variables
_DEFAULT_EMBEDDING_PROVIDER = os.getenv("COGTRIX_EMBEDDING_PROVIDER", "openai")
_DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Runtime configuration (set by configure_rag())
_rag_config = {
    "embedding_provider": _DEFAULT_EMBEDDING_PROVIDER,
    "embedding_model": _DEFAULT_OLLAMA_EMBEDDING_MODEL,
    "ollama_base_url": _DEFAULT_OLLAMA_BASE_URL,
}


def configure_rag(config: dict) -> None:
    """
    Configure RAG tool with runtime settings.

    Called from cogtrix.py to pass configuration from .cogtrix.json.

    Args:
        config: Dictionary with keys:
            - embedding_provider: "openai" or "ollama"
            - embedding_model: Model name for embeddings
            - ollama_base_url: Ollama server URL (if using Ollama)
    """
    if "embedding_provider" in config:
        _rag_config["embedding_provider"] = config["embedding_provider"]
    if "embedding_model" in config:
        _rag_config["embedding_model"] = config["embedding_model"]
    if "ollama_base_url" in config:
        _rag_config["ollama_base_url"] = config["ollama_base_url"]


class KnowledgeQueryInput(BaseModel):
    """Input schema for querying the knowledge base."""

    question: str = Field(description="The question or topic to search for")
    k: int = Field(default=4, description="Number of results to return (1-10)")


def _get_embeddings():
    """Get embeddings instance based on configured provider."""
    provider = _rag_config["embedding_provider"].lower()
    model = _rag_config["embedding_model"]
    base_url = _rag_config["ollama_base_url"]

    if provider == "ollama":
        if not OLLAMA_EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "Ollama embeddings not available. Install: pip install langchain-ollama"
            )
        return OllamaEmbeddings(
            model=model or "nomic-embed-text",
            base_url=base_url or "http://localhost:11434",
        )
    else:  # openai
        if not OPENAI_EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "OpenAI embeddings not available. Install: pip install langchain-openai"
            )
        return OpenAIEmbeddings(model=model or "text-embedding-3-small")


def query_knowledge_base(
    question: str,
    k: int = 4,
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
        return "Error: FAISS not available. Install: pip install faiss-cpu"

    if not VECTOR_DIR.exists():
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

        # Load the vector store
        store = FAISS.load_local(
            str(VECTOR_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Perform similarity search
        docs = store.similarity_search(question, k=k)

        if not docs:
            return "No relevant documents found for your question."

        # Format results
        results = []
        results.append(f"Found {len(docs)} relevant document(s):\n")

        for i, doc in enumerate(docs, 1):
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
    if not VECTOR_DIR.exists():
        return "No knowledge base found. Run 'python cogtrix.py --ingest' to create one."

    try:
        # Check for index files
        index_files = list(VECTOR_DIR.glob("*"))
        if not index_files:
            return "Knowledge base directory exists but appears empty."

        info = []
        info.append(f"Knowledge base location: {VECTOR_DIR.absolute()}")
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
            info.append(f"Ollama URL: {_rag_config['ollama_base_url']}")
            info.append(f"Ollama model: {_rag_config['embedding_model']}")

        return "\n".join(info)

    except Exception as e:
        return f"Error getting knowledge base info: {e}"


# Main tool config
TOOL_CONFIG = {
    "name": "query_knowledge_base",
    "description": (
        "Search the knowledge base for information. "
        "Use this to find answers from uploaded documents."
    ),
    "input_schema": KnowledgeQueryInput,
    "requires_confirmation": False,
}

__all__ = [
    "query_knowledge_base",
    "get_knowledge_base_info",
    "configure_rag",
    "KnowledgeQueryInput",
    "TOOL_CONFIG",
]
