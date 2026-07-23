"""Document ingestion for RAG knowledge base.

Loads documents from a directory, splits them into chunks,
creates embeddings, and stores in a FAISS vector database.
"""

from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    from langchain_community.document_loaders import (
        CSVLoader,
        PyPDFLoader,
        TextLoader,
    )

    suffix = path.suffix.lower()

    if suffix == ".pdf":
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
