"""RAG (Retrieval-Augmented Generation) module.

Provides document ingestion and vector store management for knowledge base queries.
"""

from .ingest import IngestConfig, IngestResult, ingest_documents

__all__ = ["ingest_documents", "IngestConfig", "IngestResult"]
