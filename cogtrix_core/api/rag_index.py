"""Shared raw FAISS persistence helpers for API RAG indexes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_core.documents import Document

from cogtrix_core.utils.atomic_write import atomic_write_json

log = logging.getLogger("cogtrix.api.rag_index")

_INDEX_FILENAME = "index.faiss"
_METADATA_FILENAME = "metadata.json"
_LEGACY_PICKLE_FILENAME = "index.pkl"


def save_faiss_store(store: FAISS, persist_dir: Path) -> None:
    """Persist a FAISS store as raw binary plus JSON metadata."""
    faiss = dependable_faiss_import()
    persist_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store.index, str(persist_dir / _INDEX_FILENAME))

    docstore_dict = getattr(store.docstore, "_dict", {})
    documents: list[dict[str, Any]] = []
    for docstore_id, doc in docstore_dict.items():
        if not isinstance(doc, Document):
            continue
        documents.append(
            {
                "id": docstore_id,
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    metadata = {
        "documents": documents,
        "index_to_docstore_id": {
            str(index): docstore_id for index, docstore_id in store.index_to_docstore_id.items()
        },
    }
    with atomic_write_json(persist_dir / _METADATA_FILENAME) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    (persist_dir / _LEGACY_PICKLE_FILENAME).unlink(missing_ok=True)


def load_faiss_store(persist_dir: Path, embeddings: Any) -> FAISS | None:
    """Load a raw FAISS store from disk (safe: faiss.read_index + JSON metadata).

    Returns None when the raw index or metadata sidecar is missing or invalid.
    """
    index_path = persist_dir / _INDEX_FILENAME
    metadata_path = persist_dir / _METADATA_FILENAME
    if not index_path.exists() or not metadata_path.exists():
        return None

    try:
        faiss = dependable_faiss_import()
        index = faiss.read_index(str(index_path))
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))

        documents_raw = raw.get("documents")
        mapping_raw = raw.get("index_to_docstore_id")
        if not isinstance(documents_raw, list) or not isinstance(mapping_raw, dict):
            raise ValueError("invalid metadata structure")

        docstore: dict[str, Document] = {}
        for item in documents_raw:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("id", ""))
            if not doc_id:
                continue
            page_content = str(item.get("page_content", ""))
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            docstore[doc_id] = Document(page_content=page_content, metadata=metadata)

        index_to_docstore_id = {int(k): str(v) for k, v in mapping_raw.items()}
        return FAISS(embeddings, index, InMemoryDocstore(docstore), index_to_docstore_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to load raw FAISS store from %s: %s", persist_dir, exc)
        return None


def load_faiss_store_safe(persist_dir: Path, embeddings: Any) -> FAISS | None:
    """Load a FAISS store from safe raw-FAISS + JSON format.

    Uses ``load_faiss_store`` (raw faiss.read_index + JSON metadata — no pickle).
    Returns None when no usable safe-format index is found or load fails.

    The legacy ``index.pkl`` pickle format is no longer supported. If only a
    ``.pkl`` file exists (no ``index.faiss`` + ``metadata.json``), the index is
    treated as absent and None is returned. This eliminates the RCE risk from
    ``FAISS.load_local(allow_dangerous_deserialization=True)`` in the legacy
    migration path (issue #930).
    """
    return load_faiss_store(persist_dir, embeddings)
