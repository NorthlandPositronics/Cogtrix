"""Per-session vector store for semantic recall of past conversation.

Wraps a FAISS index that stores embeddings of conversation message
pairs (human + AI together).  On each query, the user's input is
embedded and the top-k most relevant past exchanges are returned.

The store is fully optional — it degrades gracefully when no embedding
provider is available.  The embedding model name is persisted alongside
the index so that a model change can be detected (and the stale index
discarded).
"""

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from cogtrix_core.api.rag_index import load_faiss_store_safe, save_faiss_store
from cogtrix_core.memory.manager import _sanitize_session_id
from cogtrix_core.utils.atomic_write import atomic_write_json

log = logging.getLogger("cogtrix")


class SessionVectorStore:
    """Per-session FAISS vector store for semantic conversation recall.

    Parameters
    ----------
    session_id:
        Used to derive the on-disk storage path.
    storage_dir:
        Root directory for vector stores (each session gets a sub-dir).
    """

    def __init__(self, session_id: str, storage_dir: str = "data/vectordb/sessions"):
        self._session_id = session_id
        self._storage_dir = Path(storage_dir)
        safe_id = _sanitize_session_id(session_id)
        candidate = (self._storage_dir / safe_id).resolve()
        base_resolved = self._storage_dir.resolve()
        try:
            candidate.relative_to(base_resolved)
        except ValueError:
            raise ValueError(f"Path traversal detected in session_id: {session_id!r}") from None
        self._index_dir = candidate

        self._lock = threading.RLock()
        self._embedding_fn: Any = None
        self._embedding_model: str | None = None
        self._vectorstore: Any = None
        self._ready = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def configure(self, embedding_fn: Any, embedding_model: str) -> None:
        """Set the embedding function and model tag.

        If the model tag differs from what was previously persisted,
        the existing index is discarded automatically.
        """
        with self._lock:
            self._embedding_fn = embedding_fn
            self._embedding_model = embedding_model

            # Load existing index (if compatible)
            self._load_or_reset()

    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Add messages
    # ------------------------------------------------------------------

    def add_messages(self, messages: list[Any]) -> None:
        """Embed a batch of messages and store them.

        Messages are paired into human/AI exchanges.  Tool messages
        and intermediate steps are collapsed into the nearest AI text.
        """
        with self._lock:
            if not self._ready or not self._embedding_fn:
                return

            texts = self._messages_to_texts(messages)
            if not texts:
                return

            t0 = time.monotonic()
            log.debug("Vector recall: embedding %d texts", len(texts))
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_core.documents import Document

                docs = [Document(page_content=t) for t in texts]
                if self._vectorstore is None:
                    self._vectorstore = FAISS.from_documents(docs, self._embedding_fn)
                else:
                    self._vectorstore.add_documents(docs)
                log.debug("Vector recall: add_messages completed in %.2fs", time.monotonic() - t0)
            except Exception as exc:
                log.warning("Vector recall: failed to add messages: %s", exc)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        k: int = 3,
        score_threshold: float | None = None,
    ) -> list[str]:
        """Return top-k relevant past exchanges for *query*.

        When *score_threshold* is provided (0–1, higher = stricter), only
        exchanges that meet the minimum cosine-like similarity are returned.
        This prevents unrelated past exchanges from bleeding into the context
        when FAISS finds no genuinely similar matches (#277).
        """
        with self._lock:
            if not self._ready or self._vectorstore is None:
                return []

            try:
                if score_threshold is not None and score_threshold > 0.0:
                    # Use L2 distance and convert: similarity = 1 / (1 + dist)
                    scored = self._vectorstore.similarity_search_with_score(query, k=k)
                    results = [doc for doc, dist in scored if 1.0 / (1.0 + dist) >= score_threshold]
                else:
                    results = self._vectorstore.similarity_search(query, k=k)
                return [doc.page_content for doc in results]
            except Exception as exc:
                log.warning("Vector recall: search failed: %s", exc)
                return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        with self._lock:
            if self._vectorstore is None or not self._ready:
                return

            try:
                self._index_dir.mkdir(parents=True, exist_ok=True)
                save_faiss_store(self._vectorstore, self._index_dir)

                meta = {"embedding_model": self._embedding_model}
                meta_path = self._index_dir / "meta.json"
                with atomic_write_json(meta_path) as fh:
                    json.dump(meta, fh)
            except Exception as exc:
                log.warning("Vector recall: save failed: %s", exc)

    def _load_or_reset(self) -> None:
        """Load a persisted index if it exists and is compatible."""
        meta_path = self._index_dir / "meta.json"
        if not meta_path.exists():
            if self._index_dir.exists() and any(self._index_dir.iterdir()):
                self._load_index_without_meta()
                return
            self._ready = True
            return

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            self._reset_index()
            self._ready = True
            return

        if meta.get("embedding_model") != self._embedding_model:
            log.info(
                "Embedding model changed (%s -> %s); discarding vector index",
                meta.get("embedding_model"),
                self._embedding_model,
            )
            self._reset_index()
            self._ready = True
            return

        # Compatible — try to load safely
        try:
            self._vectorstore = load_faiss_store_safe(self._index_dir, self._embedding_fn)
            self._ready = True
        except Exception as exc:
            log.warning("Vector recall: failed to load index: %s", exc)
            self._reset_index()
            self._ready = True

    def _load_index_without_meta(self) -> None:
        """Attempt to load a persisted index even if meta.json is missing."""
        try:
            self._vectorstore = load_faiss_store_safe(self._index_dir, self._embedding_fn)
            self._ready = True
        except Exception as exc:
            log.warning("Vector recall: failed to load index without meta: %s", exc)
            self._reset_index()
            self._ready = True

    def _reset_index(self) -> None:
        """Delete existing index on disk."""
        if self._index_dir.exists():
            shutil.rmtree(self._index_dir, ignore_errors=True)
        self._vectorstore = None

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all stored embeddings."""
        with self._lock:
            self._reset_index()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_texts(messages: list[Any]) -> list[str]:
        """Convert messages into text chunks for embedding.

        Groups consecutive human-AI exchanges into single chunks so
        that semantic search retrieves coherent conversation fragments.
        """
        texts: list[str] = []
        current_parts: list[str] = []

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("type", "")
                content = msg.get("content", "")
            elif hasattr(msg, "content"):
                role = type(msg).__name__.replace("Message", "").lower()
                content = msg.content or ""
            else:
                continue

            if role in ("tool", "toolmessage"):
                continue

            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            content = str(content).strip()
            if not content:
                continue

            # Truncate very long content for embedding efficiency
            if len(content) > 1500:
                content = content[:750] + " [...] " + content[-500:]

            label = "User" if role in ("human", "humanmessage") else "Assistant"
            current_parts.append(f"{label}: {content}")

            # End of an exchange (AI turn) → flush
            if label == "Assistant":
                texts.append("\n".join(current_parts))
                current_parts = []

        # Flush any remaining parts
        if current_parts:
            texts.append("\n".join(current_parts))

        return texts
