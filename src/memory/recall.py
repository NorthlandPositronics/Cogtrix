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
from pathlib import Path
from typing import Any

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
        self._index_dir = self._storage_dir / session_id.replace("/", "_")

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
        if not self._ready or not self._embedding_fn:
            return

        texts = self._messages_to_texts(messages)
        if not texts:
            return

        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document

            docs = [Document(page_content=t) for t in texts]
            if self._vectorstore is None:
                self._vectorstore = FAISS.from_documents(docs, self._embedding_fn)
            else:
                self._vectorstore.add_documents(docs)
        except Exception as exc:
            log.warning("Vector recall: failed to add messages: %s", exc)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(self, query: str, k: int = 3) -> list[str]:
        """Return top-k relevant past exchanges for *query*."""
        if not self._ready or self._vectorstore is None:
            return []

        try:
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
        if self._vectorstore is None or not self._ready:
            return

        try:
            self._index_dir.mkdir(parents=True, exist_ok=True)
            self._vectorstore.save_local(str(self._index_dir))

            meta = {"embedding_model": self._embedding_model}
            meta_path = self._index_dir / "meta.json"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except Exception as exc:
            log.warning("Vector recall: save failed: %s", exc)

    def _load_or_reset(self) -> None:
        """Load a persisted index if it exists and is compatible."""
        meta_path = self._index_dir / "meta.json"
        if not meta_path.exists():
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

        # Compatible — try to load
        try:
            from langchain_community.vectorstores import FAISS

            self._vectorstore = FAISS.load_local(
                str(self._index_dir),
                self._embedding_fn,
                allow_dangerous_deserialization=True,
            )
            self._ready = True
        except Exception as exc:
            log.warning("Vector recall: failed to load index: %s", exc)
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
