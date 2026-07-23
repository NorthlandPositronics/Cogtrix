"""#2142 — SharedKnowledgeStore.recall() must scope to the source chat.

Facts are stored with a ``source_session`` tag but ``recall()`` used to ignore
it, so a fact learned from contact A could surface in contact B's conversation
(cross-contact PII leak). Recall now filters by ``source_session`` (strict
per-chat isolation), over-fetching FAISS candidates so the per-chat filter
doesn't starve semantic recall.

All construction I/O is mocked (no FAISS, no embeddings, no network).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.assistant.knowledge import Fact, SharedKnowledgeStore, _compute_hash


def _make_store() -> SharedKnowledgeStore:
    config = MagicMock()
    config.services = {"assistant": {"knowledge": {}}}
    with (
        patch.object(SharedKnowledgeStore, "_load", return_value=None),
        patch.object(SharedKnowledgeStore, "_setup_embeddings", return_value=None),
        patch("src.assistant.knowledge.threading.Thread"),
    ):
        store = SharedKnowledgeStore(config=config, llm=MagicMock())
    store._embeddings_ready.set()
    return store


def _fact(entity: str, text: str, session: str) -> Fact:
    return Fact(
        entity=entity,
        fact=text,
        source_session=session,
        timestamp=time.time(),
        fact_hash=_compute_hash(entity, text),
    )


class TestRecallPerChatIsolation:
    """Keyword path (no FAISS) — deterministic exact-scoping checks."""

    def _store(self) -> SharedKnowledgeStore:
        store = _make_store()
        store._vectorstore = None  # force the keyword recall path
        store._facts = [
            _fact("Alice", "is a vet in Portland", "whatsapp:AAA"),
            _fact("Bob", "is a vet in Portland", "whatsapp:BBB"),
        ]
        return store

    def test_recall_scoped_to_own_chat_only(self) -> None:
        out = self._store().recall("vet Portland", k=5, source_session="whatsapp:AAA")
        assert out is not None
        assert "Alice" in out
        assert "Bob" not in out  # the other contact's fact must NOT leak in

    def test_recall_other_chat_isolated(self) -> None:
        out = self._store().recall("vet Portland", k=5, source_session="whatsapp:BBB")
        assert out is not None
        assert "Bob" in out
        assert "Alice" not in out

    def test_unknown_session_recalls_nothing(self) -> None:
        assert self._store().recall("vet Portland", k=5, source_session="whatsapp:CCC") is None

    def test_none_session_is_unscoped_global(self) -> None:
        # Backward compatibility for non-assistant callers (CLI/tests).
        out = self._store().recall("vet Portland", k=5, source_session=None)
        assert out is not None
        assert "Alice" in out and "Bob" in out
