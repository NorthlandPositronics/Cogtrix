"""#2250 — knowledge fact-extraction must not crash when the LLM returns null content.

``_extract_facts`` called ``.strip()`` on ``response.content``, which can be
``None`` for an ``AIMessage`` (tool-call-only / empty-content message) →
``AttributeError: 'NoneType' object has no attribute 'strip'``. It now coerces
``None`` (and non-str/multimodal content) before ``.strip()`` and degrades to
"no facts".
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.assistant.knowledge import SharedKnowledgeStore


def _make_store() -> SharedKnowledgeStore:
    """Lightweight store with all construction I/O patched out (no FAISS/network)."""
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


def _store_with_response(response) -> SharedKnowledgeStore:
    store = _make_store()
    store._extraction_llm = MagicMock()
    store._extraction_llm.invoke = MagicMock(return_value=response)
    return store


class TestExtractionNullContent:
    def test_none_content_returns_empty_no_crash(self) -> None:
        # The #2250 repro: AIMessage-like response whose .content is None.
        store = _store_with_response(SimpleNamespace(content=None))
        assert store._extract_facts("user said something", "agent replied") == []

    def test_list_content_coerced_no_crash(self) -> None:
        # Multimodal/list content must be stringified, not .strip()-ed directly.
        store = _store_with_response(SimpleNamespace(content=[{"type": "text", "text": "x"}]))
        assert store._extract_facts("u", "a") == []

    def test_no_content_attribute_uses_str(self) -> None:
        # An object without a .content attr falls back to str(response).
        store = _store_with_response("[]")
        assert store._extract_facts("u", "a") == []

    def test_valid_json_array_still_parses(self) -> None:
        resp = SimpleNamespace(content='[{"entity": "Alice", "fact": "Is a vet in Portland"}]')
        store = _store_with_response(resp)
        facts = store._extract_facts("u", "a")
        assert facts == [{"entity": "Alice", "fact": "Is a vet in Portland"}]

    def test_empty_string_content_returns_empty(self) -> None:
        store = _store_with_response(SimpleNamespace(content="   "))
        assert store._extract_facts("u", "a") == []
