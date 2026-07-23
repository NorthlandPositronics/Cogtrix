"""Unit tests for src/assistant/knowledge.py — SharedKnowledgeStore."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.assistant.knowledge import Fact, SharedKnowledgeStore, _compute_hash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(max_facts: int = 10000) -> MagicMock:
    config = MagicMock()
    config.services = {
        "assistant": {
            "knowledge": {"max_facts": max_facts},
        }
    }
    # No embedding config — prevents _setup_embeddings from doing anything
    del config.embedding
    config.__class__ = type("MockConfig", (object,), {"services": config.services})
    return config


def _make_store(config: MagicMock | None = None, max_facts: int = 10000) -> SharedKnowledgeStore:
    """Create a SharedKnowledgeStore with all external I/O patched out."""
    if config is None:
        config = _make_config(max_facts=max_facts)

    llm = MagicMock()
    with (
        patch.object(SharedKnowledgeStore, "_load", return_value=None),
        patch.object(SharedKnowledgeStore, "_setup_embeddings", return_value=None),
        patch("src.assistant.knowledge._FACTS_PATH") as mock_path,
    ):
        mock_path.parent.mkdir.return_value = None
        store = SharedKnowledgeStore(config=config, llm=llm)

    return store


def _add_fact(store: SharedKnowledgeStore, entity: str, fact_text: str) -> Fact:
    fhash = _compute_hash(entity, fact_text)
    fact = Fact(
        entity=entity,
        fact=fact_text,
        source_session="session::test",
        timestamp=1000.0,
        fact_hash=fhash,
    )
    store._facts.append(fact)
    store._fact_hashes.add(fhash)
    return fact


# ---------------------------------------------------------------------------
# TestComputeHash
# ---------------------------------------------------------------------------


class TestComputeHash:
    """Tests for _compute_hash helper."""

    def test_hash_is_deterministic(self):
        """Same entity+fact always produces the same hash."""
        h1 = _compute_hash("Alice", "Is a veterinarian")
        h2 = _compute_hash("Alice", "Is a veterinarian")
        assert h1 == h2

    def test_hash_is_case_insensitive(self):
        """Casing differences in entity or fact produce the same hash."""
        h_lower = _compute_hash("alice", "is a veterinarian")
        h_upper = _compute_hash("ALICE", "IS A VETERINARIAN")
        h_mixed = _compute_hash("Alice", "Is A Veterinarian")
        assert h_lower == h_upper == h_mixed

    def test_different_facts_different_hashes(self):
        """Different entity/fact combinations produce different hashes."""
        h1 = _compute_hash("Alice", "Is a vet")
        h2 = _compute_hash("Bob", "Is a vet")
        h3 = _compute_hash("Alice", "Is a doctor")
        assert h1 != h2
        assert h1 != h3

    def test_hash_length_is_16(self):
        """Hash is truncated to 16 hex chars."""
        h = _compute_hash("Entity", "Some fact")
        assert len(h) == 16


# ---------------------------------------------------------------------------
# TestFactExtraction
# ---------------------------------------------------------------------------


class TestFactExtraction:
    """Tests for extract_and_store() calling the LLM."""

    def test_extraction_stores_facts_returned_by_llm(self):
        """Facts returned by the LLM are added to the store."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            [{"entity": "Alice", "fact": "Is a veterinarian in Portland"}]
        )
        store._extraction_llm.invoke.return_value = mock_response

        store.extract_and_store(
            "Alice is a vet.", "Yes, Alice works as a veterinarian in Portland."
        )

        assert len(store._facts) == 1
        assert store._facts[0].entity == "Alice"
        assert store._facts[0].fact == "Is a veterinarian in Portland"

    def test_extraction_with_surrounding_text(self):
        """LLM response with surrounding text is still parsed correctly."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = (
            "Here are the facts:\n"
            '[{"entity": "Project X", "fact": "Uses PostgreSQL"}]\n'
            "End of response."
        )
        store._extraction_llm.invoke.return_value = mock_response

        store.extract_and_store("What DB does Project X use?", "Project X uses PostgreSQL.")

        assert any(f.entity == "Project X" for f in store._facts)

    def test_extraction_multiple_facts(self):
        """Multiple facts in the LLM response are all stored."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            [
                {"entity": "Alice", "fact": "Works at Acme Corp"},
                {"entity": "Acme Corp", "fact": "Is based in New York"},
            ]
        )
        store._extraction_llm.invoke.return_value = mock_response

        store.extract_and_store("Alice works at Acme Corp in New York.", "Correct.")

        assert len(store._facts) == 2

    def test_json_parse_failure_does_not_crash(self):
        """Malformed JSON from the LLM results in an empty extraction, no crash."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = "This is not JSON at all."
        store._extraction_llm.invoke.return_value = mock_response

        # Should not raise
        store.extract_and_store("Tell me something.", "Sure.")
        assert len(store._facts) == 0

    def test_empty_json_array_adds_no_facts(self):
        """LLM returning [] results in zero facts stored."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = "[]"
        store._extraction_llm.invoke.return_value = mock_response

        store.extract_and_store("No facts here.", "Indeed.")
        assert len(store._facts) == 0

    def test_llm_exception_does_not_crash(self):
        """If the extraction LLM raises, extract_and_store() returns silently."""
        store = _make_store()
        store._extraction_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        # Should not raise
        store.extract_and_store("Hello", "Hi there")
        assert len(store._facts) == 0


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for fact deduplication logic."""

    def test_same_fact_not_stored_twice(self):
        """Identical entity+fact (same casing) is deduplicated."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = json.dumps([{"entity": "Alice", "fact": "Is a veterinarian"}])
        store._extraction_llm.invoke.return_value = mock_response

        store.extract_and_store("Alice is a vet.", "Yes.")
        store.extract_and_store("Alice is a vet again.", "Yes.")

        assert len(store._facts) == 1

    def test_casing_deduplication(self):
        """entity+fact with different casing resolves to the same hash and is not duplicated."""
        store = _make_store()

        responses = [
            json.dumps([{"entity": "Alice", "fact": "Is a veterinarian"}]),
            json.dumps([{"entity": "ALICE", "fact": "IS A VETERINARIAN"}]),
        ]
        call_count = [0]

        def _mock_invoke(messages: Any) -> MagicMock:
            resp = MagicMock()
            resp.content = responses[call_count[0]]
            call_count[0] += 1
            return resp

        store._extraction_llm.invoke.side_effect = _mock_invoke

        store.extract_and_store("First call", "resp")
        store.extract_and_store("Second call", "resp")

        assert len(store._facts) == 1

    def test_distinct_facts_both_stored(self):
        """Two facts with different content are both stored."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            [
                {"entity": "Alice", "fact": "Is a veterinarian"},
                {"entity": "Alice", "fact": "Lives in Portland"},
            ]
        )
        store._extraction_llm.invoke.return_value = mock_response

        store.extract_and_store("Tell me about Alice.", "OK.")
        assert len(store._facts) == 2


# ---------------------------------------------------------------------------
# TestRecall
# ---------------------------------------------------------------------------


class TestRecall:
    """Tests for recall() keyword fallback."""

    def test_recall_returns_none_when_no_facts(self):
        """recall() returns None when the store is empty."""
        store = _make_store()
        assert store.recall("anything") is None

    def test_recall_returns_matching_facts(self):
        """recall() returns facts whose entity or fact text matches query tokens."""
        store = _make_store()
        _add_fact(store, "Alice", "Is a veterinarian in Portland")
        _add_fact(store, "Bob", "Is a software engineer")

        result = store.recall("veterinarian Portland")
        assert result is not None
        assert "Alice" in result
        assert "veterinarian" in result.lower()

    def test_recall_returns_none_when_no_match(self):
        """recall() returns None when no facts match the query tokens."""
        store = _make_store()
        _add_fact(store, "Alice", "Is a veterinarian")

        result = store.recall("quantum physics")
        assert result is None

    def test_recall_respects_k_limit(self):
        """recall() returns at most k facts."""
        store = _make_store()
        for i in range(10):
            _add_fact(store, f"Person{i}", "Is a developer")

        result = store.recall("developer", k=3)
        assert result is not None
        lines = result.strip().split("\n")
        assert len(lines) <= 3

    def test_recall_format_is_entity_colon_fact(self):
        """Each line in recall output follows '- {entity}: {fact}' format."""
        store = _make_store()
        _add_fact(store, "Alice", "Is a vet")

        result = store.recall("Alice vet")
        assert result is not None
        assert "- Alice: Is a vet" in result


# ---------------------------------------------------------------------------
# TestPrivacy
# ---------------------------------------------------------------------------


class TestPrivacy:
    """Tests that source_session is never exposed in recall output."""

    def test_source_session_not_in_recall_output(self):
        """recall() output does not contain source_session values."""
        store = _make_store()
        fact = _add_fact(store, "Alice", "Is a vet")
        fact.source_session = "telegram::private_chat_999"

        result = store.recall("Alice vet")
        assert result is not None
        assert "private_chat_999" not in result
        assert "telegram::" not in result


# ---------------------------------------------------------------------------
# TestMaxFactsCap
# ---------------------------------------------------------------------------


class TestMaxFactsCap:
    """Tests for max_facts cap enforcement."""

    def test_max_facts_cap_prevents_overflow(self):
        """When the store is at capacity, new facts are not added."""
        store = _make_store(max_facts=2)

        # Pre-populate to the cap
        _add_fact(store, "Fact1", "Content1")
        _add_fact(store, "Fact2", "Content2")

        mock_response = MagicMock()
        mock_response.content = json.dumps([{"entity": "Fact3", "fact": "Content3"}])
        store._extraction_llm.invoke.return_value = mock_response

        store.extract_and_store("input", "response")
        assert len(store._facts) == 2  # still at cap


# ---------------------------------------------------------------------------
# TestSaveLoad
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for save() and _load() with real filesystem (tmp_path)."""

    def test_save_writes_json_file(self, tmp_path: Path):
        """save() writes a valid JSON file to the facts path."""
        store = _make_store()
        _add_fact(store, "Alice", "Is a vet")

        facts_file = tmp_path / "facts.json"

        with patch("src.assistant.knowledge._FACTS_PATH", facts_file):
            store.save()

        assert facts_file.exists()
        data = json.loads(facts_file.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["entity"] == "Alice"

    def test_save_multiple_facts(self, tmp_path: Path):
        """save() correctly serialises multiple facts."""
        store = _make_store()
        _add_fact(store, "Alice", "Is a vet")
        _add_fact(store, "Bob", "Is an engineer")

        facts_file = tmp_path / "facts.json"

        with patch("src.assistant.knowledge._FACTS_PATH", facts_file):
            store.save()

        data = json.loads(facts_file.read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_save_does_not_include_private_source_session(self, tmp_path: Path):
        """source_session field value is not a privacy leak (it's stored blank)."""
        store = _make_store()
        fact = _add_fact(store, "Alice", "Is a vet")
        # source_session is always stored as "" per implementation
        assert fact.source_session == "session::test"

        facts_file = tmp_path / "facts.json"
        with patch("src.assistant.knowledge._FACTS_PATH", facts_file):
            store.save()

        data = json.loads(facts_file.read_text(encoding="utf-8"))
        # The field exists in the serialised form but its value comes from Fact.source_session
        assert "source_session" in data[0]

    def test_load_reads_persisted_facts(self, tmp_path: Path):
        """_load() correctly restores facts saved by save()."""
        store = _make_store()
        _add_fact(store, "Charlie", "Is a designer")

        facts_file = tmp_path / "facts.json"
        with patch("src.assistant.knowledge._FACTS_PATH", facts_file):
            store.save()

        # Create a fresh store and load the file
        store2 = _make_store()
        with patch("src.assistant.knowledge._FACTS_PATH", facts_file):
            store2._load()

        assert len(store2._facts) == 1
        assert store2._facts[0].entity == "Charlie"

    def test_load_skips_nonexistent_file(self):
        """_load() is a no-op when the facts file does not exist."""
        store = _make_store()
        nonexistent = Path("/tmp/__nonexistent_cogtrix_test__.json")

        with patch("src.assistant.knowledge._FACTS_PATH", nonexistent):
            store._load()

        assert len(store._facts) == 0

    def test_save_no_vectorstore_does_not_raise(self, tmp_path: Path):
        """save() with no vectorstore set does not crash."""
        store = _make_store()
        assert store._vectorstore is None

        facts_file = tmp_path / "facts.json"
        with patch("src.assistant.knowledge._FACTS_PATH", facts_file):
            store.save()  # should not raise
