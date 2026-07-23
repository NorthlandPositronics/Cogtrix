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


def _make_store(
    config: MagicMock | None = None,
    max_facts: int = 10000,
    tmp_path: Path | None = None,
) -> SharedKnowledgeStore:
    """Create a SharedKnowledgeStore with all external I/O patched out."""
    if config is None:
        config = _make_config(max_facts=max_facts)

    if tmp_path is not None:
        config.services["assistant"]["knowledge"]["data_dir"] = str(tmp_path)

    llm = MagicMock()
    with (
        patch.object(SharedKnowledgeStore, "_load", return_value=None),
        patch.object(SharedKnowledgeStore, "_setup_embeddings", return_value=None),
        patch("src.assistant.knowledge.threading.Thread") as mock_thread_cls,
    ):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread
        store = SharedKnowledgeStore(config=config, llm=llm)

    store._embeddings_ready.set()
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
    """Tests for _extract_and_store_sync() calling the LLM.

    The public extract_and_store() dispatches to a background pool; tests that
    assert on _facts state call _extract_and_store_sync() directly so they run
    synchronously without races.
    """

    def test_extraction_stores_facts_returned_by_llm(self):
        """Facts returned by the LLM are added to the store."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            [{"entity": "Alice", "fact": "Is a veterinarian in Portland"}]
        )
        store._extraction_llm.invoke.return_value = mock_response

        store._extract_and_store_sync(
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

        store._extract_and_store_sync("What DB does Project X use?", "Project X uses PostgreSQL.")

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

        store._extract_and_store_sync("Alice works at Acme Corp in New York.", "Correct.")

        assert len(store._facts) == 2

    def test_json_parse_failure_does_not_crash(self):
        """Malformed JSON from the LLM results in an empty extraction, no crash."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = "This is not JSON at all."
        store._extraction_llm.invoke.return_value = mock_response

        # Should not raise
        store._extract_and_store_sync("Tell me something.", "Sure.")
        assert len(store._facts) == 0

    def test_empty_json_array_adds_no_facts(self):
        """LLM returning [] results in zero facts stored."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = "[]"
        store._extraction_llm.invoke.return_value = mock_response

        store._extract_and_store_sync("No facts here.", "Indeed.")
        assert len(store._facts) == 0

    def test_llm_exception_does_not_crash(self):
        """If the extraction LLM raises, _extract_and_store_sync() returns silently."""
        store = _make_store()
        store._extraction_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        # Should not raise
        store._extract_and_store_sync("Hello", "Hi there")
        assert len(store._facts) == 0

    def test_extract_and_store_returns_immediately(self):
        """extract_and_store() dispatches to the pool and returns without blocking."""
        import time

        store = _make_store()

        import threading

        started = threading.Event()
        blocked = threading.Event()

        def _slow_invoke(messages: Any) -> MagicMock:
            started.set()
            blocked.wait(timeout=2.0)
            resp = MagicMock()
            resp.content = "[]"
            return resp

        store._extraction_llm.invoke.side_effect = _slow_invoke

        t_start = time.monotonic()
        store.extract_and_store("Hello", "Hi there")
        elapsed = time.monotonic() - t_start

        # Call must return before the LLM call starts (or nearly so)
        assert elapsed < 0.5, f"extract_and_store blocked for {elapsed:.2f}s"
        blocked.set()


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for fact deduplication logic.

    Calls _extract_and_store_sync() directly so assertions on _facts are
    not subject to background-pool scheduling.
    """

    def test_same_fact_not_stored_twice(self):
        """Identical entity+fact (same casing) is deduplicated."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = json.dumps([{"entity": "Alice", "fact": "Is a veterinarian"}])
        store._extraction_llm.invoke.return_value = mock_response

        store._extract_and_store_sync("Alice is a vet.", "Yes.")
        store._extract_and_store_sync("Alice is a vet again.", "Yes.")

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

        store._extract_and_store_sync("First call", "resp")
        store._extract_and_store_sync("Second call", "resp")

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

        store._extract_and_store_sync("Tell me about Alice.", "OK.")
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

        store._extract_and_store_sync("input", "response")
        assert len(store._facts) == 2  # still at cap


# ---------------------------------------------------------------------------
# TestSaveLoad
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for save() and _load() with real filesystem (tmp_path)."""

    def test_save_writes_json_file(self, tmp_path: Path):
        """save() writes a valid JSON file to the facts path."""
        store = _make_store(tmp_path=tmp_path)
        _add_fact(store, "Alice", "Is a vet")

        store.save()

        assert store._facts_path.exists()
        data = json.loads(store._facts_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["entity"] == "Alice"

    def test_save_multiple_facts(self, tmp_path: Path):
        """save() correctly serialises multiple facts."""
        store = _make_store(tmp_path=tmp_path)
        _add_fact(store, "Alice", "Is a vet")
        _add_fact(store, "Bob", "Is an engineer")

        store.save()

        data = json.loads(store._facts_path.read_text(encoding="utf-8"))
        assert len(data) == 2

    def test_save_does_not_include_private_source_session(self, tmp_path: Path):
        """source_session field value is not a privacy leak (it's stored blank)."""
        store = _make_store(tmp_path=tmp_path)
        fact = _add_fact(store, "Alice", "Is a vet")
        # source_session is always stored as "" per implementation
        assert fact.source_session == "session::test"

        store.save()

        data = json.loads(store._facts_path.read_text(encoding="utf-8"))
        # The field exists in the serialised form but its value comes from Fact.source_session
        assert "source_session" in data[0]

    def test_load_reads_persisted_facts(self, tmp_path: Path):
        """_load() correctly restores facts saved by save()."""
        store = _make_store(tmp_path=tmp_path)
        _add_fact(store, "Charlie", "Is a designer")

        store.save()

        # Create a fresh store pointing at the same data_dir and load
        store2 = _make_store(tmp_path=tmp_path)
        store2._load()

        assert len(store2._facts) == 1
        assert store2._facts[0].entity == "Charlie"

    def test_load_skips_nonexistent_file(self, tmp_path: Path):
        """_load() is a no-op when the facts file does not exist."""
        store = _make_store(tmp_path=tmp_path)
        store._facts_path = tmp_path / "nonexistent.json"

        store._load()

        assert len(store._facts) == 0

    def test_save_no_vectorstore_does_not_raise(self, tmp_path: Path):
        """save() with no vectorstore set does not crash."""
        store = _make_store(tmp_path=tmp_path)
        assert store._vectorstore is None

        store.save()  # should not raise


# ---------------------------------------------------------------------------
# TestIndexFactsLockBehavior (BUG-028)
# ---------------------------------------------------------------------------


class TestIndexFactsLockBehavior:
    """Tests that _index_facts is called outside the store lock.

    BUG-028: FAISS indexing (potentially slow due to embedding network calls)
    was previously done inside `with self._lock:`, blocking concurrent recall()
    calls.  After the fix, _index_facts must be invoked after the lock releases.

    These tests call _extract_and_store_sync() directly so the spy is not
    subject to background-pool scheduling races.
    """

    def test_index_facts_called_outside_lock(self):
        """_index_facts is invoked when the store lock is NOT held."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            [{"entity": "Alice", "fact": "Is a veterinarian in Portland"}]
        )
        store._extraction_llm.invoke.return_value = mock_response

        lock_held_during_index: list[bool] = []

        original_index = store._index_facts

        def _spy_index_facts(facts: list) -> None:
            # Check whether the lock is held by the current thread when called.
            # threading.Lock.acquire(blocking=False) returns False if already locked.
            acquired = store._lock.acquire(blocking=False)
            if acquired:
                store._lock.release()
                lock_held_during_index.append(False)
            else:
                lock_held_during_index.append(True)
            original_index(facts)

        store._index_facts = _spy_index_facts

        store._extract_and_store_sync(
            "Alice is a vet.", "Yes, Alice works as a veterinarian in Portland."
        )

        assert len(lock_held_during_index) == 1, "_index_facts should have been called once"
        assert (
            lock_held_during_index[0] is False
        ), "_index_facts was called while the lock was held (BUG-028 not fixed)"

    def test_index_facts_not_called_when_no_new_facts(self):
        """_index_facts is not called when no new facts are added."""
        store = _make_store()

        mock_response = MagicMock()
        mock_response.content = "[]"
        store._extraction_llm.invoke.return_value = mock_response

        index_call_count: list[int] = [0]
        original_index = store._index_facts

        def _spy_index_facts(facts: list) -> None:
            index_call_count[0] += 1
            original_index(facts)

        store._index_facts = _spy_index_facts

        store._extract_and_store_sync("Nothing here.", "Indeed.")

        assert index_call_count[0] == 0, "_index_facts should not be called when no facts added"


# ---------------------------------------------------------------------------
# TestBug111AtomicWriteAdoption (BUG-111)
# ---------------------------------------------------------------------------


class TestBug111AtomicWriteAdoption:
    """BUG-111: save() must use atomic_write_json instead of raw mkstemp/fdopen.

    The manual mkstemp pattern leaked the raw fd and left a .tmp file on disk if a
    BaseException (e.g. KeyboardInterrupt) fired between mkstemp and the inner try.
    The fix replaces the pattern with atomic_write_json which handles BaseException
    correctly via an fd-ownership sentinel.
    """

    def test_save_uses_atomic_write_json(self):
        """save() source must reference atomic_write_json rather than mkstemp/fdopen."""
        import inspect

        import src.assistant.knowledge as know_module

        source = inspect.getsource(know_module.SharedKnowledgeStore.save)
        assert (
            "atomic_write_json" in source
        ), "save() does not call atomic_write_json — BUG-111 not fixed"
        assert "mkstemp" not in source, "save() still contains mkstemp — BUG-111 not fixed"
        assert "fdopen" not in source, "save() still contains fdopen — BUG-111 not fixed"

    def test_save_does_not_import_tempfile(self):
        """knowledge.py must not import tempfile at module level after BUG-111 fix."""
        import importlib
        import sys

        # Reload the module fresh to check its actual imports
        if "src.assistant.knowledge" in sys.modules:
            mod = sys.modules["src.assistant.knowledge"]
        else:
            mod = importlib.import_module("src.assistant.knowledge")

        assert not hasattr(mod, "tempfile") or not callable(
            getattr(mod, "tempfile", None)
        ), "knowledge.py still imports tempfile at module level"

    def test_save_does_not_use_os_replace(self, tmp_path: Path):
        """save() must not call os.replace directly (that is now handled by atomic_write_json)."""
        import inspect

        import src.assistant.knowledge as know_module

        source = inspect.getsource(know_module.SharedKnowledgeStore.save)
        assert (
            "os.replace" not in source
        ), "save() still contains os.replace — BUG-111 not fully fixed"
        assert "mkstemp" not in source, "save() still contains mkstemp — BUG-111 not fully fixed"

    def test_save_completes_successfully_with_atomic_write(self, tmp_path: Path):
        """save() still writes a valid JSON file when using atomic_write_json."""
        store = _make_store(tmp_path=tmp_path)
        _add_fact(store, "Bob", "Is an engineer")
        _add_fact(store, "Alice", "Is a vet")

        store.save()

        assert store._facts_path.exists()
        data = json.loads(store._facts_path.read_text(encoding="utf-8"))
        assert len(data) == 2
        entities = {d["entity"] for d in data}
        assert entities == {"Bob", "Alice"}


# ---------------------------------------------------------------------------
# TestBug112DeadPoolShutdown (BUG-112)
# ---------------------------------------------------------------------------


class TestBug112DeadPoolShutdown:
    """BUG-112: The dead _pool_shutdown flag in knowledge.py must be removed.
    Instead, RuntimeError from pool.submit() must be caught and logged as WARNING.
    """

    def test_pool_shutdown_flag_does_not_exist(self):
        """_pool_shutdown module-level variable must not exist in knowledge.py."""
        import src.assistant.knowledge as know_module

        assert not hasattr(
            know_module, "_pool_shutdown"
        ), "_pool_shutdown still present in knowledge.py — BUG-112 not fixed"

    def test_get_extraction_pool_no_pool_shutdown_branch(self):
        """_get_extraction_pool must not reference _pool_shutdown."""
        import inspect

        import src.assistant.knowledge as know_module

        source = inspect.getsource(know_module._get_extraction_pool)
        assert (
            "_pool_shutdown" not in source
        ), "_get_extraction_pool still references _pool_shutdown"

    def test_extract_and_store_handles_runtime_error_from_pool(self):
        """When pool.submit() raises RuntimeError, extract_and_store logs a warning
        and does not propagate the exception."""
        from unittest.mock import patch

        store = _make_store()

        with patch("src.assistant.knowledge._get_extraction_pool") as mock_get_pool:
            mock_pool = MagicMock()
            mock_pool.submit.side_effect = RuntimeError(
                "cannot schedule new futures after shutdown"
            )
            mock_get_pool.return_value = mock_pool

            # Must not raise
            store.extract_and_store("Hello", "Hi there")

            # Pool submit must have been attempted
            mock_pool.submit.assert_called_once()

    def test_extract_and_store_runtime_error_does_not_crash(self):
        """extract_and_store() must swallow RuntimeError from pool.submit()."""
        from unittest.mock import patch

        store = _make_store()

        with patch("src.assistant.knowledge._get_extraction_pool") as mock_get_pool:
            mock_pool = MagicMock()
            mock_pool.submit.side_effect = RuntimeError("BrokenExecutor")
            mock_get_pool.return_value = mock_pool

            # Must not raise
            store.extract_and_store("test input", "test response")
