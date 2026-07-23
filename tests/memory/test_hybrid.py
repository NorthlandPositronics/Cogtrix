"""Tests for hybrid memory: summarization + vector recall."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.memory.json_store import JsonFileMemoryStore
from src.memory.modes.code import CodeDevelopmentMemoryManager
from src.memory.modes.conversation import ConversationMemoryManager
from src.memory.modes.reasoning import ReasoningMemoryManager
from src.memory.recall import SessionVectorStore
from src.memory.summarizer import _format_messages_text, generate_summary

# ── Summarizer unit tests ───────────────────────────────────────────


class TestFormatMessagesText:
    def test_formats_human_and_ai(self):
        msgs = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        text = _format_messages_text(msgs)
        assert "User: Hello" in text
        assert "Assistant: Hi there!" in text

    def test_skips_tool_messages(self):
        msgs = [
            {"type": "tool", "content": "tool output"},
            HumanMessage(content="Hello"),
        ]
        text = _format_messages_text(msgs)
        assert "tool output" not in text
        assert "User: Hello" in text

    def test_skips_empty_content(self):
        msgs = [
            HumanMessage(content=""),
            AIMessage(content="response"),
        ]
        text = _format_messages_text(msgs)
        assert "User:" not in text
        assert "Assistant: response" in text

    def test_truncates_long_messages(self):
        long_text = "x" * 5000
        msgs = [HumanMessage(content=long_text)]
        text = _format_messages_text(msgs)
        assert "[...]" in text
        assert len(text) < 5000


class TestGenerateSummary:
    def test_returns_existing_when_no_messages(self):
        result = generate_summary(MagicMock(), [], "existing summary")
        assert result == "existing summary"

    def test_returns_none_when_no_messages(self):
        result = generate_summary(MagicMock(), [])
        assert result is None

    def test_calls_llm_invoke(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Summary of conversation")

        msgs = [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
        ]

        result = generate_summary(mock_llm, msgs)
        assert result == "Summary of conversation"
        mock_llm.invoke.assert_called_once()

    def test_includes_existing_summary_in_prompt(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Updated summary")

        msgs = [HumanMessage(content="New topic")]
        result = generate_summary(mock_llm, msgs, existing_summary="Old summary")
        assert result == "Updated summary"

        # Verify existing summary was passed in the prompt
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[1].content  # HumanMessage content
        assert "Old summary" in prompt_text

    def test_returns_existing_on_llm_failure(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API error")

        msgs = [HumanMessage(content="Hello")]
        result = generate_summary(mock_llm, msgs, existing_summary="existing")
        assert result == "existing"

    def test_returns_existing_on_empty_response(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")

        msgs = [HumanMessage(content="Hello")]
        result = generate_summary(mock_llm, msgs, existing_summary="existing")
        assert result == "existing"


# ── SessionVectorStore unit tests ───────────────────────────────────


class TestSessionVectorStore:
    def test_not_ready_before_configure(self):
        store = SessionVectorStore("test-session")
        assert not store.ready

    def test_recall_empty_when_not_ready(self):
        store = SessionVectorStore("test-session")
        assert store.recall("query") == []

    def test_messages_to_texts(self):
        msgs = [
            HumanMessage(content="How do I use Python?"),
            AIMessage(content="Python is easy to learn."),
            HumanMessage(content="What about Java?"),
            AIMessage(content="Java is also popular."),
        ]
        texts = SessionVectorStore._messages_to_texts(msgs)
        assert len(texts) == 2
        assert "User: How do I use Python?" in texts[0]
        assert "Assistant: Python is easy to learn." in texts[0]
        assert "User: What about Java?" in texts[1]

    def test_messages_to_texts_skips_tool_messages(self):
        msgs = [
            HumanMessage(content="Search for X"),
            {"type": "tool", "content": "results..."},
            AIMessage(content="Found it."),
        ]
        texts = SessionVectorStore._messages_to_texts(msgs)
        assert len(texts) == 1
        assert "results..." not in texts[0]

    def test_messages_to_texts_truncates_long_content(self):
        msgs = [
            HumanMessage(content="Q"),
            AIMessage(content="A" * 3000),
        ]
        texts = SessionVectorStore._messages_to_texts(msgs)
        assert len(texts) == 1
        assert "[...]" in texts[0]

    def test_clear_resets(self, tmp_path):
        store = SessionVectorStore("test", storage_dir=str(tmp_path))
        store.clear()
        assert store._vectorstore is None


# ── Hybrid integration in memory modes ──────────────────────────────


class TestHybridMemoryIntegration:
    """Test that hybrid features integrate correctly with memory modes."""

    def _make_manager(self, mode="conversation", tmp_path=None):
        store = JsonFileMemoryStore(base_dir=str(tmp_path) if tmp_path else "/tmp/test_hybrid")
        if mode == "conversation":
            mgr = ConversationMemoryManager(store, "test-session", {"working_memory_size": 4})
        elif mode == "code":
            mgr = CodeDevelopmentMemoryManager(store, "test-session", {"working_memory_size": 4})
        else:
            mgr = ReasoningMemoryManager(store, "test-session", {"working_memory_size": 4})
        return mgr

    def test_no_summary_when_no_llm(self):
        mgr = self._make_manager()
        for i in range(10):
            mgr.update(f"msg {i}", f"response {i}")
        assert mgr._summary is None

    def test_summary_generated_after_threshold(self):
        mgr = self._make_manager()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Summary: user asked about several topics."
        )
        mgr.set_llm(mock_llm)

        # Window is 4, batch threshold is 6.
        # Need at least 4 + 6 = 10 messages to trigger summary.
        for i in range(12):
            mgr.update(f"msg {i}", f"response {i}")

        # Summarization runs in a background thread; wait for it.
        mgr.join_background()

        assert mgr._summary is not None
        assert "Summary" in mgr._summary

    def test_summary_injected_into_context(self):
        mgr = self._make_manager()
        mgr._summary = "User discussed Python programming."
        ctx = mgr.prepare_context("new question")

        assert ctx.context_prefix is not None
        assert "Python programming" in ctx.context_prefix

    def test_summary_persists_via_to_dict(self):
        mgr = self._make_manager()
        mgr._summary = "Test summary"
        mgr._summary_msg_idx = 5

        data = mgr.to_dict()
        assert data["_summary"] == "Test summary"
        assert data["_summary_msg_idx"] == 5

    def test_summary_restores_via_from_dict(self):
        mgr = self._make_manager()
        data = {
            "mode": "conversation",
            "version": 1,
            "session_id": "test-session",
            "config": {},
            "messages": [],
            "_summary": "Restored summary",
            "_summary_msg_idx": 3,
        }
        mgr.from_dict(data)
        assert mgr._summary == "Restored summary"
        assert mgr._summary_msg_idx == 3

    def test_clear_resets_summary(self):
        mgr = self._make_manager()
        mgr._summary = "Some summary"
        mgr._summary_msg_idx = 5
        mgr.clear()
        assert mgr._summary is None
        assert mgr._summary_msg_idx == 0

    def test_set_llm_replaces_llm(self):
        mgr = self._make_manager()
        llm1 = MagicMock()
        llm2 = MagicMock()
        mgr.set_llm(llm1)
        assert mgr._llm is llm1
        mgr.set_llm(llm2)
        assert mgr._llm is llm2

    def test_summarization_disabled_via_config(self):
        store = JsonFileMemoryStore(base_dir="/tmp/test_hybrid_disabled")
        mgr = ConversationMemoryManager(
            store,
            "test-session",
            {"working_memory_size": 4, "summarization": False},
        )
        mock_llm = MagicMock()
        mgr.set_llm(mock_llm)

        for i in range(15):
            mgr.update(f"msg {i}", f"response {i}")

        # LLM should never have been called for summarization
        mock_llm.invoke.assert_not_called()
        assert mgr._summary is None

    def test_code_mode_has_hybrid_prefix(self):
        mgr = self._make_manager(mode="code")
        mgr._summary = "Previously discussed file refactoring."
        ctx = mgr.prepare_context("next step")
        assert ctx.context_prefix is not None
        assert "file refactoring" in ctx.context_prefix

    def test_reasoning_mode_has_hybrid_prefix(self):
        mgr = self._make_manager(mode="reasoning")
        mgr._summary = "Strategic analysis of market trends."
        ctx = mgr.prepare_context("what next?")
        assert ctx.context_prefix is not None
        assert "market trends" in ctx.context_prefix

    def test_stats_include_hybrid_info(self):
        mgr = self._make_manager()
        mgr._summary = "test"
        mgr._summary_msg_idx = 3
        stats = mgr.get_stats()
        assert stats["has_summary"] is True
        assert stats["summary_coverage"] == 3

    def test_legacy_summary_key_migrated(self):
        """Old sessions stored summary under 'summary' key."""
        mgr = self._make_manager()
        data = {
            "mode": "conversation",
            "version": 1,
            "session_id": "test-session",
            "config": {},
            "messages": [],
            "summary": "Legacy summary text",
        }
        mgr.from_dict(data)
        assert mgr._summary == "Legacy summary text"


class TestHybridPersistence:
    """Test that hybrid state survives save/load round-trips."""

    def test_save_load_round_trip(self, tmp_path):
        """Summary state persists across save() / load() cycle."""
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mgr = ConversationMemoryManager(store, "persist-test", {"working_memory_size": 4})

        # Populate some messages and set summary state
        for i in range(5):
            mgr.update(f"msg {i}", f"response {i}")
        mgr._summary = "Previously discussed Python."
        mgr._summary_msg_idx = 6
        mgr.save()

        # Create a fresh manager and load
        mgr2 = ConversationMemoryManager(store, "persist-test", {"working_memory_size": 4})
        mgr2.load()

        assert mgr2._summary == "Previously discussed Python."
        assert mgr2._summary_msg_idx == 6

    def test_clear_removes_meta_file(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mgr = ConversationMemoryManager(store, "clear-test", {"working_memory_size": 4})
        mgr._summary = "To be cleared"
        mgr._summary_msg_idx = 3
        mgr.save()

        meta_path = mgr._hybrid_meta_path()
        assert meta_path.exists()

        mgr.clear()
        assert not meta_path.exists()
        assert mgr._summary is None
        assert mgr._summary_msg_idx == 0

    def test_load_without_meta_file(self, tmp_path):
        """Loading when no meta file exists is graceful."""
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mgr = ConversationMemoryManager(store, "no-meta-test")
        mgr.load()
        assert mgr._summary is None
        assert mgr._summary_msg_idx == 0


class TestSummaryIdxClamping:
    """Test that _clamp_summary_idx handles stale indices."""

    def test_clamp_after_sanitization(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mgr = ConversationMemoryManager(store, "clamp-test", {"working_memory_size": 4})

        # Add 10 messages, then simulate a stale summary index
        for i in range(5):
            mgr.update(f"msg {i}", f"response {i}")
        mgr._summary = "Old summary"
        mgr._summary_msg_idx = 20  # Way past actual message count
        mgr.save()

        # Reload — should clamp
        mgr2 = ConversationMemoryManager(store, "clamp-test", {"working_memory_size": 4})
        mgr2.load()

        assert mgr2._summary == "Old summary"
        assert mgr2._summary_msg_idx <= mgr2.get_message_count()

    def test_clamp_no_op_when_valid(self, tmp_path):
        store = JsonFileMemoryStore(base_dir=str(tmp_path))
        mgr = ConversationMemoryManager(store, "clamp-valid", {"working_memory_size": 4})

        for i in range(5):
            mgr.update(f"msg {i}", f"response {i}")
        mgr._summary = "Valid summary"
        mgr._summary_msg_idx = 3  # Within bounds
        mgr.save()

        mgr2 = ConversationMemoryManager(store, "clamp-valid", {"working_memory_size": 4})
        mgr2.load()
        assert mgr2._summary_msg_idx == 3  # Unchanged
