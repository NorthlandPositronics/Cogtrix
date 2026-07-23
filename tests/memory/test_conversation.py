"""Unit tests for ConversationMemoryManager."""

from datetime import datetime

# Import to trigger registration
from src.memory import modes  # noqa: F401
from src.memory.context import MemoryContext
from src.memory.factory import MemoryFactory
from src.memory.json_store import JsonFileMemoryStore
from src.memory.modes.conversation import ConversationMemoryManager


class MockStore:
    """Mock storage for testing."""

    def __init__(self):
        self.data = {}

    def load_history(self, session_id: str):
        return self.data.get(session_id, [])

    def save_history(self, session_id: str, messages):
        self.data[session_id] = list(messages)


class TestConversationMemoryManager:
    """Tests for ConversationMemoryManager."""

    def test_mode_name(self):
        """Test that mode_name returns 'conversation'."""
        manager = ConversationMemoryManager(MockStore(), "test")
        assert manager.mode_name == "conversation"

    def test_factory_registration(self):
        """Test that conversation mode is registered with factory."""
        assert MemoryFactory.is_registered("conversation")

        store = MockStore()
        manager = MemoryFactory.create("conversation", store, "session")
        assert isinstance(manager, ConversationMemoryManager)

    def test_default_config(self):
        """Test default configuration values."""
        manager = ConversationMemoryManager(MockStore(), "test")
        assert manager._mode_config["working_memory_size"] == 25
        assert manager._mode_config["summary_threshold"] == 35
        assert manager._mode_config["summary_max_age_hours"] is None
        assert manager._mode_config["entity_extraction"] is False
        assert manager._mode_config["rag_enabled"] is False

    def test_custom_config(self):
        """Test custom configuration overrides defaults."""
        config = {"working_memory_size": 10, "custom_option": True}
        manager = ConversationMemoryManager(MockStore(), "test", config)

        assert manager._mode_config["working_memory_size"] == 10
        assert manager._mode_config["custom_option"] is True
        # Default values still present
        assert manager._mode_config["summary_threshold"] == 35

    def test_prepare_context_empty(self):
        """Test prepare_context with no messages."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()

        context = manager.prepare_context("hello")

        assert isinstance(context, MemoryContext)
        assert context.mode == "conversation"
        assert context.messages == []
        assert context.total_messages_stored == 0
        assert context.context_messages_count == 0

    def test_update_adds_messages(self):
        """Test that update adds messages to memory."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()

        manager.update("Hello", "Hi there!")

        assert len(manager._messages) == 2
        assert manager.get_message_count() == 2

    def test_update_and_prepare_context(self):
        """Test update followed by prepare_context."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()

        # Add some turns
        manager.update("Hello", "Hi there!")
        manager.update("How are you?", "I'm doing well!")

        context = manager.prepare_context("next question")

        assert context.total_messages_stored == 4  # 2 turns = 4 messages
        assert context.context_messages_count == 4
        assert len(context.messages) == 4

    def test_cold_cache_preserves_all_messages(self):
        """Test that cold-cache path returns ALL messages (no truncation).

        After the sliding-window amnesia fix (#657), the cold-cache first-turn
        path no longer truncates to working_memory_size. Instead, all messages
        are returned and background summarization compresses them into tier cache.
        """
        config = {"working_memory_size": 4}  # Only keep 4 messages
        manager = ConversationMemoryManager(MockStore(), "test", config)
        manager.load()

        # Add 10 messages (5 turns) - exceeds window size
        for i in range(5):
            manager.update(f"Question {i}", f"Answer {i}")

        # Cold-cache path: all messages returned, no truncation
        context = manager.prepare_context("next")

        assert context.total_messages_stored == 10
        assert context.context_messages_count == 10  # NOT truncated
        assert len(context.messages) == 10

    def test_save_and_load(self):
        """Test persistence via save and load."""
        store = MockStore()

        # Create and populate
        manager1 = ConversationMemoryManager(store, "test-session")
        manager1.load()
        manager1.update("Hello", "Hi!")
        manager1.save()

        # Load in new manager
        manager2 = ConversationMemoryManager(store, "test-session")
        manager2.load()

        context = manager2.prepare_context("next")
        assert context.total_messages_stored == 2

    def test_clear(self):
        """Test clearing memory."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Hello", "Hi!")

        assert len(manager._messages) == 2

        manager.clear()

        assert len(manager._messages) == 0
        assert manager._summary is None
        assert manager._entities == {}

    def test_get_stats(self):
        """Test get_stats returns correct information."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Hello", "Hi!")

        stats = manager.get_stats()

        assert stats["mode"] == "conversation"
        assert stats["total_messages"] == 2
        assert stats["working_memory_size"] == 25
        assert stats["has_summary"] is False
        assert stats["entity_count"] == 0

    def test_get_message_count(self):
        """Test get_message_count returns correct count."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()

        assert manager.get_message_count() == 0

        manager.update("Q1", "A1")
        assert manager.get_message_count() == 2

        manager.update("Q2", "A2")
        assert manager.get_message_count() == 4

    def test_to_dict(self):
        """Test serialization to dictionary."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Question", "Answer")

        data = manager.to_dict()

        assert data["mode"] == "conversation"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["type"] == "human"
        assert data["messages"][0]["content"] == "Question"
        assert data["messages"][1]["type"] == "ai"
        assert data["messages"][1]["content"] == "Answer"

    def test_from_dict(self):
        """Test restoration from dictionary."""
        manager1 = ConversationMemoryManager(MockStore(), "test")
        manager1.load()
        manager1.update("Question", "Answer")

        data = manager1.to_dict()

        # Restore to new manager
        manager2 = ConversationMemoryManager(MockStore(), "test")
        manager2.from_dict(data)

        assert len(manager2._messages) == 2
        assert manager2._loaded is True

    def test_token_estimation(self):
        """Test rough token estimation (content + timestamp overhead)."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        # 100 chars each = 200 total content + ~19 chars timestamp per message
        manager.update("A" * 100, "B" * 100)

        context = manager.prepare_context("next")

        # 200 content chars + 38 timestamp chars ≈ 238 / 4 ≈ 59
        assert context.token_estimate is not None
        assert context.token_estimate > 50
        assert context.token_estimate < 70

    def test_context_prefix_with_summary(self):
        """Test that summary appears in context prefix."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager._summary = "We discussed weather and coding."

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        assert "We discussed weather and coding." in (context.context_prefix or "")

    def test_context_prefix_with_entities(self):
        """Test that entities appear in context prefix."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager._entities = {"user_name": "Alice", "preference": "detailed"}

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        prefix = context.context_prefix or ""
        assert "Alice" in prefix
        assert "detailed" in prefix

    def test_system_prompt_additions(self):
        """Test that conversation mode provides system prompt additions."""
        manager = ConversationMemoryManager(MockStore(), "test")
        additions = manager.get_system_prompt_additions()
        assert additions is not None
        assert isinstance(additions, str)
        assert "conversation mode" in additions.lower()

    def test_metadata_in_context(self):
        """Test that metadata is included in context."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager._summary = "Test summary"
        manager._entities = {"key": "value"}

        context = manager.prepare_context("next")

        assert context.metadata["has_summary"] is True
        assert context.metadata["entity_count"] == 1


class TestConversationModeBackwardCompatibility:
    """Ensure conversation mode maintains backward compatibility."""

    def test_works_with_json_store(self, tmp_path):
        """Test with actual JsonFileMemoryStore."""
        store = JsonFileMemoryStore(str(tmp_path))
        manager = ConversationMemoryManager(store, "compat-test")
        manager.load()

        manager.update("Test input", "Test output")
        manager.save()

        # Verify file was created
        session_file = tmp_path / "compat-test.json"
        assert session_file.exists()

        # Load in new manager
        manager2 = ConversationMemoryManager(store, "compat-test")
        manager2.load()

        assert len(manager2._messages) == 2

    def test_multiple_sessions(self, tmp_path):
        """Test that different sessions are isolated."""
        store = JsonFileMemoryStore(str(tmp_path))

        # Session 1
        m1 = ConversationMemoryManager(store, "session-1")
        m1.load()
        m1.update("Q1", "A1")
        m1.save()

        # Session 2
        m2 = ConversationMemoryManager(store, "session-2")
        m2.load()
        m2.update("Q2", "A2")
        m2.update("Q3", "A3")
        m2.save()

        # Reload and verify isolation
        m1_reload = ConversationMemoryManager(store, "session-1")
        m1_reload.load()
        assert m1_reload.get_message_count() == 2

        m2_reload = ConversationMemoryManager(store, "session-2")
        m2_reload.load()
        assert m2_reload.get_message_count() == 4


class TestFactoryCreation:
    """Test creating conversation manager through factory."""

    def test_create_with_default_config(self):
        """Test factory creation with no config."""
        manager = MemoryFactory.create("conversation", MockStore(), "test-session")
        assert isinstance(manager, ConversationMemoryManager)
        assert manager._mode_config["working_memory_size"] == 25

    def test_create_with_custom_config(self):
        """Test factory creation with custom config."""
        config = {"working_memory_size": 50}
        manager = MemoryFactory.create("conversation", MockStore(), "test-session", config)
        assert manager._mode_config["working_memory_size"] == 50

    def test_available_modes_includes_conversation(self):
        """Test that conversation is in available modes."""
        available = MemoryFactory.available_modes()
        assert "conversation" in available


class TestTimestamps:
    """Tests for message timestamp functionality."""

    def test_update_stamps_messages(self):
        """Test that update attaches timestamps to messages."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Hello", "Hi!")

        for msg in manager._messages:
            ts = manager._get_msg_ts(msg)
            assert ts is not None
            datetime.fromisoformat(ts)

    def test_prepare_context_injects_timestamps(self):
        """Timestamps are prepended to HumanMessages only (not AI — the LLM mimics them)."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Hello", "Hi!")

        context = manager.prepare_context("next")

        for msg in context.messages:
            content = msg.content if hasattr(msg, "content") else msg["content"]
            if type(msg).__name__ == "HumanMessage":
                assert content.startswith("[")
                assert "] " in content
            elif type(msg).__name__ == "AIMessage":
                assert not content.startswith("[")

    def test_to_dict_preserves_timestamps(self):
        """Test that serialization includes timestamps."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Hello", "Hi!")

        data = manager.to_dict()
        for msg_data in data["messages"]:
            assert "timestamp" in msg_data
            datetime.fromisoformat(msg_data["timestamp"])

    def test_from_dict_restores_timestamps(self):
        """Test that deserialization restores timestamps."""
        manager1 = ConversationMemoryManager(MockStore(), "test")
        manager1.load()
        manager1.update("Q", "A")

        data = manager1.to_dict()

        manager2 = ConversationMemoryManager(MockStore(), "test")
        manager2.from_dict(data)

        for msg in manager2._messages:
            ts = manager2._get_msg_ts(msg)
            assert ts is not None

    def test_backward_compat_no_timestamps(self):
        """Test loading data without timestamps (backward compatibility)."""
        manager = ConversationMemoryManager(MockStore(), "test")
        data = {
            "mode": "conversation",
            "version": 1,
            "session_id": "test",
            "config": {},
            "messages": [
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi!"},
            ],
            "summary": None,
            "entities": {},
            "topics": [],
        }

        manager.from_dict(data)

        context = manager.prepare_context("next")
        assert len(context.messages) == 2
        # Messages without timestamps pass through unmodified
        content = (
            context.messages[0].content
            if hasattr(context.messages[0], "content")
            else context.messages[0]["content"]
        )
        assert not content.startswith("[")

    def test_json_store_roundtrip_preserves_timestamps(self, tmp_path):
        """Test that save/load through JsonFileMemoryStore keeps timestamps."""
        store = JsonFileMemoryStore(str(tmp_path))

        m1 = ConversationMemoryManager(store, "ts-test")
        m1.load()
        m1.update("Hello", "Hi!")
        m1.save()

        m2 = ConversationMemoryManager(store, "ts-test")
        m2.load()

        for msg in m2._messages:
            ts = m2._get_msg_ts(msg)
            assert ts is not None
            datetime.fromisoformat(ts)

    # --- Layer-1a: token-based summary TTL (#380) ------------------------

    def test_tokens_since_summary_increments_on_update(self):
        """Each update() call adds tokens from human + ai messages."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        assert manager._tokens_since_summary == 0

        manager.update("Hello there", "Hi!")
        # "Hello there" = 11 chars → 5 tokens; "Hi!" = 3 chars → 1 token
        expected = 5 + 1
        assert manager._tokens_since_summary == expected

        manager.update("How are you today?", "I am fine, thanks.")
        # "How are you today?" = 18 chars → 9 tokens; "I am fine, thanks." = 18 chars → 9 tokens
        expected += 9 + 9
        assert manager._tokens_since_summary == expected

    def test_token_ttl_resets_summary_at_threshold(self):
        """When _tokens_since_summary reaches threshold, summary resets."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager._mode_config["summary_max_uncovered_tokens"] = 10
        manager._summary = "Old summary"
        manager._summary_last_updated_at = datetime.now()

        # One large message pushes us over the 10-token threshold
        manager.update("x" * 40, "y" * 40)  # 20 + 20 = 40 tokens
        assert manager._summary is None
        assert manager._tokens_since_summary == 0

    def test_token_ttl_does_not_reset_below_threshold(self):
        """Summary survives when token churn is below threshold."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager._mode_config["summary_max_uncovered_tokens"] = 100
        manager._summary = "Old summary"
        manager._summary_last_updated_at = datetime.now()

        manager.update("Hello", "Hi!")  # 2 + 1 = 3 tokens
        assert manager._summary == "Old summary"
        assert manager._tokens_since_summary == 3

    def test_reset_summary_state_clears_token_counter(self):
        """_reset_summary_state() must zero _tokens_since_summary."""
        manager = ConversationMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Hello", "Hi!")
        assert manager._tokens_since_summary > 0
        manager._reset_summary_state()
        assert manager._tokens_since_summary == 0

    def test_default_config_includes_summary_max_uncovered_tokens(self):
        """DEFAULT_CONFIG includes summary_max_uncovered_tokens."""
        manager = ConversationMemoryManager(MockStore(), "test")
        assert manager._mode_config["summary_max_uncovered_tokens"] == 16_000

    def test_cold_cache_first_turn_preserves_all_messages(self):
        """Regression test for sliding-window amnesia (#657).

        The cold-cache first-turn path was truncating messages to working_memory_size
        BEFORE summarization completed, causing message loss. After the fix, all messages
        should be returned on cold-cache paths (background summarization will compress
        them into tier cache on subsequent turns).
        """
        # Use a small working memory size (3 messages) to trigger the cold-cache path
        config = {"working_memory_size": 3}
        manager = ConversationMemoryManager(MockStore(), "test", config)
        manager.load()

        # Add 5 turns (10 messages total) - this exceeds the window size
        for i in range(5):
            manager.update(f"Question {i}", f"Answer {i}")

        # First prepare_context call (cold cache) should return ALL messages,
        # not truncate to working_memory_size
        context = manager.prepare_context("next")

        # All messages should be present (no truncation on cold-cache path)
        assert context.total_messages_stored == 10
        assert context.context_messages_count == 10  # NOT truncated to 3
        assert len(context.messages) == 10

        # Verify the oldest messages are present (not dropped)
        first_msg = context.messages[0]
        first_content = first_msg.content if hasattr(first_msg, "content") else first_msg["content"]
        assert "Question 0" in first_content

        # Verify the newest messages are present
        last_msg = context.messages[-1]
        last_content = last_msg.content if hasattr(last_msg, "content") else last_msg["content"]
        assert "Answer 4" in last_content

    def test_cold_cache_preserves_all_messages_with_larger_window(self):
        """Verify the fix works with different window sizes."""
        # Test with working_memory_size=5 and 15 messages (3 turns)
        config = {"working_memory_size": 5}
        manager = ConversationMemoryManager(MockStore(), "test", config)
        manager.load()

        for i in range(3):
            manager.update(f"Q{i}", f"A{i}")

        context = manager.prepare_context("next")

        # All 6 messages (3 turns) should be present
        assert context.total_messages_stored == 6
        assert context.context_messages_count == 6
        assert len(context.messages) == 6

        # Verify oldest message is present (was dropped in the bug)
        first_content = (
            context.messages[0].content
            if hasattr(context.messages[0], "content")
            else context.messages[0]["content"]
        )
        assert "Q0" in first_content
