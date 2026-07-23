"""Unit tests for memory system foundation."""

import pytest

from src.memory.base import BaseMemoryStore
from src.memory.context import MemoryContext
from src.memory.factory import MemoryFactory
from src.memory.manager import BaseMemoryManager, _sanitize_session_id


class MockStore(BaseMemoryStore):
    """Mock storage for testing."""

    def __init__(self):
        self.data = {}

    def load_history(self, session_id: str):
        return self.data.get(session_id, [])

    def save_history(self, session_id: str, messages):
        self.data[session_id] = list(messages)


class MockMemoryManager(BaseMemoryManager):
    """Mock memory manager for testing."""

    @property
    def mode_name(self) -> str:
        return "mock"

    def prepare_context(self, user_input: str) -> MemoryContext:
        return MemoryContext(
            messages=[],
            mode=self.mode_name,
        )

    def update(self, user_input: str, ai_response: str) -> None:
        pass


class TestMemoryContext:
    """Tests for MemoryContext dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        ctx = MemoryContext()
        assert ctx.messages == []
        assert ctx.system_additions is None
        assert ctx.context_prefix is None
        assert ctx.mode == "unknown"
        assert ctx.total_messages_stored == 0
        assert ctx.context_messages_count == 0
        assert ctx.token_estimate is None
        assert ctx.metadata == {}

    def test_with_values(self):
        """Test creation with explicit values."""
        ctx = MemoryContext(
            messages=["msg1", "msg2"],
            system_additions="Be helpful",
            context_prefix="Previous context",
            mode="test",
            total_messages_stored=10,
            token_estimate=100,
            metadata={"key": "value"},
        )
        assert len(ctx.messages) == 2
        assert ctx.system_additions == "Be helpful"
        assert ctx.context_prefix == "Previous context"
        assert ctx.mode == "test"
        assert ctx.total_messages_stored == 10
        assert ctx.token_estimate == 100
        assert ctx.metadata == {"key": "value"}

    def test_auto_count_messages(self):
        """Test that context_messages_count is auto-set from messages."""
        ctx = MemoryContext(messages=["a", "b", "c"])
        assert ctx.context_messages_count == 3

    def test_explicit_count_preserved(self):
        """Test that explicit context_messages_count is preserved."""
        ctx = MemoryContext(messages=["a", "b"], context_messages_count=5)
        assert ctx.context_messages_count == 5

    def test_has_context_prefix(self):
        """Test has_context_prefix helper."""
        ctx1 = MemoryContext()
        assert ctx1.has_context_prefix() is False

        ctx2 = MemoryContext(context_prefix="")
        assert ctx2.has_context_prefix() is False

        ctx3 = MemoryContext(context_prefix="Some context")
        assert ctx3.has_context_prefix() is True

    def test_has_system_additions(self):
        """Test has_system_additions helper."""
        ctx1 = MemoryContext()
        assert ctx1.has_system_additions() is False

        ctx2 = MemoryContext(system_additions="")
        assert ctx2.has_system_additions() is False

        ctx3 = MemoryContext(system_additions="Be concise")
        assert ctx3.has_system_additions() is True


class TestMemoryFactory:
    """Tests for MemoryFactory."""

    def setup_method(self):
        """Clear registry before each test."""
        MemoryFactory.clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        MemoryFactory.clear_registry()

    def test_register_and_create(self):
        """Test basic registration and creation."""
        MemoryFactory.register("mock", MockMemoryManager)

        store = MockStore()
        manager = MemoryFactory.create("mock", store, "test-session")

        assert manager.mode_name == "mock"
        assert manager.session_id == "test-session"
        assert manager.store is store

    def test_register_with_config(self):
        """Test creation with config."""
        MemoryFactory.register("mock", MockMemoryManager)

        config = {"option1": True, "option2": 42}
        manager = MemoryFactory.create("mock", MockStore(), "session", config=config)

        assert manager.config == config

    def test_available_modes(self):
        """Test listing available modes."""
        MemoryFactory.register("mode_b", MockMemoryManager)
        MemoryFactory.register("mode_a", MockMemoryManager)

        modes = MemoryFactory.available_modes()
        assert modes == ["mode_a", "mode_b"]  # Sorted

    def test_available_modes_empty(self):
        """Test available_modes when registry is empty."""
        modes = MemoryFactory.available_modes()
        assert modes == []

    def test_unknown_mode_raises(self):
        """Test that unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown memory mode"):
            MemoryFactory.create("nonexistent", MockStore(), "session")

    def test_unknown_mode_shows_available(self):
        """Test that error message shows available modes."""
        MemoryFactory.register("valid", MockMemoryManager)

        with pytest.raises(ValueError) as exc_info:
            MemoryFactory.create("invalid", MockStore(), "session")

        assert "valid" in str(exc_info.value)

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises ValueError."""
        MemoryFactory.register("test", MockMemoryManager)
        with pytest.raises(ValueError, match="already registered"):
            MemoryFactory.register("test", MockMemoryManager)

    def test_invalid_class_raises_type_error(self):
        """Test that non-class raises TypeError."""
        with pytest.raises(TypeError, match="must be a class"):
            MemoryFactory.register("bad", "not a class")  # type: ignore

    def test_non_manager_class_raises(self):
        """Test that class not inheriting BaseMemoryManager raises."""
        with pytest.raises(TypeError, match="must inherit from"):
            MemoryFactory.register("bad", str)

    def test_empty_mode_name_raises(self):
        """Test that empty mode name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            MemoryFactory.register("", MockMemoryManager)

    def test_is_registered(self):
        """Test is_registered check."""
        assert MemoryFactory.is_registered("mock") is False
        MemoryFactory.register("mock", MockMemoryManager)
        assert MemoryFactory.is_registered("mock") is True

    def test_unregister(self):
        """Test unregistration."""
        MemoryFactory.register("mock", MockMemoryManager)
        assert MemoryFactory.is_registered("mock") is True

        MemoryFactory.unregister("mock")
        assert MemoryFactory.is_registered("mock") is False

    def test_unregister_nonexistent_silent(self):
        """Test that unregistering nonexistent mode doesn't raise."""
        MemoryFactory.unregister("nonexistent")  # Should not raise

    def test_get_manager_class(self):
        """Test getting manager class without instantiation."""
        MemoryFactory.register("mock", MockMemoryManager)

        cls = MemoryFactory.get_manager_class("mock")
        assert cls is MockMemoryManager

    def test_get_manager_class_nonexistent(self):
        """Test get_manager_class returns None for unknown mode."""
        cls = MemoryFactory.get_manager_class("nonexistent")
        assert cls is None

    def test_clear_registry(self):
        """Test clearing the registry."""
        MemoryFactory.register("a", MockMemoryManager)
        MemoryFactory.register("b", MockMemoryManager)

        MemoryFactory.clear_registry()

        assert MemoryFactory.available_modes() == []


class TestBaseMemoryManager:
    """Tests for BaseMemoryManager interface."""

    def test_initialization(self):
        """Test manager initialization."""
        store = MockStore()
        manager = MockMemoryManager(store, "test-session", {"key": "value"})

        assert manager.store is store
        assert manager.session_id == "test-session"
        assert manager.config == {"key": "value"}
        assert manager._loaded is False

    def test_default_config(self):
        """Test that config defaults to empty dict."""
        manager = MockMemoryManager(MockStore(), "session")
        assert manager.config == {}

    def test_mode_name_abstract(self):
        """Test that mode_name is implemented by subclass."""
        manager = MockMemoryManager(MockStore(), "session")
        assert manager.mode_name == "mock"

    def test_prepare_context_returns_memory_context(self):
        """Test that prepare_context returns MemoryContext."""
        manager = MockMemoryManager(MockStore(), "session")
        context = manager.prepare_context("hello")

        assert isinstance(context, MemoryContext)
        assert context.mode == "mock"

    def test_load_sets_loaded_flag(self):
        """Test that load() sets _loaded flag."""
        manager = MockMemoryManager(MockStore(), "session")
        assert manager._loaded is False

        manager.load()
        assert manager._loaded is True

    def test_get_stats(self):
        """Test get_stats returns basic info."""
        manager = MockMemoryManager(MockStore(), "session")
        stats = manager.get_stats()

        assert stats["mode"] == "mock"
        assert stats["session_id"] == "session"
        assert stats["loaded"] is False

        manager.load()
        stats = manager.get_stats()
        assert stats["loaded"] is True

    def test_get_message_count_default(self):
        """Test default message count is 0."""
        manager = MockMemoryManager(MockStore(), "session")
        assert manager.get_message_count() == 0

    def test_get_system_prompt_additions_default(self):
        """Test default system prompt additions is None."""
        manager = MockMemoryManager(MockStore(), "session")
        assert manager.get_system_prompt_additions() is None

    def test_to_dict(self):
        """Test serialization to dict."""
        manager = MockMemoryManager(MockStore(), "session", {"opt": True})
        data = manager.to_dict()

        assert data["mode"] == "mock"
        assert data["session_id"] == "session"
        assert data["config"] == {"opt": True}
        assert "version" in data
        assert data["version"] == 1

    def test_from_dict_matching_mode(self):
        """Test restoration from dict."""
        manager = MockMemoryManager(MockStore(), "session")
        data = {
            "mode": "mock",
            "version": 1,
            "session_id": "session",
            "config": {"restored": True},
        }

        manager.from_dict(data)
        assert manager.config == {"restored": True}

    def test_from_dict_mode_mismatch_raises(self):
        """Test that mode mismatch in from_dict raises ValueError."""
        manager = MockMemoryManager(MockStore(), "session")
        data = {"mode": "different_mode"}

        with pytest.raises(ValueError, match="Mode mismatch"):
            manager.from_dict(data)


class TestIntegration:
    """Integration tests for the foundation components."""

    def setup_method(self):
        """Clear registry before each test."""
        MemoryFactory.clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        MemoryFactory.clear_registry()

    def test_full_workflow(self):
        """Test complete workflow: register, create, use, serialize."""
        # Register
        MemoryFactory.register("mock", MockMemoryManager)

        # Create
        store = MockStore()
        manager = MemoryFactory.create("mock", store, "workflow-test")

        # Load
        manager.load()
        assert manager._loaded is True

        # Prepare context
        context = manager.prepare_context("Hello")
        assert isinstance(context, MemoryContext)

        # Update
        manager.update("Hello", "Hi there!")

        # Get stats
        stats = manager.get_stats()
        assert stats["mode"] == "mock"

        # Serialize
        data = manager.to_dict()
        assert data["mode"] == "mock"

        # Create new manager and restore
        manager2 = MemoryFactory.create("mock", store, "workflow-test")
        manager2.from_dict(data)


class TestSanitizeSessionId:
    """Tests for _sanitize_session_id path-traversal prevention."""

    def test_normal_session_id_passes_through(self):
        assert _sanitize_session_id("my-session-123") == "my-session-123"

    def test_forward_slash_replaced(self):
        result = _sanitize_session_id("user/session")
        assert "/" not in result
        assert result == "user_session"

    def test_backslash_replaced(self):
        result = _sanitize_session_id("user\\session")
        assert "\\" not in result
        assert result == "user_session"

    def test_double_dot_replaced(self):
        result = _sanitize_session_id("../secrets")
        assert ".." not in result
        assert "/" not in result

    def test_complex_traversal_replaced(self):
        result = _sanitize_session_id("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_null_bytes_replaced(self):
        result = _sanitize_session_id("session\x00evil")
        assert "\x00" not in result
        assert result == "session_evil"

    def test_length_capped_at_200(self):
        long_id = "a" * 300
        result = _sanitize_session_id(long_id)
        assert len(result) == 200

    def test_combined_attack_sanitized(self):
        result = _sanitize_session_id("..\\..\\windows\\system32")
        assert ".." not in result
        assert "\\" not in result


class TestHybridMetaPathTraversal:
    """Tests that _hybrid_meta_path() stays within the base directory."""

    def test_normal_session_id_within_base(self, tmp_path):
        store = MockStore()
        store.base_path = tmp_path  # type: ignore[attr-defined]
        manager = MockMemoryManager(store, "normal-session")
        path = manager._hybrid_meta_path()
        assert str(path).startswith(str(tmp_path.resolve()))

    def test_dotdot_session_id_sanitized(self, tmp_path):
        store = MockStore()
        store.base_path = tmp_path  # type: ignore[attr-defined]
        manager = MockMemoryManager(store, "../secrets")
        path = manager._hybrid_meta_path()
        assert str(path).startswith(str(tmp_path.resolve()))

    def test_slash_session_id_sanitized(self, tmp_path):
        store = MockStore()
        store.base_path = tmp_path  # type: ignore[attr-defined]
        manager = MockMemoryManager(store, "sub/dir/session")
        path = manager._hybrid_meta_path()
        assert str(path).startswith(str(tmp_path.resolve()))

    def test_backslash_session_id_sanitized(self, tmp_path):
        store = MockStore()
        store.base_path = tmp_path  # type: ignore[attr-defined]
        manager = MockMemoryManager(store, "sub\\dir\\session")
        path = manager._hybrid_meta_path()
        assert str(path).startswith(str(tmp_path.resolve()))


class TestSessionVectorStoreTraversal:
    """Tests that SessionVectorStore rejects path-traversal session IDs."""

    def test_normal_session_id_accepted(self, tmp_path):
        from src.memory.recall import SessionVectorStore

        store = SessionVectorStore("my-session", storage_dir=str(tmp_path))
        assert str(store._index_dir).startswith(str(tmp_path.resolve()))

    def test_dotdot_session_id_sanitized(self, tmp_path):
        from src.memory.recall import SessionVectorStore

        store = SessionVectorStore("../outside", storage_dir=str(tmp_path))
        assert str(store._index_dir).startswith(str(tmp_path.resolve()))

    def test_slash_session_id_sanitized(self, tmp_path):
        from src.memory.recall import SessionVectorStore

        store = SessionVectorStore("sub/dir", storage_dir=str(tmp_path))
        assert str(store._index_dir).startswith(str(tmp_path.resolve()))

    def test_backslash_session_id_sanitized(self, tmp_path):
        from src.memory.recall import SessionVectorStore

        store = SessionVectorStore("sub\\dir", storage_dir=str(tmp_path))
        assert str(store._index_dir).startswith(str(tmp_path.resolve()))

    def test_null_byte_session_id_sanitized(self, tmp_path):
        from src.memory.recall import SessionVectorStore

        store = SessionVectorStore("session\x00evil", storage_dir=str(tmp_path))
        assert "\x00" not in str(store._index_dir)
        assert str(store._index_dir).startswith(str(tmp_path.resolve()))
