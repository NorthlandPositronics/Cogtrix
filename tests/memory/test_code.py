"""Unit tests for CodeDevelopmentMemoryManager."""

from datetime import datetime

# Import to trigger registration
from src.memory import modes  # noqa: F401
from src.memory.factory import MemoryFactory
from src.memory.json_store import JsonFileMemoryStore
from src.memory.modes.code import CodeDevelopmentMemoryManager


class MockStore:
    """Mock storage for testing."""

    def __init__(self):
        self.data = {}

    def load_history(self, session_id: str):
        return self.data.get(session_id, [])

    def save_history(self, session_id: str, messages):
        self.data[session_id] = list(messages)


class TestCodeDevelopmentMemoryManager:
    """Tests for CodeDevelopmentMemoryManager."""

    def test_mode_name(self):
        """Test that mode_name returns 'code'."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        assert manager.mode_name == "code"

    def test_factory_registration(self):
        """Test that code mode is registered with factory."""
        assert MemoryFactory.is_registered("code")

        store = MockStore()
        manager = MemoryFactory.create("code", store, "session")
        assert isinstance(manager, CodeDevelopmentMemoryManager)

    def test_default_config(self):
        """Test default configuration values."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        assert manager._mode_config["working_memory_size"] == 30
        assert manager._mode_config["track_files"] is True
        assert manager._mode_config["track_errors"] is True
        assert manager._mode_config["max_errors"] == 5
        assert manager._mode_config["max_files"] == 20
        assert manager._mode_config["summary_max_age_hours"] is None

    def test_custom_config(self):
        """Test custom configuration overrides defaults."""
        config = {"working_memory_size": 4, "max_files": 10}
        manager = CodeDevelopmentMemoryManager(MockStore(), "test", config)

        assert manager._mode_config["working_memory_size"] == 4
        assert manager._mode_config["max_files"] == 10
        # Defaults preserved
        assert manager._mode_config["track_files"] is True

    def test_working_memory_window(self):
        """Test that code mode respects working memory window."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        # Add 80 messages (40 turns) to exceed the 30-message window
        for i in range(40):
            manager.update(f"Q{i}", f"A{i}")

        context = manager.prepare_context("next")

        # Should only have 30 messages in context (default)
        assert context.context_messages_count == 30
        assert context.total_messages_stored == 80

    def test_system_prompt_additions(self):
        """Test that code mode adds system prompt."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        prompt = manager.get_system_prompt_additions()

        assert prompt is not None
        assert "programmer" in prompt.lower()
        assert "code" in prompt.lower()


class TestTaskTracking:
    """Tests for task tracking functionality."""

    def test_set_task(self):
        """Test setting a task."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_task("Implement user authentication")

        assert manager._current_task is not None
        expected = "Implement user authentication"
        assert manager._current_task.description == expected

    def test_add_progress(self):
        """Test adding progress steps."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_task("Build API")
        manager.add_progress("Created models")
        manager.add_progress("Added routes")

        assert len(manager._current_task.steps_completed) == 2
        assert "Created models" in manager._current_task.steps_completed

    def test_set_current_step(self):
        """Test setting current step."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_task("Build API")
        manager.set_current_step("Writing tests")

        assert manager._current_task.current_step == "Writing tests"

    def test_add_blocker(self):
        """Test adding blockers."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_task("Build API")
        manager.add_blocker("Missing database credentials")

        assert len(manager._current_task.blockers) == 1
        assert "database credentials" in manager._current_task.blockers[0]

    def test_task_in_context(self):
        """Test that task appears in context prefix."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_task("Implement authentication")
        manager.add_progress("Created User model")
        manager.set_current_step("Adding login endpoint")

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        prefix = context.context_prefix or ""
        assert "Implement authentication" in prefix
        assert "Created User model" in prefix
        assert "Adding login endpoint" in prefix


class TestFileTracking:
    """Tests for file tracking functionality."""

    def test_set_file_context(self):
        """Test setting file context."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_file_context("src/auth.py", "def login():\n    pass")

        assert "src/auth.py" in manager._files
        assert manager._current_file == "src/auth.py"
        assert manager._files["src/auth.py"].snippet is not None

    def test_file_extraction_from_message(self):
        """Test automatic file extraction from messages."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.update(
            "Please check `src/auth.py` and `models/user.py`",
            "I'll look at those files.",
        )

        assert "src/auth.py" in manager._files
        assert "models/user.py" in manager._files

    def test_file_extraction_quoted(self):
        """Test file extraction from quoted paths."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.update(
            'The file "config/settings.json" needs updating',
            "I'll update it.",
        )

        assert "config/settings.json" in manager._files

    def test_max_files_limit(self):
        """Test that max files limit is enforced."""
        config = {"max_files": 3}
        manager = CodeDevelopmentMemoryManager(MockStore(), "test", config)
        manager.load()

        # Add more than max files
        for i in range(5):
            manager.set_file_context(f"file{i}.py")

        assert len(manager._files) == 3

    def test_get_tracked_files(self):
        """Test getting list of tracked files."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_file_context("a.py")
        manager.set_file_context("b.py")

        files = manager.get_tracked_files()
        assert "a.py" in files
        assert "b.py" in files

    def test_files_in_context(self):
        """Test that files appear in context prefix."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_file_context("src/main.py", "print('hello')")

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        assert "src/main.py" in (context.context_prefix or "")


class TestErrorTracking:
    """Tests for error tracking functionality."""

    def test_add_error(self):
        """Test adding errors manually."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_error("TypeError: 'NoneType' has no attribute 'save'")

        assert len(manager._recent_errors) == 1
        assert "TypeError" in manager._recent_errors[0]

    def test_error_extraction_from_message(self):
        """Test automatic error extraction from messages."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.update(
            "I'm getting TypeError: 'NoneType' has no attribute 'save'",
            "Let me help with that error.",
        )

        assert len(manager._recent_errors) > 0
        assert any("TypeError" in e for e in manager._recent_errors)

    def test_max_errors_limit(self):
        """Test that max errors limit is enforced."""
        config = {"max_errors": 3}
        manager = CodeDevelopmentMemoryManager(MockStore(), "test", config)
        manager.load()

        for i in range(5):
            manager.add_error(f"Error {i}")

        assert len(manager._recent_errors) == 3
        # Should keep most recent
        assert "Error 4" in manager._recent_errors[-1]

    def test_get_recent_errors(self):
        """Test getting list of recent errors."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_error("Error 1")
        manager.add_error("Error 2")

        errors = manager.get_recent_errors()
        assert len(errors) == 2

    def test_errors_in_context(self):
        """Test that errors appear in context prefix."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.add_error("ImportError: No module named 'foo'")

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        assert "ImportError" in (context.context_prefix or "")


class TestChangeTracking:
    """Tests for change tracking functionality."""

    def test_record_change(self):
        """Test recording changes."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.record_change("Modified auth.py: added login function")

        assert len(manager._changes_made) == 1
        assert "auth.py" in manager._changes_made[0]

    def test_changes_in_context(self):
        """Test that changes appear in context prefix."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.record_change("Created new file: utils.py")

        context = manager.prepare_context("next")

        assert context.has_context_prefix()
        assert "utils.py" in (context.context_prefix or "")


class TestPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load(self):
        """Test persistence via save and load."""
        store = MockStore()

        # Create and populate
        manager1 = CodeDevelopmentMemoryManager(store, "test-session")
        manager1.load()
        manager1.update("Hello", "Hi!")
        manager1.set_task("Test task")
        manager1.set_file_context("test.py", "code")
        manager1.add_error("Test error")
        manager1.save()

        # Load in new manager
        manager2 = CodeDevelopmentMemoryManager(store, "test-session")
        manager2.load()

        assert manager2.get_message_count() == 2

    def test_to_dict(self):
        """Test serialization to dictionary."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.update("Q", "A")
        manager.set_task("Test task")
        manager.set_file_context("test.py", "code", (1, 10))
        manager.add_error("Test error")
        manager.record_change("Changed file")

        data = manager.to_dict()

        assert data["mode"] == "code"
        assert len(data["messages"]) == 2
        assert data["task"]["description"] == "Test task"
        assert "test.py" in data["files"]
        assert "Test error" in data["recent_errors"]
        assert any("Changed file" in c for c in data["changes_made"])

    def test_from_dict(self):
        """Test restoration from dictionary."""
        manager1 = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager1.load()
        manager1.update("Q", "A")
        manager1.set_task("Test task")
        manager1.set_file_context("test.py")
        manager1.add_error("Test error")

        data = manager1.to_dict()

        manager2 = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager2.from_dict(data)

        assert manager2.get_message_count() == 2
        assert manager2._current_task.description == "Test task"
        assert "test.py" in manager2._files
        assert "Test error" in manager2._recent_errors


class TestCodeModeWithJsonStore:
    """Test code mode with actual JsonFileMemoryStore."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow with real storage."""
        store = JsonFileMemoryStore(str(tmp_path))

        # Session 1
        m1 = CodeDevelopmentMemoryManager(store, "code-test")
        m1.load()
        m1.set_task("Build feature X")
        m1.update("Working on `feature.py`", "I'll help with that file.")
        m1.add_progress("Created initial structure")
        m1.save()

        # Verify file created
        session_file = tmp_path / "code-test.json"
        assert session_file.exists()

        # Session 2 - Continue
        m2 = CodeDevelopmentMemoryManager(store, "code-test")
        m2.load()

        context = m2.prepare_context("What's next?")
        # Messages persist across sessions
        assert context.total_messages_stored == 2
        # Note: File tracking is session-local (only messages persist)
        # Files would be re-extracted when processing loaded messages if needed


class TestMetadata:
    """Tests for metadata in context."""

    def test_metadata_includes_code_specific_info(self):
        """Test that metadata includes code-specific information."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.set_task("Test")
        manager.set_file_context("file.py")
        manager.add_error("Error")

        context = manager.prepare_context("next")

        assert context.metadata["has_task"] is True
        assert context.metadata["files_tracked"] == 1
        assert context.metadata["current_file"] == "file.py"
        assert context.metadata["error_count"] == 1


class TestStats:
    """Tests for statistics."""

    def test_get_stats(self):
        """Test get_stats returns comprehensive info."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.update("Q", "A")
        manager.set_task("Task")
        manager.set_file_context("file.py")
        manager.add_error("Error")
        manager.record_change("Change")

        stats = manager.get_stats()

        assert stats["mode"] == "code"
        assert stats["total_messages"] == 2
        assert stats["working_memory_size"] == 30
        assert stats["files_tracked"] == 1
        assert stats["has_task"] is True
        assert stats["error_count"] == 1
        assert stats["changes_count"] == 1


class TestClear:
    """Tests for clear functionality."""

    def test_clear_resets_all(self):
        """Test that clear resets all state."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()

        manager.update("Q", "A")
        manager.set_task("Task")
        manager.set_file_context("file.py")
        manager.add_error("Error")
        manager.record_change("Change")

        manager.clear()

        assert manager.get_message_count() == 0
        assert manager._current_task is None
        assert len(manager._files) == 0
        assert len(manager._recent_errors) == 0
        assert len(manager._changes_made) == 0


class TestCodeTimestamps:
    """Tests for timestamp support in code memory mode."""

    def test_update_stamps_messages(self):
        """Test that update attaches timestamps to messages."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Fix the bug", "Done.")

        for msg in manager._messages:
            ts = manager._get_msg_ts(msg)
            assert ts is not None
            datetime.fromisoformat(ts)

    def test_prepare_context_injects_timestamps(self):
        """Timestamps are prepended to HumanMessages only (not AI — the LLM mimics them)."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Fix the bug", "Done.")

        context = manager.prepare_context("next")

        for msg in context.messages:
            content = msg.content if hasattr(msg, "content") else msg["content"]
            if type(msg).__name__ == "HumanMessage":
                assert content.startswith("[")

    def test_to_dict_preserves_timestamps(self):
        """Test that serialization includes timestamps."""
        manager = CodeDevelopmentMemoryManager(MockStore(), "test")
        manager.load()
        manager.update("Q", "A")

        data = manager.to_dict()
        for msg_data in data["messages"]:
            assert "timestamp" in msg_data

    def test_from_dict_restores_timestamps(self):
        """Test that deserialization restores timestamps."""
        m1 = CodeDevelopmentMemoryManager(MockStore(), "test")
        m1.load()
        m1.update("Q", "A")

        data = m1.to_dict()

        m2 = CodeDevelopmentMemoryManager(MockStore(), "test")
        m2.from_dict(data)

        for msg in m2._messages:
            assert m2._get_msg_ts(msg) is not None
