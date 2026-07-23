"""Integration tests for memory system with CLI and config."""

import json
from argparse import Namespace
from unittest.mock import patch

from src.config import Config, load_config
from src.memory import JsonFileMemoryStore, MemoryFactory
from src.memory.modes.conversation import ConversationMemoryManager


class TestMemoryConfigIntegration:
    """Test memory configuration loading."""

    @patch("src.config.find_config_file", return_value=None)
    def test_default_memory_mode(self, mock_find):
        """Default memory mode should be 'conversation'."""
        config = load_config(None)
        assert config.memory_mode == "conversation"

    @patch("src.config.find_config_file", return_value=None)
    def test_default_memory_config_is_none(self, mock_find):
        """Default memory config should be None."""
        config = load_config(None)
        assert config.memory_config is None

    def test_cli_memory_mode(self):
        """CLI should override memory mode."""
        args = Namespace(
            provider=None,
            model=None,
            session=None,
            memory_mode="code",
        )
        config = load_config(args)
        assert config.memory_mode == "code"

    def test_cli_memory_mode_reasoning(self):
        """CLI should accept reasoning mode."""
        args = Namespace(
            provider=None,
            model=None,
            session=None,
            memory_mode="reasoning",
        )
        config = load_config(args)
        assert config.memory_mode == "reasoning"

    def test_env_memory_mode(self):
        """Environment variable should set memory mode."""
        with patch.dict("os.environ", {"COGTRIX_MEMORY_MODE": "code"}):
            config = load_config(None)
            assert config.memory_mode == "code"

    def test_config_file_memory_mode(self, tmp_path):
        """Config file should set memory mode."""
        config_file = tmp_path / ".cogtrix.json"
        config_file.write_text(
            json.dumps(
                {
                    "memory": {
                        "mode": "code",
                    }
                }
            )
        )

        with patch("src.config.find_config_file", return_value=config_file):
            config = load_config(None)
            assert config.memory_mode == "code"

    def test_config_file_memory_config(self, tmp_path):
        """Config file should set mode-specific config."""
        config_file = tmp_path / ".cogtrix.json"
        config_file.write_text(
            json.dumps(
                {
                    "memory": {
                        "mode": "conversation",
                        "modes": {
                            "conversation": {
                                "working_memory_size": 50,
                            }
                        },
                    }
                }
            )
        )

        with patch("src.config.find_config_file", return_value=config_file):
            config = load_config(None)
            assert config.memory_mode == "conversation"
            assert config.memory_config == {"working_memory_size": 50}

    def test_cli_overrides_config_file(self, tmp_path):
        """CLI memory mode should override config file."""
        config_file = tmp_path / ".cogtrix.json"
        config_file.write_text(json.dumps({"memory": {"mode": "conversation"}}))

        args = Namespace(
            provider=None,
            model=None,
            session=None,
            memory_mode="code",
        )

        with patch("src.config.find_config_file", return_value=config_file):
            config = load_config(args)
            assert config.memory_mode == "code"

    def test_cli_overrides_env_var(self):
        """CLI should override environment variable."""
        args = Namespace(
            provider=None,
            model=None,
            session=None,
            memory_mode="reasoning",
        )

        with patch.dict("os.environ", {"COGTRIX_MEMORY_MODE": "code"}):
            config = load_config(args)
            assert config.memory_mode == "reasoning"


class TestMemoryManagerIntegration:
    """Test memory manager creation and usage."""

    def setup_method(self):
        """Ensure conversation mode is registered before each test."""
        if not MemoryFactory.is_registered("conversation"):
            MemoryFactory.register("conversation", ConversationMemoryManager)

    def test_create_from_config(self, tmp_path):
        """Create memory manager from config."""
        config = Config(
            memory_mode="conversation",
            memory_config={"working_memory_size": 5},
            session="test",
        )

        store = JsonFileMemoryStore(str(tmp_path))
        manager = MemoryFactory.create(
            mode=config.memory_mode,
            store=store,
            session_id=config.session,
            config=config.memory_config,
        )

        assert manager.mode_name == "conversation"
        assert manager._mode_config["working_memory_size"] == 5

    def test_full_flow(self, tmp_path):
        """Test complete memory flow: create, update, save, load."""
        store = JsonFileMemoryStore(str(tmp_path))

        # Session 1: Create and use
        manager1 = MemoryFactory.create("conversation", store, "flow-test")
        manager1.load()

        manager1.update("Hello", "Hi there!")
        manager1.update("How are you?", "I'm well!")
        manager1.save()

        # Session 2: Load and continue
        manager2 = MemoryFactory.create("conversation", store, "flow-test")
        manager2.load()

        context = manager2.prepare_context("What's next?")
        assert context.total_messages_stored == 4
        assert len(context.messages) == 4

    def test_different_sessions_isolated(self, tmp_path):
        """Different sessions should be isolated."""
        store = JsonFileMemoryStore(str(tmp_path))

        # Session A
        ma = MemoryFactory.create("conversation", store, "session-a")
        ma.load()
        ma.update("A1", "A2")
        ma.save()

        # Session B
        mb = MemoryFactory.create("conversation", store, "session-b")
        mb.load()
        mb.update("B1", "B2")
        mb.update("B3", "B4")
        mb.save()

        # Reload and verify
        ma_reload = MemoryFactory.create("conversation", store, "session-a")
        ma_reload.load()
        assert ma_reload.get_message_count() == 2

        mb_reload = MemoryFactory.create("conversation", store, "session-b")
        mb_reload.load()
        assert mb_reload.get_message_count() == 4

    def test_context_preparation(self, tmp_path):
        """Test context preparation returns valid MemoryContext."""
        store = JsonFileMemoryStore(str(tmp_path))
        manager = MemoryFactory.create("conversation", store, "context-test")
        manager.load()

        manager.update("Test question", "Test answer")

        context = manager.prepare_context("Next question")

        assert context.mode == "conversation"
        assert context.total_messages_stored == 2
        assert len(context.messages) == 2

    def test_working_memory_window(self, tmp_path):
        """Test that working memory window returns all messages on cold cache.

        On the cold-cache path (before the tier cache is warm), all stored
        messages are returned so that no messages are lost before background
        summarisation compresses them into the tier cache.  Window limiting
        (`working_memory_size`) takes effect once the tier cache is warm.
        """
        config = {"working_memory_size": 4}
        store = JsonFileMemoryStore(str(tmp_path))
        manager = MemoryFactory.create("conversation", store, "window-test", config)
        manager.load()

        # Add 10 messages (5 Q&A turns)
        for i in range(5):
            manager.update(f"Q{i}", f"A{i}")

        context = manager.prepare_context("next")

        assert context.total_messages_stored == 10
        # Cold-cache path returns all messages; window limit applies once tier
        # cache is warm (see conversation.py sliding-window fallback comment).
        assert context.context_messages_count == 10
        assert len(context.messages) == 10


class TestConfigPriority:
    """Test configuration priority: CLI > env > file > defaults."""

    def test_priority_order(self, tmp_path):
        """Verify priority: CLI > ENV > Config file > Default."""
        # Setup config file
        config_file = tmp_path / ".cogtrix.json"
        config_file.write_text(json.dumps({"memory": {"mode": "reasoning"}}))

        # Test 1: Default (no overrides)
        with patch("src.config.find_config_file", return_value=None):
            config = load_config(None)
            assert config.memory_mode == "conversation"  # Default

        # Test 2: Config file overrides default
        with patch("src.config.find_config_file", return_value=config_file):
            config = load_config(None)
            assert config.memory_mode == "reasoning"  # From file

        # Test 3: ENV overrides config file
        with patch("src.config.find_config_file", return_value=config_file):
            with patch.dict("os.environ", {"COGTRIX_MEMORY_MODE": "code"}):
                config = load_config(None)
                assert config.memory_mode == "code"  # From env

        # Test 4: CLI overrides everything
        args = Namespace(
            provider=None,
            model=None,
            session=None,
            memory_mode="conversation",
        )
        with patch("src.config.find_config_file", return_value=config_file):
            with patch.dict("os.environ", {"COGTRIX_MEMORY_MODE": "code"}):
                config = load_config(args)
                assert config.memory_mode == "conversation"  # From CLI


class TestFactoryErrorHandling:
    """Test error handling in factory."""

    def setup_method(self):
        """Ensure conversation mode is registered for error message checks."""
        if not MemoryFactory.is_registered("conversation"):
            MemoryFactory.register("conversation", ConversationMemoryManager)

    def test_unknown_mode_raises_value_error(self, tmp_path):
        """Unknown mode should raise ValueError."""
        store = JsonFileMemoryStore(str(tmp_path))

        try:
            MemoryFactory.create("unknown_mode", store, "test")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "Unknown memory mode" in str(e)
            assert "conversation" in str(e)  # Shows available modes
