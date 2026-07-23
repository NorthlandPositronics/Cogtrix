"""Tests for the data_dir config option and resolve_data_path method."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import Config, RAGConfig, _apply_cli_args, _apply_config_file, _apply_env_vars


class TestDefaultDataDir:
    def test_default_data_dir(self):
        config = Config()
        assert config.data_dir == "data"


class TestResolveDataPath:
    def test_resolve_data_path_relative(self):
        config = Config()
        result = config.resolve_data_path("history")
        assert result == Path("data/history")

    def test_resolve_data_path_absolute(self):
        config = Config()
        result = config.resolve_data_path("/abs/path")
        assert result == Path("/abs/path")

    def test_resolve_data_path_legacy_prefix(self):
        config = Config()
        assert config.data_dir == "data"
        result = config.resolve_data_path("data/vectordb")
        assert result == Path("data/vectordb")

    def test_resolve_data_path_legacy_prefix_not_double_nested(self):
        config = Config()
        result = config.resolve_data_path("data/vectordb")
        assert result != Path("data/data/vectordb")

    def test_resolve_data_path_custom_data_dir(self):
        config = Config()
        config.data_dir = "/mnt/storage"
        result = config.resolve_data_path("history")
        assert result == Path("/mnt/storage/history")

    def test_resolve_data_path_custom_dir_no_legacy_strip(self):
        config = Config()
        config.data_dir = "/mnt/storage"
        result = config.resolve_data_path("data/foo")
        assert result == Path("/mnt/storage/data/foo")

    def test_resolve_data_path_absolute_ignores_data_dir(self):
        config = Config()
        config.data_dir = "/mnt/storage"
        result = config.resolve_data_path("/abs/path")
        assert result == Path("/abs/path")

    def test_resolve_data_path_non_legacy_relative(self):
        config = Config()
        result = config.resolve_data_path("sessions/foo")
        assert result == Path("data/sessions/foo")


class TestRAGConfigDefaults:
    def test_rag_vectordb_dir_default(self):
        rag = RAGConfig()
        assert rag.vectordb_dir == "vectordb"

    def test_rag_vectordb_dir_not_prefixed_with_data(self):
        rag = RAGConfig()
        assert not rag.vectordb_dir.startswith("data/")


class TestConfigFileDataDir:
    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        cfg_file = tmp_path / ".cogtrix.yaml"
        cfg_file.write_text(content)
        return cfg_file

    def test_config_file_data_dir(self, tmp_path):
        cfg_file = self._write_yaml(tmp_path, "data_dir: /var/cogtrix/data\n")
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.data_dir == "/var/cogtrix/data"

    def test_config_file_data_dir_relative(self, tmp_path):
        cfg_file = self._write_yaml(tmp_path, "data_dir: mydata\n")
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.data_dir == "mydata"

    def test_config_file_missing_data_dir_keeps_default(self, tmp_path):
        cfg_file = self._write_yaml(tmp_path, "provider: ollama\n")
        config = Config()
        _apply_config_file(config, cfg_file)
        assert config.data_dir == "data"


class TestEnvDataDir:
    def test_env_data_dir(self, monkeypatch):
        monkeypatch.setenv("COGTRIX_DATA_DIR", "/env/data")
        config = Config()
        _apply_env_vars(config)
        assert config.data_dir == "/env/data"

    def test_env_data_dir_relative(self, monkeypatch):
        monkeypatch.setenv("COGTRIX_DATA_DIR", "custom_data")
        config = Config()
        _apply_env_vars(config)
        assert config.data_dir == "custom_data"

    def test_env_data_dir_not_set_keeps_default(self, monkeypatch):
        monkeypatch.delenv("COGTRIX_DATA_DIR", raising=False)
        config = Config()
        _apply_env_vars(config)
        assert config.data_dir == "data"


class TestCLIDataDir:
    def test_cli_data_dir(self):
        config = Config()
        args = SimpleNamespace(data_dir="/custom")
        _apply_cli_args(config, args)
        assert config.data_dir == "/custom"

    def test_cli_data_dir_none_keeps_existing(self):
        config = Config()
        config.data_dir = "existing"
        args = SimpleNamespace(data_dir=None)
        _apply_cli_args(config, args)
        assert config.data_dir == "existing"

    def test_cli_data_dir_missing_attr_keeps_existing(self):
        config = Config()
        args = SimpleNamespace()
        _apply_cli_args(config, args)
        assert config.data_dir == "data"

    def test_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("COGTRIX_DATA_DIR", "/from/env")
        config = Config()
        _apply_env_vars(config)
        assert config.data_dir == "/from/env"

        args = SimpleNamespace(data_dir="/from/cli")
        _apply_cli_args(config, args)
        assert config.data_dir == "/from/cli"


class TestSetEmbeddingsVectorStoreDir:
    def test_set_embeddings_vector_store_dir(self):
        from src.memory.base import BaseMemoryStore
        from src.memory.modes.conversation import ConversationMemoryManager

        store = MagicMock(spec=BaseMemoryStore)
        manager = ConversationMemoryManager(store=store, session_id="test-session")

        embedding_fn = MagicMock()

        with patch("src.memory.recall.SessionVectorStore") as MockVS:
            mock_vs_instance = MagicMock()
            MockVS.return_value = mock_vs_instance

            manager.set_embeddings(
                embedding_fn=embedding_fn,
                embedding_model="nomic-embed-text",
                vector_store_dir="/custom/vectordb",
            )

            MockVS.assert_called_once_with(
                "test-session",
                storage_dir="/custom/vectordb",
            )
            mock_vs_instance.configure.assert_called_once_with(embedding_fn, "nomic-embed-text")

    def test_set_embeddings_default_vector_store_dir(self):
        from src.memory.base import BaseMemoryStore
        from src.memory.modes.conversation import ConversationMemoryManager

        store = MagicMock(spec=BaseMemoryStore)
        manager = ConversationMemoryManager(store=store, session_id="test-session-default")

        embedding_fn = MagicMock()

        with patch("src.memory.recall.SessionVectorStore") as MockVS:
            mock_vs_instance = MagicMock()
            MockVS.return_value = mock_vs_instance

            manager.set_embeddings(
                embedding_fn=embedding_fn,
                embedding_model="nomic-embed-text",
            )

            MockVS.assert_called_once_with(
                "test-session-default",
                storage_dir="data/vectordb/sessions",
            )

    def test_set_embeddings_vector_store_dir_none_uses_default(self):
        from src.memory.base import BaseMemoryStore
        from src.memory.modes.conversation import ConversationMemoryManager

        store = MagicMock(spec=BaseMemoryStore)
        manager = ConversationMemoryManager(store=store, session_id="test-session-none")

        embedding_fn = MagicMock()

        with patch("src.memory.recall.SessionVectorStore") as MockVS:
            mock_vs_instance = MagicMock()
            MockVS.return_value = mock_vs_instance

            manager.set_embeddings(
                embedding_fn=embedding_fn,
                embedding_model="nomic-embed-text",
                vector_store_dir=None,
            )

            MockVS.assert_called_once_with(
                "test-session-none",
                storage_dir="data/vectordb/sessions",
            )
