"""Tests for multi-provider configuration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Config, ProviderConfig, _parse_providers_section, load_config


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_provider_config_defaults(self):
        """Test ProviderConfig with minimal fields."""
        cfg = ProviderConfig(name="test", type="openai")
        assert cfg.name == "test"
        assert cfg.type == "openai"
        assert cfg.base_url is None
        assert cfg.model is None
        assert cfg.api_key is None

    def test_get_base_url_openai_default(self):
        """Test OpenAI provider returns None for default URL."""
        cfg = ProviderConfig(name="openai", type="openai")
        assert cfg.get_base_url() is None

    def test_get_base_url_ollama_default(self):
        """Test Ollama provider returns default localhost URL."""
        cfg = ProviderConfig(name="ollama", type="ollama")
        assert cfg.get_base_url() == "http://localhost:11434"

    def test_get_base_url_custom(self):
        """Test custom base_url is returned."""
        cfg = ProviderConfig(
            name="custom",
            type="openai",
            base_url="http://custom:8000/v1",
        )
        assert cfg.get_base_url() == "http://custom:8000/v1"

    def test_get_model_openai_default(self):
        """Test OpenAI default model."""
        cfg = ProviderConfig(name="openai", type="openai")
        assert cfg.get_model() == "gpt-4o-mini"

    def test_get_model_ollama_default(self):
        """Test Ollama default model."""
        cfg = ProviderConfig(name="ollama", type="ollama")
        assert cfg.get_model() == "qwen3:32b"

    def test_get_model_custom(self):
        """Test custom model is returned."""
        cfg = ProviderConfig(name="custom", type="openai", model="gpt-4o")
        assert cfg.get_model() == "gpt-4o"

    def test_to_dict_hides_api_key(self):
        """Test that to_dict masks the API key."""
        cfg = ProviderConfig(
            name="openai",
            type="openai",
            api_key="sk-secret-key",
        )
        d = cfg.to_dict()
        assert d["api_key"] == "***"


class TestConfigProviders:
    """Tests for Config.providers functionality."""

    def test_get_provider_config_named(self):
        """Test getting a named provider config."""
        config = Config()
        config.providers["my-server"] = ProviderConfig(
            name="my-server",
            type="ollama",
            base_url="http://192.168.1.100:11434",
            model="llama3:70b",
        )
        config.provider = "my-server"

        prov_cfg = config.get_provider_config()
        assert prov_cfg.name == "my-server"
        assert prov_cfg.type == "ollama"
        assert prov_cfg.base_url == "http://192.168.1.100:11434"

    def test_get_provider_config_legacy_openai(self):
        """Test getting legacy openai provider."""
        config = Config()
        config.openai_api_key = "sk-test"
        config.openai_model = "gpt-4o"

        prov_cfg = config.get_provider_config("openai")
        assert prov_cfg.name == "openai"
        assert prov_cfg.type == "openai"
        assert prov_cfg.api_key == "sk-test"
        assert prov_cfg.model == "gpt-4o"

    def test_get_provider_config_legacy_ollama(self):
        """Test getting legacy ollama provider."""
        config = Config()
        config.ollama_base_url = "http://custom:11434"
        config.ollama_model = "qwen3:8b"

        prov_cfg = config.get_provider_config("ollama")
        assert prov_cfg.name == "ollama"
        assert prov_cfg.type == "ollama"
        assert prov_cfg.base_url == "http://custom:11434"
        assert prov_cfg.model == "qwen3:8b"

    def test_get_provider_config_unknown(self):
        """Test getting unknown provider raises ValueError."""
        config = Config()
        with pytest.raises(ValueError) as exc_info:
            config.get_provider_config("unknown-provider")
        assert "Unknown provider" in str(exc_info.value)

    def test_list_providers(self):
        """Test listing available providers."""
        config = Config()
        config.providers["gpu-server"] = ProviderConfig(name="gpu-server", type="ollama")
        config.providers["vllm"] = ProviderConfig(name="vllm", type="openai")

        providers = config.list_providers()
        assert "gpu-server" in providers
        assert "vllm" in providers
        assert "openai" in providers  # Legacy
        assert "ollama" in providers  # Legacy


class TestProvidersConfigFile:
    """Tests for parsing providers from config file."""

    def test_parse_providers_section(self):
        """Test parsing providers section."""
        config = Config()
        providers_data = {
            "gpu-server": {
                "type": "ollama",
                "base_url": "http://192.168.1.100:11434",
                "model": "llama3:70b",
            },
            "openai": {
                "type": "openai",
                "model": "gpt-4o",
            },
        }

        _parse_providers_section(config, providers_data)

        assert "gpu-server" in config.providers
        assert config.providers["gpu-server"].type == "ollama"
        assert config.providers["gpu-server"].base_url == "http://192.168.1.100:11434"

        assert "openai" in config.providers
        assert config.providers["openai"].type == "openai"
        assert config.providers["openai"].model == "gpt-4o"

    def test_load_config_with_providers(self):
        """Test loading config file with providers section."""
        config_data = {
            "provider": "gpu-server",
            "providers": {
                "gpu-server": {
                    "type": "ollama",
                    "base_url": "http://192.168.1.100:11434",
                    "model": "llama3:70b",
                },
                "groq": {
                    "type": "openai",
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key": "gsk-test",
                    "model": "llama-3.3-70b-versatile",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config()

            assert config.provider == "gpu-server"
            assert "gpu-server" in config.providers
            assert "groq" in config.providers
            assert config.providers["groq"].api_key == "gsk-test"
        finally:
            Path(config_path).unlink()

    def test_legacy_config_still_works(self):
        """Test that legacy config format still works."""
        config_data = {
            "provider": "ollama",
            "ollama": {
                "base_url": "http://custom:11434",
                "model": "qwen3:32b",
            },
            "openai": {
                "api_key": "sk-test",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config()

            assert config.provider == "ollama"
            assert config.ollama_base_url == "http://custom:11434"
            assert config.openai_api_key == "sk-test"

            # Legacy providers should be in providers dict too
            assert "ollama" in config.providers
            assert "openai" in config.providers
        finally:
            Path(config_path).unlink()

    def test_mixed_config_format(self):
        """Test config with both providers section and legacy format."""
        config_data = {
            "provider": "gpu-server",
            "providers": {
                "gpu-server": {
                    "type": "ollama",
                    "base_url": "http://192.168.1.100:11434",
                },
            },
            # Legacy format should also work alongside
            "openai": {
                "api_key": "sk-legacy",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config()

            # Named provider should work
            assert "gpu-server" in config.providers

            # Legacy openai should also be in providers
            assert "openai" in config.providers
            assert config.openai_api_key == "sk-legacy"
        finally:
            Path(config_path).unlink()


class TestModelResolution:
    """Tests for model resolution from provider config."""

    def test_model_from_provider_config(self):
        """Test that model is resolved from provider config."""
        config_data = {
            "provider": "custom",
            "providers": {
                "custom": {
                    "type": "ollama",
                    "model": "custom-model:7b",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config()

            # Model should be resolved from provider config
            assert config.model == "custom-model:7b"
        finally:
            Path(config_path).unlink()

    def test_cli_model_overrides_provider_config(self):
        """Test that CLI model overrides provider config model."""
        config_data = {
            "provider": "custom",
            "providers": {
                "custom": {
                    "type": "ollama",
                    "model": "default-model",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            # Create mock CLI args
            class MockArgs:
                provider = None
                model = "cli-model"
                session = None
                memory_mode = None
                debug = False
                log = None

            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config(MockArgs())

            # CLI model should override
            assert config.model == "cli-model"
        finally:
            Path(config_path).unlink()
