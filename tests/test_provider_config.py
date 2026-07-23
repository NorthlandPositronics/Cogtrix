"""Tests for multi-provider configuration."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    Config,
    ProviderConfig,
    _parse_ollama_address,
    _parse_providers_section,
    load_config,
)


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
        assert cfg.get_model() == "gpt-4.1-mini"

    def test_get_model_ollama_default(self):
        """Test Ollama default model."""
        cfg = ProviderConfig(name="ollama", type="ollama")
        assert cfg.get_model() == "qwen3:8b"

    def test_get_model_custom(self):
        """Test custom model is returned."""
        cfg = ProviderConfig(name="custom", type="openai", model="gpt-4.1")
        assert cfg.get_model() == "gpt-4.1"

    def test_to_dict_hides_api_key(self):
        """Test that to_dict masks the API key."""
        cfg = ProviderConfig(
            name="openai",
            type="openai",
            api_key="sk-secret-key",
        )
        d = cfg.to_dict()
        assert d["api_key"] == "***"

    def test_tool_instructions_default_none(self):
        """Test that tool_instructions defaults to None."""
        cfg = ProviderConfig(name="test", type="openai")
        assert cfg.tool_instructions is None

    def test_tool_instructions_custom(self):
        """Test custom tool_instructions."""
        cfg = ProviderConfig(
            name="test",
            type="openai",
            tool_instructions="Use JSON for tool calls.",
        )
        assert cfg.tool_instructions == "Use JSON for tool calls."

    def test_to_dict_includes_tool_instructions(self):
        """Test that to_dict includes tool_instructions."""
        cfg = ProviderConfig(
            name="test",
            type="openai",
            tool_instructions="Custom instructions",
        )
        d = cfg.to_dict()
        assert d["tool_instructions"] == "Custom instructions"


class TestConfigProviders:
    """Tests for Config.providers functionality."""

    def test_get_provider_config_named(self):
        """Test getting a named provider config."""
        config = Config()
        config.providers["my-server"] = ProviderConfig(
            name="my-server",
            type="ollama",
            base_url="http://192.168.1.100:11434",
            model="llama4:scout",
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
        config.openai_model = "gpt-4.1"

        prov_cfg = config.get_provider_config("openai")
        assert prov_cfg.name == "openai"
        assert prov_cfg.type == "openai"
        assert prov_cfg.api_key == "sk-test"
        assert prov_cfg.model == "gpt-4.1"

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
                "model": "llama4:scout",
            },
            "openai": {
                "type": "openai",
                "model": "gpt-4.1",
            },
        }

        _parse_providers_section(config, providers_data)

        assert "gpu-server" in config.providers
        assert config.providers["gpu-server"].type == "ollama"
        assert config.providers["gpu-server"].base_url == "http://192.168.1.100:11434"

        assert "openai" in config.providers
        assert config.providers["openai"].type == "openai"
        assert config.providers["openai"].model == "gpt-4.1"

    def test_load_config_with_providers(self):
        """Test loading config file with providers section."""
        config_data = {
            "provider": "gpu-server",
            "providers": {
                "gpu-server": {
                    "type": "ollama",
                    "base_url": "http://192.168.1.100:11434",
                    "model": "llama4:scout",
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
                "model": "qwen3:8b",
            },
            "openai": {
                "api_key": "sk-test",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            # Clear OPENAI_API_KEY so the env var (if set in CI) does not
            # override the config-file value.  Documented priority is
            # CLI > Env > Config, and this test validates config-file parsing.
            with (
                patch("src.config.find_config_file", return_value=Path(config_path)),
                patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
            ):
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
            with (
                patch("src.config.find_config_file", return_value=Path(config_path)),
                patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False),
            ):
                config = load_config()

            # Named provider should work
            assert "gpu-server" in config.providers

            # Legacy openai should also be in providers
            assert "openai" in config.providers
            assert config.openai_api_key == "sk-legacy"
        finally:
            Path(config_path).unlink()

    def test_env_var_overrides_config_file(self):
        """Verify documented priority: env var wins over config file value.

        The config loading order is CLI > Env > Config > Defaults.
        When OPENAI_API_KEY is set in the environment, it must take
        precedence over the legacy ``openai.api_key`` in the config file.
        """
        config_data = {
            "openai": {
                "api_key": "sk-from-file",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with (
                patch("src.config.find_config_file", return_value=Path(config_path)),
                patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}, clear=False),
            ):
                config = load_config()

            assert config.openai_api_key == "sk-from-env"
        finally:
            Path(config_path).unlink()

    def test_parse_providers_with_tool_instructions(self):
        """Test parsing providers section with tool_instructions."""
        config = Config()
        providers_data = {
            "vllm": {
                "type": "openai",
                "base_url": "http://localhost:8000/v1",
                "tool_instructions": "Always use JSON for tool calls.",
            },
        }
        _parse_providers_section(config, providers_data)
        assert config.providers["vllm"].tool_instructions == "Always use JSON for tool calls."


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


class TestDefaultProvider:
    """Tests for default-provider-is-ollama behavior."""

    def test_default_provider_is_ollama(self):
        """Test that the default provider is Ollama (works without API keys)."""
        config = Config()
        assert config.provider == "ollama"

    def test_default_model_resolved_to_ollama(self):
        """Test that load_config resolves to Ollama model when no config exists."""
        with patch("src.config.find_config_file", return_value=None):
            config = load_config()
        assert config.provider == "ollama"
        assert config.model == "qwen3:8b"


class TestParseOllamaAddress:
    """Tests for _parse_ollama_address helper."""

    def test_host_only(self):
        """Test bare hostname gets default port."""
        assert _parse_ollama_address("192.168.1.100") == "http://192.168.1.100:11434"

    def test_host_and_port(self):
        """Test host:port is wrapped in http://."""
        assert _parse_ollama_address("192.168.1.100:8080") == "http://192.168.1.100:8080"

    def test_full_url_passthrough(self):
        """Test full http:// URL is returned unchanged."""
        assert _parse_ollama_address("http://my-server:11434") == "http://my-server:11434"

    def test_https_url_passthrough(self):
        """Test https:// URL is returned unchanged."""
        assert _parse_ollama_address("https://ollama.example.com") == "https://ollama.example.com"

    def test_localhost(self):
        """Test localhost shorthand."""
        assert _parse_ollama_address("localhost") == "http://localhost:11434"

    def test_whitespace_stripped(self):
        """Test leading/trailing whitespace is stripped."""
        assert _parse_ollama_address("  192.168.1.100  ") == "http://192.168.1.100:11434"

    def test_bare_ipv6_not_split(self):
        """Test bare IPv6 address is not incorrectly split as host:port."""
        result = _parse_ollama_address("::1")
        assert result == "http://::1:11434"

    def test_bracketed_ipv6_with_port(self):
        """Test [IPv6]:port is correctly parsed."""
        result = _parse_ollama_address("[::1]:11434")
        assert result == "http://[::1]:11434"

    def test_cogtrix_ollama_env_var(self):
        """Test COGTRIX_OLLAMA env var is parsed and applied."""
        env = {"COGTRIX_OLLAMA": "10.0.0.5:9999"}
        with (
            patch("src.config.find_config_file", return_value=None),
            patch.dict("os.environ", env, clear=False),
        ):
            config = load_config()
        assert config.ollama_base_url == "http://10.0.0.5:9999"

    def test_cogtrix_ollama_overrides_legacy(self):
        """Test COGTRIX_OLLAMA takes priority over OLLAMA_BASE_URL."""
        env = {
            "COGTRIX_OLLAMA": "10.0.0.5",
            "OLLAMA_BASE_URL": "http://old-server:11434",
        }
        with (
            patch("src.config.find_config_file", return_value=None),
            patch.dict("os.environ", env, clear=False),
        ):
            config = load_config()
        assert config.ollama_base_url == "http://10.0.0.5:11434"


class TestBuildSystemPrompt:
    """Tests for build_system_prompt tool_instructions parameter."""

    def test_default_no_tool_instructions(self):
        """Tool instructions are NOT injected by default.

        bind_tools() handles tool-call formatting at the API level, so the
        system prompt should not include raw-JSON formatting examples that
        can conflict with the structured tool_calls response format.
        """
        from src.agent.core import DEFAULT_TOOL_INSTRUCTIONS, build_system_prompt

        prompt = build_system_prompt()
        assert DEFAULT_TOOL_INSTRUCTIONS not in prompt

    def test_custom_tool_instructions(self):
        """Custom tool instructions are included when explicitly provided."""
        from src.agent.core import build_system_prompt

        prompt = build_system_prompt(tool_instructions="Custom instructions")
        assert "Custom instructions" in prompt

    def test_empty_tool_instructions(self):
        """Empty string does not inject instructions."""
        from src.agent.core import DEFAULT_TOOL_INSTRUCTIONS, build_system_prompt

        prompt = build_system_prompt(tool_instructions="")
        assert DEFAULT_TOOL_INSTRUCTIONS not in prompt


class TestContextCompressionConfig:
    """Tests for context_compression config parsing."""

    def test_context_compression_bool_false(self):
        """context_compression: false disables compression."""
        from src.config import Config

        config = Config()
        assert config.context_compression is True  # default

        config.context_compression = False
        assert config.context_compression is False

    def test_context_compression_dict_config(self):
        """Dict form sets enabled, min_age, and min_chars."""
        import json
        import tempfile
        from pathlib import Path

        from src.config import Config, _apply_config_file

        config_data = {
            "context_compression": {
                "enabled": True,
                "min_age": 10,
                "min_chars": 4000,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = Config()
            _apply_config_file(config, Path(config_path))
            assert config.context_compression is True
            assert config.context_compression_min_age == 10
            assert config.context_compression_min_chars == 4000
        finally:
            Path(config_path).unlink()

    def test_context_compression_model_config(self):
        """context_compression.model is parsed correctly."""
        import json
        import tempfile
        from pathlib import Path

        from src.config import Config, _apply_config_file

        config = Config()
        data = {"context_compression": {"enabled": True, "model": "fast", "min_age": 4}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            config_path = f.name
        _apply_config_file(config, Path(config_path))
        assert config.context_compression is True
        assert config.context_compression_model == "fast"
        assert config.context_compression_min_age == 4
