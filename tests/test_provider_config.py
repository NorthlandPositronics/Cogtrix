"""Tests for multi-provider configuration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    Config,
    ConfigError,
    ModelConfig,
    ProviderConfig,
    _parse_models_section,
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
        assert cfg.temperature is None
        assert cfg.num_ctx is None

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

    def test_temperature_and_num_ctx_parsed_from_providers(self):
        """temperature and num_ctx are parsed as provider-level defaults."""
        config = Config()
        providers_data = {
            "spark": {
                "type": "openai",
                "base_url": "http://spark:8080/v1",
                "model": "gpt-oss",
                "temperature": 0.7,
                "num_ctx": 32768,
            }
        }
        _parse_providers_section(config, providers_data)
        assert config.providers["spark"].temperature == 0.7
        assert config.providers["spark"].num_ctx == 32768


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_creation(self):
        """Test basic ModelConfig creation."""
        mc = ModelConfig(provider="ollama", model="qwen3:8b")
        assert mc.provider == "ollama"
        assert mc.model == "qwen3:8b"
        assert mc.num_ctx is None
        assert mc.temperature is None

    def test_model_config_with_all_fields(self):
        """Test ModelConfig with all optional fields."""
        mc = ModelConfig(
            provider="openai",
            model="gpt-4.1",
            num_ctx=16384,
            temperature=0.2,
        )
        assert mc.num_ctx == 16384
        assert mc.temperature == 0.2

    def test_model_config_invalid_temperature_high(self):
        """Test ModelConfig rejects temperature > 2.0."""
        with pytest.raises(ConfigError, match="Temperature"):
            ModelConfig(provider="ollama", model="test", temperature=2.1)

    def test_model_config_invalid_temperature_low(self):
        """Test ModelConfig rejects temperature < 0.0."""
        with pytest.raises(ConfigError, match="Temperature"):
            ModelConfig(provider="ollama", model="test", temperature=-0.1)

    def test_model_config_invalid_num_ctx(self):
        """Test ModelConfig rejects num_ctx < 256."""
        with pytest.raises(ConfigError, match="num_ctx"):
            ModelConfig(provider="ollama", model="test", num_ctx=100)

    def test_model_config_boundary_temperature(self):
        """Test ModelConfig accepts boundary temperature values."""
        mc0 = ModelConfig(provider="ollama", model="test", temperature=0.0)
        assert mc0.temperature == 0.0
        mc2 = ModelConfig(provider="ollama", model="test", temperature=2.0)
        assert mc2.temperature == 2.0

    def test_model_config_boundary_num_ctx(self):
        """Test ModelConfig accepts minimum valid num_ctx."""
        mc = ModelConfig(provider="ollama", model="test", num_ctx=256)
        assert mc.num_ctx == 256


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
        # No legacy built-ins added automatically
        assert providers == sorted(["gpu-server", "vllm"])

    def test_get_model_config_found(self):
        """Test get_model_config returns ModelConfig when found."""
        config = Config()
        config.models["fast"] = ModelConfig(provider="ollama", model="qwen3:8b")
        config.model = "fast"

        mc = config.get_model_config()
        assert mc is not None
        assert mc.provider == "ollama"
        assert mc.model == "qwen3:8b"

    def test_get_model_config_not_found(self):
        """Test get_model_config returns None for unknown model names."""
        config = Config()
        config.model = "literal-model-name"
        mc = config.get_model_config()
        assert mc is None

    def test_get_model_config_by_name(self):
        """Test get_model_config with explicit name argument."""
        config = Config()
        config.models["reasoning"] = ModelConfig(
            provider="openai", model="gpt-4.1", temperature=0.2
        )
        mc = config.get_model_config("reasoning")
        assert mc is not None
        assert mc.model == "gpt-4.1"


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

    def test_inference_key_as_alias_for_providers(self):
        """Test that 'inference' key is accepted as alias for 'providers'."""
        config_data = {
            "provider": "gpu-server",
            "inference": {
                "gpu-server": {
                    "type": "ollama",
                    "base_url": "http://192.168.1.100:11434",
                    "model": "llama4:scout",
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config()

            assert "gpu-server" in config.providers
        finally:
            Path(config_path).unlink()


class TestModelResolution:
    """Tests for model resolution from models registry."""

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

            class MockArgs:
                provider = None
                model = "cli-model"
                session = None
                memory_mode = None
                debug = False
                log = None

            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config(MockArgs())

            assert config.model == "cli-model"
        finally:
            Path(config_path).unlink()

    def test_model_name_resolves_from_registry(self):
        """Test that model name in models registry resolves provider and model."""
        config_data = {
            "provider": "ollama",
            "model": "fast",
            "providers": {
                "ollama": {"type": "ollama", "model": "qwen3:8b"},
                "openai": {"type": "openai", "model": "gpt-4.1-mini", "api_key": "sk-x"},
            },
            "models": {
                "fast": {"provider": "ollama", "model": "qwen3:4b"},
                "reasoning": {"provider": "openai", "model": "gpt-4.1", "temperature": 0.2},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config()

            assert config.model == "qwen3:4b"
            assert config.provider == "ollama"
        finally:
            Path(config_path).unlink()

    def test_model_registry_merges_params_into_provider(self):
        """Test that num_ctx from ModelConfig is merged via resolve_provider_config()."""
        config_data = {
            "provider": "ollama",
            "model": "big",
            "providers": {
                "ollama": {"type": "ollama", "model": "qwen3:8b"},
            },
            "models": {
                "big": {"provider": "ollama", "model": "qwen3:32b", "num_ctx": 65536},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config()

            assert config.model == "qwen3:32b"
            # provider in config.providers is NOT mutated
            assert config.providers["ollama"].num_ctx is None
            # resolve_provider_config() returns a clone with model params merged
            resolved = config.resolve_provider_config()
            assert resolved.num_ctx == 65536
            assert resolved.model == "qwen3:32b"
        finally:
            Path(config_path).unlink()

    def test_resolve_provider_config_does_not_mutate_original(self):
        """Verify that resolve_provider_config() returns a clone, not the original."""
        config_data = {
            "provider": "ollama",
            "model": "big",
            "providers": {
                "ollama": {"type": "ollama", "model": "qwen3:8b"},
            },
            "models": {
                "big": {
                    "provider": "ollama",
                    "model": "qwen3:32b",
                    "num_ctx": 32768,
                    "temperature": 0.5,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config()

            original = config.providers["ollama"]
            resolved = config.resolve_provider_config()

            # Clone has merged values
            assert resolved.num_ctx == 32768
            assert resolved.temperature == 0.5
            assert resolved.model == "qwen3:32b"

            # Original is unchanged
            assert original.num_ctx is None
            assert original.temperature is None
            assert original.model == "qwen3:8b"
        finally:
            Path(config_path).unlink()

    def test_model_aliases_key_accepted(self):
        """Test that 'model_aliases' key is accepted as alias for 'models'."""
        config_data = {
            "provider": "ollama",
            "providers": {
                "ollama": {"type": "ollama", "model": "qwen3:8b"},
            },
            "model_aliases": {
                "fast": {"provider": "ollama", "model": "qwen3:4b"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with patch("src.config.find_config_file", return_value=Path(config_path)):
                config = load_config()

            assert "fast" in config.models
        finally:
            Path(config_path).unlink()


class TestParseModelsSection:
    """Tests for _parse_models_section."""

    def test_parse_models_dict_format(self):
        """Test parsing models in dict format."""
        config = Config()
        config.providers["ollama"] = ProviderConfig(name="ollama", type="ollama")
        config.provider = "ollama"

        models_data = {
            "fast": {"provider": "ollama", "model": "qwen3:4b"},
            "reasoning": {
                "provider": "openai",
                "model": "gpt-4.1",
                "temperature": 0.2,
                "num_ctx": 16384,
            },
        }
        _parse_models_section(config, models_data)

        assert "fast" in config.models
        assert config.models["fast"].provider == "ollama"
        assert config.models["fast"].model == "qwen3:4b"

        assert "reasoning" in config.models
        assert config.models["reasoning"].temperature == 0.2
        assert config.models["reasoning"].num_ctx == 16384

    def test_parse_models_string_format_with_slash(self):
        """Test parsing models in 'provider/model' string format."""
        config = Config()
        config.provider = "ollama"
        models_data = {"fast": "ollama/qwen3:4b"}
        _parse_models_section(config, models_data)

        assert "fast" in config.models
        assert config.models["fast"].provider == "ollama"
        assert config.models["fast"].model == "qwen3:4b"

    def test_parse_models_string_format_plain(self):
        """Test parsing models in plain string format (uses current provider)."""
        config = Config()
        config.provider = "ollama"
        models_data = {"quick": "qwen3:4b"}
        _parse_models_section(config, models_data)

        assert "quick" in config.models
        assert config.models["quick"].provider == "ollama"
        assert config.models["quick"].model == "qwen3:4b"

    def test_parse_models_missing_provider_warns(self, caplog):
        """Test that missing provider field emits a warning and skips entry."""
        import logging

        config = Config()
        models_data = {"bad": {"model": "some-model"}}  # missing 'provider'
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            _parse_models_section(config, models_data)
        assert "bad" not in config.models


class TestDefaultProvider:
    """Tests for default-provider-is-ollama behavior."""

    def test_default_provider_is_ollama(self):
        """Test that the default provider is Ollama (works without API keys)."""
        config = Config()
        assert config.provider == "ollama"

    def test_default_model_resolved_to_ollama(self):
        """Test that load_config uses Ollama provider when no config exists.

        config.model stays None when there are no configured providers;
        the actual model is resolved from the provider default at LLM
        creation time via resolve_provider_config().
        """
        with patch("src.config.find_config_file", return_value=None):
            config = load_config()
        assert config.provider == "ollama"
        # model is None because there are no configured providers to query
        assert config.model is None


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

    def test_bare_ipv6_loopback(self):
        """Test bare IPv6 loopback address is wrapped in brackets."""
        assert _parse_ollama_address("::1") == "http://[::1]:11434"

    def test_bare_ipv6_link_local(self):
        """Test bare IPv6 link-local address is wrapped in brackets."""
        assert _parse_ollama_address("fe80::1") == "http://[fe80::1]:11434"

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
        # The ollama provider should be created with the custom URL
        assert "ollama" in config.providers
        assert config.providers["ollama"].base_url == "http://10.0.0.5:9999"

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
        # COGTRIX_OLLAMA wins
        assert "ollama" in config.providers
        assert config.providers["ollama"].base_url == "http://10.0.0.5:11434"


class TestResolveEmbeddingConfig:
    """Tests for Config.resolve_embedding_config()."""

    def test_resolve_from_models_registry(self):
        """Test resolve_embedding_config uses rag.model from models registry."""
        config = Config()
        config.providers["ollama"] = ProviderConfig(
            name="ollama", type="ollama", base_url="http://localhost:11434"
        )
        config.models["embed-local"] = ModelConfig(provider="ollama", model="nomic-embed-text")
        config.rag.model = "embed-local"

        emb_type, emb_model, emb_base_url, emb_api_key = config.resolve_embedding_config()
        assert emb_type == "ollama"
        assert emb_model == "nomic-embed-text"
        assert emb_base_url == "http://localhost:11434"
        assert emb_api_key is None

    def test_resolve_fallback_to_active_provider(self):
        """Test resolve_embedding_config falls back to active provider."""
        config = Config()
        config.provider = "openai"
        config.providers["openai"] = ProviderConfig(name="openai", type="openai", api_key="sk-x")
        config.rag.model = None  # not set

        emb_type, emb_model, emb_base_url, emb_api_key = config.resolve_embedding_config()
        assert emb_type == "openai"
        assert emb_api_key == "sk-x"

    def test_resolve_model_not_in_registry_falls_back(self):
        """Test resolve_embedding_config falls back when rag.model not found in registry."""
        config = Config()
        config.provider = "ollama"
        config.providers["ollama"] = ProviderConfig(name="ollama", type="ollama")
        config.rag.model = "nonexistent-model"

        emb_type, emb_model, emb_base_url, emb_api_key = config.resolve_embedding_config()
        assert emb_type == "ollama"


class TestBuildSystemPrompt:
    """Tests for build_system_prompt tool_instructions parameter."""

    def test_default_no_tool_instructions(self):
        """Tool instructions are NOT injected by default.

        bind_tools() handles tool-call formatting at the API level, so the
        system prompt should not include raw-JSON formatting examples that
        can conflict with the structured tool_calls response format.
        """
        from src.agent.core import build_system_prompt

        prompt = build_system_prompt()
        # Raw JSON formatting instructions should not appear in the system
        # prompt — bind_tools() handles tool-call formatting at the API level.
        assert "output ONLY a valid tool call" not in prompt

    def test_custom_tool_instructions(self):
        """Custom tool instructions are included when explicitly provided."""
        from src.agent.core import build_system_prompt

        prompt = build_system_prompt(tool_instructions="Custom instructions")
        assert "Custom instructions" in prompt

    def test_empty_tool_instructions(self):
        """Empty string does not inject instructions."""
        from src.agent.core import build_system_prompt

        prompt = build_system_prompt(tool_instructions="")
        assert "output ONLY a valid tool call" not in prompt

    def test_models_appear_in_system_prompt(self):
        """Models registry entries appear in system prompt when provided."""
        from src.agent.core import build_system_prompt
        from src.config import ModelConfig

        models = {
            "fast": ModelConfig(provider="ollama", model="qwen3:4b"),
            "reasoning": ModelConfig(provider="openai", model="gpt-4.1"),
        }
        prompt = build_system_prompt(models=models)
        assert "fast" in prompt
        assert "reasoning" in prompt
        assert "ollama/qwen3:4b" in prompt


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


class TestConfigureRagVectordbDir:
    """Tests for configure_rag() vectordb_dir support."""

    def test_configure_rag_vectordb_dir_updates_config(self):
        """configure_rag() with vectordb_dir updates _rag_config used by query functions."""
        from src.tools.rag import _rag_config, configure_rag

        original = _rag_config["vectordb_dir"]
        try:
            configure_rag({"vectordb_dir": "/tmp/test_vectordb"})
            assert _rag_config["vectordb_dir"] == "/tmp/test_vectordb"
        finally:
            configure_rag({"vectordb_dir": original})

    def test_configure_rag_vectordb_dir_ignored_when_absent(self):
        """configure_rag() without vectordb_dir leaves the existing value intact."""
        from src.tools.rag import _rag_config, configure_rag

        configure_rag({"vectordb_dir": "/tmp/before"})
        configure_rag({"embedding_provider": "ollama"})
        assert _rag_config["vectordb_dir"] == "/tmp/before"
        configure_rag({"vectordb_dir": str(configure_rag.__module__)})

    def test_query_knowledge_base_uses_configured_dir(self, tmp_path):
        """query_knowledge_base() checks the configured vectordb_dir, not the default."""
        from src.tools.rag import configure_rag, query_knowledge_base

        missing_dir = str(tmp_path / "nonexistent_index")
        configure_rag({"vectordb_dir": missing_dir})
        result = query_knowledge_base("test question")
        assert "No knowledge base found" in result

    def test_get_knowledge_base_info_uses_configured_dir(self, tmp_path):
        """get_knowledge_base_info() checks the configured vectordb_dir, not the default."""
        from src.tools.rag import configure_rag, get_knowledge_base_info

        missing_dir = str(tmp_path / "nonexistent_index")
        configure_rag({"vectordb_dir": missing_dir})
        result = get_knowledge_base_info()
        assert "No knowledge base found" in result


class TestNegativeValueValidation:
    """Tests for range validation of integer config fields (Bug m1)."""

    def test_negative_chunk_size_uses_default(self):
        """chunk_size: -1 in config should be rejected and default 2000 retained."""
        import tempfile
        from pathlib import Path

        from src.config import Config, _apply_config_file

        config_data = {"rag": {"chunk_size": -1}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = Config()
            _apply_config_file(config, Path(config_path))
            assert config.rag.chunk_size == 2000
        finally:
            Path(config_path).unlink()

    def test_zero_chunk_size_uses_default(self):
        """chunk_size: 0 must be rejected (must be > 0) and default 2000 retained."""
        import tempfile
        from pathlib import Path

        from src.config import Config, _apply_config_file

        config_data = {"rag": {"chunk_size": 0}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = Config()
            _apply_config_file(config, Path(config_path))
            assert config.rag.chunk_size == 2000
        finally:
            Path(config_path).unlink()

    def test_negative_chunk_overlap_uses_default(self):
        """chunk_overlap: -1 should be rejected and default 200 retained."""
        import tempfile
        from pathlib import Path

        from src.config import Config, _apply_config_file

        config_data = {"rag": {"chunk_overlap": -1}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = Config()
            _apply_config_file(config, Path(config_path))
            assert config.rag.chunk_overlap == 200
        finally:
            Path(config_path).unlink()

    def test_negative_delegate_timeout_uses_default(self):
        """delegate.default_timeout: -5 should be rejected and default 60 retained."""
        import tempfile
        from pathlib import Path

        from src.config import Config, _apply_config_file

        config_data = {"delegate": {"default_timeout": -5}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = Config()
            _apply_config_file(config, Path(config_path))
            assert config.delegate_default_timeout == 60
        finally:
            Path(config_path).unlink()


class TestProviderTypeCaseInsensitive:
    """Tests for case-insensitive provider type parsing (Bug m2)."""

    def test_provider_type_case_insensitive(self):
        """Provider type 'OpenAI' (mixed case) should be accepted and normalized to 'openai'."""
        config = Config()
        providers_data = {
            "myopenai": {
                "type": "OpenAI",
                "model": "gpt-4.1-mini",
                "api_key": "sk-test",
            }
        }
        _parse_providers_section(config, providers_data)
        assert "myopenai" in config.providers
        assert config.providers["myopenai"].type == "openai"

    def test_provider_type_uppercase_ollama(self):
        """Provider type 'OLLAMA' should be normalized to 'ollama'."""
        config = Config()
        providers_data = {
            "local": {
                "type": "OLLAMA",
                "base_url": "http://localhost:11434",
            }
        }
        _parse_providers_section(config, providers_data)
        assert "local" in config.providers
        assert config.providers["local"].type == "ollama"


class TestResolveModelRespectsCliProvider:
    """Tests that _resolve_model() respects CLI --provider override (Bug n2)."""

    def test_resolve_model_respects_cli_provider(self):
        """When _cli_provider_override=True, provider from ModelConfig is not applied."""
        from src.config import Config, ModelConfig, ProviderConfig, _resolve_model

        config = Config()
        config.providers["ollama"] = ProviderConfig(name="ollama", type="ollama")
        config.providers["openai"] = ProviderConfig(name="openai", type="openai", api_key="sk-x")
        config.models["reasoning"] = ModelConfig(provider="openai", model="gpt-4.1")
        config.provider = "ollama"
        config.model = "reasoning"
        config._cli_provider_override = True

        _resolve_model(config)

        assert config.provider == "ollama"
        assert config.model == "gpt-4.1"
        assert config._active_model is not None

    def test_resolve_model_switches_provider_without_cli_override(self):
        """Without CLI override, _resolve_model() updates provider from ModelConfig."""
        from src.config import Config, ModelConfig, ProviderConfig, _resolve_model

        config = Config()
        config.providers["ollama"] = ProviderConfig(name="ollama", type="ollama")
        config.providers["openai"] = ProviderConfig(name="openai", type="openai", api_key="sk-x")
        config.models["reasoning"] = ModelConfig(provider="openai", model="gpt-4.1")
        config.provider = "ollama"
        config.model = "reasoning"
        config._cli_provider_override = False

        _resolve_model(config)

        assert config.provider == "openai"
        assert config.model == "gpt-4.1"


class TestCreateEmbeddingsFromConfig:
    """Tests for create_embeddings_from_config in the provider registry."""

    def test_delegates_to_create_embeddings(self):
        """Verify it delegates to the low-level create_embeddings()."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from src.providers import create_embeddings_from_config

        mock_emb = MagicMock()
        with mock_patch("src.providers.create_embeddings", return_value=mock_emb) as mock_create:
            fn, tag = create_embeddings_from_config(
                "openai", model="text-embedding-3-small", api_key="sk-x"
            )

        mock_create.assert_called_once_with(
            "openai", model="text-embedding-3-small", base_url=None, api_key="sk-x"
        )
        assert fn is mock_emb
        assert tag == "openai/text-embedding-3-small"

    def test_tag_uses_default_model_when_none(self):
        """When model is None, tag should use the provider's default embedding model."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from src.providers import create_embeddings_from_config

        with mock_patch("src.providers.create_embeddings", return_value=MagicMock()):
            _, tag = create_embeddings_from_config("ollama")

        assert tag == "ollama/nomic-embed-text"

    def test_unknown_provider_raises_value_error(self):
        """Unknown provider type should raise ValueError."""
        from src.providers import create_embeddings_from_config

        with pytest.raises(ValueError, match="Unknown provider type"):
            create_embeddings_from_config("nonexistent")

    def test_anthropic_raises_not_implemented(self):
        """Anthropic has no embedding API and should raise NotImplementedError."""
        from src.providers import create_embeddings_from_config

        with pytest.raises(NotImplementedError):
            create_embeddings_from_config("anthropic")

    def test_base_url_forwarded(self):
        """Verify base_url is passed through to create_embeddings."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        from src.providers import create_embeddings_from_config

        with mock_patch("src.providers.create_embeddings", return_value=MagicMock()) as mock_create:
            create_embeddings_from_config("ollama", base_url="http://10.0.0.1:11434")

        mock_create.assert_called_once_with(
            "ollama", model=None, base_url="http://10.0.0.1:11434", api_key=None
        )


class TestProviderConfigValidation:
    """Tests for ProviderConfig.__post_init__ validation (BUG-056)."""

    def test_valid_temperature_accepted(self):
        """Valid temperature in [0.0, 2.0] is accepted."""
        cfg = ProviderConfig(name="test", type="openai", temperature=1.0)
        assert cfg.temperature == 1.0

    def test_temperature_zero_accepted(self):
        """Boundary value temperature=0.0 is accepted."""
        cfg = ProviderConfig(name="test", type="openai", temperature=0.0)
        assert cfg.temperature == 0.0

    def test_temperature_two_accepted(self):
        """Boundary value temperature=2.0 is accepted."""
        cfg = ProviderConfig(name="test", type="openai", temperature=2.0)
        assert cfg.temperature == 2.0

    def test_temperature_too_high_raises(self):
        """temperature > 2.0 raises ConfigError."""
        from src.config import ConfigError

        with pytest.raises(ConfigError, match="temperature"):
            ProviderConfig(name="test", type="openai", temperature=5.0)

    def test_temperature_negative_raises(self):
        """temperature < 0.0 raises ConfigError."""
        from src.config import ConfigError

        with pytest.raises(ConfigError, match="temperature"):
            ProviderConfig(name="test", type="openai", temperature=-0.1)

    def test_num_ctx_positive_accepted(self):
        """Positive num_ctx is accepted."""
        cfg = ProviderConfig(name="test", type="ollama", num_ctx=8192)
        assert cfg.num_ctx == 8192

    def test_num_ctx_zero_raises(self):
        """num_ctx=0 raises ConfigError."""
        from src.config import ConfigError

        with pytest.raises(ConfigError, match="num_ctx"):
            ProviderConfig(name="test", type="ollama", num_ctx=0)

    def test_num_ctx_negative_raises(self):
        """Negative num_ctx raises ConfigError."""
        from src.config import ConfigError

        with pytest.raises(ConfigError, match="num_ctx"):
            ProviderConfig(name="test", type="ollama", num_ctx=-1)

    def test_max_tokens_positive_accepted(self):
        """Positive max_tokens is accepted."""
        cfg = ProviderConfig(name="test", type="openai", max_tokens=2048)
        assert cfg.max_tokens == 2048

    def test_max_tokens_zero_raises(self):
        """max_tokens=0 raises ConfigError."""
        from src.config import ConfigError

        with pytest.raises(ConfigError, match="max_tokens"):
            ProviderConfig(name="test", type="openai", max_tokens=0)

    def test_max_tokens_negative_raises(self):
        """Negative max_tokens raises ConfigError."""
        from src.config import ConfigError

        with pytest.raises(ConfigError, match="max_tokens"):
            ProviderConfig(name="test", type="openai", max_tokens=-100)

    def test_none_fields_no_validation(self):
        """None values for optional fields skip validation entirely."""
        cfg = ProviderConfig(name="test", type="openai")
        assert cfg.temperature is None
        assert cfg.num_ctx is None
        assert cfg.max_tokens is None

    def test_invalid_type_raises(self):
        """Unknown provider type raises ConfigError."""
        from src.config import ConfigError

        with pytest.raises(ConfigError, match="not a recognized provider type"):
            ProviderConfig(name="test", type="bogus")

    def test_valid_provider_types_accepted(self):
        """All recognized provider types are accepted without error."""
        for ptype in ("openai", "ollama", "anthropic", "google"):
            cfg = ProviderConfig(name="test", type=ptype)
            assert cfg.type == ptype

    def test_invalid_temperature_in_providers_section_skips(self):
        """temperature: 5.0 in providers config logs a warning and skips the entry."""
        from src.config import Config

        config = Config()
        providers_data = {
            "bad": {
                "type": "openai",
                "temperature": 5.0,
            }
        }
        _parse_providers_section(config, providers_data)
        assert "bad" not in config.providers

    def test_invalid_num_ctx_in_providers_section_skips(self):
        """num_ctx: -1 in providers config logs a warning and skips the entry."""
        from src.config import Config

        config = Config()
        providers_data = {
            "bad": {
                "type": "ollama",
                "num_ctx": -1,
            }
        }
        _parse_providers_section(config, providers_data)
        assert "bad" not in config.providers

    def test_invalid_max_tokens_in_providers_section_skips(self):
        """max_tokens: 0 in providers config logs a warning and skips the entry."""
        from src.config import Config

        config = Config()
        providers_data = {
            "bad": {
                "type": "openai",
                "max_tokens": 0,
            }
        }
        _parse_providers_section(config, providers_data)
        assert "bad" not in config.providers
