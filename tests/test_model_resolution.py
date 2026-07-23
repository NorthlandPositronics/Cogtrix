"""Unit tests for Phase 1 of the Provider/Model Separation Refactor.

Covers:
- Config.get_active_model()
- Config.get_active_provider()
- Config.resolve_llm_config()
- Config.resolve_llm_config_for()
- _parse_models_section() handling of models.default
- create_chat_model_from_configs() in src.providers
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import (
    Config,
    ConfigError,
    ModelConfig,
    ProviderConfig,
    _apply_config_file,
    _parse_models_section,
    _parse_providers_section,
    _resolve_model,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_config(**overrides: object) -> Config:
    """Create a Config with common test providers and models."""
    providers = {
        "spark": ProviderConfig(
            name="spark", type="openai", base_url="http://spark:8080/v1", api_key="sk-test"
        ),
        "openai": ProviderConfig(name="openai", type="openai", api_key="sk-openai"),
        "local": ProviderConfig(name="local", type="ollama", base_url="http://localhost:11434"),
    }
    models = {
        "oss": ModelConfig(provider="spark", model="gpt-oss", temperature=0.5),
        "smart": ModelConfig(provider="openai", model="gpt-4o"),
        "embed": ModelConfig(provider="spark", model="qwen3-embedding", temperature=0.0),
        "fast-local": ModelConfig(provider="local", model="qwen3:8b", context_window=8192),
    }
    defaults: dict = dict(
        providers=providers,
        models=models,
    )
    defaults.update(overrides)
    return Config(**defaults)


# ── Tests: Config.get_active_model() ─────────────────────────────────


class TestGetActiveModel:
    def test_alias_set_and_found_returns_model_config(self) -> None:
        cfg = _make_config(active_model_alias="oss")
        mc = cfg.get_active_model()
        assert mc.model == "gpt-oss"
        assert mc.provider == "spark"
        assert mc.temperature == 0.5

    def test_alias_set_and_missing_raises_config_error(self) -> None:
        cfg = _make_config(active_model_alias="nonexistent")
        with pytest.raises(ConfigError, match="nonexistent"):
            cfg.get_active_model()

    def test_alias_none_returns_first_model_in_registry(self) -> None:
        cfg = _make_config(active_model_alias=None)
        result = cfg.get_active_model()
        # models dict is ordered; first key is "oss"
        assert result.model == "gpt-oss"
        assert result.provider == "spark"

    def test_alias_none_no_models_raises_config_error(self) -> None:
        cfg = _make_config(active_model_alias=None, models={})
        with pytest.raises(ConfigError, match="No models configured"):
            cfg.get_active_model()

    def test_alias_none_multiple_models_returns_first_inserted(self) -> None:
        cfg = _make_config(active_model_alias=None)
        result = cfg.get_active_model()
        first_alias = next(iter(cfg.models))
        assert result is cfg.models[first_alias]

    def test_alias_set_to_different_entry_returns_correct_one(self) -> None:
        cfg = _make_config(active_model_alias="smart")
        mc = cfg.get_active_model()
        assert mc.model == "gpt-4o"
        assert mc.provider == "openai"


# ── Tests: Config.get_active_provider() ──────────────────────────────


class TestGetActiveProvider:
    def test_returns_provider_for_active_model(self) -> None:
        cfg = _make_config(active_model_alias="oss")
        pc = cfg.get_active_provider()
        assert pc.name == "spark"
        assert pc.type == "openai"

    def test_raises_value_error_when_provider_not_configured(self) -> None:
        cfg = _make_config(
            active_model_alias="orphan",
            models={"orphan": ModelConfig(provider="missing_provider", model="some-model")},
        )
        with pytest.raises(ValueError, match="missing_provider"):
            cfg.get_active_provider()

    def test_different_model_alias_resolves_correct_provider(self) -> None:
        cfg = _make_config(active_model_alias="smart")
        pc = cfg.get_active_provider()
        assert pc.name == "openai"

    def test_ollama_provider_resolved_for_local_model(self) -> None:
        cfg = _make_config(active_model_alias="fast-local")
        pc = cfg.get_active_provider()
        assert pc.type == "ollama"
        assert pc.name == "local"


# ── Tests: Config.resolve_llm_config() ───────────────────────────────


class TestResolveLlmConfig:
    def test_returns_provider_and_model_tuple(self) -> None:
        cfg = _make_config(active_model_alias="oss")
        pc, mc = cfg.resolve_llm_config()
        assert pc.name == "spark"
        assert mc.model == "gpt-oss"

    def test_returned_provider_is_a_copy(self) -> None:
        cfg = _make_config(active_model_alias="oss")
        pc, _ = cfg.resolve_llm_config()
        assert pc is not cfg.providers["spark"]

    def test_mutating_returned_provider_does_not_affect_registry(self) -> None:
        cfg = _make_config(active_model_alias="oss")
        pc, _ = cfg.resolve_llm_config()
        pc.api_key = "mutated-key"
        assert cfg.providers["spark"].api_key == "sk-test"

    def test_model_config_fields_are_correct(self) -> None:
        cfg = _make_config(active_model_alias="oss")
        _, mc = cfg.resolve_llm_config()
        assert mc.temperature == 0.5
        assert mc.context_window is None

    def test_active_model_alias_none_falls_back_to_first_model(self) -> None:
        cfg = _make_config(active_model_alias=None)
        pc, mc = cfg.resolve_llm_config()
        # first model in registry is "oss" → spark/gpt-oss
        assert mc.model == "gpt-oss"
        assert pc is not cfg.providers["spark"]


# ── Tests: Config.resolve_llm_config_for() ───────────────────────────


class TestResolveLlmConfigFor:
    def test_alias_in_models_returns_correct_pair(self) -> None:
        cfg = _make_config()
        pc, mc = cfg.resolve_llm_config_for("oss")
        assert mc.model == "gpt-oss"
        assert pc.name == "spark"

    def test_provider_model_shorthand_synthesizes_model_config(self) -> None:
        cfg = _make_config()
        pc, mc = cfg.resolve_llm_config_for("openai/gpt-4.1-mini")
        assert mc.model == "gpt-4.1-mini"
        assert mc.provider == "openai"
        assert pc.name == "openai"

    def test_unknown_alias_raises_config_error(self) -> None:
        cfg = _make_config()
        with pytest.raises(ConfigError, match="unknown-alias"):
            cfg.resolve_llm_config_for("unknown-alias")

    def test_shorthand_with_unknown_provider_raises_value_error(self) -> None:
        cfg = _make_config()
        with pytest.raises((ConfigError, ValueError)):
            cfg.resolve_llm_config_for("ghost_provider/some-model")

    def test_returned_provider_is_a_copy_for_alias(self) -> None:
        cfg = _make_config()
        pc, _ = cfg.resolve_llm_config_for("smart")
        assert pc is not cfg.providers["openai"]

    def test_returned_provider_is_a_copy_for_shorthand(self) -> None:
        cfg = _make_config()
        pc, _ = cfg.resolve_llm_config_for("openai/gpt-3.5-turbo")
        assert pc is not cfg.providers["openai"]

    def test_shorthand_num_ctx_not_inherited_from_provider(self) -> None:
        cfg = _make_config()
        _, mc = cfg.resolve_llm_config_for("local/custom-model")
        assert mc.context_window is None

    def test_alias_ollama_provider_resolves_correctly(self) -> None:
        cfg = _make_config()
        pc, mc = cfg.resolve_llm_config_for("fast-local")
        assert pc.type == "ollama"
        assert mc.context_window == 8192


# ── Tests: _parse_models_section() with models.default ───────────────


class TestParseModelsSectionDefault:
    def test_default_key_sets_active_model_alias(self) -> None:
        cfg = Config(providers={})
        models_data = {
            "default": "fast",
            "fast": {"provider": "openai", "model": "gpt-4o-mini"},
        }
        _parse_models_section(cfg, models_data)
        assert cfg.active_model_alias == "fast"

    def test_default_key_not_parsed_as_model_entry(self) -> None:
        cfg = Config(providers={})
        models_data = {
            "default": "fast",
            "fast": {"provider": "openai", "model": "gpt-4o-mini"},
        }
        _parse_models_section(cfg, models_data)
        assert "default" not in cfg.models

    def test_other_models_still_parsed_when_default_present(self) -> None:
        cfg = Config(providers={})
        models_data = {
            "default": "fast",
            "fast": {"provider": "openai", "model": "gpt-4o-mini"},
            "slow": {"provider": "openai", "model": "gpt-4o"},
        }
        _parse_models_section(cfg, models_data)
        assert "fast" in cfg.models
        assert "slow" in cfg.models

    def test_non_string_default_is_ignored_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = Config(providers={})
        models_data = {
            "default": 42,
            "fast": {"provider": "openai", "model": "gpt-4o-mini"},
        }
        with caplog.at_level(logging.WARNING):
            _parse_models_section(cfg, models_data)
        assert cfg.active_model_alias is None
        assert any("default" in r.message.lower() for r in caplog.records)

    def test_empty_string_default_is_ignored_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = Config(providers={})
        models_data = {
            "default": "",
            "fast": {"provider": "openai", "model": "gpt-4o-mini"},
        }
        with caplog.at_level(logging.WARNING):
            _parse_models_section(cfg, models_data)
        assert cfg.active_model_alias is None

    def test_no_default_key_does_not_set_alias(self) -> None:
        cfg = Config(providers={})
        models_data = {
            "fast": {"provider": "openai", "model": "gpt-4o-mini"},
        }
        _parse_models_section(cfg, models_data)
        assert cfg.active_model_alias is None


# ── Tests: create_chat_model_from_configs() ───────────────────────────


class TestCreateChatModelFromConfigs:
    def test_basic_call_passes_correct_params(self) -> None:
        from src.providers import create_chat_model_from_configs

        pc = ProviderConfig(
            name="spark", type="openai", base_url="http://spark:8080/v1", api_key="sk-test"
        )
        mc = ModelConfig(provider="spark", model="gpt-oss", temperature=0.7)

        with patch("src.providers.create_chat_model") as mock_create:
            mock_create.return_value = MagicMock()
            create_chat_model_from_configs(pc, mc)
            mock_create.assert_called_once_with(
                "openai",
                model="gpt-oss",
                api_key="sk-test",
                base_url="http://spark:8080/v1",
                temperature=0.7,
                num_ctx=None,
                max_tokens=None,
                streaming=False,
            )

    def test_temperature_defaults_to_half_when_none(self) -> None:
        from src.providers import create_chat_model_from_configs

        pc = ProviderConfig(name="openai", type="openai", api_key="sk-openai")
        mc = ModelConfig(provider="openai", model="gpt-4o", temperature=None)

        with patch("src.providers.create_chat_model") as mock_create:
            mock_create.return_value = MagicMock()
            create_chat_model_from_configs(pc, mc)
            _, call_kwargs = mock_create.call_args
            assert call_kwargs["temperature"] == 0.5

    def test_num_ctx_passed_only_for_ollama_provider(self) -> None:
        from src.providers import create_chat_model_from_configs

        pc = ProviderConfig(name="local", type="ollama", base_url="http://localhost:11434")
        mc = ModelConfig(provider="local", model="qwen3:8b", context_window=8192)

        with patch("src.providers.create_chat_model") as mock_create:
            mock_create.return_value = MagicMock()
            create_chat_model_from_configs(pc, mc)
            _, call_kwargs = mock_create.call_args
            assert call_kwargs["num_ctx"] == 8192

    def test_num_ctx_not_passed_for_openai_provider(self) -> None:
        from src.providers import create_chat_model_from_configs

        pc = ProviderConfig(name="openai", type="openai", api_key="sk-openai")
        mc = ModelConfig(provider="openai", model="gpt-4o", context_window=8192)

        with patch("src.providers.create_chat_model") as mock_create:
            mock_create.return_value = MagicMock()
            create_chat_model_from_configs(pc, mc)
            _, call_kwargs = mock_create.call_args
            assert call_kwargs["num_ctx"] is None

    def test_streaming_flag_forwarded(self) -> None:
        from src.providers import create_chat_model_from_configs

        pc = ProviderConfig(name="openai", type="openai", api_key="sk-openai")
        mc = ModelConfig(provider="openai", model="gpt-4o")

        with patch("src.providers.create_chat_model") as mock_create:
            mock_create.return_value = MagicMock()
            create_chat_model_from_configs(pc, mc, streaming=True)
            _, call_kwargs = mock_create.call_args
            assert call_kwargs["streaming"] is True

    def test_max_tokens_from_model_config_passed(self) -> None:
        from src.providers import create_chat_model_from_configs

        pc = ProviderConfig(name="openai", type="openai", api_key="sk-openai")
        mc = ModelConfig(provider="openai", model="gpt-4o", max_tokens=2048)

        with patch("src.providers.create_chat_model") as mock_create:
            mock_create.return_value = MagicMock()
            create_chat_model_from_configs(pc, mc)
            _, call_kwargs = mock_create.call_args
            assert call_kwargs["max_tokens"] == 2048

    def test_returns_value_from_create_chat_model(self) -> None:
        from src.providers import create_chat_model_from_configs

        pc = ProviderConfig(name="openai", type="openai", api_key="sk-openai")
        mc = ModelConfig(provider="openai", model="gpt-4o")
        sentinel = MagicMock()

        with patch("src.providers.create_chat_model", return_value=sentinel):
            result = create_chat_model_from_configs(pc, mc)
            assert result is sentinel


# ── Tests: _resolve_model() ───────────────────────────────────────────


class TestResolveModel:
    def test_alias_matches_key_leaves_alias_unchanged(self) -> None:
        cfg = _make_config(active_model_alias="oss")
        _resolve_model(cfg)
        assert cfg.active_model_alias == "oss"

    def test_alias_none_returns_immediately_without_change(self) -> None:
        cfg = _make_config(active_model_alias=None)
        _resolve_model(cfg)
        assert cfg.active_model_alias is None

    def test_alias_is_model_name_remaps_to_alias_key(self) -> None:
        # active_model_alias set to "gpt-oss" (the .model value), not the key "oss"
        cfg = _make_config(active_model_alias="gpt-oss")
        _resolve_model(cfg)
        assert cfg.active_model_alias == "oss"

    def test_alias_is_model_name_of_second_entry_remaps_correctly(self) -> None:
        cfg = _make_config(active_model_alias="gpt-4o")
        _resolve_model(cfg)
        assert cfg.active_model_alias == "smart"

    def test_unknown_alias_with_models_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = _make_config(active_model_alias="totally-nonexistent")
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            _resolve_model(cfg)
        assert any("totally-nonexistent" in r.message for r in caplog.records)
        # alias is left as-is (not cleared)
        assert cfg.active_model_alias == "totally-nonexistent"

    def test_unknown_alias_with_no_models_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = _make_config(active_model_alias="ghost", models={})
        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            _resolve_model(cfg)
        assert not any("ghost" in r.message for r in caplog.records)

    def test_model_name_scan_picks_first_match_when_multiple_entries_share_model(
        self,
    ) -> None:
        providers = {
            "p1": ProviderConfig(name="p1", type="openai", api_key="k1"),
            "p2": ProviderConfig(name="p2", type="openai", api_key="k2"),
        }
        models = {
            "alias-a": ModelConfig(provider="p1", model="shared-model"),
            "alias-b": ModelConfig(provider="p2", model="shared-model"),
        }
        cfg = Config(providers=providers, models=models, active_model_alias="shared-model")
        _resolve_model(cfg)
        assert cfg.active_model_alias == "alias-a"


# ── Tests: Config.find_model_entry() ─────────────────────────────────


class TestFindModelEntry:
    def test_exact_alias_key_returns_alias_and_config(self) -> None:
        cfg = _make_config()
        alias, mc = cfg.find_model_entry("oss")
        assert alias == "oss"
        assert mc is not None
        assert mc.model == "gpt-oss"

    def test_model_name_scan_returns_correct_alias(self) -> None:
        cfg = _make_config()
        alias, mc = cfg.find_model_entry("gpt-oss")
        assert alias == "oss"
        assert mc is not None
        assert mc.provider == "spark"

    def test_model_name_scan_second_entry(self) -> None:
        cfg = _make_config()
        alias, mc = cfg.find_model_entry("gpt-4o")
        assert alias == "smart"
        assert mc is not None

    def test_not_found_returns_none_none(self) -> None:
        cfg = _make_config()
        alias, mc = cfg.find_model_entry("does-not-exist")
        assert alias is None
        assert mc is None

    def test_empty_models_returns_none_none(self) -> None:
        cfg = _make_config(models={})
        alias, mc = cfg.find_model_entry("anything")
        assert alias is None
        assert mc is None

    def test_alias_key_takes_priority_over_model_name_scan(self) -> None:
        providers = {"p": ProviderConfig(name="p", type="openai", api_key="k")}
        models = {
            "my-alias": ModelConfig(provider="p", model="my-alias"),
        }
        cfg = Config(providers=providers, models=models)
        alias, mc = cfg.find_model_entry("my-alias")
        assert alias == "my-alias"
        assert mc is not None


# ── Tests: Config.get_model_config() ─────────────────────────────────


class TestGetModelConfig:
    def test_explicit_name_returns_correct_config(self) -> None:
        cfg = _make_config()
        mc = cfg.get_model_config("smart")
        assert mc is not None
        assert mc.model == "gpt-4o"

    def test_none_name_falls_back_to_active_model_alias(self) -> None:
        cfg = _make_config(active_model_alias="oss")
        mc = cfg.get_model_config(None)
        assert mc is not None
        assert mc.model == "gpt-oss"

    def test_none_name_and_none_alias_returns_none(self) -> None:
        cfg = _make_config(active_model_alias=None)
        result = cfg.get_model_config(None)
        assert result is None

    def test_explicit_name_not_in_registry_returns_none(self) -> None:
        cfg = _make_config()
        result = cfg.get_model_config("no-such-model")
        assert result is None

    def test_active_alias_not_in_registry_returns_none(self) -> None:
        cfg = _make_config(active_model_alias="missing-alias")
        result = cfg.get_model_config(None)
        assert result is None


# ── Tests: _apply_config_file() — legacy top-level provider/model ────


class TestApplyConfigFileLegacyMigration:
    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "cogtrix.yaml"
        p.write_text(content)
        return p

    def test_legacy_provider_model_synthesizes_model_entry(self, tmp_path: Path) -> None:
        yaml_content = """
provider: openai
model: gpt-4o-mini
providers:
  openai:
    type: openai
    api_key: sk-test
"""
        cfg_file = self._write_yaml(tmp_path, yaml_content)
        cfg = Config()
        _apply_config_file(cfg, cfg_file)
        assert cfg.active_model_alias == "gpt-4o-mini"
        assert "gpt-4o-mini" in cfg.models
        assert cfg.models["gpt-4o-mini"].provider == "openai"
        assert cfg.models["gpt-4o-mini"].model == "gpt-4o-mini"

    def test_legacy_model_matches_existing_model_entry_sets_alias(self, tmp_path: Path) -> None:
        yaml_content = """
provider: openai
model: gpt-4o
providers:
  openai:
    type: openai
    api_key: sk-test
models:
  smart:
    provider: openai
    model: gpt-4o
"""
        cfg_file = self._write_yaml(tmp_path, yaml_content)
        cfg = Config()
        _apply_config_file(cfg, cfg_file)
        assert cfg.active_model_alias == "smart"

    def test_explicit_models_default_beats_legacy_top_level(self, tmp_path: Path) -> None:
        yaml_content = """
provider: openai
model: gpt-4o
providers:
  openai:
    type: openai
    api_key: sk-test
models:
  default: preferred
  preferred:
    provider: openai
    model: gpt-4.1-mini
"""
        cfg_file = self._write_yaml(tmp_path, yaml_content)
        cfg = Config()
        _apply_config_file(cfg, cfg_file)
        # models.default was applied; legacy migration is skipped
        assert cfg.active_model_alias == "preferred"

    def test_legacy_provider_only_no_model_does_not_synthesize(self, tmp_path: Path) -> None:
        yaml_content = """
provider: openai
providers:
  openai:
    type: openai
    api_key: sk-test
"""
        cfg_file = self._write_yaml(tmp_path, yaml_content)
        cfg = Config()
        _apply_config_file(cfg, cfg_file)
        # No _legacy_model → no synthesis
        assert cfg.active_model_alias is None


# ── Tests: _parse_providers_section() — auto-migration skip ──────────


class TestParseProvidersSectionAutoMigration:
    def test_provider_model_migrated_when_no_existing_models_entry(self) -> None:
        cfg = Config(providers={}, models={})
        providers_data = {
            "openai": {
                "type": "openai",
                "api_key": "sk-test",
                "model": "gpt-4o",
            }
        }
        _parse_providers_section(cfg, providers_data)
        assert "gpt-4o" in cfg.models
        assert cfg.models["gpt-4o"].model == "gpt-4o"

    def test_provider_model_skipped_when_alias_already_in_models(self) -> None:
        existing_mc = ModelConfig(provider="openai", model="gpt-4.1")
        cfg = Config(providers={}, models={"openai": existing_mc})
        providers_data = {
            "openai": {
                "type": "openai",
                "api_key": "sk-test",
                "model": "gpt-4o",
            }
        }
        _parse_providers_section(cfg, providers_data)
        # Pre-existing entry must be preserved unchanged
        assert cfg.models["openai"].model == "gpt-4.1"

    def test_provider_without_model_field_does_not_create_models_entry(self) -> None:
        cfg = Config(providers={}, models={})
        providers_data = {
            "openai": {
                "type": "openai",
                "api_key": "sk-test",
            }
        }
        _parse_providers_section(cfg, providers_data)
        assert "openai" not in cfg.models

    def test_migration_preserves_temperature_and_num_ctx(self) -> None:
        cfg = Config(providers={}, models={})
        providers_data = {
            "local": {
                "type": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen3:8b",
                "temperature": 0.3,
                "num_ctx": 16384,
            }
        }
        _parse_providers_section(cfg, providers_data)
        mc = cfg.models.get("qwen3:8b")
        assert mc is not None
        assert mc.temperature == 0.3
        assert mc.context_window == 16384


# ── Tests: load_config() — synthetic default generation ──────────────


class TestLoadConfigSyntheticDefault:
    def test_synthetic_default_created_when_alias_none_and_providers_exist(
        self,
    ) -> None:
        from src.config import load_config

        yaml_content = """
providers:
  myollama:
    type: ollama
    base_url: http://localhost:11434
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = Path(f.name)

        try:
            with patch("src.config.find_config_file", return_value=tmp_path):
                with patch("src.providers.get_default_model", return_value="qwen3:8b"):
                    cfg = load_config()
            assert cfg.active_model_alias == "myollama/qwen3:8b"
            assert "myollama/qwen3:8b" in cfg.models
            assert cfg.models["myollama/qwen3:8b"].provider == "myollama"
            assert cfg.models["myollama/qwen3:8b"].model == "qwen3:8b"
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_synthetic_default_not_duplicated_if_already_present(self) -> None:
        from src.config import load_config

        yaml_content = """
providers:
  myollama:
    type: ollama
    base_url: http://localhost:11434
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = Path(f.name)

        try:
            with patch("src.config.find_config_file", return_value=tmp_path):
                with patch("src.providers.get_default_model", return_value="qwen3:8b"):
                    cfg1 = load_config()
                    cfg2 = load_config()
            # Both runs should produce identical alias
            assert cfg1.active_model_alias == cfg2.active_model_alias
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_no_synthetic_default_when_no_providers(self) -> None:
        from src.config import load_config

        with patch("src.config.find_config_file", return_value=None):
            cfg = load_config()
        # No providers → no synthetic default created
        assert cfg.active_model_alias is None


# ── Tests: create_chat_model() — max_tokens parameter mapping ────────


class TestCreateChatModelMaxTokensMapping:
    """Verify max_tokens is mapped to the correct kwarg name per provider."""

    def test_openai_uses_max_tokens(self) -> None:
        from src.providers import create_chat_model

        with patch("src.providers._load_provider") as mock_load:
            mock_mod = MagicMock()
            mock_load.return_value = mock_mod
            create_chat_model("openai", model="gpt-4o", max_tokens=1024)
            call_kwargs = mock_mod.create_chat_model.call_args[1]
            assert call_kwargs["max_tokens"] == 1024
            assert "num_predict" not in call_kwargs
            assert "max_output_tokens" not in call_kwargs

    def test_ollama_uses_num_predict(self) -> None:
        from src.providers import create_chat_model

        with patch("src.providers._load_provider") as mock_load:
            mock_mod = MagicMock()
            mock_load.return_value = mock_mod
            create_chat_model("ollama", model="qwen3:8b", max_tokens=2048)
            call_kwargs = mock_mod.create_chat_model.call_args[1]
            assert call_kwargs["num_predict"] == 2048
            assert "max_tokens" not in call_kwargs

    def test_google_uses_max_output_tokens(self) -> None:
        from src.providers import create_chat_model

        with patch("src.providers._load_provider") as mock_load:
            mock_mod = MagicMock()
            mock_load.return_value = mock_mod
            create_chat_model("google", model="gemini-2.5-flash", max_tokens=4096)
            call_kwargs = mock_mod.create_chat_model.call_args[1]
            assert call_kwargs["max_output_tokens"] == 4096
            assert "max_tokens" not in call_kwargs

    def test_anthropic_uses_max_tokens(self) -> None:
        from src.providers import create_chat_model

        with patch("src.providers._load_provider") as mock_load:
            mock_mod = MagicMock()
            mock_load.return_value = mock_mod
            create_chat_model("anthropic", model="claude-sonnet-4-5", max_tokens=512)
            call_kwargs = mock_mod.create_chat_model.call_args[1]
            assert call_kwargs["max_tokens"] == 512

    def test_none_max_tokens_not_added_to_kwargs(self) -> None:
        from src.providers import create_chat_model

        with patch("src.providers._load_provider") as mock_load:
            mock_mod = MagicMock()
            mock_load.return_value = mock_mod
            create_chat_model("openai", model="gpt-4o", max_tokens=None)
            call_kwargs = mock_mod.create_chat_model.call_args[1]
            assert "max_tokens" not in call_kwargs
            assert "num_predict" not in call_kwargs
            assert "max_output_tokens" not in call_kwargs

    def test_num_ctx_only_passed_for_ollama(self) -> None:
        from src.providers import create_chat_model

        with patch("src.providers._load_provider") as mock_load:
            mock_mod = MagicMock()
            mock_load.return_value = mock_mod
            create_chat_model("openai", model="gpt-4o", num_ctx=8192)
            call_kwargs = mock_mod.create_chat_model.call_args[1]
            assert "num_ctx" not in call_kwargs

    def test_num_ctx_passed_for_ollama(self) -> None:
        from src.providers import create_chat_model

        with patch("src.providers._load_provider") as mock_load:
            mock_mod = MagicMock()
            mock_load.return_value = mock_mod
            create_chat_model("ollama", model="qwen3:8b", num_ctx=16384)
            call_kwargs = mock_mod.create_chat_model.call_args[1]
            assert call_kwargs["num_ctx"] == 16384

    def test_api_key_none_not_added_to_kwargs(self) -> None:
        from src.providers import create_chat_model

        with patch("src.providers._load_provider") as mock_load:
            mock_mod = MagicMock()
            mock_load.return_value = mock_mod
            create_chat_model("openai", model="gpt-4o", api_key=None)
            call_kwargs = mock_mod.create_chat_model.call_args[1]
            assert "api_key" not in call_kwargs


# ── Tests: Provider availability checks ──────────────────────────────


class TestProviderAvailability:
    def test_is_chat_available_unknown_type_returns_false(self) -> None:
        from src.providers import is_chat_available

        assert is_chat_available("nonexistent_provider") is False

    def test_is_embeddings_available_unknown_type_returns_false(self) -> None:
        from src.providers import is_embeddings_available

        assert is_embeddings_available("nonexistent_provider") is False

    def test_is_chat_available_known_type(self) -> None:
        from src.providers import is_chat_available

        # openai should always be available in the test env
        assert is_chat_available("openai") is True

    def test_is_embeddings_available_known_type(self) -> None:
        from src.providers import is_embeddings_available

        assert is_embeddings_available("openai") is True


# ── Tests: Provider helper functions ─────────────────────────────────


class TestProviderHelpers:
    def test_get_default_model_all_types(self) -> None:
        from src.providers import get_default_model

        assert get_default_model("openai") == "gpt-4.1-mini"
        assert get_default_model("ollama") == "qwen3:8b"
        assert get_default_model("anthropic") == "claude-sonnet-4-5"
        assert get_default_model("google") == "gemini-2.5-flash"

    def test_get_default_model_unknown_falls_back_to_openai(self) -> None:
        from src.providers import get_default_model

        assert get_default_model("unknown") == "gpt-4.1-mini"

    def test_get_default_embedding_model_anthropic_is_none(self) -> None:
        from src.providers import get_default_embedding_model

        assert get_default_embedding_model("anthropic") is None

    def test_get_default_embedding_model_openai(self) -> None:
        from src.providers import get_default_embedding_model

        assert get_default_embedding_model("openai") == "text-embedding-3-small"

    def test_get_default_base_url_ollama(self) -> None:
        from src.providers import get_default_base_url

        assert get_default_base_url("ollama") == "http://localhost:11434"

    def test_get_default_base_url_openai_is_none(self) -> None:
        from src.providers import get_default_base_url

        assert get_default_base_url("openai") is None
