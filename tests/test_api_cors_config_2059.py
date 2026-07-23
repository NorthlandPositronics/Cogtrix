"""Regression tests for #2059 — CORS origins flow through the Config hierarchy.

Covers:
  - APIConfig.cors_origins default is localhost-only (no real-domain
    placeholder ships, so a misconfigured prod fails loudly).
  - api.cors_origins from a config file (YAML list and comma-separated string).
  - COGTRIX_CORS_ORIGINS env var override.
  - Precedence: env overrides the config file.
  - Invalid cors_origins raises ConfigError at load.
  - _get_cors_origins() resolves through Config and never returns the old
    https://app.cogtrix.ai placeholder by default.
"""

from __future__ import annotations

import pytest

from src.config import APIConfig, ConfigError, load_config


def _load_with_yaml(monkeypatch, yaml_path):
    """Point load_config at a temp YAML and clear stray CORS env overrides."""
    monkeypatch.setenv("COGTRIX_CONFIG_FILE", str(yaml_path))
    monkeypatch.delenv("COGTRIX_CORS_ORIGINS", raising=False)
    return load_config()


class TestCorsOriginsDefault:
    def test_default_is_localhost_only(self) -> None:
        cfg = APIConfig()
        assert cfg.cors_origins == [
            "http://localhost:5173",
            "http://localhost:3000",
        ]
        # The removed placeholder must not reappear.
        assert "https://app.cogtrix.ai" not in cfg.cors_origins

    def test_invalid_origins_raise(self) -> None:
        with pytest.raises(ConfigError, match="cors_origins"):
            APIConfig(cors_origins=["", "  "])
        with pytest.raises(ConfigError, match="cors_origins"):
            APIConfig(cors_origins="https://cogtrix.ai")  # type: ignore[arg-type]


class TestCorsOriginsConfigFile:
    def test_yaml_list(self, tmp_path, monkeypatch) -> None:
        yaml_path = tmp_path / "cogtrix.yaml"
        yaml_path.write_text(
            "api:\n"
            "  cors_origins:\n"
            "    - 'https://cogtrix.ai'\n"
            "    - 'https://www.cogtrix.ai'\n"
        )
        cfg = _load_with_yaml(monkeypatch, yaml_path)
        assert cfg.api.cors_origins == ["https://cogtrix.ai", "https://www.cogtrix.ai"]

    def test_yaml_comma_separated_string(self, tmp_path, monkeypatch) -> None:
        yaml_path = tmp_path / "cogtrix.yaml"
        yaml_path.write_text("api:\n  cors_origins: 'https://cogtrix.ai, https://www.cogtrix.ai'\n")
        cfg = _load_with_yaml(monkeypatch, yaml_path)
        assert cfg.api.cors_origins == ["https://cogtrix.ai", "https://www.cogtrix.ai"]


class TestCorsOriginsEnv:
    def test_env_override(self, tmp_path, monkeypatch) -> None:
        # Hermetic: point at an empty config file so the machine's own
        # config search is bypassed; the env var should win over the default.
        empty = tmp_path / "cogtrix.yaml"
        empty.write_text("{}\n")
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", str(empty))
        monkeypatch.setenv("COGTRIX_CORS_ORIGINS", "https://a.example, https://b.example")
        cfg = load_config()
        assert cfg.api.cors_origins == ["https://a.example", "https://b.example"]

    def test_env_overrides_config_file(self, tmp_path, monkeypatch) -> None:
        yaml_path = tmp_path / "cogtrix.yaml"
        yaml_path.write_text("api:\n  cors_origins:\n    - 'https://from-file.example'\n")
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", str(yaml_path))
        monkeypatch.setenv("COGTRIX_CORS_ORIGINS", "https://from-env.example")
        cfg = load_config()
        assert cfg.api.cors_origins == ["https://from-env.example"]


class TestGetCorsOriginsHelper:
    def test_resolves_env_through_config(self, tmp_path, monkeypatch) -> None:
        from src.api.app import _get_cors_origins

        empty = tmp_path / "cogtrix.yaml"
        empty.write_text("{}\n")
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", str(empty))
        monkeypatch.setenv("COGTRIX_CORS_ORIGINS", "https://cogtrix.ai")
        assert _get_cors_origins() == ["https://cogtrix.ai"]

    def test_default_has_no_placeholder(self, tmp_path, monkeypatch) -> None:
        from src.api.app import _get_cors_origins

        empty = tmp_path / "cogtrix.yaml"
        empty.write_text("{}\n")
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", str(empty))
        monkeypatch.delenv("COGTRIX_CORS_ORIGINS", raising=False)
        origins = _get_cors_origins()
        # Built-in default is localhost-only — the old real-domain placeholder
        # must never appear.
        assert "https://app.cogtrix.ai" not in origins
        assert origins == ["http://localhost:5173", "http://localhost:3000"]
