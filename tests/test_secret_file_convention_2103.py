"""#2103 — `<name>_FILE` secret-file convention (Docker/K8s/Vault secrets).

Secrets should be deliverable as a *file* (Docker/Swarm secrets at
``/run/secrets/``, a Kubernetes secret volume, a Vault-agent sidecar) instead of
an environment variable — keeping the value out of ``docker inspect`` /
``/proc/<pid>/environ`` / child inheritance. For every secret-bearing setting,
``<name>_FILE=<path>`` reads the secret from that path.

Precedence (highest → lowest): explicit ``<name>`` env value → ``<name>_FILE`` →
config-file value → default. A missing/empty ``_FILE`` target fails loudly.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogtrix_core.config import (
    ConfigError,
    _read_secret_from_file,
    _secret_env,
    load_config,
    secret_from_env_or_file,
)


def _secret_file(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def _load(cfg_file: str):
    return load_config(SimpleNamespace(config_file=cfg_file))


_YAML_OPENAI = textwrap.dedent("""
    providers:
      openai: {type: openai, model: gpt-4.1-mini}
    models:
      m: {provider: openai, model: gpt-4.1-mini}
    model: m
    """)


# ── _read_secret_from_file unit behaviour ───────────────────────────────────


class TestReadSecretFromFile:
    def test_returns_none_when_file_var_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
        assert _read_secret_from_file("OPENAI_API_KEY") is None

    def test_reads_value(self, tmp_path, monkeypatch) -> None:
        path = _secret_file(tmp_path, "k", "sk-abc")
        monkeypatch.setenv("OPENAI_API_KEY_FILE", path)
        assert _read_secret_from_file("OPENAI_API_KEY") == "sk-abc"

    def test_trims_single_trailing_newline(self, tmp_path, monkeypatch) -> None:
        path = _secret_file(tmp_path, "k", "sk-abc\n")
        monkeypatch.setenv("OPENAI_API_KEY_FILE", path)
        assert _read_secret_from_file("OPENAI_API_KEY") == "sk-abc"

    def test_trims_single_crlf(self, tmp_path, monkeypatch) -> None:
        path = _secret_file(tmp_path, "k", "sk-abc\r\n")
        monkeypatch.setenv("OPENAI_API_KEY_FILE", path)
        assert _read_secret_from_file("OPENAI_API_KEY") == "sk-abc"

    def test_preserves_interior_whitespace(self, tmp_path, monkeypatch) -> None:
        # Only ONE trailing newline is trimmed; a secret may legitimately
        # contain other characters, so don't over-strip.
        path = _secret_file(tmp_path, "k", "a b\nc\n")
        monkeypatch.setenv("OPENAI_API_KEY_FILE", path)
        assert _read_secret_from_file("OPENAI_API_KEY") == "a b\nc"

    def test_missing_file_raises_named_error(self, tmp_path, monkeypatch) -> None:
        missing = str(tmp_path / "nope")
        monkeypatch.setenv("OPENAI_API_KEY_FILE", missing)
        with pytest.raises(ConfigError) as exc:
            _read_secret_from_file("OPENAI_API_KEY")
        assert "OPENAI_API_KEY_FILE" in str(exc.value)
        assert missing in str(exc.value)

    def test_empty_file_raises_named_error(self, tmp_path, monkeypatch) -> None:
        path = _secret_file(tmp_path, "k", "\n")
        monkeypatch.setenv("OPENAI_API_KEY_FILE", path)
        with pytest.raises(ConfigError) as exc:
            _read_secret_from_file("OPENAI_API_KEY")
        assert "empty" in str(exc.value).lower()


# ── Precedence via _secret_env ──────────────────────────────────────────────


class TestSecretEnvPrecedence:
    def test_file_used_when_env_absent(self, tmp_path, monkeypatch) -> None:
        path = _secret_file(tmp_path, "k", "from-file")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY_FILE", path)
        assert _secret_env("OPENAI_API_KEY") == "from-file"

    def test_explicit_env_wins_over_file(self, tmp_path, monkeypatch) -> None:
        path = _secret_file(tmp_path, "k", "from-file")
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        monkeypatch.setenv("OPENAI_API_KEY_FILE", path)
        assert _secret_env("OPENAI_API_KEY") == "from-env"


# ── End-to-end through load_config ──────────────────────────────────────────


class TestLoadConfigFileConvention:
    def test_provider_key_from_file(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_YAML_OPENAI)
        path = _secret_file(tmp_path, "openai_key", "sk-file-openai\n")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY_FILE", path)

        config = _load(str(cfg))
        assert config.providers["openai"].api_key == "sk-file-openai"

    def test_file_overrides_inline_config_value(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(textwrap.dedent("""
                providers:
                  openai: {type: openai, model: gpt-4.1-mini, api_key: inline-key}
                models:
                  m: {provider: openai, model: gpt-4.1-mini}
                model: m
                """))
        path = _secret_file(tmp_path, "openai_key", "file-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY_FILE", path)

        config = _load(str(cfg))
        # Precedence: _FILE > config-file inline value.
        assert config.providers["openai"].api_key == "file-key"

    def test_generic_provider_key_from_file(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(textwrap.dedent("""
                providers:
                  spark: {type: openai, base_url: http://x/v1}
                models:
                  coder: {provider: spark, model: qwen3-coder}
                model: coder
                """))
        path = _secret_file(tmp_path, "spark_key", "sk-spark-file")
        monkeypatch.delenv("COGTRIX_PROVIDER_SPARK_API_KEY", raising=False)
        monkeypatch.setenv("COGTRIX_PROVIDER_SPARK_API_KEY_FILE", path)

        config = _load(str(cfg))
        assert config.providers["spark"].api_key == "sk-spark-file"

    def test_service_key_from_file(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_YAML_OPENAI)
        path = _secret_file(tmp_path, "tavily_key", "tvly-file\n")
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setenv("TAVILY_API_KEY_FILE", path)

        config = _load(str(cfg))
        assert config.services["tavily"]["api_key"] == "tvly-file"

    def test_messaging_token_from_file(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_YAML_OPENAI)
        path = _secret_file(tmp_path, "tg_token", "12345:bot-token\n")
        monkeypatch.delenv("COGTRIX_TELEGRAM_TOKEN", raising=False)
        monkeypatch.setenv("COGTRIX_TELEGRAM_TOKEN_FILE", path)

        config = _load(str(cfg))
        assert config.services["telegram"]["bot_token"] == "12345:bot-token"

    def test_missing_file_target_fails_load(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_YAML_OPENAI)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY_FILE", str(tmp_path / "absent"))

        with pytest.raises(ConfigError):
            _load(str(cfg))


# ── Public wrapper for JWT / DB consumers ───────────────────────────────────


class TestSecretFromEnvOrFile:
    def test_jwt_secret_from_file(self, tmp_path, monkeypatch) -> None:
        path = _secret_file(tmp_path, "jwt", "x" * 40 + "\n")
        monkeypatch.delenv("COGTRIX_JWT_SECRET", raising=False)
        monkeypatch.setenv("COGTRIX_JWT_SECRET_FILE", path)
        assert secret_from_env_or_file("COGTRIX_JWT_SECRET") == "x" * 40

    def test_db_url_env_wins_over_file(self, tmp_path, monkeypatch) -> None:
        path = _secret_file(tmp_path, "db", "postgres://from-file/db")
        monkeypatch.setenv("COGTRIX_DB_URL", "postgres://from-env/db")
        monkeypatch.setenv("COGTRIX_DB_URL_FILE", path)
        assert secret_from_env_or_file("COGTRIX_DB_URL") == "postgres://from-env/db"

    def test_returns_none_when_neither_set(self, monkeypatch) -> None:
        monkeypatch.delenv("COGTRIX_DB_URL", raising=False)
        monkeypatch.delenv("COGTRIX_DB_URL_FILE", raising=False)
        assert secret_from_env_or_file("COGTRIX_DB_URL") is None
