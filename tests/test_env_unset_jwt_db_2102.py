"""#2102 / #2101 — unset COGTRIX_JWT_SECRET after read (+ opt-out).

#2223 already unsets provider/service/messaging secrets from ``os.environ`` after
they are copied into ``Config``. This closes the remaining gap for the JWT signing
secret. It is consumed outside the config loader (``app.py`` / ``auth.py``) via
``secret_from_env_or_file`` (#2103), so ``_apply_env_vars`` seeds the
survives-unset process cache before the unset — the documented read-once
dependency (#2101). A ``COGTRIX_KEEP_ENV_SECRETS`` flag opts out for debugging.

``COGTRIX_DB_URL`` is intentionally NOT unset: the engine layer resolves it
through its own ``data_dir``-aware, lazily-reimported path that reads
``os.environ`` directly, so unsetting it from the loader would rebind the engine.
The default SQLite URL carries no secret anyway.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import (
    _SECRETS_UNSET_AFTER_READ,
    _keep_env_secrets,
    load_config,
    secret_from_env_or_file,
)

_MIN_YAML = textwrap.dedent("""
    providers:
      openai: {type: openai, model: gpt-4.1-mini}
    models:
      m: {provider: openai, model: gpt-4.1-mini}
    model: m
    """)


def _cfg(tmp_path: Path) -> str:
    p = tmp_path / "c.yaml"
    p.write_text(_MIN_YAML)
    return str(p)


def _load(cfg_file: str, **kw):
    return load_config(SimpleNamespace(config_file=cfg_file), **kw)


class TestDenylist:
    def test_jwt_in_denylist(self) -> None:
        assert "COGTRIX_JWT_SECRET" in _SECRETS_UNSET_AFTER_READ

    def test_db_url_not_in_denylist(self) -> None:
        # Intentionally not unset — resolved by the engine layer's own
        # data_dir-aware path; unsetting it from the loader rebinds the engine.
        assert "COGTRIX_DB_URL" not in _SECRETS_UNSET_AFTER_READ


class TestKeepEnvSecretsFlag:
    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "Yes"])
    def test_truthy(self, val, monkeypatch) -> None:
        monkeypatch.setenv("COGTRIX_KEEP_ENV_SECRETS", val)
        assert _keep_env_secrets() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", ""])
    def test_falsey(self, val, monkeypatch) -> None:
        monkeypatch.setenv("COGTRIX_KEEP_ENV_SECRETS", val)
        assert _keep_env_secrets() is False

    def test_absent_is_false(self, monkeypatch) -> None:
        monkeypatch.delenv("COGTRIX_KEEP_ENV_SECRETS", raising=False)
        assert _keep_env_secrets() is False


class TestUnsetAfterRead:
    def test_jwt_and_db_unset_from_environ(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("COGTRIX_JWT_SECRET", "x" * 40)

        _load(_cfg(tmp_path))

        assert "COGTRIX_JWT_SECRET" not in os.environ

    def test_db_url_is_preserved(self, tmp_path, monkeypatch) -> None:
        """COGTRIX_DB_URL must NOT be unset by the loader (engine-layer concern)."""
        monkeypatch.setenv("COGTRIX_DB_URL", "postgresql+asyncpg://u:pw@h/db")

        _load(_cfg(tmp_path))

        assert os.environ.get("COGTRIX_DB_URL") == "postgresql+asyncpg://u:pw@h/db"

    def test_values_survive_unset_via_cache(self, tmp_path, monkeypatch) -> None:
        """Read-once (#2101): the value is still resolvable after the env is gone,
        because _apply_env_vars seeds the cache before unsetting (this is what the
        app.py / auth.py consumers rely on)."""
        monkeypatch.setenv("COGTRIX_JWT_SECRET", "y" * 40)

        _load(_cfg(tmp_path))

        assert "COGTRIX_JWT_SECRET" not in os.environ  # popped
        # ...but the consumer-facing accessor still returns the value (cache).
        assert secret_from_env_or_file("COGTRIX_JWT_SECRET") == "y" * 40

    def test_keep_flag_preserves_environ(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("COGTRIX_JWT_SECRET", "z" * 40)
        monkeypatch.setenv("COGTRIX_KEEP_ENV_SECRETS", "1")

        _load(_cfg(tmp_path))

        # Opt-out: the secret stays in the environment for debugging.
        assert os.environ.get("COGTRIX_JWT_SECRET") == "z" * 40

    def test_unset_secrets_false_preserves_environ(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("COGTRIX_JWT_SECRET", "w" * 40)

        _load(_cfg(tmp_path), unset_secrets=False)

        assert os.environ.get("COGTRIX_JWT_SECRET") == "w" * 40
