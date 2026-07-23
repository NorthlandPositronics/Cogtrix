"""#2101 — environment variables are read exactly once per process.

``load_config()`` re-applies ``os.environ`` on every call. Runtime paths that
re-invoke it (RAG ingest, the DB-engine default-URL resolver, the API CORS
resolver, the weather/WhatsApp/Telegram tool loaders) therefore re-read the
environment — and, after the #2102/#2223 unset, observe missing secrets.

``get_cached_config()`` resolves config ONCE and returns that instance to every
later caller, so the environment is read a single time. ``reload_cached_config()``
is the single sanctioned re-read path (the admin reload endpoint).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import src.config as cfgmod
from src.config import (
    get_cached_config,
    load_config,
    reload_cached_config,
    reset_cached_config,
)

_YAML = textwrap.dedent("""
    providers:
      openai: {type: openai, model: gpt-4.1-mini}
    models:
      m: {provider: openai, model: gpt-4.1-mini}
    model: m
    """)


def _write_cfg(tmp_path: Path) -> str:
    p = tmp_path / "c.yaml"
    p.write_text(_YAML)
    return str(p)


class TestCachedConfigIdentity:
    def test_returns_same_instance(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", _write_cfg(tmp_path))
        reset_cached_config()
        a = get_cached_config()
        b = get_cached_config()
        assert a is b

    def test_reset_forces_new_instance(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", _write_cfg(tmp_path))
        reset_cached_config()
        a = get_cached_config()
        reset_cached_config()
        b = get_cached_config()
        assert a is not b


class TestReadOnce:
    def test_env_read_once_secret_survives_change(self, monkeypatch, tmp_path) -> None:
        """Once resolved, a later env change does NOT alter the cached config —
        the environment was read exactly once."""
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", _write_cfg(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-first")
        reset_cached_config()

        cfg1 = get_cached_config()
        assert cfg1.providers["openai"].api_key == "sk-first"

        # Change the env AND clear the secret cache: the cached Config must not move.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-second")
        cfgmod._reset_secret_env_cache()
        cfg2 = get_cached_config()
        assert cfg2 is cfg1
        assert cfg2.providers["openai"].api_key == "sk-first"

    def test_only_one_load_config_call(self, monkeypatch, tmp_path) -> None:
        """Repeated get_cached_config() resolves via load_config exactly once."""
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", _write_cfg(tmp_path))
        reset_cached_config()
        real = cfgmod.load_config
        with patch.object(cfgmod, "load_config", side_effect=real) as spy:
            get_cached_config()
            get_cached_config()
            get_cached_config()
        assert spy.call_count == 1


class TestReload:
    def test_reload_picks_up_change(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", _write_cfg(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-orig")
        reset_cached_config()
        cfg1 = get_cached_config()
        assert cfg1.providers["openai"].api_key == "sk-orig"

        # The sanctioned re-read path (admin reload) DOES re-read the environment.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-rotated")
        cfgmod._reset_secret_env_cache()
        cfg2 = reload_cached_config()
        assert cfg2 is not cfg1
        assert cfg2.providers["openai"].api_key == "sk-rotated"
        # And subsequent get_cached_config() returns the reloaded instance.
        assert get_cached_config() is cfg2


class TestRuntimeCallersReuseCache:
    """The #2101 re-read sites must reuse get_cached_config (env read once)."""

    def test_engine_default_url_uses_cached_config(self, monkeypatch, tmp_path) -> None:
        # No COGTRIX_DB_URL / COGTRIX_DATA_DIR → engine falls back to data_dir
        # from config, which must come from the cached config.
        monkeypatch.delenv("COGTRIX_DB_URL", raising=False)
        monkeypatch.delenv("COGTRIX_DATA_DIR", raising=False)
        reset_cached_config()
        get_cached_config()  # seed the cache
        with patch.object(cfgmod, "load_config", side_effect=AssertionError("re-read!")):
            # Importing the resolver and calling it must NOT re-invoke load_config.
            from src.api.db.engine import _resolve_default_db_url

            url = _resolve_default_db_url()
        assert url.startswith("sqlite+aiosqlite:///")

    def test_cors_resolver_uses_cached_config(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", _write_cfg(tmp_path))
        reset_cached_config()
        get_cached_config()  # seed
        with patch.object(cfgmod, "load_config", side_effect=AssertionError("re-read!")):
            from src.api.app import _get_cors_origins

            origins = _get_cors_origins()
        assert isinstance(origins, list) and origins


def test_load_config_still_independent(monkeypatch, tmp_path) -> None:
    """load_config() itself is unchanged — direct callers still get a fresh read
    (the caching lives only in get_cached_config)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-direct")
    c = load_config(SimpleNamespace(config_file=_write_cfg(tmp_path)))
    assert c.providers["openai"].api_key == "sk-direct"
