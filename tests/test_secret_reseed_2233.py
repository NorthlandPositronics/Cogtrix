"""Regression test for #2233 — env-only secret keys must survive config
re-resolution after the #2223 unset.

The API re-resolves config (per session / turn / reload). #2223 pops secret env
vars from os.environ after the first load, so a provider whose key is ONLY in
the env (no inline ``api_key:`` in the YAML) resolved to an empty key on the
second load → provider 401 on agent turns. The fix caches secret env values in
process and re-seeds them on later loads, so the key stays available regardless
of how many times config is resolved. The cache is in-process (not inherited by
subprocesses), preserving the #2223 hardening.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from src.config import _reset_secret_env_cache, load_config

# Provider keyed ONLY by env (no inline api_key) + one with an inline key.
_YAML = textwrap.dedent("""
    providers:
      deepseek: {type: openai, base_url: https://api.deepseek.com/v1}
      spark: {type: openai, base_url: http://192.168.70.254:8080/v1, api_key: inline-spark-key}
    models:
      ds: {provider: deepseek, model: deepseek-v4}
    model: ds
    """)


def _cfg_file(tmp_path: Path) -> str:
    p = tmp_path / "c.yaml"
    p.write_text(_YAML)
    return str(p)


def _load(cfg_file: str, **kw):
    return load_config(SimpleNamespace(config_file=cfg_file), **kw)


def test_env_only_key_survives_second_load(tmp_path, monkeypatch):
    """The #2233 bug: env-only key empty on the 2nd load. Must now persist."""
    cfg_file = _cfg_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-env-only")

    cfg1 = _load(cfg_file)
    assert cfg1.providers["deepseek"].api_key == "sk-deepseek-env-only"
    # #2223 popped the env var after the first load.
    assert "DEEPSEEK_API_KEY" not in os.environ

    # The API re-resolves config; the env is now gone — key must still resolve.
    cfg2 = _load(cfg_file)
    assert (
        cfg2.providers["deepseek"].api_key == "sk-deepseek-env-only"
    ), "env-only provider key must survive re-resolution after #2223 unset"
    # Inline-key provider unaffected on both loads.
    assert cfg2.providers["spark"].api_key == "inline-spark-key"


def test_generic_provider_env_key_survives_second_load(tmp_path, monkeypatch):
    """Same invariant for the generic COGTRIX_PROVIDER_<NAME>_API_KEY (#2222)."""
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
            providers:
              spark: {type: openai, base_url: http://x/v1}
            models:
              coder: {provider: spark, model: qwen3-coder}
            model: coder
            """))
    monkeypatch.setenv("COGTRIX_PROVIDER_SPARK_API_KEY", "sk-spark-env-only")

    cfg1 = _load(str(p))
    assert cfg1.providers["spark"].api_key == "sk-spark-env-only"
    assert "COGTRIX_PROVIDER_SPARK_API_KEY" not in os.environ

    cfg2 = _load(str(p))
    assert cfg2.providers["spark"].api_key == "sk-spark-env-only"


def test_cache_does_not_leak_across_reset(tmp_path, monkeypatch):
    """The cache must be resettable so it can't pollute unrelated loads/tests."""
    cfg_file = _cfg_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-leaky")
    _load(cfg_file)
    assert load_config(SimpleNamespace(config_file=cfg_file)).providers["deepseek"].api_key == (
        "sk-leaky"
    )

    # After an explicit reset (what the autouse conftest fixture does per test),
    # a fresh load with no env must NOT see the previously-cached key.
    _reset_secret_env_cache()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    cfg = load_config(SimpleNamespace(config_file=cfg_file))
    assert cfg.providers["deepseek"].api_key in (None, "")


def test_subprocess_still_does_not_inherit_secret(tmp_path, monkeypatch):
    """#2223 invariant intact: the cached secret stays in-process, never in the
    inheritable environment a child would see."""
    cfg_file = _cfg_file(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-not-leak")
    _load(cfg_file)
    out = subprocess.run(
        [sys.executable, "-c", "import os; print('DEEPSEEK_API_KEY' in os.environ)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "False", "secret must not leak into a subprocess env"
