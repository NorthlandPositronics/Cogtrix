"""Read-once + unset of config-authoritative secret env vars (#2223).

After ``load_config`` copies a secret into the ``Config``, that env var is
removed from ``os.environ`` so a shell/code-exec tool subprocess can't inherit
it.  Phase 2 extends this to the messaging/weather tool secrets: each tool
now declares ``TOOL_SETUP`` which captures its token from the ``Config``
before the env var is unset, so the token is available post-unset.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import load_config

# Minimal config file so the test never resolves a machine config, and so the
# `spark` provider exists for the generic-key path.
_YAML = textwrap.dedent("""
    providers:
      spark: {type: openai, base_url: http://x/v1}
    models:
      coder: {provider: spark, model: qwen3-coder}
    model: coder
    """)

_UNSET = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "TAVILY_API_KEY",
    "BRAVE_API_KEY",
    "EXA_API_KEY",
    "SERPAPI_API_KEY",
    "GOOGLE_API_KEY",
    "COGTRIX_PROVIDER_SPARK_API_KEY",
    # Phase 2: messaging / weather secrets now also config-authoritative
    "OPENWEATHER_API_KEY",
    "COGTRIX_WHATSAPP_API_KEY",
    "COGTRIX_TELEGRAM_TOKEN",
    "COGTRIX_SLACK_BOT_TOKEN",
]
_DEFERRED: list[str] = []  # all deferred secrets resolved in phase 2


@pytest.fixture
def cfg_file(tmp_path: Path) -> str:
    p = tmp_path / "c.yaml"
    p.write_text(_YAML)
    return str(p)


def _set_all(monkeypatch):
    for k in _UNSET + _DEFERRED:
        monkeypatch.setenv(k, f"secret-{k.lower()}")
    monkeypatch.setenv("COGTRIX_MODEL", "coder")  # non-secret settings var


def _load(cfg_file, **kw):
    return load_config(SimpleNamespace(config_file=cfg_file), **kw)


def test_authoritative_secrets_unset_after_load(cfg_file, monkeypatch):
    _set_all(monkeypatch)
    _load(cfg_file)
    for k in _UNSET:
        assert k not in os.environ, f"{k} should be unset after load"


def test_config_still_carries_the_keys(cfg_file, monkeypatch):
    _set_all(monkeypatch)
    cfg = _load(cfg_file)
    # Read happened before unset → the Config has them.
    assert cfg.providers["spark"].api_key == "secret-cogtrix_provider_spark_api_key"
    assert cfg.providers["openai"].api_key == "secret-openai_api_key"
    assert cfg.providers["deepseek"].api_key == "secret-deepseek_api_key"
    assert cfg.tavily_api_key == "secret-tavily_api_key"


def test_deferred_secrets_remain(cfg_file, monkeypatch):
    """Phase 2 complete: _DEFERRED is empty — all former deferred secrets are now unset."""
    _set_all(monkeypatch)
    _load(cfg_file)
    # No deferred secrets remain; verify the list is empty as a contract check.
    assert _DEFERRED == [], "All secrets should be config-authoritative after phase 2"


def test_non_secret_settings_var_remains(cfg_file, monkeypatch):
    _set_all(monkeypatch)
    _load(cfg_file)
    assert os.environ.get("COGTRIX_MODEL") == "coder"


def test_opt_out_keeps_everything(cfg_file, monkeypatch):
    _set_all(monkeypatch)
    _load(cfg_file, unset_secrets=False)
    for k in _UNSET:
        assert os.environ.get(k) == f"secret-{k.lower()}"


def test_subprocess_does_not_inherit_unset_secrets(cfg_file, monkeypatch):
    """The core property: a child process sees none of the unset secrets."""
    _set_all(monkeypatch)
    _load(cfg_file)
    out = subprocess.run(
        [sys.executable, "-c", "import os; print('\\n'.join(os.environ))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    leaked = [k for k in _UNSET if k in out]
    assert not leaked, f"secrets leaked to subprocess env: {leaked}"
