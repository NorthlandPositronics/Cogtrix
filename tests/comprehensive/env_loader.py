"""Load comprehensive-test secrets + pin the dedicated config, at test start.

The comprehensive harnesses (agent-complexity fleet, PM role test, Gate-2 eval)
share one secret-free config — ``cogtrix.comprehensive.yaml`` — whose API keys
are injected from the environment. This module is the "loaded upon tests start"
step:

* it ``python-dotenv``-loads ``tests/comprehensive/.env`` (gitignored secrets)
  into ``os.environ`` so Cogtrix's ``_apply_env_vars`` can resolve the keys, and
* it points ``COGTRIX_CONFIG_FILE`` at the dedicated config so ``load_config`` /
  ``find_config_file`` resolve it **deterministically** — avoiding the
  ``~/.config/cogtrix`` vs ``~/.cogtrix/config`` ambiguity that produced a stale
  config (and a dead key) in the 2026-06-24 cycle (finding F-01).

Usage
-----
At the start of a harness (or in a session-scoped fixture)::

    from tests.comprehensive.env_loader import load_comprehensive_env
    load_comprehensive_env()

For the Docker fleet, also pass ``config_path()`` as ``--config-path`` and
forward the loaded keys into the container (see #2219).
"""

from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: The secret-free dedicated config (committed).
CONFIG_PATH = _HERE / "cogtrix.comprehensive.yaml"
#: The real secrets file (gitignored; copy from the .example template).
ENV_PATH = _HERE / ".env"
#: The committed template, for the error message when ``.env`` is missing.
ENV_EXAMPLE = _HERE / "cogtrix.comprehensive.env.example"

#: Keys this loader is responsible for (for diagnostics / a missing-key check).
EXPECTED_KEYS = (
    "COGTRIX_PROVIDER_SPARK_API_KEY",
    "DEEPSEEK_API_KEY",
    "TAVILY_API_KEY",
    "OPENWEATHER_API_KEY",
)


def config_path() -> Path:
    """Absolute path to the dedicated comprehensive-test config."""
    return CONFIG_PATH


def load_comprehensive_env(
    *,
    set_config_file: bool = True,
    override: bool = False,
    require_env: bool = False,
) -> Path:
    """Load ``.env`` into the environment and pin the comprehensive config.

    Parameters
    ----------
    set_config_file:
        Also set ``COGTRIX_CONFIG_FILE`` to :data:`CONFIG_PATH` (via
        ``setdefault`` unless ``override``), so Cogtrix resolves this config.
    override:
        Let ``.env`` values overwrite variables already present in the
        environment. Default ``False`` (existing env wins — e.g. CI secrets).
    require_env:
        Raise ``FileNotFoundError`` if ``.env`` is absent. Default ``False`` so
        environments that inject keys another way (CI, docker ``env_file``) work
        without a local file.

    Returns
    -------
    Path
        :data:`CONFIG_PATH`.
    """
    if ENV_PATH.exists():
        from dotenv import load_dotenv  # python-dotenv (already a dependency)

        load_dotenv(ENV_PATH, override=override)
    elif require_env:
        raise FileNotFoundError(
            f"Comprehensive-test secrets not found: {ENV_PATH}\n"
            f"Copy the template and fill it in:\n"
            f"    cp {ENV_EXAMPLE.name} .env   # in {_HERE}"
        )

    if set_config_file:
        if override:
            os.environ["COGTRIX_CONFIG_FILE"] = str(CONFIG_PATH)
        else:
            os.environ.setdefault("COGTRIX_CONFIG_FILE", str(CONFIG_PATH))

    return CONFIG_PATH


def missing_keys() -> list[str]:
    """Return any :data:`EXPECTED_KEYS` not present (or empty) in the environment.

    Call after :func:`load_comprehensive_env` to decide whether a key-dependent
    harness (e.g. Gate-2 needs ``DEEPSEEK_API_KEY``) can run or should skip.
    """
    return [k for k in EXPECTED_KEYS if not os.environ.get(k)]
