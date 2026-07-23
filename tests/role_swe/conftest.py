"""Pytest config for the role_swe harness.

LOCAL-ONLY: the software-development holistic test runs locally only, never in CI
(operator directive 2026-06-27). The primary CI guard is the unit-test shard
resolver (``.github/scripts/resolve-unit-test-shard.sh``), which excludes the whole
``tests/role_swe`` tree from enumeration. This conftest adds a defensive second
guard: when running under GitHub Actions, skip collecting the harness self-tests
entirely — so no future CI step can pick them up via a broad ``pytest tests/``.

The ``project/`` subtree is the **ledgerlite SUT** fixture — exercised by the
harness against per-scenario workspace copies, never collected (it imports the
uninstalled ``ledgerlite`` package). It is ignored in every environment.
"""

from __future__ import annotations

import os

collect_ignore = ["project"]

# Defence in depth: role_swe is never collected in CI, by any path.
if os.environ.get("GITHUB_ACTIONS"):
    collect_ignore_glob = ["test_*.py"]
