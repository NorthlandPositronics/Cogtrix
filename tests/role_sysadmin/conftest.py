"""Pytest config for the role_sysadmin harness.

LOCAL-ONLY: the systems-administration holistic test runs locally only, never in
CI (it needs Docker + a privileged container + live models). The CI unit-test
shard resolver excludes the whole ``tests/role_sysadmin`` tree; this conftest adds
a defensive second guard (skip self-tests under GitHub Actions) and registers the
``docker`` marker used by the live-container tests.

The ``target/`` (Dockerfile build context) and ``scenarios/`` (YAML + check
scripts) subtrees are fixtures, never collected as tests.
"""

from __future__ import annotations

import os

import pytest

collect_ignore = ["target", "scenarios"]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "docker: needs a working Docker daemon (boots the live SUT container)"
    )


# Defence in depth: role_sysadmin is never collected in CI, by any path.
if os.environ.get("GITHUB_ACTIONS"):
    collect_ignore_glob = ["test_*.py"]
