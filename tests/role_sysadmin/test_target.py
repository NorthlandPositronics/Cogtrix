"""Live-container smoke test for the SUT target (requires Docker).

Marked ``docker`` and skipped when no Docker daemon is reachable. Builds the
image, boots a privileged systemd container, and asserts the basics the whole
harness depends on: sshd reachable on the ephemeral key, systemd up, harness-side
verification + check-script execution work, and teardown removes the container.

Run with:  uv run pytest tests/role_sysadmin/test_target.py -q -m docker
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.role_sysadmin.target import Target, docker_available

pytestmark = pytest.mark.docker


@pytest.fixture(autouse=True)
def _require_docker() -> None:
    if not docker_available():
        pytest.skip("docker daemon not available")


def _container_exists(name: str) -> bool:
    out = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return name in out.stdout.split()


@pytest.mark.timeout(600)  # first run builds the image (apt) — well over the 30s default
def test_target_boots_runs_and_tears_down(tmp_path: Path) -> None:
    name = "role_sa_pytest_smoke"
    target = Target.create(name)
    try:
        # sshd reachable on the ephemeral key.
        assert target.reachable()
        # systemd is PID 1 and up (running/degraded both acceptable in a container).
        sysd = target.run("systemctl is-system-running")
        assert ("running" in sysd.output) or ("degraded" in sysd.output)
        # privileged service control works.
        assert target.run("sudo systemctl is-active ssh").ok
        # the check-script path works end-to-end (scp + run as root).
        check = tmp_path / "ok_check.sh"
        check.write_text("#!/usr/bin/env bash\necho PASS\nexit 0\n", encoding="utf-8")
        assert target.run_check(check).ok
        # the agent connection string is well-formed.
        inv = target.agent_ssh_invocation()
        assert "ssh -i" in inv and f"-p {target.port}" in inv and "ops@127.0.0.1" in inv
    finally:
        target.teardown()
    assert not _container_exists(name)
