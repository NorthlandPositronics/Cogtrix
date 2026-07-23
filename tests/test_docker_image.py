"""Docker image integration tests.

These tests build and run the Cogtrix Docker image. They require a Docker
daemon and are skipped automatically when none is available.

Run with:
    uv run pytest tests/test_docker_image.py -m docker -v

Skip in regular runs:
    uv run pytest tests/ -m "not docker"
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
IMAGE_TAG = "cogtrix-test:pytest-session"
JWT_SECRET = "thisisaverylongsecretkey1234567890xx"  # 36 chars, meets >=32
HEALTHCHECK_PORT = 18000


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=False)
def docker_available():
    """Skip entire module if Docker daemon is not running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("Docker CLI not found or timed out -- skipping container integration tests")
    if result.returncode != 0:
        pytest.skip("Docker daemon not available -- skipping container integration tests")


@pytest.fixture(scope="session")
def docker_image(docker_available):
    """Build the image once; yield tag; remove after all tests."""
    result = subprocess.run(
        ["docker", "build", "-f", str(ROOT / "docker" / "Dockerfile"), "-t", IMAGE_TAG, str(ROOT)],
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, f"docker build failed:\n{result.stderr.decode()}"
    yield IMAGE_TAG
    subprocess.run(["docker", "rmi", "-f", IMAGE_TAG], capture_output=True, timeout=30)


def _run_container(
    image: str,
    args: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    entrypoint: str | None = None,
    stdin_data: bytes | None = b"",
    timeout: int = 15,
    extra_docker_args: list[str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Helper: run a container and return the CompletedProcess."""
    cmd = ["docker", "run", "--rm"]
    if entrypoint is not None:
        cmd += ["--entrypoint", entrypoint]
    if env:
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
    if extra_docker_args:
        cmd += extra_docker_args
    cmd.append(image)
    if args:
        cmd += args
    return subprocess.run(
        cmd,
        input=stdin_data,
        capture_output=True,
        timeout=timeout,
    )


# ── Build tests ───────────────────────────────────────────────────


@pytest.mark.docker
@pytest.mark.timeout(600)
class TestDockerBuild:
    def test_image_builds_successfully(self, docker_image: str) -> None:
        result = subprocess.run(
            ["docker", "images", "-q", docker_image],
            capture_output=True,
            timeout=10,
        )
        assert result.stdout.strip(), f"Image {docker_image} not found after build"

    def test_image_has_expected_labels(self, docker_image: str) -> None:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Config.Labels}}", docker_image],
            capture_output=True,
            timeout=10,
        )
        labels = json.loads(result.stdout.decode())
        assert (
            labels.get("org.opencontainers.image.title") == "Cogtrix"
        ), f"Expected label 'org.opencontainers.image.title'='Cogtrix', got: {labels}"

    def test_image_runs_as_nonroot(self, docker_image: str) -> None:
        r = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "id", docker_image],
            capture_output=True,
            timeout=10,
        )
        output = r.stdout.decode()
        assert "uid=1000" in output, f"Expected uid=1000 (non-root), got: {output}"
        assert "cogtrix" in output, f"Expected user 'cogtrix', got: {output}"


# ── Data directory tests ──────────────────────────────────────────


@pytest.mark.docker
@pytest.mark.timeout(30)
class TestDockerDataDirectories:
    def test_data_directory_tree_exists(self, docker_image: str) -> None:
        # The Dockerfile creates the runtime data tree at /data (see
        # docker/Dockerfile: ``COGTRIX_DATA_DIR=/data``, ``mkdir -p /data/...``,
        # ``VOLUME /data``).  Commit dca9e30 updated the expected list to
        # reference /data/* but left the find argument pointing at the old
        # /app/data path, producing empty output and a self-defeating
        # assertion.  The find target is aligned with the actual layout.
        r = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "find",
                docker_image,
                "/data",
                "-type",
                "d",
            ],
            capture_output=True,
            timeout=15,
        )
        output = r.stdout.decode()
        expected = [
            "/data/history",
            "/data/knowledge",
            "/data/vectordb",
            "/data/api/uploads",
            "/data/assistant",
            "/data/workflows",
        ]
        for d in expected:
            assert d in output, f"Missing data directory: {d}\nActual:\n{output}"

    def test_data_dir_is_writable(self, docker_image: str) -> None:
        r = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "sh",
                docker_image,
                "-c",
                "touch /data/test_write && echo ok",
            ],
            capture_output=True,
            timeout=10,
        )
        assert r.returncode == 0, f"Data dir not writable: {r.stderr.decode()}"
        assert b"ok" in r.stdout

    def test_volume_declared_at_data(self, docker_image: str) -> None:
        r = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .Config.Volumes}}",
                docker_image,
            ],
            capture_output=True,
            timeout=10,
        )
        volumes = json.loads(r.stdout.decode())
        assert volumes and "/data" in volumes, f"Expected VOLUME at /data, got: {volumes}"


# ── CLI mode tests ────────────────────────────────────────────────


@pytest.mark.docker
@pytest.mark.timeout(30)
class TestDockerCliMode:
    def test_cli_help_flag_exits_zero(self, docker_image: str) -> None:
        r = _run_container(docker_image, ["--help"], timeout=20)
        combined = (r.stdout + r.stderr).decode().lower()
        assert r.returncode == 0, f"--help exited {r.returncode}: {combined}"
        assert (
            "cogtrix" in combined or "usage" in combined
        ), f"--help output does not mention 'cogtrix' or 'usage': {combined[:500]}"

    def test_no_wizard_without_tty(self, docker_image: str) -> None:
        # No -t flag, so [ -t 0 ] is false; pipe empty stdin so it exits
        r = subprocess.run(
            ["docker", "run", "--rm", "-i", docker_image],
            input=b"",
            capture_output=True,
            timeout=20,
        )
        combined = (r.stdout + r.stderr).decode().lower()
        # Wizard should NOT have started (no TTY)
        assert (
            "setup wizard" not in combined and "cogtrix setup" not in combined
        ), f"Wizard should not start without TTY, but output contains wizard text: {combined[:500]}"


# ── Wizard auto-start tests ──────────────────────────────────────


@pytest.mark.docker
@pytest.mark.timeout(30)
class TestDockerWizardAutoStart:
    def test_wizard_starts_with_setup_flag(self, docker_image: str) -> None:
        # Pass --setup; wizard will fail on stdin but should print preamble
        r = subprocess.run(
            ["docker", "run", "--rm", "-i", docker_image, "--setup"],
            input=b"",
            capture_output=True,
            timeout=20,
        )
        combined = (r.stdout + r.stderr).decode().lower()
        # The wizard should show some setup-related output
        assert any(
            kw in combined
            for kw in ("provider", "wizard", "setup", "ollama", "openai", "configure")
        ), f"--setup did not produce wizard output: {combined[:500]}"

    def test_wizard_skipped_with_api_key_env(self, docker_image: str) -> None:
        r = _run_container(
            docker_image,
            ["--help"],
            env={"OPENAI_API_KEY": "sk-test"},
            timeout=20,
        )
        combined = (r.stdout + r.stderr).decode().lower()
        # --help should work normally; wizard should not appear
        assert r.returncode == 0, f"Unexpected exit code {r.returncode}: {combined[:500]}"


# ── API mode tests ────────────────────────────────────────────────


@pytest.mark.docker
@pytest.mark.timeout(60)
class TestDockerApiMode:
    def test_api_mode_starts_uvicorn(self, docker_image: str) -> None:
        proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-e",
                f"COGTRIX_JWT_SECRET={JWT_SECRET}",
                docker_image,
                "api",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=5)
        combined = (stdout + stderr).decode().lower()
        assert (
            "uvicorn" in combined or "application startup" in combined
        ), f"API mode did not produce uvicorn output: {combined[:1000]}"

    def test_api_mode_healthcheck(self, docker_image: str) -> None:
        container_name = "cogtrix-pytest-healthcheck"
        # Clean up any stale container with the same name
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=10,
        )
        # Start container in detached mode
        r = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                f"{HEALTHCHECK_PORT}:8000",
                "-e",
                f"COGTRIX_JWT_SECRET={JWT_SECRET}",
                docker_image,
                "api",
            ],
            capture_output=True,
            timeout=30,
        )
        assert r.returncode == 0, f"Failed to start container: {r.stderr.decode()}"

        try:
            # Poll until healthy (max 30s)
            url = f"http://localhost:{HEALTHCHECK_PORT}/api/v1/health"
            deadline = time.monotonic() + 30
            last_error = None
            while time.monotonic() < deadline:
                try:
                    resp = urllib.request.urlopen(url, timeout=3)
                    if resp.status == 200:
                        body = json.loads(resp.read().decode())
                        assert "data" in body, f"Health response missing 'data': {body}"
                        return  # success
                except Exception as exc:
                    last_error = exc
                time.sleep(1)
            pytest.fail(f"Health endpoint not ready after 30s. Last error: {last_error}")
        finally:
            subprocess.run(
                ["docker", "stop", "-t", "5", container_name],
                capture_output=True,
                timeout=15,
            )
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
            )

    def test_api_mode_runs_alembic(self, docker_image: str) -> None:
        container_name = "cogtrix-pytest-alembic"
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=10,
        )
        r = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-e",
                f"COGTRIX_JWT_SECRET={JWT_SECRET}",
                docker_image,
                "api",
            ],
            capture_output=True,
            timeout=30,
        )
        assert r.returncode == 0, f"Failed to start container: {r.stderr.decode()}"

        try:
            # Wait briefly for alembic to run
            time.sleep(5)
            logs = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True,
                timeout=10,
            )
            combined = (logs.stdout + logs.stderr).decode().lower()
            assert (
                "alembic" in combined or "upgrade" in combined or "migration" in combined
            ), f"Container logs do not mention alembic/migration: {combined[:1000]}"
        finally:
            subprocess.run(
                ["docker", "stop", "-t", "3", container_name],
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
            )


# ── Graceful shutdown tests ──────────────────────────────────────


@pytest.mark.docker
@pytest.mark.timeout(30)
class TestDockerSigterm:
    def test_sigterm_graceful_shutdown(self, docker_image: str) -> None:
        container_name = "cogtrix-pytest-sigterm"
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            timeout=10,
        )
        r = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-e",
                f"COGTRIX_JWT_SECRET={JWT_SECRET}",
                docker_image,
                "api",
            ],
            capture_output=True,
            timeout=30,
        )
        assert r.returncode == 0, f"Failed to start: {r.stderr.decode()}"

        try:
            # Give server time to start
            time.sleep(5)
            # Send SIGTERM via docker stop (default signal)
            stop = subprocess.run(
                ["docker", "stop", "-t", "10", container_name],
                capture_output=True,
                timeout=20,
            )
            assert stop.returncode == 0, f"docker stop failed: {stop.stderr.decode()}"

            # Inspect exit code (0 = graceful, 137 = SIGKILL)
            inspect = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.ExitCode}}",
                    container_name,
                ],
                capture_output=True,
                timeout=10,
            )
            exit_code = int(inspect.stdout.decode().strip())
            assert exit_code == 0, (
                f"Container exited with code {exit_code} (137=SIGKILL). "
                "Expected 0 for graceful SIGTERM shutdown."
            )
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
            )
