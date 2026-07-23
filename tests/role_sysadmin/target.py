"""Per-run, disposable "system" container the agent configures over SSH.

Each scenario run gets a fresh systemd Ubuntu container (the SUT). The harness
generates an ephemeral SSH keypair, boots the container ``--privileged``, injects
the public key into the ``ops`` user, and waits for sshd. The agent-under-test
then drives the box with its real ``execute_shell_command`` tool
(``ssh ops@127.0.0.1 -p <port> '...'``); the harness verifies the result by
SSHing in independently (:meth:`Target.run` / :meth:`Target.run_check`) and
inspecting the live system state. Everything outside the model is deterministic.

This is the systems-administration analog of ``tests/role_swe/workspace.py`` — a
Docker target replaces the git workspace. Validated in issue #2337.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

#: The build context for the SUT image (target/Dockerfile).
_TARGET_DIR = Path(__file__).parent / "target"
#: The image tag the harness builds + runs.
IMAGE = "cogtrix-role-sysadmin:latest"
#: The unprivileged user the agent + harness SSH in as.
USER = "ops"


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a command run on (or about) the target."""

    ok: bool
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


def docker_available() -> bool:
    """True if a working Docker daemon is reachable (gates the docker tests)."""
    try:
        proc = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10, check=False
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _free_port() -> int:
    """Pick a currently-free localhost TCP port for the container's SSH map."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Target:
    """A disposable systemd container the agent configures over SSH.

    Use as a context manager so the container + temp keys are always torn down::

        with Target.create("role_sa_demo") as t:
            t.run("sudo systemctl is-active ssh")   # harness-side verification
            t.agent_ssh_invocation()                # handed to the agent
    """

    def __init__(self, name: str, port: int, key_path: Path, workdir: Path) -> None:
        self.name = name
        self.port = port
        self.key_path = key_path
        #: Per-run scratch dir (ephemeral keypair + the agent's known_hosts).
        self.workdir = workdir

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def build(cls, *, force: bool = False) -> None:
        """Build the SUT image if it is missing (cheap no-op when cached)."""
        if not force:
            present = subprocess.run(
                ["docker", "image", "inspect", IMAGE],
                capture_output=True,
                text=True,
                check=False,
            )
            if present.returncode == 0:
                return
        proc = subprocess.run(
            ["docker", "build", "-t", IMAGE, str(_TARGET_DIR)],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"docker build failed:\n{proc.stdout}\n{proc.stderr}")

    @classmethod
    def create(
        cls,
        name: str,
        *,
        seed_setup: Path | None = None,
        ready_timeout: int = 40,
    ) -> Target:
        """Boot a fresh SUT container and return a ready :class:`Target`.

        Args:
            name: Container name (also used to pre-clean any stale container).
            seed_setup: Optional root setup script run inside the container
                *before* the agent (break-fix scenarios plant a broken state).
            ready_timeout: Seconds to wait for sshd to accept the harness key.

        Raises:
            RuntimeError: If the image build, container start, or sshd readiness
                fails.
        """
        cls.build()
        # Pre-clean any stale container of the same name so re-runs are idempotent.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

        workdir = Path(tempfile.mkdtemp(prefix=f"{name}_"))
        key_path = workdir / "id_target"
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(key_path), "-C", name],
            capture_output=True,
            check=True,
        )
        port = _free_port()

        start = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                "--privileged",
                "--cgroupns=host",
                "-p",
                f"127.0.0.1:{port}:22",
                IMAGE,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if start.returncode != 0:
            shutil.rmtree(workdir, ignore_errors=True)
            raise RuntimeError(f"docker run failed:\n{start.stdout}\n{start.stderr}")

        target = cls(name=name, port=port, key_path=key_path, workdir=workdir)
        try:
            target._inject_key()
            target._wait_ready(ready_timeout)
            if seed_setup is not None:
                target._apply_seed(seed_setup)
        except Exception:
            target.teardown()
            raise
        return target

    def __enter__(self) -> Target:
        return self

    def __exit__(self, *exc: object) -> None:
        self.teardown()

    def teardown(self) -> None:
        """Remove the container and the per-run temp keys (best-effort)."""
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, check=False)
        shutil.rmtree(self.workdir, ignore_errors=True)

    # -- harness-side verification (independent of the agent) --------------

    def run(self, command: str, *, timeout: int = 60) -> CommandResult:
        """Run *command* on the target over SSH (harness side) and capture output."""
        return self._ssh_exec([command], timeout=timeout)

    def reachable(self) -> bool:
        """True if the harness can still SSH in — i.e. the agent didn't lock us out.

        The headline safety signal: a hardening/firewall task that breaks key auth
        or blocks port 22 fails here even if every other check passed.
        """
        return self._ssh_exec(["true"], timeout=8).ok

    def put_file(self, local: Path, remote: str, *, timeout: int = 30) -> CommandResult:
        """Copy a local file to *remote* on the target via scp (harness side)."""
        argv = [
            "scp",
            "-q",
            "-P",
            str(self.port),
            "-i",
            str(self.key_path),
            *self._ssh_opts(),
            str(local),
            f"{USER}@127.0.0.1:{remote}",
        ]
        return self._subprocess(argv, timeout=timeout)

    def run_check(self, check_file: Path, *, timeout: int = 180) -> CommandResult:
        """Copy a scenario check script to the target and run it as root.

        The check (``scenarios/checks/sa_XX_check.sh``) asserts the task was
        actually done against live state (``systemctl is-active``, ``curl`` ...)
        and exits non-zero on any failure. ``ok`` ⇔ task achieved.
        """
        remote = "/tmp/role_sa_check.sh"
        put = self.put_file(check_file, remote, timeout=30)
        if not put.ok:
            return CommandResult(
                False, put.returncode, put.stdout, f"scp check failed: {put.stderr}"
            )
        return self.run(f"sudo bash {remote}", timeout=timeout)

    # -- the agent's connection -------------------------------------------

    def agent_ssh_invocation(self) -> str:
        """The exact ``ssh`` prefix the agent uses to drive the target.

        Handed to the agent in the assignment so the model never fights host-key
        or auth prompts. The agent runs::

            execute_shell_command("<this> 'sudo systemctl enable --now nginx'")

        Uses a per-run ``known_hosts`` (``accept-new``) so a reused host port with
        a new host key never causes a verification conflict.
        """
        known_hosts = self.workdir / "agent_known_hosts"
        return (
            f"ssh -i {self.key_path} -p {self.port} "
            f"-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile={known_hosts} "
            f"-o ConnectTimeout=8 {USER}@127.0.0.1"
        )

    # -- internals ---------------------------------------------------------

    def _ssh_opts(self) -> list[str]:
        # Harness side never cares about the host key (fresh container each run).
        return [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "LogLevel=ERROR",
        ]

    def _ssh_exec(self, remote_argv: list[str], *, timeout: int) -> CommandResult:
        argv = [
            "ssh",
            "-p",
            str(self.port),
            "-i",
            str(self.key_path),
            *self._ssh_opts(),
            "-o",
            "ConnectTimeout=5",
            f"{USER}@127.0.0.1",
            *remote_argv,
        ]
        return self._subprocess(argv, timeout=timeout)

    def _subprocess(self, argv: list[str], *, timeout: int) -> CommandResult:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return CommandResult(False, 124, "", f"timeout after {timeout}s")
        return CommandResult(proc.returncode == 0, proc.returncode, proc.stdout, proc.stderr)

    def _inject_key(self) -> None:
        pub = (self.key_path.with_suffix(".pub")).read_bytes()
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self.name,
                "bash",
                "-c",
                "cat > /home/ops/.ssh/authorized_keys "
                "&& chmod 600 /home/ops/.ssh/authorized_keys "
                "&& chown ops:ops /home/ops/.ssh/authorized_keys",
            ],
            input=pub,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"key injection failed: {proc.stderr.decode(errors='replace')}")

    def _wait_ready(self, timeout: int) -> None:
        deadline = time.monotonic() + timeout
        last = ""
        while time.monotonic() < deadline:
            res = self._ssh_exec(["true"], timeout=6)
            if res.ok:
                return
            last = res.output
            time.sleep(1.0)
        raise RuntimeError(f"target {self.name} sshd not ready in {timeout}s; last: {last[:200]}")

    def _apply_seed(self, seed_setup: Path) -> None:
        remote = "/tmp/role_sa_seed.sh"
        put = self.put_file(seed_setup, remote, timeout=30)
        if not put.ok:
            raise RuntimeError(f"seed scp failed: {put.stderr}")
        res = self.run(f"sudo bash {remote}", timeout=180)
        if not res.ok:
            raise RuntimeError(f"seed setup failed (rc={res.returncode}): {res.output[:300]}")
