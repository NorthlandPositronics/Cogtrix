"""Per-run, git-isolated workspace for an SWE scenario.

Each scenario run gets a fresh copy of the ledgerlite SUT in a temp directory,
seeded as a git repo at a pristine baseline commit. The agent edits the working
copy; the harness then asks the workspace what changed (``changed_files`` /
``diff``) and whether it still works (``run_tests`` / ``run_lint``). All of this is
deterministic — no LLM — so reviewer/QA decisions and the scorecard are
reproducible.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The pristine SUT template shipped with the harness.
PROJECT_TEMPLATE = Path(__file__).parent / "project"


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a subprocess run in the workspace."""

    ok: bool
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


class Workspace:
    """A git-isolated copy of the SUT the agent works in.

    Use as a context manager so the temp tree is always cleaned up::

        with Workspace.create(tmp_path) as ws:
            (ws.root / "src/ledgerlite/x.py").write_text(...)
            ws.changed_files()  # -> {"src/ledgerlite/x.py"}
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def create(
        cls, dest: Path, template: Path = PROJECT_TEMPLATE, seed_dir: Path | None = None
    ) -> Workspace:
        """Copy the pristine SUT into *dest* and seed a baseline git commit.

        Args:
            dest: Directory to create the workspace in (must not yet exist).
            template: The pristine project to copy (defaults to the shipped SUT).
            seed_dir: Optional overlay applied **before** the baseline commit, so
                its files become part of the pristine starting state (not a change
                the agent is blamed for). Used by bug-fix scenarios (swe_02) to
                plant a pre-existing defect — the agent's diff then shows only the
                fix.

        Returns:
            A ready :class:`Workspace` whose ``HEAD`` is the (optionally seeded)
            baseline.
        """
        # Skip build/cache artifacts an operator may have left in the template
        # (a local ``.venv`` / ``.ruff_cache`` / ``.pytest_cache`` from running
        # tools in ``project/``). Copying them would bloat every workspace and —
        # for the ``.pyc`` case — pollute the post-run diff. The shipped SUT also
        # carries a ``.gitignore`` so any artifacts the agent *creates* during the
        # run (pytest/ruff caches) stay out of ``changed_files`` / ``diff``.
        shutil.copytree(
            template,
            dest,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.egg-info",
                "*.pyc",
                ".venv",
                ".ruff_cache",
                ".pytest_cache",
                ".mypy_cache",
            ),
        )
        if seed_dir is not None:
            # Overlay the scenario seed onto the pristine copy (full-file replace)
            # so the planted defect is part of the baseline, not the agent's diff.
            shutil.copytree(seed_dir, dest, dirs_exist_ok=True)
        ws = cls(dest)
        ws._git("init", "-q")
        ws._git("config", "user.email", "harness@cogtrix.test")
        ws._git("config", "user.name", "SWE Harness")
        ws._git("add", "-A")
        ws._git("commit", "-q", "-m", "baseline")
        return ws

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *exc: object) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    # -- inspection (deterministic) ---------------------------------------

    def changed_files(self) -> set[str]:
        """Return the set of files changed vs the pristine baseline (posix paths)."""
        # Staged + unstaged + untracked, relative to the repo root.
        self._git("add", "-A")
        out = self._git("diff", "--cached", "--name-only", "HEAD").stdout
        return {line.strip() for line in out.splitlines() if line.strip()}

    def diff(self) -> str:
        """Return the unified diff of all changes vs the pristine baseline."""
        self._git("add", "-A")
        return self._git("diff", "--cached", "HEAD").stdout

    # -- verification (deterministic) -------------------------------------

    def run_tests(self) -> CommandResult:
        """Run the SUT's pytest suite against the working copy."""
        return self._run(
            ["python", "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"],
            env_pythonpath="src",
        )

    def run_lint(self) -> CommandResult:
        """Run ruff + black --check on the SUT's source (the project's own config)."""
        ruff = self._run(["ruff", "check", "src", "tests"])
        black = self._run(["black", "--check", "src", "tests"])
        ok = ruff.ok and black.ok
        return CommandResult(ok, 0 if ok else 1, ruff.output, black.output)

    def run_behavioural_check(self, check_file: Path) -> CommandResult:
        """Run the harness's own pytest check against the agent's final code.

        This is the harness asserting the feature *actually works* — independent
        of the agent's own tests (which may be weak or missing). Used by scenarios
        where a naive implementation passes the conventions but is wrong, e.g.
        swe_07 (an unbalanced transfer breaks the double-entry invariant) and
        swe_02 (was the reported bug really fixed?).

        The check imports the workspace's ``ledgerlite`` via ``PYTHONPATH=src`` but
        runs from a temp dir **outside** the workspace tree, so it never appears in
        ``changed_files()`` / ``diff()`` and can't pollute the graded result.

        Args:
            check_file: A pytest file (one or more ``test_*`` functions) shipped
                with the scenario.

        Returns:
            The pytest run result (``ok`` iff the behavioural check passed).
        """
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "test_behavioural_check.py"
            shutil.copyfile(check_file, dst)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(self.root / "src")
            proc = subprocess.run(
                ["python", "-m", "pytest", str(dst), "-q", "-p", "no:cacheprovider"],
                cwd=td,
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
                check=False,
            )
            return CommandResult(proc.returncode == 0, proc.returncode, proc.stdout, proc.stderr)

    # -- internals ---------------------------------------------------------

    def _git(self, *args: str) -> CommandResult:
        return self._run(["git", *args])

    def _run(self, cmd: list[str], env_pythonpath: str | None = None) -> CommandResult:
        import os

        env = dict(os.environ)
        if env_pythonpath:
            env["PYTHONPATH"] = str(self.root / env_pythonpath)
        proc = subprocess.run(
            cmd,
            cwd=self.root,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )
        return CommandResult(proc.returncode == 0, proc.returncode, proc.stdout, proc.stderr)
