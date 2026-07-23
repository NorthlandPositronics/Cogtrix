from __future__ import annotations

import os
import subprocess

__version__ = "0.4.1"  # x-release-please-version
__copyright__ = "© 2025–2026 Northland Positronics (FZE)"


def get_commit_hash() -> str | None:
    """Return the short git commit hash (7 chars), or None if unavailable.

    Checks ``GIT_COMMIT`` env var first (for Docker builds), then falls
    back to ``git rev-parse --short HEAD``.
    """
    commit = os.environ.get("GIT_COMMIT")
    if commit and len(commit) >= 7:
        return commit[:7]

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass

    return None


def get_version_string() -> str:
    """Return the full version string including commit hash when available.

    Format: ``<major>.<minor>.<patch>+<short-hash>`` (e.g. ``0.2.6+abc1234``).
    Falls back to the bare ``__version__`` when the commit hash is unavailable.
    """
    commit = get_commit_hash()
    if commit:
        return f"{__version__}+{commit}"
    return __version__
