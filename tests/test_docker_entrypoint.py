"""Regression tests for Docker entrypoint and Dockerfile correctness.

BUG-DOCKER-001: docker-entrypoint.sh used `python -m alembic upgrade head`
    which fails because alembic has no __main__.py. Fix: use `alembic upgrade head`.

BUG-DOCKER-002: Dockerfile used python:3.14-slim base image while
    pyproject.toml requires-python = "~=3.13.0". uv builds the venv with its
    own managed Python 3.13 whose binary is not copied to the runtime stage,
    causing the venv's python symlink to be broken and falling back to the
    system Python 3.14 which has no installed packages (no uvicorn, no fastapi).
    Fix: align the base image with requires-python (python:3.13-slim).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Shared helpers ────────────────────────────────────────────────


def _dockerfile_text() -> str:
    return (ROOT / "Dockerfile").read_text()


def _entrypoint_text() -> str:
    return (ROOT / "docker-entrypoint.sh").read_text()


class TestEntrypointAlembicInvocation:
    """BUG-DOCKER-001 — alembic must be called as CLI binary, not python -m alembic."""

    def _entrypoint_text(self) -> str:
        return (ROOT / "docker-entrypoint.sh").read_text()

    def test_does_not_use_python_m_alembic(self) -> None:
        assert "python -m alembic" not in self._entrypoint_text(), (
            "docker-entrypoint.sh must not call 'python -m alembic' — "
            "alembic has no __main__.py. Use 'alembic upgrade head' instead."
        )

    def test_calls_alembic_binary_directly(self) -> None:
        text = self._entrypoint_text()
        assert re.search(r"\balembic upgrade head\b", text), (
            "docker-entrypoint.sh must call 'alembic upgrade head' directly "
            "so the venv binary on PATH is used."
        )


class TestDockerfilePythonVersionAlignment:
    """BUG-DOCKER-002 — Dockerfile base image must match requires-python in pyproject.toml."""

    def _dockerfile_text(self) -> str:
        return (ROOT / "Dockerfile").read_text()

    def _required_minor(self) -> str:
        """Return the minor version required by pyproject.toml, e.g. '3.13'."""
        with open(ROOT / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        spec = data.get("project", {}).get("requires-python", "")
        # Handles "~=3.13.0", ">=3.13", "~=3.13", etc.
        m = re.search(r"(\d+\.\d+)", spec)
        assert m, f"Could not parse minor version from requires-python: {spec!r}"
        return m.group(1)

    def test_builder_stage_matches_requires_python(self) -> None:
        minor = self._required_minor()
        dockerfile = self._dockerfile_text()
        # Extract the python:X.Y-slim image used in the builder FROM line.
        m = re.search(r"FROM python:(\d+\.\d+)-slim AS builder", dockerfile)
        assert m, "Could not find 'FROM python:X.Y-slim AS builder' in Dockerfile"
        assert m.group(1) == minor, (
            f"Dockerfile builder stage uses python:{m.group(1)}-slim but "
            f"requires-python = '~={minor}.x'. "
            "Misalignment causes uv to download a managed Python whose binary "
            "is not copied to the runtime stage, breaking the venv symlink."
        )

    def test_runtime_stage_matches_requires_python(self) -> None:
        minor = self._required_minor()
        dockerfile = self._dockerfile_text()
        m = re.search(r"FROM python:(\d+\.\d+)-slim AS runtime", dockerfile)
        assert m, "Could not find 'FROM python:X.Y-slim AS runtime' in Dockerfile"
        assert m.group(1) == minor, (
            f"Dockerfile runtime stage uses python:{m.group(1)}-slim but "
            f"requires-python = '~={minor}.x'. "
            "The system Python in the runtime image must match the venv Python."
        )

    def test_builder_and_runtime_use_same_python_version(self) -> None:
        dockerfile = self._dockerfile_text()
        builder = re.search(r"FROM python:(\d+\.\d+)-slim AS builder", dockerfile)
        runtime = re.search(r"FROM python:(\d+\.\d+)-slim AS runtime", dockerfile)
        assert builder and runtime, "Could not parse both FROM lines in Dockerfile"
        assert builder.group(1) == runtime.group(1), (
            f"Builder uses python:{builder.group(1)}-slim but runtime uses "
            f"python:{runtime.group(1)}-slim — they must match."
        )


# ── Dockerfile structure best practices ──────────────────────────


class TestDockerfileStructure:
    """Verify Dockerfile follows container best practices."""

    def test_healthcheck_present_and_probes_health_endpoint(self) -> None:
        text = _dockerfile_text()
        assert "HEALTHCHECK" in text, "Dockerfile must declare a HEALTHCHECK instruction"
        assert "/api/v1/health" in text, "HEALTHCHECK must probe the /api/v1/health endpoint"

    def test_nonroot_user_is_last_user_before_entrypoint(self) -> None:
        text = _dockerfile_text()
        # Find all USER instructions
        user_lines = re.findall(r"^USER\s+(\S+)", text, re.MULTILINE)
        assert user_lines, "Dockerfile must contain at least one USER instruction"
        assert (
            user_lines[-1] != "root"
        ), f"Last USER before ENTRYPOINT must be non-root, got '{user_lines[-1]}'"

    def test_expose_8000(self) -> None:
        text = _dockerfile_text()
        assert re.search(
            r"^EXPOSE\s+8000\b", text, re.MULTILINE
        ), "Dockerfile must EXPOSE 8000 for the API server"

    def test_volume_app_data(self) -> None:
        text = _dockerfile_text()
        assert re.search(
            r"^VOLUME\s+/app/data\b", text, re.MULTILINE
        ), "Dockerfile must declare VOLUME /app/data"

    def test_stopsignal_sigterm(self) -> None:
        text = _dockerfile_text()
        assert re.search(
            r"^STOPSIGNAL\s+SIGTERM\b", text, re.MULTILINE
        ), "Dockerfile must set STOPSIGNAL SIGTERM for graceful shutdown"

    def test_path_includes_venv_bin(self) -> None:
        text = _dockerfile_text()
        assert (
            "/app/.venv/bin" in text
        ), "PATH must include /app/.venv/bin so venv binaries are on PATH"

    def test_pythonpath_includes_app(self) -> None:
        text = _dockerfile_text()
        # Match PYTHONPATH="/app" or PYTHONPATH=/app in an ENV instruction
        assert re.search(
            r'PYTHONPATH="?/app"?', text
        ), "PYTHONPATH must include /app for src.* imports"

    def test_entrypoint_points_to_entrypoint_script(self) -> None:
        text = _dockerfile_text()
        assert re.search(
            r'ENTRYPOINT\s+\["/app/docker-entrypoint\.sh"\]', text
        ), "ENTRYPOINT must point to /app/docker-entrypoint.sh"


# ── Entrypoint script logic ──────────────────────────────────────


class TestEntrypointLogic:
    """Verify docker-entrypoint.sh branching logic."""

    def test_api_mode_triggers_on_api_arg(self) -> None:
        text = _entrypoint_text()
        assert (
            '"api"' in text or "'api'" in text
        ), "Entrypoint must check for 'api' as first argument"

    def test_api_mode_triggers_on_dash_dash_api_arg(self) -> None:
        text = _entrypoint_text()
        assert (
            '"--api"' in text or "'--api'" in text
        ), "Entrypoint must check for '--api' as first argument"

    def test_alembic_runs_before_api_server(self) -> None:
        text = _entrypoint_text()
        alembic_pos = text.find("alembic upgrade head")
        api_pos = text.find("exec python -m src.api")
        assert alembic_pos != -1, "Entrypoint must run 'alembic upgrade head'"
        assert api_pos != -1, "Entrypoint must exec 'python -m src.api'"
        assert alembic_pos < api_pos, "alembic upgrade head must run BEFORE exec python -m src.api"

    def test_api_server_command(self) -> None:
        text = _entrypoint_text()
        assert (
            "exec python -m src.api" in text
        ), "API mode must use 'exec python -m src.api' as the server command"

    def test_wizard_skipped_when_args_present(self) -> None:
        text = _entrypoint_text()
        assert (
            "$# -eq 0" in text
        ), "Wizard guard must check $# -eq 0 so explicit CLI args pass straight through"

    def test_wizard_checks_no_yaml_config(self) -> None:
        text = _entrypoint_text()
        assert (
            "! -f /app/.cogtrix.yaml" in text
        ), "Wizard guard must check for absence of /app/.cogtrix.yaml"

    def test_wizard_checks_no_yml_config(self) -> None:
        text = _entrypoint_text()
        assert (
            "! -f /app/.cogtrix.yml" in text
        ), "Wizard guard must check for absence of /app/.cogtrix.yml"

    def test_wizard_checks_no_json_config(self) -> None:
        text = _entrypoint_text()
        assert (
            "! -f /app/.cogtrix.json" in text
        ), "Wizard guard must check for absence of /app/.cogtrix.json"

    def test_wizard_checks_tty(self) -> None:
        text = _entrypoint_text()
        assert "-t 0" in text, "Wizard guard must check '[ -t 0 ]' (stdin is a TTY)"

    def test_wizard_checks_all_six_api_key_env_vars(self) -> None:
        text = _entrypoint_text()
        expected_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "XAI_API_KEY",
            "COGTRIX_OLLAMA",
            "OLLAMA_BASE_URL",
        ]
        for var in expected_vars:
            assert var in text, f"Wizard guard must check env var {var} before skipping wizard"

    def test_wizard_runs_setup(self) -> None:
        text = _entrypoint_text()
        assert (
            "exec python cogtrix.py --setup" in text
        ), "Wizard branch must run 'exec python cogtrix.py --setup'"

    def test_final_fallback_is_cogtrix(self) -> None:
        text = _entrypoint_text()
        lines = text.strip().splitlines()
        last_line = lines[-1].strip()
        assert (
            last_line == 'exec python cogtrix.py "$@"'
        ), f"Final fallback must be 'exec python cogtrix.py \"$@\"', got: {last_line!r}"


# ── Data directory consistency ────────────────────────────────────


class TestDockerfileDataDirs:
    """Verify data directory declarations are consistent between Dockerfile and entrypoint."""

    EXPECTED_SUBDIRS = {
        "history",
        "knowledge",
        "vectordb",
        "api/uploads",
        "assistant",
        "workflows",
    }

    @staticmethod
    def _extract_mkdir_dirs(text: str) -> set[str]:
        """Extract relative data subdirs from mkdir -p lines."""
        dirs: set[str] = set()
        # Match paths like /app/data/foo or "$DATA_DIR/foo"
        for m in re.finditer(r'(?:/app/data|"\$DATA_DIR)/([a-zA-Z0-9_/]+)', text):
            dirs.add(m.group(1))
        return dirs

    def test_dockerfile_has_all_expected_subdirs(self) -> None:
        dockerfile_dirs = self._extract_mkdir_dirs(_dockerfile_text())
        for subdir in self.EXPECTED_SUBDIRS:
            assert subdir in dockerfile_dirs, f"Dockerfile mkdir -p is missing /app/data/{subdir}"

    def test_entrypoint_has_all_expected_subdirs(self) -> None:
        entrypoint_dirs = self._extract_mkdir_dirs(_entrypoint_text())
        for subdir in self.EXPECTED_SUBDIRS:
            assert (
                subdir in entrypoint_dirs
            ), f"docker-entrypoint.sh mkdir -p is missing $DATA_DIR/{subdir}"

    def test_dockerfile_and_entrypoint_subdirs_match(self) -> None:
        dockerfile_dirs = self._extract_mkdir_dirs(_dockerfile_text())
        entrypoint_dirs = self._extract_mkdir_dirs(_entrypoint_text())
        assert dockerfile_dirs == entrypoint_dirs, (
            f"Dockerfile and entrypoint data subdirs differ.\n"
            f"  Dockerfile only: {dockerfile_dirs - entrypoint_dirs}\n"
            f"  Entrypoint only: {entrypoint_dirs - dockerfile_dirs}"
        )

    def test_entrypoint_references_cogtrix_data_dir(self) -> None:
        text = _entrypoint_text()
        assert (
            "COGTRIX_DATA_DIR" in text
        ), "Entrypoint must reference COGTRIX_DATA_DIR env var for the data root"
