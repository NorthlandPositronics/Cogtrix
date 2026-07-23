"""Unit tests for the Docker-free pieces of the fleet runner.

The CLI / Docker-orchestration paths are exercised manually (running
the fleet costs real LLM tokens, so they're not in CI).  This module
covers the bits that ARE testable in isolation:

  * ``resolve_config_path`` — override path + ``find_config_file``
    fallback + legacy-path fallback chain.
  * ``_parse_log`` — counter aggregation and tool-histogram extraction
    from a synthetic log file.
  * ``_select_scenarios`` CLI helper — slug filtering + error on
    unknown slugs.

These ARE collected by pytest because they share the
``tests/agent_complexity/`` package and follow the ``test_*.py``
convention.  Keep the heavy fleet-runner glue (``runner.run_fleet``
etc.) out of this module — it requires a Docker daemon.
"""

from __future__ import annotations

import pytest

from tests.agent_complexity.runner import (
    ScenarioResult,
    _build_run_cmd,
    _format_summary,
    _parse_log,
    _resolve_env_file,
    _select_scenarios,
    resolve_config_path,
)

# ── secrets env-file (#2219) ──────────────────────────────────────────


class TestBuildRunCmd:
    def _kw(self, tmp_path, **extra):
        return {
            "name": "fleet-1-gas",
            "image": "cogtrix:test",
            "config_path": tmp_path / "cfg.yaml",
            "log_path": tmp_path / "t.log",
            "prompt": "do a thing",
            "verbosity": 3,
            **extra,
        }

    def test_no_env_file_by_default(self, tmp_path):
        cmd = _build_run_cmd(**self._kw(tmp_path))
        assert "--env-file" not in cmd

    def test_env_file_injected_before_image(self, tmp_path):
        envf = tmp_path / ".env"
        cmd = _build_run_cmd(**self._kw(tmp_path, env_file=envf))
        assert "--env-file" in cmd
        i = cmd.index("--env-file")
        assert cmd[i + 1] == str(envf)
        # must precede the image (a docker run flag, not a container arg)
        assert i < cmd.index("cogtrix:test")

    def test_prompt_and_config_still_present(self, tmp_path):
        envf = tmp_path / ".env"
        cmd = _build_run_cmd(**self._kw(tmp_path, env_file=envf))
        assert cmd[cmd.index("--prompt") + 1] == "do a thing"
        assert any(str(tmp_path / "cfg.yaml") in part for part in cmd)


class TestResolveEnvFile:
    def test_explicit_override_used_when_present(self, tmp_path):
        envf = tmp_path / "secrets.env"
        envf.write_text("X=1\n")
        assert _resolve_env_file(envf, tmp_path / "cfg.yaml") == envf

    def test_explicit_missing_override_is_skip(self, tmp_path):
        assert _resolve_env_file(tmp_path / "nope.env", tmp_path / "cfg.yaml") is None

    def test_autodetect_sibling_dotenv(self, tmp_path):
        (tmp_path / ".env").write_text("X=1\n")
        assert _resolve_env_file(None, tmp_path / "cfg.yaml") == tmp_path / ".env"

    def test_no_sibling_returns_none(self, tmp_path):
        assert _resolve_env_file(None, tmp_path / "cfg.yaml") is None


# ── resolve_config_path ───────────────────────────────────────────────


class TestResolveConfigPath:
    def test_override_returns_existing_file(self, tmp_path):
        cfg = tmp_path / "custom.yaml"
        cfg.write_text("providers: {}\n")
        result = resolve_config_path(cfg)
        assert result == cfg

    def test_override_missing_raises(self, tmp_path):
        cfg = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_config_path(cfg)

    def test_override_directory_raises(self, tmp_path):
        cfg_dir = tmp_path / "stale_bind_mount_artefact"
        cfg_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="directory"):
            resolve_config_path(cfg_dir)

    def test_no_override_falls_through_to_canonical_resolver(self, monkeypatch, tmp_path):
        """When no override is given, the canonical ``find_config_file``
        is consulted first."""
        sentinel = tmp_path / "canonical.yaml"
        sentinel.write_text("providers: {}\n")
        monkeypatch.setattr(
            "cogtrix_core.config.find_config_file",
            lambda: sentinel,
        )
        # Block the legacy fallback paths too (they shouldn't be touched
        # when the canonical resolver wins).
        monkeypatch.setattr(
            "tests.agent_complexity.runner._LEGACY_CONFIG_FALLBACKS",
            (),
        )
        assert resolve_config_path() == sentinel

    def test_no_canonical_match_falls_back_to_legacy_paths(self, monkeypatch, tmp_path):
        """When the canonical resolver returns None and a legacy
        fallback exists, that wins."""
        legacy = tmp_path / "legacy.yaml"
        legacy.write_text("providers: {}\n")
        monkeypatch.setattr(
            "cogtrix_core.config.find_config_file",
            lambda: None,
        )
        monkeypatch.setattr(
            "tests.agent_complexity.runner._LEGACY_CONFIG_FALLBACKS",
            (legacy,),
        )
        assert resolve_config_path() == legacy

    def test_nothing_found_raises_with_tried_paths(self, monkeypatch):
        """The error message enumerates every path consulted — the
        operator needs to know *where* to put the config."""
        monkeypatch.setattr("cogtrix_core.config.find_config_file", lambda: None)
        monkeypatch.setattr(
            "tests.agent_complexity.runner._LEGACY_CONFIG_FALLBACKS",
            (),
        )
        with pytest.raises(FileNotFoundError, match="No cogtrix config file found"):
            resolve_config_path()


# ── _parse_log ────────────────────────────────────────────────────────


class TestParseLog:
    def test_empty_log_returns_zeros(self, tmp_path):
        log_path = tmp_path / "empty.log"
        log_path.write_text("")
        result = _parse_log(log_path, expected_tools=None)
        assert result["turns"] == 0
        assert result["tool_calls"] == 0
        assert result["completed"] is False
        assert result["top_tools"] == []

    def test_missing_log_returns_zeros(self, tmp_path):
        """Log file doesn't exist — runner survives, returns zeros."""
        log_path = tmp_path / "does_not_exist.log"
        result = _parse_log(log_path, expected_tools=("write_file",))
        assert result["turns"] == 0
        assert result["missing_expected_tools"] == ["write_file"]

    def test_counts_llm_chat_start_as_turns(self, tmp_path):
        log_path = tmp_path / "turns.log"
        log_path.write_text(
            "2026-05-30 LLM_CHAT_START: ...\n"
            "2026-05-30 LLM_CHAT_START: ...\n"
            "2026-05-30 LLM_CHAT_START: ...\n"
        )
        result = _parse_log(log_path, expected_tools=None)
        assert result["turns"] == 3

    def test_extracts_tool_calls_and_histogram(self, tmp_path):
        log_path = tmp_path / "tools.log"
        log_path.write_text(
            "LLM_TOOL_CALL: write_file args={'path': '/tmp/x'}\n"
            "LLM_TOOL_CALL: write_file args={'path': '/tmp/y'}\n"
            "LLM_TOOL_CALL: execute_shell_command args={'command': 'ls'}\n"
            "LLM_TOOL_CALL: write_file args={'path': '/tmp/z'}\n"
        )
        result = _parse_log(log_path, expected_tools=None)
        assert result["tool_calls"] == 4
        assert result["top_tools"][0] == ("write_file", 3)
        assert ("execute_shell_command", 1) in result["top_tools"]

    def test_counts_tool_failures(self, tmp_path):
        log_path = tmp_path / "failures.log"
        log_path.write_text(
            "Tool failed: write_file - Error: Permission denied\n"
            "Tool failed: write_file - Error: Path outside\n"
        )
        result = _parse_log(log_path, expected_tools=None)
        assert result["tool_failures"] == 2

    def test_counts_error_and_warning_levels(self, tmp_path):
        log_path = tmp_path / "levels.log"
        log_path.write_text(
            "2026 [ERROR] something broke\n"
            "2026 [WARNING] something looks off\n"
            "2026 [WARNING] another warning\n"
            "2026 [INFO] this is fine\n"
        )
        result = _parse_log(log_path, expected_tools=None)
        assert result["errors"] == 1
        assert result["warnings"] == 2

    def test_detects_agent_response_as_completion(self, tmp_path):
        log_path = tmp_path / "done.log"
        log_path.write_text("[INFO] Agent response\n[DEBUG] Agent: All done\n")
        result = _parse_log(log_path, expected_tools=None)
        assert result["completed"] is True

    def test_reports_missing_expected_tools(self, tmp_path):
        log_path = tmp_path / "missing.log"
        log_path.write_text("LLM_TOOL_CALL: write_file args={}\n")
        result = _parse_log(log_path, expected_tools=("write_file", "execute_shell_command"))
        # write_file was invoked; execute_shell_command was not.
        assert result["missing_expected_tools"] == ["execute_shell_command"]

    def test_counts_duplicate_cache_hits(self, tmp_path):
        log_path = tmp_path / "dup.log"
        log_path.write_text(
            "[DEBUG] Tool output: [Duplicate call — returning cached result]\n"
            "[DEBUG] Tool output: [Duplicate call — returning cached result]\n"
        )
        result = _parse_log(log_path, expected_tools=None)
        assert result["duplicate_cache_hits"] == 2


# ── _select_scenarios ─────────────────────────────────────────────────


class TestSelectScenarios:
    def test_empty_slugs_returns_all_defaults(self):
        from tests.agent_complexity.scenarios import DEFAULT_SCENARIOS

        result = _select_scenarios("")
        assert len(result) == len(DEFAULT_SCENARIOS)

    def test_subset_by_slug(self):
        result = _select_scenarios("gas,sec")
        assert [s.slug for s in result] == ["gas", "sec"]

    def test_unknown_slug_raises_systemexit(self):
        with pytest.raises(SystemExit, match="Unknown scenario slug"):
            _select_scenarios("not_a_real_slug")

    def test_whitespace_around_slugs_tolerated(self):
        result = _select_scenarios(" gas , sec ")
        assert [s.slug for s in result] == ["gas", "sec"]


# ── _format_summary ───────────────────────────────────────────────────


class TestFormatSummary:
    def _make(self, **overrides) -> ScenarioResult:
        defaults = {
            "slug": "x",
            "container_name": "fleet-1-x",
            "container_id": "deadbeef",
            "log_path": "/tmp/x.log",
            "elapsed_s": 100.0,
            "timed_out": False,
            "turns": 5,
            "tool_calls": 10,
            "tool_failures": 0,
            "errors": 0,
            "warnings": 0,
            "duplicate_cache_hits": 0,
            "checkpoints": 1,
            "completed": True,
            "top_tools": [],
            "missing_expected_tools": [],
        }
        defaults.update(overrides)
        # ScenarioResult.log_path is typed Path, but the summary formatter
        # never reads it — pass a str to keep test setup simple.
        from pathlib import Path

        defaults["log_path"] = Path(defaults["log_path"])  # type: ignore[arg-type]
        return ScenarioResult(**defaults)  # type: ignore[arg-type]

    def test_passing_scenario_shows_check(self):
        out = _format_summary([self._make()])
        assert "✓ x" in out

    def test_failing_scenario_shows_cross(self):
        out = _format_summary([self._make(tool_failures=2)])
        assert "✗ x" in out

    def test_timed_out_scenario_annotated(self):
        out = _format_summary([self._make(timed_out=True, completed=False)])
        assert "[TIMEOUT]" in out

    def test_missing_expected_tools_listed(self):
        out = _format_summary([self._make(missing_expected_tools=["write_file"])])
        assert "missing expected tools: write_file" in out

    def test_top_tools_rendered(self):
        out = _format_summary([self._make(top_tools=[("write_file", 3), ("read_file", 1)])])
        assert "write_file×3" in out
        assert "read_file×1" in out
