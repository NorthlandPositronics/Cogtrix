"""Tests for src/tools/self_improve.py — M4.1 self-improvement tool.

Covers:
- detect phase: ruff findings parsed correctly from JSON output
- detect phase: bandit HIGH/MEDIUM findings parsed, LOW skipped
- detect phase: no findings → returns "No issues found" immediately
- dry_run=True returns findings without patching
- patch phase: LLM called with file content + issue details
- patch phase: invalid Python from LLM (ast.parse fails) → skipped, file unchanged
- verify phase: pytest passes → fix retained, recorded in summary
- verify phase: pytest fails → file reverted to original content
- auto_commit=False → no git commands run even after successful patches
- auto_commit=True → git add + git commit run after successful patches
- target path traversal rejected
- max_fixes cap respected
- TOOL_CONFIGS has requires_confirmation=True
- TOOL_SETUP sets _config
- Config self_improve_auto_commit default is False
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUFF_ISSUE = {
    "filename": "src/foo.py",
    "location": {"row": 42, "column": 1},
    "code": "E501",
    "message": "Line too long (101 > 100 characters)",
}

_BANDIT_HIGH = {
    "filename": "src/bar.py",
    "line_number": 17,
    "test_id": "B105",
    "issue_text": "Hardcoded password string",
    "issue_severity": "HIGH",
}

_BANDIT_MEDIUM = {
    "filename": "src/baz.py",
    "line_number": 5,
    "test_id": "B601",
    "issue_text": "Shell injection risk",
    "issue_severity": "MEDIUM",
}

_BANDIT_LOW = {
    "filename": "src/qux.py",
    "line_number": 9,
    "test_id": "B110",
    "issue_text": "Try-except-pass",
    "issue_severity": "LOW",
}

_VALID_PYTHON = "x = 1\n"
_INVALID_PYTHON = "def broken(\n"  # SyntaxError

_DUMMY_CONFIG = MagicMock()
_DUMMY_CONFIG.resolve_llm_config.return_value = (MagicMock(), MagicMock())


def _ruff_stdout(issues: list[dict]) -> str:
    return json.dumps(issues)


def _bandit_stdout(issues: list[dict]) -> str:
    return json.dumps({"results": issues})


def _make_subprocess_side_effect(
    ruff_out: str = "[]",
    bandit_out: str = '{"results": []}',
    pytest_rc: int = 0,
    git_add_rc: int = 0,
    git_commit_rc: int = 0,
) -> Any:
    """Return a side_effect callable for subprocess.run that dispatches by command."""

    def _side_effect(cmd, **_kwargs):
        result = MagicMock()
        result.stderr = ""
        if "ruff" in cmd:
            result.returncode = 1 if ruff_out != "[]" else 0
            result.stdout = ruff_out
        elif "bandit" in cmd:
            result.returncode = 0
            result.stdout = bandit_out
        elif "pytest" in cmd:
            result.returncode = pytest_rc
            result.stdout = "1 passed"
        elif cmd[:2] == ["git", "add"]:
            result.returncode = git_add_rc
            result.stdout = ""
        elif cmd[:2] == ["git", "commit"]:
            result.returncode = git_commit_rc
            result.stdout = ""
        else:
            result.returncode = 0
            result.stdout = ""
        return result

    return _side_effect


# ---------------------------------------------------------------------------
# Tests: parsing
# ---------------------------------------------------------------------------


class TestParseRuff:
    def test_ruff_findings_parsed_correctly(self):
        from src.tools.self_improve import _parse_ruff

        findings = _parse_ruff(_ruff_stdout([_RUFF_ISSUE]), cap=10)
        assert len(findings) == 1
        f = findings[0]
        assert f.linter == "ruff"
        assert f.file == "src/foo.py"
        assert f.line == 42
        assert f.code == "E501"
        assert "Line too long" in f.message

    def test_ruff_empty_output_returns_empty_list(self):
        from src.tools.self_improve import _parse_ruff

        assert _parse_ruff("", cap=10) == []
        assert _parse_ruff("[]", cap=10) == []

    def test_ruff_cap_respected(self):
        from src.tools.self_improve import _parse_ruff

        issues = [
            {
                "filename": f"src/f{i}.py",
                "location": {"row": i, "column": 1},
                "code": "E501",
                "message": "too long",
            }
            for i in range(10)
        ]
        findings = _parse_ruff(_ruff_stdout(issues), cap=3)
        assert len(findings) == 3

    def test_ruff_malformed_json_returns_empty(self):
        from src.tools.self_improve import _parse_ruff

        assert _parse_ruff("not json", cap=10) == []


class TestParseBandit:
    def test_bandit_high_included(self):
        from src.tools.self_improve import _parse_bandit

        findings = _parse_bandit(_bandit_stdout([_BANDIT_HIGH]), cap=10)
        assert len(findings) == 1
        assert findings[0].code == "B105"
        assert findings[0].linter == "bandit"

    def test_bandit_medium_included(self):
        from src.tools.self_improve import _parse_bandit

        findings = _parse_bandit(_bandit_stdout([_BANDIT_MEDIUM]), cap=10)
        assert len(findings) == 1
        assert findings[0].code == "B601"

    def test_bandit_low_skipped(self):
        from src.tools.self_improve import _parse_bandit

        findings = _parse_bandit(_bandit_stdout([_BANDIT_LOW]), cap=10)
        assert findings == []

    def test_bandit_mixed_severity_filters_correctly(self):
        from src.tools.self_improve import _parse_bandit

        findings = _parse_bandit(
            _bandit_stdout([_BANDIT_HIGH, _BANDIT_LOW, _BANDIT_MEDIUM]), cap=10
        )
        assert len(findings) == 2
        codes = {f.code for f in findings}
        assert "B105" in codes
        assert "B601" in codes
        assert "B110" not in codes


# ---------------------------------------------------------------------------
# Tests: detect phase (no findings)
# ---------------------------------------------------------------------------


class TestDetectPhase:
    def test_no_findings_returns_no_issues_message(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG
        with patch("src.tools.self_improve._run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (
                (0, "[]", "") if "ruff" in cmd else (0, '{"results":[]}', "")
            )
            result = mod.self_improve(target=str(tmp_path))
        assert "No issues found" in result

    def test_no_findings_skips_patch_phase(self, tmp_path: Path):
        import src.tools.self_improve as mod

        mod._config = _DUMMY_CONFIG
        with (
            patch("src.tools.self_improve._run") as mock_run,
            patch("src.tools.self_improve.create_chat_model_from_configs") as mock_llm,
        ):
            mock_run.return_value = (0, "[]", "")
            mod.self_improve(target=str(tmp_path))
        mock_llm.assert_not_called()

    def test_bandit_nonzero_rc_empty_output_warns(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG
        with patch("src.tools.self_improve._run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (
                (0, "[]", "") if "ruff" in cmd else (127, "", "command not found: bandit")
            )
            result = mod.self_improve(target=str(tmp_path))
        assert "No issues found" not in result
        assert "bandit exited with code 127" in result
        assert "command not found" in result

    def test_ruff_nonzero_rc_empty_output_warns(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG
        with patch("src.tools.self_improve._run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (
                (127, "", "command not found: ruff") if "ruff" in cmd else (0, '{"results":[]}', "")
            )
            result = mod.self_improve(target=str(tmp_path))
        assert "No issues found" not in result
        assert "ruff exited with code 127" in result
        assert "command not found" in result

    def test_both_linters_nonzero_rc_warns(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG
        with patch("src.tools.self_improve._run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (
                (127, "", "command not found: ruff") if "ruff" in cmd else (1, "", "bandit crashed")
            )
            result = mod.self_improve(target=str(tmp_path))
        assert "ruff exited with code 127" in result
        assert "bandit exited with code 1" in result

    def test_nonzero_rc_with_valid_findings_no_warning(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG
        ruff_out = _ruff_stdout([_RUFF_ISSUE])
        with (
            patch("src.tools.self_improve._run") as mock_run,
            patch("src.tools.self_improve.create_chat_model_from_configs") as mock_llm,
        ):
            mock_run.side_effect = lambda cmd, **kw: (
                (1, ruff_out, "") if "ruff" in cmd else (0, '{"results":[]}', "")
            )
            result = mod.self_improve(target=str(tmp_path), dry_run=True)
        assert "Warning: ruff exited" not in result
        assert "E501" in result
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: dry_run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_returns_findings_without_patching(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG
        ruff_out = _ruff_stdout([_RUFF_ISSUE])

        with (
            patch("src.tools.self_improve._run") as mock_run,
            patch("src.tools.self_improve.create_chat_model_from_configs") as mock_llm,
        ):
            mock_run.side_effect = lambda cmd, **kw: (
                (1, ruff_out, "") if "ruff" in cmd else (0, '{"results":[]}', "")
            )
            result = mod.self_improve(target=str(tmp_path), dry_run=True)

        assert "Dry-run" in result
        assert "E501" in result
        mock_llm.assert_not_called()

    def test_dry_run_does_not_write_files(self, tmp_path: Path):
        import src.tools.self_improve as mod

        mod._config = _DUMMY_CONFIG
        src_file = tmp_path / "src" / "foo.py"
        src_file.parent.mkdir(parents=True)
        src_file.write_text("x = 1\n", encoding="utf-8")
        ruff_issue = dict(_RUFF_ISSUE, filename=str(src_file))
        ruff_out = _ruff_stdout([ruff_issue])

        with patch("src.tools.self_improve._run") as mock_run:
            mock_run.side_effect = lambda cmd, **kw: (
                (1, ruff_out, "") if "ruff" in cmd else (0, '{"results":[]}', "")
            )
            mod.self_improve(target=str(tmp_path), dry_run=True)

        assert src_file.read_text(encoding="utf-8") == "x = 1\n"


# ---------------------------------------------------------------------------
# Tests: path traversal
# ---------------------------------------------------------------------------


class TestPathSecurity:
    def test_target_traversal_rejected(self):
        import src.tools.self_improve as mod

        mod._config = _DUMMY_CONFIG
        result = mod.self_improve(target="../etc")
        assert "Error" in result
        assert "traversal" in result.lower() or "outside" in result.lower()

    def test_absolute_path_outside_cwd_rejected(self):
        import src.tools.self_improve as mod

        mod._config = _DUMMY_CONFIG
        result = mod.self_improve(target="/etc/passwd")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Tests: patch phase
# ---------------------------------------------------------------------------


class TestPatchPhase:
    def test_llm_called_with_file_content_and_issue(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG

        src_file = tmp_path / "src" / "foo.py"
        src_file.parent.mkdir(parents=True)
        src_file.write_text("x = 1  # original\n", encoding="utf-8")

        ruff_issue = dict(_RUFF_ISSUE, filename=str(src_file))
        ruff_out = _ruff_stdout([ruff_issue])

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "x = 1\n"
        mock_llm.invoke.return_value = response

        with (
            patch("src.tools.self_improve._run") as mock_run,
            patch("src.tools.self_improve.create_chat_model_from_configs", return_value=mock_llm),
        ):
            mock_run.side_effect = lambda cmd, **kw: (
                (1, ruff_out, "")
                if "ruff" in cmd
                else ((0, '{"results":[]}', "") if "bandit" in cmd else (0, "1 passed", ""))
            )
            with patch("src.tools.self_improve._safe_patch_target", return_value=True):
                mod.self_improve(target=str(tmp_path))

        assert mock_llm.invoke.called
        call_arg = mock_llm.invoke.call_args[0][0][0]
        prompt_text = call_arg.content
        assert "E501" in prompt_text
        assert "x = 1  # original" in prompt_text

    def test_invalid_python_from_llm_skipped_file_unchanged(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG

        src_file = tmp_path / "src" / "foo.py"
        src_file.parent.mkdir(parents=True)
        original = "x = 1\n"
        src_file.write_text(original, encoding="utf-8")

        ruff_issue = dict(_RUFF_ISSUE, filename=str(src_file))
        ruff_out = _ruff_stdout([ruff_issue])

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = _INVALID_PYTHON  # SyntaxError
        mock_llm.invoke.return_value = response

        with (
            patch("src.tools.self_improve._run") as mock_run,
            patch("src.tools.self_improve.create_chat_model_from_configs", return_value=mock_llm),
        ):
            mock_run.side_effect = lambda cmd, **kw: (
                (1, ruff_out, "") if "ruff" in cmd else (0, '{"results":[]}', "")
            )
            with patch("src.tools.self_improve._safe_patch_target", return_value=True):
                result = mod.self_improve(target=str(tmp_path))

        assert src_file.read_text(encoding="utf-8") == original
        assert "Skipped" in result or "skipped" in result.lower()

    def test_max_fixes_cap_respected(self, tmp_path: Path):
        import src.tools.self_improve as mod

        mod._config = _DUMMY_CONFIG

        # 5 ruff issues, but max_fixes=2
        issues = [
            {
                "filename": f"src/f{i}.py",
                "location": {"row": i + 1, "column": 1},
                "code": "E501",
                "message": "too long",
            }
            for i in range(5)
        ]
        ruff_out = _ruff_stdout(issues)

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = _VALID_PYTHON
        mock_llm.invoke.return_value = response

        with (
            patch("src.tools.self_improve._run") as mock_run,
            patch("src.tools.self_improve.create_chat_model_from_configs", return_value=mock_llm),
            patch("src.tools.self_improve._safe_patch_target", return_value=True),
            patch("pathlib.Path.read_text", return_value=_VALID_PYTHON),
            patch("pathlib.Path.write_text"),
        ):
            mock_run.side_effect = lambda cmd, **kw: (
                (1, ruff_out, "")
                if "ruff" in cmd
                else ((0, '{"results":[]}', "") if "bandit" in cmd else (0, "ok", ""))
            )
            mod.self_improve(target=str(tmp_path), max_fixes=2)

        # LLM called at most 2 times (max_fixes=2)
        assert mock_llm.invoke.call_count <= 2


# ---------------------------------------------------------------------------
# Tests: verify phase
# ---------------------------------------------------------------------------


class TestVerifyPhase:
    def _setup_file(self, tmp_path: Path) -> tuple[Path, str]:
        src_file = tmp_path / "src" / "module.py"
        src_file.parent.mkdir(parents=True)
        original = "x = 1\n"
        src_file.write_text(original, encoding="utf-8")
        return src_file, original

    def test_pytest_passes_fix_retained(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG
        src_file, _ = self._setup_file(tmp_path)

        ruff_issue = {
            "filename": str(src_file),
            "location": {"row": 1, "column": 1},
            "code": "E501",
            "message": "too long",
        }

        mock_llm = MagicMock()
        response = MagicMock()
        patched = "x = 2\n"
        response.content = patched
        mock_llm.invoke.return_value = response

        with (
            patch("src.tools.self_improve._run") as mock_run,
            patch("src.tools.self_improve.create_chat_model_from_configs", return_value=mock_llm),
            patch("src.tools.self_improve._safe_patch_target", return_value=True),
        ):
            mock_run.side_effect = lambda cmd, **kw: (
                (1, _ruff_stdout([ruff_issue]), "")
                if "ruff" in cmd
                else ((0, '{"results":[]}', "") if "bandit" in cmd else (0, "1 passed", ""))
            )
            result = mod.self_improve(target=str(tmp_path))

        # _extract_code strips trailing whitespace, so "x = 2\n" → "x = 2"
        assert src_file.read_text(encoding="utf-8") == patched.rstrip()
        assert "Patched:  1" in result

    def test_pytest_fails_file_reverted(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG
        src_file, original = self._setup_file(tmp_path)

        ruff_issue = {
            "filename": str(src_file),
            "location": {"row": 1, "column": 1},
            "code": "E501",
            "message": "too long",
        }

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "x = BROKEN_CHANGE\n"
        mock_llm.invoke.return_value = response

        with (
            patch("src.tools.self_improve._run") as mock_run,
            patch("src.tools.self_improve.create_chat_model_from_configs", return_value=mock_llm),
            patch("src.tools.self_improve._safe_patch_target", return_value=True),
        ):
            mock_run.side_effect = lambda cmd, **kw: (
                (1, _ruff_stdout([ruff_issue]), "")
                if "ruff" in cmd
                else ((0, '{"results":[]}', "") if "bandit" in cmd else (1, "1 failed", ""))
            )
            result = mod.self_improve(target=str(tmp_path))

        assert src_file.read_text(encoding="utf-8") == original
        assert "Reverted" in result


# ---------------------------------------------------------------------------
# Tests: commit phase
# ---------------------------------------------------------------------------


class TestCommitPhase:
    def _setup(self, tmp_path: Path) -> tuple[Path, dict]:
        src_file = tmp_path / "src" / "module.py"
        src_file.parent.mkdir(parents=True)
        src_file.write_text("x = 1\n", encoding="utf-8")
        issue = {
            "filename": str(src_file),
            "location": {"row": 1, "column": 1},
            "code": "E501",
            "message": "too long",
        }
        return src_file, issue

    def test_auto_commit_false_no_git_commands(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG
        src_file, issue = self._setup(tmp_path)

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "x = 2\n"
        mock_llm.invoke.return_value = response

        git_calls: list = []

        def _side_effect(cmd, **kw):
            if "ruff" in cmd:
                return (1, _ruff_stdout([issue]), "")
            elif "bandit" in cmd:
                return (0, '{"results":[]}', "")
            elif "pytest" in cmd:
                return (0, "1 passed", "")
            elif "git" in cmd:
                git_calls.append(list(cmd))
                return (0, "", "")
            return (0, "", "")

        with (
            patch("src.tools.self_improve._run", side_effect=_side_effect),
            patch("src.tools.self_improve.create_chat_model_from_configs", return_value=mock_llm),
            patch("src.tools.self_improve._safe_patch_target", return_value=True),
        ):
            mod.self_improve(target=str(tmp_path), auto_commit=False)

        assert git_calls == [], f"Expected no git calls but got: {git_calls}"

    def test_auto_commit_true_runs_git_add_and_commit(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG
        src_file, issue = self._setup(tmp_path)

        mock_llm = MagicMock()
        response = MagicMock()
        response.content = "x = 2\n"
        mock_llm.invoke.return_value = response

        git_cmds: list[list[str]] = []

        def _side_effect(cmd, **kw):
            if "ruff" in cmd:
                return (1, _ruff_stdout([issue]), "")
            elif "bandit" in cmd:
                return (0, '{"results":[]}', "")
            elif "pytest" in cmd:
                return (0, "1 passed", "")
            elif "git" in cmd:
                git_cmds.append(list(cmd))
                return (0, "", "")
            return (0, "", "")

        with (
            patch("src.tools.self_improve._run", side_effect=_side_effect),
            patch("src.tools.self_improve.create_chat_model_from_configs", return_value=mock_llm),
            patch("src.tools.self_improve._safe_patch_target", return_value=True),
        ):
            result = mod.self_improve(target=str(tmp_path), auto_commit=True)

        git_add_seen = any(c[1] == "add" for c in git_cmds)
        git_commit_seen = any(c[1] == "commit" for c in git_cmds)
        assert git_add_seen, f"git add not called; git_cmds={git_cmds}"
        assert git_commit_seen, f"git commit not called; git_cmds={git_cmds}"
        assert "Committed" in result


# ---------------------------------------------------------------------------
# Tests: tool registry and configuration
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_tool_configs_requires_confirmation_true(self):
        from src.tools.self_improve import TOOL_CONFIGS

        cfg = TOOL_CONFIGS[0]
        assert cfg["requires_confirmation"] is True

    def test_tool_configs_name(self):
        from src.tools.self_improve import TOOL_CONFIGS

        assert TOOL_CONFIGS[0]["name"] == "self_improve"

    def test_tool_setup_sets_config(self):
        import src.tools.self_improve as mod

        original = mod._config
        try:
            mock_cfg = MagicMock()
            mod.TOOL_SETUP(mock_cfg)
            assert mod._config is mock_cfg
            assert mod.is_configured() is True
        finally:
            mod._config = original

    def test_is_configured_false_when_no_config(self):
        import src.tools.self_improve as mod

        original = mod._config
        try:
            mod._config = None
            assert mod.is_configured() is False
        finally:
            mod._config = original

    def test_self_improve_returns_error_when_not_configured(self):
        import src.tools.self_improve as mod

        original = mod._config
        try:
            mod._config = None
            result = mod.self_improve()
        finally:
            mod._config = original
        assert "Error" in result
        assert "not configured" in result


# ---------------------------------------------------------------------------
# Tests: LLM timeout
# ---------------------------------------------------------------------------


class TestPatchTimeout:
    def test_llm_invoke_timeout_skips_finding(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG

        src_file = tmp_path / "src" / "foo.py"
        src_file.parent.mkdir(parents=True)
        src_file.write_text("x = 1\n", encoding="utf-8")

        ruff_issue = dict(_RUFF_ISSUE, filename=str(src_file))
        ruff_out = _ruff_stdout([ruff_issue])

        mock_llm = MagicMock()
        stop_event = threading.Event()

        def _never_return(prompt):
            stop_event.wait(timeout=60)

        mock_llm.invoke.side_effect = _never_return

        with (
            patch("src.tools.self_improve._run") as mock_run,
            patch(
                "src.tools.self_improve.create_chat_model_from_configs",
                return_value=mock_llm,
            ),
            patch("src.tools.self_improve._safe_patch_target", return_value=True),
            patch("src.tools.self_improve._SELF_IMPROVE_LLM_TIMEOUT_SECONDS", 0.1),
        ):
            mock_run.side_effect = lambda cmd, **kw: (
                (1, ruff_out, "")
                if "ruff" in cmd
                else ((0, '{"results":[]}', "") if "bandit" in cmd else (0, "1 passed", ""))
            )
            start = time.monotonic()
            result = mod.self_improve(target=str(tmp_path))
            elapsed = time.monotonic() - start

        stop_event.set()

        assert elapsed < 5, f"Expected return within 5s, took {elapsed}s"
        assert "timed out" in result.lower() or "timeout" in result.lower()

    def test_hung_thread_returns_within_guard_timeout(self, tmp_path: Path, monkeypatch):
        import src.tools.self_improve as mod

        monkeypatch.chdir(tmp_path)
        mod._config = _DUMMY_CONFIG

        src_file = tmp_path / "src" / "foo.py"
        src_file.parent.mkdir(parents=True)
        src_file.write_text("x = 1\n", encoding="utf-8")

        ruff_issue = dict(_RUFF_ISSUE, filename=str(src_file))
        ruff_out = _ruff_stdout([ruff_issue])

        mock_llm = MagicMock()
        stop_event = threading.Event()
        mock_llm.invoke.side_effect = lambda *_a, **_kw: stop_event.wait(timeout=60)

        with (
            patch("src.tools.self_improve._run") as mock_run,
            patch(
                "src.tools.self_improve.create_chat_model_from_configs",
                return_value=mock_llm,
            ),
            patch("src.tools.self_improve._safe_patch_target", return_value=True),
            patch("src.tools.self_improve._SELF_IMPROVE_LLM_TIMEOUT_SECONDS", 0.1),
        ):
            mock_run.side_effect = lambda cmd, **kw: (
                (1, ruff_out, "")
                if "ruff" in cmd
                else ((0, '{"results":[]}', "") if "bandit" in cmd else (0, "1 passed", ""))
            )
            start = time.monotonic()
            mod.self_improve(target=str(tmp_path))
            elapsed = time.monotonic() - start

        stop_event.set()

        assert elapsed < 5, f"Expected return within 5s, took {elapsed}s"


# ---------------------------------------------------------------------------
# Tests: Config field
# ---------------------------------------------------------------------------


class TestConfigField:
    def test_self_improve_auto_commit_default_is_false(self):
        from src.config import Config

        cfg = Config()
        assert cfg.self_improve_auto_commit is False

    def test_self_improve_auto_commit_can_be_set_true(self):
        from src.config import Config

        cfg = Config(self_improve_auto_commit=True)
        assert cfg.self_improve_auto_commit is True
