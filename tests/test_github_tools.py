"""Tests for src/tools/github_tools.py."""

from __future__ import annotations

import base64
import json
import subprocess
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    """Return a mock CompletedProcess-like object."""
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = returncode
    return cp


def _b64(text: str) -> str:
    """Encode *text* as base64 with embedded newlines (matches GitHub API format)."""
    raw = base64.b64encode(text.encode()).decode()
    # Insert newlines every 60 chars to mimic GitHub's format
    return "\n".join(raw[i : i + 60] for i in range(0, len(raw), 60))


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


class TestIsConfigured:
    def test_returns_false_when_gh_not_in_path(self):
        from src.tools.github_tools import is_configured

        with patch("src.tools.github_tools.shutil.which", return_value=None):
            assert is_configured() is False

    def test_returns_true_when_gh_available(self):
        from src.tools.github_tools import is_configured

        with patch("src.tools.github_tools.shutil.which", return_value="/usr/bin/gh"):
            assert is_configured() is True


# ---------------------------------------------------------------------------
# TOOL_SETUP / configure_github_tools
# ---------------------------------------------------------------------------


class TestToolSetup:
    def test_tool_setup_sets_default_repo(self):
        import src.tools.github_tools as mod
        from src.tools.github_tools import TOOL_SETUP

        config = MagicMock()
        config.services = {"github": {"default_repo": "myorg/myrepo"}}

        TOOL_SETUP(config)
        assert mod._default_repo == "myorg/myrepo"

    def test_tool_setup_empty_services(self):
        import src.tools.github_tools as mod
        from src.tools.github_tools import TOOL_SETUP

        config = MagicMock()
        config.services = {}

        TOOL_SETUP(config)
        assert mod._default_repo == ""


# ---------------------------------------------------------------------------
# TOOL_CONFIGS structure
# ---------------------------------------------------------------------------


class TestToolConfigs:
    def test_tool_configs_has_four_entries(self):
        from src.tools.github_tools import TOOL_CONFIGS

        assert len(TOOL_CONFIGS) == 4

    def test_tool_configs_names(self):
        from src.tools.github_tools import TOOL_CONFIGS

        names = {t["name"] for t in TOOL_CONFIGS}
        assert names == {"gh_create_issue", "gh_comment_issue", "gh_list_prs", "gh_get_file"}

    def test_create_issue_requires_confirmation(self):
        from src.tools.github_tools import TOOL_CONFIGS

        cfg = next(t for t in TOOL_CONFIGS if t["name"] == "gh_create_issue")
        assert cfg["requires_confirmation"] is True

    def test_comment_issue_requires_confirmation(self):
        from src.tools.github_tools import TOOL_CONFIGS

        cfg = next(t for t in TOOL_CONFIGS if t["name"] == "gh_comment_issue")
        assert cfg["requires_confirmation"] is True

    def test_list_prs_no_confirmation(self):
        from src.tools.github_tools import TOOL_CONFIGS

        cfg = next(t for t in TOOL_CONFIGS if t["name"] == "gh_list_prs")
        assert cfg["requires_confirmation"] is False

    def test_get_file_no_confirmation(self):
        from src.tools.github_tools import TOOL_CONFIGS

        cfg = next(t for t in TOOL_CONFIGS if t["name"] == "gh_get_file")
        assert cfg["requires_confirmation"] is False


# ---------------------------------------------------------------------------
# gh_create_issue
# ---------------------------------------------------------------------------


class TestGhCreateIssue:
    def test_create_issue_success(self):
        from src.tools.github_tools import gh_create_issue

        payload = json.dumps({"number": 42, "url": "https://github.com/o/r/issues/42"})
        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stdout=payload),
        ):
            result = gh_create_issue(title="My issue", body="Description", repo="owner/repo")
        assert "#42" in result
        assert "My issue" in result
        assert "https://github.com/o/r/issues/42" in result

    def test_create_issue_empty_repo_no_default(self):
        import src.tools.github_tools as mod
        from src.tools.github_tools import gh_create_issue

        original = mod._default_repo
        mod._default_repo = ""
        try:
            result = gh_create_issue(title="T", repo="")
        finally:
            mod._default_repo = original

        assert "Error" in result
        assert "repo not specified" in result

    def test_create_issue_invalid_repo(self):
        from src.tools.github_tools import gh_create_issue

        result = gh_create_issue(title="T", repo="../../evil")
        assert "Error" in result
        assert "invalid repo format" in result

    def test_create_issue_with_labels(self):
        from src.tools.github_tools import gh_create_issue

        payload = json.dumps({"number": 7, "url": "https://github.com/o/r/issues/7"})
        captured: list[list[str]] = []

        def _capture(cmd, **_kw):
            captured.append(cmd)
            return _make_completed(stdout=payload)

        with patch("src.tools.github_tools.subprocess.run", side_effect=_capture):
            gh_create_issue(title="T", repo="owner/repo", labels="bug,enhancement")

        assert "--label" in captured[0]
        label_idx = captured[0].index("--label")
        assert captured[0][label_idx + 1] == "bug,enhancement"

    def test_create_issue_gh_error(self):
        from src.tools.github_tools import gh_create_issue

        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stderr="authentication required", returncode=1),
        ):
            result = gh_create_issue(title="T", repo="owner/repo")
        assert "gh error" in result
        assert "authentication required" in result

    def test_create_issue_strips_null_bytes(self):
        from src.tools.github_tools import gh_create_issue

        payload = json.dumps({"number": 1, "url": "https://github.com/o/r/issues/1"})
        captured: list[list[str]] = []

        def _capture(cmd, **_kw):
            captured.append(cmd)
            return _make_completed(stdout=payload)

        with patch("src.tools.github_tools.subprocess.run", side_effect=_capture):
            gh_create_issue(title="T\x00itle", body="bo\x00dy", repo="owner/repo")

        title_idx = captured[0].index("--title")
        body_idx = captured[0].index("--body")
        assert "\x00" not in captured[0][title_idx + 1]
        assert "\x00" not in captured[0][body_idx + 1]


# ---------------------------------------------------------------------------
# gh_comment_issue
# ---------------------------------------------------------------------------


class TestGhCommentIssue:
    def test_comment_success(self):
        from src.tools.github_tools import gh_comment_issue

        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stdout=""),
        ):
            result = gh_comment_issue(issue_number=5, body="LGTM!", repo="owner/repo")
        assert "Comment added to #5" in result

    def test_comment_empty_repo_no_default(self):
        import src.tools.github_tools as mod
        from src.tools.github_tools import gh_comment_issue

        original = mod._default_repo
        mod._default_repo = ""
        try:
            result = gh_comment_issue(issue_number=1, body="hi", repo="")
        finally:
            mod._default_repo = original

        assert "Error" in result

    def test_comment_gh_error(self):
        from src.tools.github_tools import gh_comment_issue

        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stderr="not found", returncode=1),
        ):
            result = gh_comment_issue(issue_number=99, body="x", repo="owner/repo")
        assert "gh error" in result


# ---------------------------------------------------------------------------
# gh_list_prs
# ---------------------------------------------------------------------------


class TestGhListPrs:
    def test_list_prs_formatted(self):
        from src.tools.github_tools import gh_list_prs

        prs = [
            {"number": 10, "title": "Fix bug", "author": {"login": "alice"}, "state": "OPEN"},
            {
                "number": 11,
                "title": "Add feature",
                "author": {"login": "bob"},
                "state": "OPEN",
            },
        ]
        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stdout=json.dumps(prs)),
        ):
            result = gh_list_prs(repo="owner/repo")

        assert "#10 | Fix bug | alice | open" in result
        assert "#11 | Add feature | bob | open" in result

    def test_list_prs_empty(self):
        from src.tools.github_tools import gh_list_prs

        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stdout=json.dumps([])),
        ):
            result = gh_list_prs(repo="owner/repo")

        assert result == "No pull requests found."

    def test_list_prs_gh_error(self):
        from src.tools.github_tools import gh_list_prs

        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stderr="rate limit exceeded", returncode=1),
        ):
            result = gh_list_prs(repo="owner/repo")

        assert "gh error" in result

    def test_list_prs_passes_state_and_limit(self):
        from src.tools.github_tools import gh_list_prs

        captured: list[list[str]] = []

        def _capture(cmd, **_kw):
            captured.append(cmd)
            return _make_completed(stdout=json.dumps([]))

        with patch("src.tools.github_tools.subprocess.run", side_effect=_capture):
            gh_list_prs(repo="owner/repo", state="merged", limit=5)

        cmd = captured[0]
        assert "--state" in cmd
        assert cmd[cmd.index("--state") + 1] == "merged"
        assert "--limit" in cmd
        assert cmd[cmd.index("--limit") + 1] == "5"


# ---------------------------------------------------------------------------
# gh_get_file
# ---------------------------------------------------------------------------


class TestGhGetFile:
    def _file_response(self, content: str) -> str:
        return json.dumps(
            {
                "type": "file",
                "name": "test.py",
                "encoding": "base64",
                "content": _b64(content),
            }
        )

    def test_get_file_success(self):
        from src.tools.github_tools import gh_get_file

        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stdout=self._file_response("hello world")),
        ):
            result = gh_get_file(path="README.md", repo="owner/repo")
        assert result == "hello world"

    def test_get_file_truncation(self):
        from src.tools.github_tools import gh_get_file

        long_content = "x" * 15_000
        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stdout=self._file_response(long_content)),
        ):
            result = gh_get_file(path="big.txt", repo="owner/repo")

        assert len(result) < 15_000
        assert "truncated" in result
        assert "5000 chars omitted" in result

    def test_get_file_404(self):
        from src.tools.github_tools import gh_get_file

        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stderr="HTTP 404: Not Found", returncode=1),
        ):
            result = gh_get_file(path="missing.py", repo="owner/repo")

        assert "file not found" in result
        assert "missing.py" in result

    def test_get_file_with_ref(self):
        from src.tools.github_tools import gh_get_file

        captured: list[list[str]] = []

        def _capture(cmd, **_kw):
            captured.append(cmd)
            return _make_completed(stdout=self._file_response("content"))

        with patch("src.tools.github_tools.subprocess.run", side_effect=_capture):
            gh_get_file(path="src/main.py", repo="owner/repo", ref="feature-branch")

        cmd = captured[0]
        assert "--raw-field" in cmd
        rf_idx = cmd.index("--raw-field")
        assert cmd[rf_idx + 1] == "ref=feature-branch"

    def test_path_validation_rejects_dotdot(self):
        from src.tools.github_tools import gh_get_file

        result = gh_get_file(path="../etc/passwd", repo="owner/repo")
        assert "Error" in result
        assert ".." in result

    def test_path_validation_rejects_leading_slash(self):
        from src.tools.github_tools import gh_get_file

        result = gh_get_file(path="/etc/passwd", repo="owner/repo")
        assert "Error" in result

    def test_get_file_repo_not_configured(self):
        import src.tools.github_tools as mod
        from src.tools.github_tools import gh_get_file

        original = mod._default_repo
        mod._default_repo = ""
        try:
            result = gh_get_file(path="README.md", repo="")
        finally:
            mod._default_repo = original

        assert "Error" in result

    def test_invalid_ref_rejected(self):
        from src.tools.github_tools import gh_get_file

        result = gh_get_file(path="README.md", repo="owner/repo", ref="bad ref!")
        assert "Error" in result
        assert "invalid ref" in result


# ---------------------------------------------------------------------------
# Repo format validation
# ---------------------------------------------------------------------------


class TestRepoValidation:
    def test_rejects_traversal_repo(self):
        from src.tools.github_tools import gh_create_issue

        result = gh_create_issue(title="T", repo="../../evil")
        assert "Error" in result
        assert "invalid repo format" in result

    def test_accepts_valid_repo(self):
        from src.tools.github_tools import gh_create_issue

        payload = json.dumps({"number": 1, "url": "https://github.com/o/r/issues/1"})
        with patch(
            "src.tools.github_tools.subprocess.run",
            return_value=_make_completed(stdout=payload),
        ):
            result = gh_create_issue(title="T", repo="my-org/my.repo_name")
        assert "Error" not in result

    def test_rejects_single_component(self):
        from src.tools.github_tools import gh_create_issue

        result = gh_create_issue(title="T", repo="noslash")
        assert "Error" in result

    def test_rejects_spaces_in_repo(self):
        from src.tools.github_tools import gh_create_issue

        result = gh_create_issue(title="T", repo="owner/ repo")
        assert "Error" in result

    def test_uses_default_repo_when_empty(self):
        import src.tools.github_tools as mod
        from src.tools.github_tools import gh_list_prs

        original = mod._default_repo
        mod._default_repo = "default-org/default-repo"
        captured: list[list[str]] = []

        def _capture(cmd, **_kw):
            captured.append(cmd)
            return _make_completed(stdout=json.dumps([]))

        try:
            with patch("src.tools.github_tools.subprocess.run", side_effect=_capture):
                gh_list_prs(repo="")
        finally:
            mod._default_repo = original

        cmd = captured[0]
        assert "--repo" in cmd
        assert cmd[cmd.index("--repo") + 1] == "default-org/default-repo"
