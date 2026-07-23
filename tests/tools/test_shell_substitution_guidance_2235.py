"""#2235 — the command-substitution block (#1104) should redirect file-content
writes to write_file instead of leaving models to spiral on shell workarounds.

The block itself is unchanged (substitution is still rejected); only the error
*message* is enriched: when the blocked command looks like a file-content write
(heredoc / redirect), it names the write_file tool. Non-write commands keep the
generic message. This is options 1+2 of #2235; quote/heredoc-aware relaxation of
the guard is deferred (phase 2).
"""

from __future__ import annotations

import cogtrix_core.tools.shell as shell

_HEREDOC_MD = "cat > /workspace/cluster-docs.md <<'EOF'\n# Doc\n```bash\nls\n```\nEOF"


class TestSubstitutionStillBlocked:
    """Security intact: substitution is rejected, message says blocked/substitution."""

    def test_backtick_still_blocked(self) -> None:
        r = shell.execute_shell_command("echo `whoami`")
        assert "blocked" in r.lower() or "substitution" in r.lower()

    def test_dollar_paren_still_blocked(self) -> None:
        r = shell.execute_shell_command("echo $(id)")
        assert "blocked" in r.lower() or "substitution" in r.lower()

    def test_heredoc_with_backticks_blocked(self) -> None:
        r = shell.execute_shell_command(_HEREDOC_MD)
        assert "blocked" in r.lower() or "substitution" in r.lower()


class TestFileWriteRedirectsToWriteFile:
    """A blocked file-content write points the model at write_file (#2235)."""

    def test_heredoc_markdown_suggests_write_file(self) -> None:
        r = shell.execute_shell_command(_HEREDOC_MD)
        assert "write_file" in r

    def test_redirect_write_with_backtick_suggests_write_file(self) -> None:
        r = shell.execute_shell_command("echo '`code`' > /workspace/x.md")
        assert "write_file" in r

    def test_dollar_paren_in_heredoc_suggests_write_file(self) -> None:
        r = shell.execute_shell_command("cat > /tmp/x.sh <<'EOF'\nx=$(date)\nEOF")
        assert "write_file" in r


class TestNonWriteKeepsGenericMessage:
    """Substitution outside a file-write context must NOT mis-suggest write_file."""

    def test_plain_backtick_command_generic(self) -> None:
        r = shell.execute_shell_command("echo `whoami`")
        assert "write_file" not in r
        assert "blocked" in r.lower() or "substitution" in r.lower()

    def test_plain_dollar_paren_generic(self) -> None:
        r = shell.execute_shell_command("FOO=$(id) env")
        assert "write_file" not in r

    def test_process_substitution_not_treated_as_file_write(self) -> None:
        # `>(` is process substitution, not a redirect to a path.
        r = shell.execute_shell_command("tee >(cat)")
        assert "blocked" in r.lower() or "substitution" in r.lower()
        assert "write_file" not in r


class TestFileWriteHeuristic:
    def test_heredoc_detected(self) -> None:
        assert shell._looks_like_file_write("cat > f <<EOF\nx\nEOF") is True

    def test_redirect_to_path_detected(self) -> None:
        assert shell._looks_like_file_write("echo hi > /tmp/f") is True

    def test_stderr_redirect_not_detected(self) -> None:
        assert shell._looks_like_file_write("foo 2> /dev/null") is False

    def test_process_substitution_not_detected(self) -> None:
        assert shell._looks_like_file_write("tee >(cat)") is False

    def test_plain_command_not_detected(self) -> None:
        assert shell._looks_like_file_write("echo `whoami`") is False
