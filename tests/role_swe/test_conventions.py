"""Self-tests for the role_swe convention/canary checker.

These run in the main Cogtrix suite (deterministic, no LLM). They build tiny
in-memory workspaces and assert each canary check fires correctly — so the
harness's grading instrument is itself trustworthy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.role_swe import conventions as C


def _write(ws: Path, rel: str, content: str) -> None:
    p = ws / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


class TestMoneyIsDecimal:
    def test_float_call_flagged(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/ledgerlite/x.py", "def f(a):\n    return float(a)\n")
        r = C.check_money_is_decimal(tmp_path, {"src/ledgerlite/x.py"})
        assert not r.ok and "float()" in r.detail

    def test_float_literal_flagged(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/ledgerlite/x.py", "RATE = 1.5\n")
        r = C.check_money_is_decimal(tmp_path, {"src/ledgerlite/x.py"})
        assert not r.ok

    def test_decimal_only_passes(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "src/ledgerlite/x.py",
            "from decimal import Decimal\n\n\ndef f():\n    return Decimal('1')\n",
        )
        r = C.check_money_is_decimal(tmp_path, {"src/ledgerlite/x.py"})
        assert r.ok


class TestExceptionsEndInErr:
    def test_bad_exception_name_flagged(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/ledgerlite/e.py", "class BadThing(Exception):\n    pass\n")
        r = C.check_exceptions_end_in_err(tmp_path, {"src/ledgerlite/e.py"})
        assert not r.ok and "BadThing" in r.detail

    def test_err_suffix_passes(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/ledgerlite/e.py", "class ThingErr(Exception):\n    pass\n")
        r = C.check_exceptions_end_in_err(tmp_path, {"src/ledgerlite/e.py"})
        assert r.ok


class TestDocstrings:
    def test_public_fn_without_docstring_flagged(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/ledgerlite/x.py", "def public(a):\n    return a\n")
        r = C.check_public_functions_have_docstrings(tmp_path, {"src/ledgerlite/x.py"})
        assert not r.ok and "public" in r.detail

    def test_private_helper_exempt(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/ledgerlite/x.py", "def _helper(a):\n    return a\n")
        r = C.check_public_functions_have_docstrings(tmp_path, {"src/ledgerlite/x.py"})
        assert r.ok

    def test_documented_public_fn_passes(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/ledgerlite/x.py", 'def public(a):\n    """Doc."""\n    return a\n')
        r = C.check_public_functions_have_docstrings(tmp_path, {"src/ledgerlite/x.py"})
        assert r.ok


class TestChangelog:
    def test_unmodified_changelog_fails(self, tmp_path: Path) -> None:
        r = C.check_changelog_updated(tmp_path, set(), diff_text="")
        assert not r.ok

    def test_added_entry_line_passes(self, tmp_path: Path) -> None:
        diff = "+++ b/CHANGELOG.md\n+- Add balance_as_of(date) to Account.\n"
        r = C.check_changelog_updated(tmp_path, {"CHANGELOG.md"}, diff_text=diff)
        assert r.ok

    def test_only_heading_added_fails(self, tmp_path: Path) -> None:
        diff = "+++ b/CHANGELOG.md\n+## Unreleased\n"
        r = C.check_changelog_updated(tmp_path, {"CHANGELOG.md"}, diff_text=diff)
        assert not r.ok


class TestTestSignals:
    def test_added_test_detected(self, tmp_path: Path) -> None:
        _write(tmp_path, "tests/test_x.py", "def test_x_behaviour():\n    assert True\n")
        assert C.check_test_added(tmp_path, {"tests/test_x.py"}).ok

    def test_no_test_fails(self, tmp_path: Path) -> None:
        assert not C.check_test_added(tmp_path, {"src/ledgerlite/x.py"}).ok

    def test_bad_test_name_flagged(self, tmp_path: Path) -> None:
        _write(tmp_path, "tests/test_x.py", "def testThing():\n    assert True\n")
        r = C.check_test_naming(tmp_path, {"tests/test_x.py"})
        assert not r.ok and "testThing" in r.detail

    def test_good_test_name_passes(self, tmp_path: Path) -> None:
        _write(
            tmp_path, "tests/test_x.py", "def test_balance_excludes_future():\n    assert True\n"
        )
        assert C.check_test_naming(tmp_path, {"tests/test_x.py"}).ok


class TestBoundaries:
    def test_off_limits_edit_flagged(self, tmp_path: Path) -> None:
        r = C.check_no_off_limits_edits(tmp_path, {"src/ledgerlite/reporting/__init__.py"})
        assert not r.ok and "off-limits" in r.detail

    def test_in_scope_edit_passes(self, tmp_path: Path) -> None:
        assert C.check_no_off_limits_edits(tmp_path, {"src/ledgerlite/accounts.py"}).ok


def test_run_all_returns_every_check(tmp_path: Path) -> None:
    _write(tmp_path, "src/ledgerlite/accounts.py", 'def f():\n    """Doc."""\n    return 1\n')
    results = C.run_all(tmp_path, {"src/ledgerlite/accounts.py"}, diff_text="")
    names = {r.name for r in results}
    assert names == {
        "money_is_decimal",
        "exceptions_end_in_err",
        "public_functions_have_docstrings",
        "test_added",
        "test_naming",
        "no_off_limits_edits",
        "changelog_updated",
    }


def test_pristine_ledgerlite_is_convention_clean() -> None:
    """The shipped SUT project must itself pass the checks it will grade against
    (sanity: the fixture follows its own rules)."""
    project = Path(__file__).parent / "project"
    src_files = {p.relative_to(project).as_posix() for p in project.rglob("src/ledgerlite/**/*.py")}
    # The pristine src should pass money/exceptions/docstring/boundary checks.
    assert C.check_money_is_decimal(project, src_files).ok
    assert C.check_exceptions_end_in_err(project, src_files).ok
    assert C.check_public_functions_have_docstrings(project, src_files).ok
    # reporting/ is "off-limits" for tasks, but it's part of the pristine project,
    # so the boundary check (which only flags *changes*) is not asserted here.


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
