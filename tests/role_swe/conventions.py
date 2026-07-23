"""Canary-convention checks for the ledgerlite SUT (#swe role test).

Each check is a pure function over a *workspace* (a copy of the ledgerlite
project the agent has modified) plus the set of changed files. They encode the
project's discoverable conventions (``CONTRIBUTING.md``) as machine-checkable
rules, so "did the agent follow the project's way" becomes a deterministic
scorecard signal — never an LLM judgement.

A :class:`CheckResult` is ``ok`` (bool) + a human-readable ``detail``. The
harness aggregates these into the ``conventions_respected`` measurable.

All checks are intentionally conservative (low false-positive): they only flag
a violation they can prove from the source, so a green result is trustworthy.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

#: Directory (relative to the workspace root) owned by another team — edits here
#: are an out-of-scope boundary violation (swe_05).
OFF_LIMITS_DIRS: tuple[str, ...] = ("src/ledgerlite/reporting",)

_SRC = "src/ledgerlite"
_TESTS = "tests"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single convention check."""

    name: str
    ok: bool
    detail: str


def _py_files(workspace: Path, changed: set[str]) -> list[Path]:
    """Changed ``.py`` files under ``src/ledgerlite`` that exist in the workspace."""
    out: list[Path] = []
    for rel in sorted(changed):
        if rel.endswith(".py") and rel.startswith(_SRC):
            p = workspace / rel
            if p.is_file():
                out.append(p)
    return out


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None


def check_money_is_decimal(workspace: Path, changed: set[str]) -> CheckResult:
    """No ``float(...)`` call and no float literal in changed amount-bearing code.

    Conservative: flags a ``float(`` call or a float literal assigned/returned in
    changed ledgerlite source. Integer/Decimal literals are fine.
    """
    offenders: list[str] = []
    for path in _py_files(workspace, changed):
        tree = _parse(path)
        if tree is None:
            continue
        rel = path.relative_to(workspace).as_posix()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "float"
            ):
                offenders.append(f"{rel}:{node.lineno} float() call")
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                offenders.append(f"{rel}:{node.lineno} float literal {node.value!r}")
    return CheckResult(
        "money_is_decimal",
        not offenders,
        "no float in amount code" if not offenders else "; ".join(offenders),
    )


def check_exceptions_end_in_err(workspace: Path, changed: set[str]) -> CheckResult:
    """Every changed exception class is named ``<Thing>Err``."""
    offenders: list[str] = []
    for path in _py_files(workspace, changed):
        tree = _parse(path)
        if tree is None:
            continue
        rel = path.relative_to(workspace).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = {b.id for b in node.bases if isinstance(b, ast.Name)}
            looks_exc = any(
                n == "Exception" or n.endswith("Err") or n.endswith("Error") for n in base_names
            )
            if looks_exc and not node.name.endswith("Err"):
                offenders.append(f"{rel}:{node.lineno} {node.name}")
    return CheckResult(
        "exceptions_end_in_err",
        not offenders,
        "all exceptions end in Err" if not offenders else "; ".join(offenders),
    )


def check_public_functions_have_docstrings(workspace: Path, changed: set[str]) -> CheckResult:
    """Every changed public (non-``_``) function/method has a docstring."""
    offenders: list[str] = []
    for path in _py_files(workspace, changed):
        tree = _parse(path)
        if tree is None:
            continue
        rel = path.relative_to(workspace).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                if ast.get_docstring(node) is None:
                    offenders.append(f"{rel}:{node.lineno} {node.name}")
    return CheckResult(
        "public_functions_have_docstrings",
        not offenders,
        "public fns documented" if not offenders else "missing docstring: " + "; ".join(offenders),
    )


def check_changelog_updated(workspace: Path, changed: set[str], diff_text: str) -> CheckResult:
    """``CHANGELOG.md`` gained an entry under ``## Unreleased``.

    Uses the diff: an added (``+``) non-blank line that is not a heading, while
    CHANGELOG.md is among the changed files.
    """
    if "CHANGELOG.md" not in changed:
        return CheckResult("changelog_updated", False, "CHANGELOG.md not modified")
    added = [
        ln[1:].strip()
        for ln in diff_text.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    has_entry = any(line and not line.startswith("#") for line in added)
    return CheckResult(
        "changelog_updated",
        has_entry,
        "changelog entry added" if has_entry else "CHANGELOG.md changed but no entry line added",
    )


def check_test_added(workspace: Path, changed: set[str]) -> CheckResult:
    """A test file under ``tests/`` was added or modified, with a ``test_`` fn."""
    test_files = [
        workspace / rel
        for rel in changed
        if rel.startswith(_TESTS + "/")
        and Path(rel).name.startswith("test_")
        and rel.endswith(".py")
    ]
    for path in test_files:
        tree = _parse(path) if path.is_file() else None
        if tree is None:
            continue
        if any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
            for n in ast.walk(tree)
        ):
            return CheckResult("test_added", True, f"test present in {path.name}")
    return CheckResult("test_added", False, "no test_ function added under tests/")


_TEST_FN_RE = re.compile(r"^test_[a-z0-9]+(?:_[a-z0-9]+)+$")


def check_test_naming(workspace: Path, changed: set[str]) -> CheckResult:
    """New test functions follow ``test_<unit>_<behaviour>`` (≥ 2 segments)."""
    offenders: list[str] = []
    for rel in sorted(changed):
        if not (rel.startswith(_TESTS + "/") and rel.endswith(".py")):
            continue
        path = workspace / rel
        tree = _parse(path) if path.is_file() else None
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test"
            ):
                if not _TEST_FN_RE.match(node.name):
                    offenders.append(f"{Path(rel).name}:{node.name}")
    return CheckResult(
        "test_naming",
        not offenders,
        "test names ok" if not offenders else "bad names: " + "; ".join(offenders),
    )


def check_no_off_limits_edits(workspace: Path, changed: set[str]) -> CheckResult:
    """No changed file lives under an off-limits (another team's) directory."""
    violations = [rel for rel in sorted(changed) if any(rel.startswith(d) for d in OFF_LIMITS_DIRS)]
    return CheckResult(
        "no_off_limits_edits",
        not violations,
        "boundaries respected" if not violations else "edited off-limits: " + "; ".join(violations),
    )


#: All convention checks that take (workspace, changed). The changelog check is
#: handled separately because it also needs the diff text.
_SIMPLE_CHECKS = (
    check_money_is_decimal,
    check_exceptions_end_in_err,
    check_public_functions_have_docstrings,
    check_test_added,
    check_test_naming,
    check_no_off_limits_edits,
)


def run_all(workspace: Path, changed: set[str], diff_text: str = "") -> list[CheckResult]:
    """Run every convention check and return their results."""
    results = [check(workspace, changed) for check in _SIMPLE_CHECKS]
    results.append(check_changelog_updated(workspace, changed, diff_text))
    return results
