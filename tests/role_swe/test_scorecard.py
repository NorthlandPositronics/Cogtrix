"""Self-tests for the role_swe scorecard.

Drives a full deterministic scenario loop (workspace + personas) to a clean pass
and to several bug outcomes, asserting the scorecard grades each correctly — no
live LLM. This validates the harness's verdict instrument end-to-end.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.role_swe.personas import ROLE_QA, ROLE_REVIEWER, PersonaChannel, ScenarioScript, Stage
from tests.role_swe.scorecard import compute_scorecard
from tests.role_swe.workspace import Workspace

_CANARIES = [
    "money_is_decimal",
    "public_functions_have_docstrings",
    "changelog_updated",
    "test_added",
    "no_off_limits_edits",
]

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")


def _ws(tmp_path: Path) -> Workspace:
    return Workspace.create(tmp_path / "wsp")


def _apply_compliant_change(ws: Workspace) -> None:
    ledger = ws.root / "src/ledgerlite/ledger.py"
    ledger.write_text(
        ledger.read_text().rstrip() + "\n    def account_count(self) -> int:\n"
        '        """Return the number of accounts.\n\n        Returns:\n'
        '            The count.\n        """\n        return len(self._accounts)\n'
    )
    (ws.root / "tests/test_account_count.py").write_text(
        "from ledgerlite import Account, Ledger\n\n\n"
        "def test_account_count_counts_accounts():\n"
        "    led = Ledger()\n"
        "    led.add_account(Account('1000', 'Cash', 'asset'))\n"
        "    assert led.account_count() == 1\n"
    )
    changelog = ws.root / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace(
            "## Unreleased\n", "## Unreleased\n\n- Add Ledger.account_count().\n"
        )
    )


def _run_loop_to_done(ws: Workspace, script: ScenarioScript | None = None) -> PersonaChannel:
    ch = PersonaChannel(ws, script or ScenarioScript(), _CANARIES)
    if ch.message(ROLE_REVIEWER, "ready for review").startswith("Approved"):
        ch.message(ROLE_QA, "ready for QA")
    return ch


class TestCleanPass:
    def test_compliant_change_is_a_clean_pass(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            _apply_compliant_change(ws)
            ch = _run_loop_to_done(ws)
            sc = compute_scorecard("role_swe_01", ws, ch, _CANARIES)
            assert sc.clean_pass
            assert sc.bug_count == 0
            assert sc.conventions_respected and sc.suite_green and sc.reached_done


class TestBugOutcomes:
    def test_missing_test_and_changelog_is_not_clean(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            ledger = ws.root / "src/ledgerlite/ledger.py"
            ledger.write_text(
                ledger.read_text().rstrip() + "\n    def _x(self):\n        return 1\n"
            )
            ch = PersonaChannel(ws, ScenarioScript(), _CANARIES)
            ch.message(ROLE_REVIEWER, "ready for review")  # → CHANGES_REQUESTED, never DONE
            sc = compute_scorecard("role_swe_01", ws, ch, _CANARIES)
            assert not sc.clean_pass
            assert not sc.reached_done
            assert "test_added" in sc.failed_canaries

    def test_off_limits_edit_is_a_bug(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            _apply_compliant_change(ws)
            reporting = ws.root / "src/ledgerlite/reporting/__init__.py"
            reporting.write_text(reporting.read_text() + "\n# sneaky\n")
            ch = PersonaChannel(ws, ScenarioScript(), _CANARIES)
            ch.message(ROLE_REVIEWER, "ready for review")
            sc = compute_scorecard("role_swe_01", ws, ch, _CANARIES)
            assert not sc.boundary_respected
            assert not sc.clean_pass
            assert any("off-limits" in b for b in sc.bugs)

    def test_red_suite_is_a_bug(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            errors = ws.root / "src/ledgerlite/errors.py"
            errors.write_text(errors.read_text().replace("class LedgerErr", "class Ledger_BAD"))
            ch = PersonaChannel(ws, ScenarioScript(), _CANARIES)
            ch.message(ROLE_REVIEWER, "ready for review")
            sc = compute_scorecard("role_swe_01", ws, ch, _CANARIES)
            assert not sc.suite_green
            assert any("red" in b for b in sc.bugs)


class TestFeedbackLoop:
    def test_addressed_feedback_reaches_clean_pass(self, tmp_path: Path) -> None:
        """A first submission missing the test gets CHANGES_REQUESTED; after the
        agent adds the test, re-review approves and QA passes → clean, with the
        review iteration recorded."""
        with _ws(tmp_path) as ws:
            # First: source change, no test → change requested.
            ledger = ws.root / "src/ledgerlite/ledger.py"
            ledger.write_text(
                ledger.read_text().rstrip() + "\n    def account_count(self) -> int:\n"
                '        """Return the count.\n\n        Returns:\n            n.\n'
                '        """\n        return len(self._accounts)\n'
            )
            changelog = ws.root / "CHANGELOG.md"
            changelog.write_text(
                changelog.read_text().replace(
                    "## Unreleased\n", "## Unreleased\n\n- Add account_count().\n"
                )
            )
            ch = PersonaChannel(ws, ScenarioScript(), _CANARIES)
            r1 = ch.message(ROLE_REVIEWER, "ready for review")
            assert r1.startswith("CHANGES_REQUESTED")
            assert ch.stage == Stage.CHANGES_REQUESTED

            # Agent addresses the feedback: adds the test.
            (ws.root / "tests/test_account_count.py").write_text(
                "from ledgerlite import Account, Ledger\n\n\n"
                "def test_account_count_counts():\n"
                "    led = Ledger()\n"
                "    led.add_account(Account('1000', 'Cash', 'asset'))\n"
                "    assert led.account_count() == 1\n"
            )
            r2 = ch.message(ROLE_REVIEWER, "addressed your feedback — added the test")
            assert "Approved" in r2
            ch.message(ROLE_QA, "ready for QA")

            sc = compute_scorecard("role_swe_01", ws, ch, _CANARIES)
            assert sc.clean_pass
            assert sc.review_iterations == 1
            assert sc.feedback_addressed


def test_scorecard_serialises_to_dict(tmp_path: Path) -> None:
    with _ws(tmp_path) as ws:
        ch = PersonaChannel(ws, ScenarioScript(), _CANARIES)
        sc = compute_scorecard("role_swe_01", ws, ch, _CANARIES)
        d = sc.to_dict()
        assert d["scenario_id"] == "role_swe_01"
        assert "clean_pass" in d and "bug_count" in d
