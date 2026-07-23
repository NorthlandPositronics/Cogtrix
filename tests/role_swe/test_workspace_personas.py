"""Self-tests for the role_swe workspace + persona simulation.

These exercise the real git-isolated workspace (the tiny SUT runs in <1s) and the
deterministic persona state machine end-to-end — no live LLM. They prove the
team-simulation grading instrument behaves reproducibly: the reviewer approves a
convention-clean change, requests changes on a dirty one, and QA gates on the
suite / files scenario-scripted defects.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.role_swe.personas import (
    ROLE_MANAGER,
    ROLE_QA,
    ROLE_REVIEWER,
    PersonaChannel,
    ScenarioScript,
    Stage,
)
from tests.role_swe.workspace import Workspace

# The canaries swe_01 grades against.
_CANARIES = [
    "money_is_decimal",
    "public_functions_have_docstrings",
    "changelog_updated",
    "test_added",
    "no_off_limits_edits",
]


def _ws(tmp_path: Path) -> Workspace:
    return Workspace.create(tmp_path / "wsp")


# Pytest may have been invoked from anywhere; the SUT needs `git` + `pytest` +
# `ruff`/`black` on PATH. Skip cleanly if the environment lacks git.
def _has_git() -> bool:
    import shutil

    return shutil.which("git") is not None


pytestmark = pytest.mark.skipif(not _has_git(), reason="git required for workspace tests")


class TestWorkspace:
    def test_pristine_workspace_is_clean_and_green(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            assert ws.changed_files() == set()
            assert ws.run_tests().ok  # the shipped SUT suite passes

    def test_edit_is_detected_in_changed_files_and_diff(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            target = ws.root / "src/ledgerlite/accounts.py"
            target.write_text(target.read_text() + "\n# a harmless comment\n")
            assert "src/ledgerlite/accounts.py" in ws.changed_files()
            assert "harmless comment" in ws.diff()

    def test_breaking_a_test_makes_suite_red(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            # Break the invariant check so the suite fails.
            errors = ws.root / "src/ledgerlite/errors.py"
            errors.write_text(errors.read_text().replace("class LedgerErr", "class Ledger_BAD"))
            assert not ws.run_tests().ok


class TestPersonasHappyPath:
    def _make_compliant_change(self, ws: Workspace) -> None:
        """Apply a convention-clean change: feature + Decimal + docstring + test +
        CHANGELOG, nothing off-limits."""
        ledger = ws.root / "src/ledgerlite/ledger.py"
        src = ledger.read_text()
        # Add a documented public method using Decimal only.
        method = (
            "\n    def account_count(self) -> int:\n"
            '        """Return the number of accounts in the chart.\n\n'
            "        Returns:\n            The count of registered accounts.\n"
            '        """\n        return len(self._accounts)\n'
        )
        ledger.write_text(src.rstrip() + "\n" + method)
        test = ws.root / "tests/test_account_count.py"
        test.write_text(
            "from ledgerlite import Account, Ledger\n\n\n"
            "def test_account_count_reflects_added_accounts():\n"
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

    def test_reviewer_approves_clean_change_then_qa_passes(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            self._make_compliant_change(ws)
            ch = PersonaChannel(ws, ScenarioScript(qa_files_defect=False), _CANARIES)

            reply = ch.message(ROLE_REVIEWER, "PR ready for review: added account_count()")
            assert "Approved" in reply
            assert ch.stage == Stage.APPROVED

            qa = ch.message(ROLE_QA, "ready for QA")
            assert "passed" in qa.lower()
            assert ch.stage == Stage.DONE
            assert ch.verdicts[-1].approved


class TestPersonasChangeRequests:
    def test_reviewer_requests_changes_on_missing_test_and_changelog(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            # A source change with NO test and NO changelog entry → canaries fail.
            ledger = ws.root / "src/ledgerlite/ledger.py"
            ledger.write_text(
                ledger.read_text().rstrip() + "\n    def _noop(self):\n        return None\n"
            )
            ch = PersonaChannel(ws, ScenarioScript(), _CANARIES)
            reply = ch.message(ROLE_REVIEWER, "ready for review")
            assert reply.startswith("CHANGES_REQUESTED")
            assert "test" in reply.lower()  # test_added canary failed
            assert ch.stage == Stage.CHANGES_REQUESTED
            assert not ch.verdicts[-1].approved
            assert ch.verdicts[-1].points  # enumerated, gradeable

    def test_off_limits_edit_is_a_change_request(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            reporting = ws.root / "src/ledgerlite/reporting/__init__.py"
            reporting.write_text(reporting.read_text() + "\n# sneaky edit\n")
            ch = PersonaChannel(ws, ScenarioScript(), _CANARIES)
            reply = ch.message(ROLE_REVIEWER, "ready for review")
            assert "off-limits" in reply.lower()


class TestPersonasQADefect:
    def test_qa_files_scripted_spurious_defect(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            ch = PersonaChannel(
                ws,
                ScenarioScript(
                    qa_files_defect=True,
                    qa_defect_is_spurious=True,
                    qa_defect_text="balance looks off on my machine",
                ),
                _CANARIES,
            )
            ch.stage = Stage.APPROVED  # reached QA
            reply = ch.message(ROLE_QA, "ready for QA")
            assert reply.startswith("DEFECT")
            assert ch.stage == Stage.QA_FAILED


class TestPersonaChannelGuards:
    def test_unknown_role_raises(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            ch = PersonaChannel(ws, ScenarioScript(), _CANARIES)
            with pytest.raises(ValueError):
                ch.message("ceo", "hi")

    def test_manager_answers_scope_question_by_keyword(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            ch = PersonaChannel(
                ws,
                ScenarioScript(scope_answers={"pending": "Settled entries only."}),
                _CANARIES,
            )
            reply = ch.message(ROLE_MANAGER, "Should it include pending entries?")
            assert "Settled entries only" in reply

    def test_exchanges_recorded_for_scorecard(self, tmp_path: Path) -> None:
        with _ws(tmp_path) as ws:
            ch = PersonaChannel(ws, ScenarioScript(), _CANARIES)
            ch.message(ROLE_MANAGER, "question?")
            assert len(ch.exchanges) == 1
            assert ch.exchanges[0].role == ROLE_MANAGER
