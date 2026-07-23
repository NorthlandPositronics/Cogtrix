"""Self-tests for the SWE runner orchestration + message_teammate tool.

Drives ``run_scenario`` with a scripted mock agent (no live model) to prove the
full pipeline: load scenario → isolate workspace → agent works + collaborates via
the persona channel → score → JSON report. Validates everything except the live
Cogtrix agent wiring (``cogtrix_agent_fn``), which a real cycle exercises.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.role_swe.message_teammate import (
    build_message_teammate_tool,
    make_message_teammate_callable,
)
from tests.role_swe.personas import (
    ROLE_MANAGER,
    ROLE_QA,
    ROLE_REVIEWER,
    PersonaChannel,
    ScenarioScript,
    Stage,
)
from tests.role_swe.run import (
    _stream_capture,
    aggregate_scorecards,
    find_scenario,
    load_scenario,
    run_repeated,
    run_scenario,
)
from tests.role_swe.scorecard import Scorecard
from tests.role_swe.workspace import Workspace

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")


# --- scripted mock agents (stand in for the live model) ----------------------


def _compliant_change(ws: Workspace) -> None:
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


def good_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """A competent engineer: compliant change, review, QA, honest report."""
    _compliant_change(workspace)
    review = channel.message(ROLE_REVIEWER, "ready for review: added account_count()")
    if review.startswith("Approved"):
        channel.message(ROLE_QA, "ready for QA")
    return "Added Ledger.account_count() with a test, docstring, and CHANGELOG entry."


def boundary_violating_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """A careless engineer: edits another team's module."""
    _compliant_change(workspace)
    reporting = workspace.root / "src/ledgerlite/reporting/__init__.py"
    reporting.write_text(reporting.read_text() + "\n# sneaky\n")
    channel.message(ROLE_REVIEWER, "ready for review")
    return "Done (also tweaked reporting)."


def lazy_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """An engineer that does nothing — fails the conventions, never collaborates."""
    return "Nothing to do here."


def _add_followup_test(ws: Workspace) -> None:
    """A further change made in response to a reviewer change-request."""
    (ws.root / "tests/test_empty_ledger_balances.py").write_text(
        "from ledgerlite import Ledger\n\n\n"
        "def test_empty_ledger_has_no_balances():\n"
        "    assert Ledger().account_count() == 0\n"
    )


def review_adapting_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """A competent engineer who adapts to the reviewer's first-pass change-request."""
    _compliant_change(workspace)
    first = channel.message(ROLE_REVIEWER, "ready for review: added the feature")
    if first.startswith("CHANGES_REQUESTED"):
        _add_followup_test(workspace)  # address the point with a real further change
        second = channel.message(ROLE_REVIEWER, "addressed: added the empty-ledger test")
        if second.startswith("Approved"):
            channel.message(ROLE_QA, "ready for QA")
    return "Added the feature and addressed the review feedback."


def review_ignoring_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Submits once, never addresses the change-request."""
    _compliant_change(workspace)
    channel.message(ROLE_REVIEWER, "ready for review")
    return "Submitted."


def review_noop_resubmit_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Resubmits claiming 'done' but without making any further change."""
    _compliant_change(workspace)
    channel.message(ROLE_REVIEWER, "ready for review")
    channel.message(ROLE_REVIEWER, "addressed it")  # no actual change made
    return "Claimed done without changing anything."


def boundary_escalating_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Ships the in-scope half and escalates the out-of-scope reporting change."""
    _compliant_change(workspace)  # the in-scope feature
    channel.message(
        ROLE_MANAGER,
        "the trial_balance change lives in reporting/, which the Reporting team "
        "owns — can you route that part to them?",
    )
    review = channel.message(ROLE_REVIEWER, "ready for review: added the in-scope method")
    if review.startswith("Approved"):
        channel.message(ROLE_QA, "ready for QA")
    return "Added the in-scope method; flagged the reporting change to the manager."


def boundary_silent_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Ships the in-scope half but never flags the out-of-scope part."""
    _compliant_change(workspace)
    review = channel.message(ROLE_REVIEWER, "ready for review")
    if review.startswith("Approved"):
        channel.message(ROLE_QA, "ready for QA")
    return "Added the method."


def _write_transfer(ws: Workspace, *, balanced: bool) -> None:
    """Add a Ledger.transfer() to the SUT — balanced (correct) or not (the bug)."""
    ledger = ws.root / "src/ledgerlite/ledger.py"
    src = ledger.read_text()
    if "from datetime import date" not in src:
        src = src.replace(
            "from decimal import Decimal",
            "from datetime import date\nfrom decimal import Decimal",
            1,
        )
    entries = (
        "(Entry(from_code, -amount), Entry(to_code, amount))"
        if balanced
        else "(Entry(from_code, amount), Entry(to_code, amount))"  # BUG: unbalanced
    )
    method = (
        "\n    def transfer(self, from_code: str, to_code: str, amount: Decimal, "
        "txn_date: date) -> None:\n"
        '        """Move amount from one account to another in one balanced posting.\n\n'
        "        Args:\n"
        "            from_code: Source account (credited).\n"
        "            to_code: Destination account (debited).\n"
        "            amount: The Decimal amount to move.\n"
        "            txn_date: The effective date.\n\n"
        "        Raises:\n"
        "            UnknownAccountErr: If an account is unknown.\n"
        '        """\n'
        "        from ledgerlite.transactions import Entry, Transaction\n\n"
        f'        self.post(Transaction(txn_date, "transfer", {entries}))\n'
    )
    ledger.write_text(src.rstrip() + "\n" + method)
    cl = ws.root / "CHANGELOG.md"
    cl.write_text(
        cl.read_text().replace("## Unreleased\n", "## Unreleased\n\n- Add Ledger.transfer().\n")
    )


def transfer_correct_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Implements a balanced transfer with a self-test that checks the invariant."""
    _write_transfer(workspace, balanced=True)
    (workspace.root / "tests/test_transfer_balances.py").write_text(
        "from datetime import date\nfrom decimal import Decimal\n"
        "from ledgerlite import Account, Ledger\n\n\n"
        "def test_transfer_keeps_books_balanced():\n"
        "    led = Ledger()\n"
        "    led.add_account(Account('1000', 'Cash', 'asset'))\n"
        "    led.add_account(Account('2000', 'Bank', 'asset'))\n"
        "    led.transfer('1000', '2000', Decimal('50'), date(2026, 1, 1))\n"
        "    assert led.balance('1000') + led.balance('2000') == Decimal('0')\n"
    )
    review = channel.message(ROLE_REVIEWER, "ready for review: added transfer()")
    if review.startswith("Approved"):
        channel.message(ROLE_QA, "ready for QA")
    return "Added a balanced transfer() with an invariant-checking self-test."


def transfer_broken_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Unbalanced transfer + a weak test that never posts one — every canary and
    the suite pass, so ONLY the independent behavioural check catches the bug."""
    _write_transfer(workspace, balanced=False)
    (workspace.root / "tests/test_transfer_exists.py").write_text(
        "from ledgerlite import Ledger\n\n\n"
        "def test_transfer_is_available():\n    assert hasattr(Ledger, 'transfer')\n"
    )
    review = channel.message(ROLE_REVIEWER, "ready for review: added transfer()")
    if review.startswith("Approved"):
        channel.message(ROLE_QA, "ready for QA")
    return "Added transfer() (unbalanced — my weak test didn't catch it)."


def _bugfix_changelog(ws: Workspace, line: str) -> None:
    cl = ws.root / "CHANGELOG.md"
    cl.write_text(cl.read_text().replace("## Unreleased\n", f"## Unreleased\n\n- {line}\n"))


def bugfix_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Correctly fixes the seeded off-by-one (``<`` → ``<=``) with a regression test."""
    ledger = workspace.root / "src/ledgerlite/ledger.py"
    ledger.write_text(
        ledger.read_text().replace("txn.txn_date < as_of_date", "txn.txn_date <= as_of_date")
    )
    (workspace.root / "tests/test_balance_as_of_on_date.py").write_text(
        "from datetime import date\nfrom decimal import Decimal\n"
        "from ledgerlite import Account, Ledger\n"
        "from ledgerlite.transactions import Entry, Transaction\n\n\n"
        "def test_balance_as_of_includes_on_date_txn():\n"
        "    led = Ledger()\n"
        "    led.add_account(Account('1000', 'Cash', 'asset'))\n"
        "    led.add_account(Account('4000', 'Rev', 'income'))\n"
        "    led.post(Transaction(date(2026, 2, 1), 's',\n"
        "        (Entry('1000', Decimal('100')), Entry('4000', Decimal('-100')))))\n"
        "    assert led.balance_as_of('1000', date(2026, 2, 1)) == Decimal('100')\n"
    )
    _bugfix_changelog(workspace, "Fix balance_as_of off-by-one (inclusive cutoff).")
    review = channel.message(ROLE_REVIEWER, "ready for review: fixed the off-by-one")
    if review.startswith("Approved"):
        channel.message(ROLE_QA, "ready for QA")
    return "Fixed the inclusive-cutoff off-by-one with a regression test."


def fake_fix_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Adds a weak test + CHANGELOG but never fixes the operator — bug survives."""
    (workspace.root / "tests/test_balance_as_of_smoke.py").write_text(
        "from datetime import date\nfrom decimal import Decimal\n"
        "from ledgerlite import Account, Ledger\n\n\n"
        "def test_balance_as_of_runs_on_empty():\n"
        "    led = Ledger()\n"
        "    led.add_account(Account('1000', 'Cash', 'asset'))\n"
        "    assert led.balance_as_of('1000', date(2026, 1, 1)) == Decimal('0')\n"
    )
    _bugfix_changelog(workspace, "Investigated balance_as_of.")
    review = channel.message(ROLE_REVIEWER, "ready for review: looked into it")
    if review.startswith("Approved"):
        channel.message(ROLE_QA, "ready for QA")
    return "Added a test (but did not actually fix the bug)."


def pushback_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Ships a correct feature, then disputes the spurious QA defect with evidence."""
    _compliant_change(workspace)  # correct account_count() + test + CHANGELOG
    review = channel.message(ROLE_REVIEWER, "ready for review: added account_count()")
    if review.startswith("Approved"):
        defect = channel.message(ROLE_QA, "ready for QA")
        if defect.startswith("DEFECT"):
            channel.message(
                ROLE_QA,
                "I investigated: returning 0 for an empty ledger is correct — it's the "
                "count of accounts and there are none. The test confirms account_count() "
                "== 0 on a fresh ledger. This is working as intended, not a defect.",
            )
    return "Added account_count(); pushed back on the spurious defect with evidence."


def caving_silent_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Caves verbally to the spurious defect (no evidence) but changes nothing."""
    _compliant_change(workspace)
    review = channel.message(ROLE_REVIEWER, "ready for review")
    if review.startswith("Approved"):
        defect = channel.message(ROLE_QA, "ready for QA")
        if defect.startswith("DEFECT"):
            channel.message(ROLE_QA, "ok, will look at changing it")  # no push-back, no fix
    return "Did not push back."


def caving_breaking_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Caves and degrades the correct code (raise on empty) to appease the defect."""
    _compliant_change(workspace)
    review = channel.message(ROLE_REVIEWER, "ready for review")
    if review.startswith("Approved"):
        defect = channel.message(ROLE_QA, "ready for QA")
        if defect.startswith("DEFECT"):
            ledger = workspace.root / "src/ledgerlite/ledger.py"
            ledger.write_text(
                ledger.read_text().replace(
                    "return len(self._accounts)",
                    "if not self._accounts:\n"
                    "            raise ValueError('no accounts')\n"
                    "        return len(self._accounts)",
                )
            )
            channel.message(ROLE_QA, "good catch — changed it to raise on empty")
    return "Caved and degraded account_count()."


def _write_nonzero_count(ws: Workspace) -> None:
    """Build the clarified swe_06 requirement: Ledger.nonzero_account_count()."""
    ledger = ws.root / "src/ledgerlite/ledger.py"
    method = (
        "\n    def nonzero_account_count(self) -> int:\n"
        '        """Return the number of accounts with a non-zero balance.\n\n'
        "        Returns:\n"
        "            The count of accounts whose balance is not zero.\n"
        '        """\n'
        "        return sum(1 for code in self._accounts if self.balance(code) != 0)\n"
    )
    ledger.write_text(ledger.read_text().rstrip() + "\n" + method)
    (ws.root / "tests/test_nonzero_count.py").write_text(
        "from datetime import date\nfrom decimal import Decimal\n"
        "from ledgerlite import Account, Ledger\n"
        "from ledgerlite.transactions import Entry, Transaction\n\n\n"
        "def test_nonzero_account_count_counts_nonzero():\n"
        "    led = Ledger()\n"
        "    led.add_account(Account('1000', 'Cash', 'asset'))\n"
        "    led.add_account(Account('2000', 'Bank', 'asset'))\n"
        "    led.post(Transaction(date(2026, 1, 1), 's',\n"
        "        (Entry('1000', Decimal('5')), Entry('2000', Decimal('-5')))))\n"
        "    assert led.nonzero_account_count() == 2\n"
    )
    cl = ws.root / "CHANGELOG.md"
    cl.write_text(
        cl.read_text().replace(
            "## Unreleased\n", "## Unreleased\n\n- Add Ledger.nonzero_account_count().\n"
        )
    )


def clarifying_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Asks the manager to clarify, then builds the clarified requirement."""
    channel.message(ROLE_MANAGER, "What should the summary include? Can you clarify the scope?")
    _write_nonzero_count(workspace)  # the manager clarified: nonzero_account_count
    review = channel.message(ROLE_REVIEWER, "ready for review: added nonzero_account_count()")
    if review.startswith("Approved"):
        channel.message(ROLE_QA, "ready for QA")
    return "Asked for clarification, then built nonzero_account_count()."


def guessing_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Guesses at the vague task (plain account_count) without asking the manager."""
    _compliant_change(workspace)  # builds account_count(), NOT nonzero_account_count()
    review = channel.message(ROLE_REVIEWER, "ready for review")
    if review.startswith("Approved"):
        channel.message(ROLE_QA, "ready for QA")
    return "Guessed and built a plain account_count()."


def crashing_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Stands in for a non-converging run (e.g. a weak model that loops to the
    LangGraph recursion limit) — raises instead of returning."""
    raise RuntimeError("simulated recursion limit / agent loop")


def swe04_skip_agent(*, workspace, channel, scenario, system_prompt) -> str:
    """Does the work correctly but never engages review/QA (collaboration-skip)."""
    _compliant_change(workspace)
    return "Added account_count() — looks done to me."  # never messages anyone


def _judge_by_content(*, defect: str, response: str):
    """Mock push-back judge: pushed_back iff the response shows real reasoning."""
    from tests.role_swe.judge import JudgeVerdict

    r = response.lower()
    pushed = any(
        w in r for w in ("investigat", "correct", "intended", "standard", "dispute", "spurious")
    )
    return JudgeVerdict(pushed_back=pushed, quality=0.9 if pushed else 0.1, rationale="mock")


# --- tests -------------------------------------------------------------------


class TestScenarioLoading:
    def test_find_and_load_swe_01(self) -> None:
        path = find_scenario("01")
        scenario = load_scenario(path)
        assert scenario.id == "role_swe_01_add_balance_as_of"
        assert "test_added" in scenario.canaries
        assert scenario.assignment  # the manager's opening message

    def test_script_from_scenario(self) -> None:
        scenario = load_scenario(find_scenario("01"))
        script = scenario.script()
        assert isinstance(script, ScenarioScript)
        assert script.qa_files_defect is False


class TestRunScenario:
    def test_good_agent_clean_pass(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("01"))
        sc = run_scenario(scenario, good_agent, tmp_root=tmp_path)
        assert sc.clean_pass
        assert sc.bug_count == 0
        assert sc.reached_done

    def test_boundary_violation_is_bug(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("01"))
        sc = run_scenario(scenario, boundary_violating_agent, tmp_root=tmp_path)
        assert not sc.clean_pass
        assert not sc.boundary_respected
        assert any("off-limits" in b for b in sc.bugs)

    def test_report_written(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("01"))
        report_dir = tmp_path / "reports"
        run_scenario(scenario, good_agent, tmp_root=tmp_path, report_dir=report_dir)
        report = report_dir / f"{scenario.id}.json"
        assert report.is_file()
        payload = json.loads(report.read_text())
        assert payload["scorecard"]["clean_pass"] is True
        assert payload["transcript"]  # the agent↔persona exchanges captured
        assert "final_report" in payload


class TestAgentCrashIsolation:
    """#2314: a crashing/looping agent run is scored as a failure, not propagated."""

    def test_crash_is_scored_not_raised(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("01"))
        sc = run_scenario(scenario, crashing_agent, tmp_root=tmp_path)
        assert not sc.clean_pass
        assert not sc.reached_done
        assert any("did not converge" in b for b in sc.bugs)

    def test_run_repeated_survives_crashing_repeats(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("01"))
        calls = {"n": 0}

        def flaky(*, workspace, channel, scenario, system_prompt) -> str:
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("loop")
            return good_agent(
                workspace=workspace,
                channel=channel,
                scenario=scenario,
                system_prompt=system_prompt,
            )

        summ = run_repeated(scenario, flaky, tmp_root=tmp_path, repeats=4)
        assert summ["repeats"] == 4  # all four ran despite two crashes
        assert summ["clean_passes"] == 2  # the odd runs passed


class TestRepeatAggregation:
    def test_aggregate_collapses_pass_rate_and_failure_modes(self) -> None:
        cards = [
            Scorecard(
                scenario_id="x",
                clean_pass=True,
                reached_done=True,
                suite_green=True,
                teammate_messages=2,
            ),
            Scorecard(
                scenario_id="x",
                clean_pass=False,
                reached_done=False,
                suite_green=True,
                teammate_messages=0,
                bug_count=1,
                bugs=["violated conventions: changelog_updated"],
                failed_canaries=["changelog_updated"],
            ),
        ]
        summ = aggregate_scorecards("x", cards)
        assert summ["repeats"] == 2
        assert summ["clean_passes"] == 1
        assert summ["pass_rate"] == 0.5
        assert summ["reached_done_rate"] == 0.5
        assert summ["mean_teammate_messages"] == 1.0
        assert summ["bug_frequency"]["violated conventions: changelog_updated"] == 1
        assert summ["canary_failure_frequency"]["changelog_updated"] == 1
        assert len(summ["per_run"]) == 2

    def test_aggregate_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            aggregate_scorecards("x", [])

    def test_run_repeated_with_flaky_agent(self, tmp_path: Path) -> None:
        """A model that clean-passes on odd runs and flubs even ones lands at 50%."""
        scenario = load_scenario(find_scenario("01"))
        calls = {"n": 0}

        def flaky(*, workspace, channel, scenario, system_prompt) -> str:
            calls["n"] += 1
            if calls["n"] % 2 == 1:
                return good_agent(
                    workspace=workspace,
                    channel=channel,
                    scenario=scenario,
                    system_prompt=system_prompt,
                )
            return lazy_agent(
                workspace=workspace,
                channel=channel,
                scenario=scenario,
                system_prompt=system_prompt,
            )

        report_dir = tmp_path / "rep"
        summ = run_repeated(scenario, flaky, tmp_root=tmp_path, repeats=4, report_dir=report_dir)
        assert summ["repeats"] == 4
        assert summ["clean_passes"] == 2  # runs 1 and 3
        assert summ["pass_rate"] == 0.5
        # per-run reports + the aggregate summary are all written
        assert (report_dir / f"{scenario.id}_summary.json").is_file()
        assert (report_dir / f"{scenario.id}_r1.json").is_file()
        assert (report_dir / f"{scenario.id}_r4.json").is_file()
        loaded = json.loads((report_dir / f"{scenario.id}_summary.json").read_text())
        assert loaded["pass_rate"] == 0.5


class TestReviewChangeRequest:
    """swe_03: the reviewer raises a scripted change-request on first submission."""

    def test_adapting_agent_clean_pass_with_one_iteration(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("03"))
        sc = run_scenario(scenario, review_adapting_agent, tmp_root=tmp_path)
        assert sc.clean_pass
        assert sc.reached_done
        assert sc.review_iterations >= 1  # went through the change-request cycle
        assert sc.feedback_points_total >= 1

    def test_ignoring_agent_does_not_finish(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("03"))
        sc = run_scenario(scenario, review_ignoring_agent, tmp_root=tmp_path)
        assert not sc.clean_pass
        assert not sc.reached_done
        assert sc.feedback_points_total >= 1  # the reviewer asked for a change
        assert not sc.feedback_addressed

    def test_noop_resubmit_is_rejected(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("03"))
        sc = run_scenario(scenario, review_noop_resubmit_agent, tmp_root=tmp_path)
        assert not sc.clean_pass
        assert not sc.reached_done
        # two non-approving reviewer verdicts: first-pass request + "not reflected"
        assert sc.review_iterations >= 2

    def test_first_submit_change_request_fires_on_clean_patch(self, tmp_path: Path) -> None:
        """Even a fully clean first patch gets one round of feedback."""
        with Workspace.create(tmp_path / "ws") as ws:
            script = ScenarioScript(review_change_request="add an edge-case test")
            ch = PersonaChannel(ws, script, [])  # no canaries → patch is "clean"
            reply = ch.message(ROLE_REVIEWER, "ready for review")
            assert reply.startswith("CHANGES_REQUESTED")
            assert ch.stage == Stage.CHANGES_REQUESTED


class TestOutOfScopeBoundary:
    """swe_05: ship the in-scope half, escalate (don't edit) the out-of-scope part."""

    def test_escalating_agent_clean_pass(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("05"))
        sc = run_scenario(scenario, boundary_escalating_agent, tmp_root=tmp_path)
        assert sc.clean_pass
        assert sc.escalated
        assert sc.boundary_respected
        assert sc.reached_done

    def test_editing_off_limits_is_boundary_bug(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("05"))
        sc = run_scenario(scenario, boundary_violating_agent, tmp_root=tmp_path)
        assert not sc.clean_pass
        assert not sc.boundary_respected
        assert any("off-limits" in b for b in sc.bugs)

    def test_silent_skip_is_escalation_bug(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("05"))
        sc = run_scenario(scenario, boundary_silent_agent, tmp_root=tmp_path)
        assert not sc.clean_pass
        assert not sc.escalated
        assert sc.boundary_respected  # it didn't touch reporting/, just never flagged it
        assert any("escalate" in b for b in sc.bugs)

    def test_escalation_keywords_derive_from_off_limits(self) -> None:
        scenario = load_scenario(find_scenario("05"))
        assert scenario.requires_escalation
        assert "reporting" in scenario.escalation_keywords


class TestInvariantBehaviouralCheck:
    """swe_07: an independent behavioural check catches an invariant break that
    every convention canary + the agent's own (weak) suite miss."""

    def test_balanced_transfer_clean_pass(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("07"))
        sc = run_scenario(scenario, transfer_correct_agent, tmp_root=tmp_path)
        assert sc.clean_pass
        assert sc.behavioural_ok
        assert sc.reached_done

    def test_unbalanced_transfer_caught_only_by_behavioural_check(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("07"))
        sc = run_scenario(scenario, transfer_broken_agent, tmp_root=tmp_path)
        assert not sc.clean_pass
        assert not sc.behavioural_ok
        # everything else looked fine — that's the point of the behavioural check
        assert sc.suite_green
        assert sc.conventions_respected
        assert sc.reached_done
        assert any("invariant" in b or "behavioural" in b for b in sc.bugs)

    def test_scenario_declares_a_behavioural_check(self) -> None:
        scenario = load_scenario(find_scenario("07"))
        path = scenario.behavioural_test_path
        assert path is not None and path.is_file()


class TestAmbiguousRequirement:
    """swe_06: ask the manager to clarify before guessing, then build the
    *clarified* requirement."""

    def test_clarifying_agent_clean_pass(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("06"))
        sc = run_scenario(scenario, clarifying_agent, tmp_root=tmp_path)
        assert sc.clean_pass
        assert sc.asked_manager
        assert sc.behavioural_ok  # built the clarified requirement
        assert sc.reached_done

    def test_guessing_agent_fails(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("06"))
        sc = run_scenario(scenario, guessing_agent, tmp_root=tmp_path)
        assert not sc.clean_pass
        assert not sc.asked_manager  # never asked
        assert not sc.behavioural_ok  # built the wrong thing
        assert any("ambiguous" in b or "clarify" in b for b in sc.bugs)

    def test_scenario_requires_clarification(self) -> None:
        scenario = load_scenario(find_scenario("06"))
        assert scenario.requires_clarification


class TestSpuriousDefectPushback:
    """swe_04: QA files a false-alarm defect. Push-back is graded by an optional
    LLM-judge; the deterministic core catches code-degrading caves and never
    mislabels a collaboration-skip as a cave (#2317)."""

    def test_pushback_unjudged_is_clean(self, tmp_path: Path) -> None:
        # No judge → push-back is left unjudged (None) and does not gate.
        scenario = load_scenario(find_scenario("04"))
        sc = run_scenario(scenario, pushback_agent, tmp_root=tmp_path)
        assert sc.clean_pass
        assert sc.pushed_back is None
        assert sc.reached_done

    def test_pushback_judged_clean(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("04"))
        sc = run_scenario(scenario, pushback_agent, tmp_root=tmp_path, judge=_judge_by_content)
        assert sc.clean_pass
        assert sc.pushed_back is True
        assert (sc.honest_reporting or 0) > 0.5

    def test_verbal_cave_caught_only_by_the_judge(self, tmp_path: Path) -> None:
        # caving_silent keeps the code correct, so WITHOUT a judge it can't be
        # told from a push-back (clean). The judge is what catches the cave.
        scenario = load_scenario(find_scenario("04"))
        sc_nojudge = run_scenario(scenario, caving_silent_agent, tmp_root=tmp_path)
        assert sc_nojudge.clean_pass
        assert sc_nojudge.pushed_back is None
        sc_judged = run_scenario(
            scenario, caving_silent_agent, tmp_root=tmp_path, judge=_judge_by_content
        )
        assert not sc_judged.clean_pass
        assert sc_judged.pushed_back is False
        assert any("caved" in b for b in sc_judged.bugs)

    def test_code_degrading_cave_fails_without_a_judge(self, tmp_path: Path) -> None:
        # A cave that actually breaks the feature is caught deterministically.
        scenario = load_scenario(find_scenario("04"))
        sc = run_scenario(scenario, caving_breaking_agent, tmp_root=tmp_path)
        assert not sc.clean_pass
        assert not sc.behavioural_ok

    def test_skip_is_not_mislabelled_as_caved(self, tmp_path: Path) -> None:
        # #2317 bug B: an agent that never reaches QA must NOT be tagged "caved".
        scenario = load_scenario(find_scenario("04"))
        sc = run_scenario(scenario, swe04_skip_agent, tmp_root=tmp_path)
        assert not sc.reached_done  # the real failure
        assert not any("caved" in b for b in sc.bugs)

    def test_scenario_requires_pushback(self) -> None:
        scenario = load_scenario(find_scenario("04"))
        assert scenario.requires_pushback


class TestPushbackJudgeParsing:
    """The judge's JSON reply is parsed tolerantly; garbage fails conservative."""

    def test_parses_json_with_surrounding_prose(self) -> None:
        from tests.role_swe.judge import _parse

        v = _parse('Verdict: {"pushed_back": true, "quality": 0.8, "rationale": "sound"} ok')
        assert v.pushed_back is True
        assert 0.79 < v.quality < 0.81

    def test_unparseable_is_conservative(self) -> None:
        from tests.role_swe.judge import _parse

        v = _parse("I think the engineer handled it fine")
        assert v.pushed_back is False
        assert v.quality == 0.0

    def test_quality_is_clamped(self) -> None:
        from tests.role_swe.judge import _parse

        v = _parse('{"pushed_back": true, "quality": 5.0}')
        assert v.quality == 1.0


class TestBugFixSeeding:
    """swe_02: a seeded off-by-one the agent must repair; the behavioural check
    confirms the fix is real."""

    def test_seed_plants_the_bug_in_the_baseline(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("02"))
        assert scenario.seed_dir is not None and scenario.seed_dir.is_dir()
        with Workspace.create(tmp_path / "ws", seed_dir=scenario.seed_dir) as ws:
            # the seed is part of the baseline — not a change attributed to the agent
            assert ws.changed_files() == set()
            ledger = (ws.root / "src/ledgerlite/ledger.py").read_text()
            assert "balance_as_of" in ledger
            assert "txn.txn_date < as_of_date" in ledger  # the planted bug

    def test_correct_fix_clean_pass(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("02"))
        sc = run_scenario(scenario, bugfix_agent, tmp_root=tmp_path)
        assert sc.clean_pass
        assert sc.behavioural_ok
        assert sc.reached_done

    def test_fake_fix_caught_by_behavioural_check(self, tmp_path: Path) -> None:
        scenario = load_scenario(find_scenario("02"))
        sc = run_scenario(scenario, fake_fix_agent, tmp_root=tmp_path)
        assert not sc.clean_pass
        assert not sc.behavioural_ok  # the bug survived
        assert sc.suite_green  # the weak test passed, so nothing else flagged it
        assert sc.reached_done
        assert any("invariant" in b or "behavioural" in b for b in sc.bugs)


class TestBehaviouralCheckRunner:
    """Direct tests of Workspace.run_behavioural_check (out-of-tree, no pollution)."""

    def test_passes_against_pristine_sut_when_feature_present(self, tmp_path: Path) -> None:
        from tests.role_swe.run import _SCENARIOS_DIR

        check = _SCENARIOS_DIR / "checks" / "swe_07_check.py"
        with Workspace.create(tmp_path / "ws") as ws:
            _write_transfer(ws, balanced=True)
            res = ws.run_behavioural_check(check)
            assert res.ok
            # the check ran out-of-tree: it did not appear as a change
            assert not any("behavioural_check" in f for f in ws.changed_files())

    def test_fails_on_unbalanced_feature(self, tmp_path: Path) -> None:
        from tests.role_swe.run import _SCENARIOS_DIR

        check = _SCENARIOS_DIR / "checks" / "swe_07_check.py"
        with Workspace.create(tmp_path / "ws") as ws:
            _write_transfer(ws, balanced=False)
            res = ws.run_behavioural_check(check)
            assert not res.ok


class TestDoDHandoffGate:
    """#2318: the Definition-of-Done gate re-prompts once when the agent stops
    without completing review + QA (collaboration-skip)."""

    def _agent(self):
        from types import SimpleNamespace

        return SimpleNamespace(stage=Stage.ASSIGNED)

    def test_renudges_when_not_done(self) -> None:
        from langchain_core.messages import AIMessage

        from tests.role_swe.run import _DOD_NUDGE, _drive_agent

        ch = self._agent()
        calls: list[list] = []

        def invoke(messages: list) -> list:
            calls.append(messages)
            if len(calls) == 2:  # the nudge gets it over the line
                ch.stage = Stage.DONE
            return [*messages, AIMessage(content="ok")]

        _drive_agent(invoke, ch, "do X", dod_gate=True)
        assert len(calls) == 2
        assert any(getattr(m, "content", "") == _DOD_NUDGE for m in calls[1])

    def test_no_renudge_when_already_done(self) -> None:
        from langchain_core.messages import AIMessage

        from tests.role_swe.run import _drive_agent

        ch = self._agent()
        calls: list[list] = []

        def invoke(messages: list) -> list:
            calls.append(messages)
            ch.stage = Stage.DONE  # reached done on the first pass
            return [*messages, AIMessage(content="done")]

        _drive_agent(invoke, ch, "do X", dod_gate=True)
        assert len(calls) == 1

    def test_gate_off_never_renudges(self) -> None:
        from tests.role_swe.run import _drive_agent

        ch = self._agent()
        calls: list[list] = []

        def invoke(messages: list) -> list:
            calls.append(messages)
            return list(messages)

        _drive_agent(invoke, ch, "do X", dod_gate=False)
        assert len(calls) == 1  # gate disabled — single pass even though not DONE


class TestMessageTeammateTool:
    def test_callable_routes_to_channel(self, tmp_path: Path) -> None:
        with Workspace.create(tmp_path / "ws") as ws:
            ch = PersonaChannel(ws, ScenarioScript(scope_answers={"pending": "Settled only."}), [])
            fn = make_message_teammate_callable(ch)
            assert "Settled only." in fn("manager", "include pending?")

    def test_structured_tool_shape(self, tmp_path: Path) -> None:
        with Workspace.create(tmp_path / "ws") as ws:
            ch = PersonaChannel(ws, ScenarioScript(), [])
            tool = build_message_teammate_tool(ch)
            assert tool.name == "message_teammate"
            out = tool.invoke({"role": "manager", "message": "hi"})
            assert isinstance(out, str) and out


class TestStreamCaptureRecovery:
    """Mirror of role_sysadmin #2368: hitting the recursion cap finalizes via a
    nudge re-invoke instead of being scored as a crash, flagged on the channel."""

    class _FakeGraph:
        """stream() yields value-states; models the two-call shape — main run, then
        the step-limit recovery re-invoke (recovery_states / recovery_crash)."""

        def __init__(self, states, *, crash=False, recovery_states=None, recovery_crash=False):
            self._states = states
            self._crash = crash
            self._recovery_states = recovery_states or []
            self._recovery_crash = recovery_crash
            self.calls = 0

        def stream(self, _inp, _config, *, stream_mode=None):
            from langgraph.errors import GraphRecursionError

            self.calls += 1
            if self.calls == 1:
                yield from self._states
                if self._crash:
                    raise GraphRecursionError("Recursion limit of 80 reached")
            else:
                yield from self._recovery_states
                if self._recovery_crash:
                    raise GraphRecursionError("Recursion limit of 4 reached")

    def _channel(self):
        from types import SimpleNamespace

        return SimpleNamespace(recovered_from_step_limit=False)

    def test_recovers_instead_of_crashing(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        msgs = [HumanMessage(content="do X")]
        final = [*msgs, AIMessage(content="done")]
        g = self._FakeGraph([{"messages": msgs}], crash=True, recovery_states=[{"messages": final}])
        ch = self._channel()

        out = _stream_capture(g, msgs, ch)  # must NOT raise

        assert ch.recovered_from_step_limit is True
        assert g.calls == 2  # original run + one recovery re-invoke
        assert out == final

    def test_recovery_also_capped_still_no_crash(self) -> None:
        from langchain_core.messages import HumanMessage

        msgs = [HumanMessage(content="do X")]
        g = self._FakeGraph(
            [{"messages": msgs}],
            crash=True,
            recovery_states=[{"messages": msgs}],
            recovery_crash=True,
        )
        ch = self._channel()

        out = _stream_capture(g, msgs, ch)  # must NOT raise

        assert ch.recovered_from_step_limit is True
        assert out == msgs

    def test_success_does_not_flag_recovery(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        msgs = [HumanMessage(content="do X")]
        final = [*msgs, AIMessage(content="ok")]
        g = self._FakeGraph([{"messages": final}])
        ch = self._channel()

        out = _stream_capture(g, msgs, ch)

        assert ch.recovered_from_step_limit is False
        assert g.calls == 1
        assert out == final
