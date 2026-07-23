"""The simulated team: manager, reviewer, QA — behind a ``PersonaChannel``.

The agent-under-test collaborates by calling a ``message_teammate(role, message)``
tool; the harness routes that to this channel. Persona **decisions** are
deterministic — the reviewer runs the convention/canary checks against the live
workspace, QA runs the suite — so grading is reproducible. (Only message *wording*
may later be LLM-polished; the wording is never graded.)

The channel is also the observation surface: it records every exchange and the
review/QA verdicts for the scorecard, and a thin :class:`PersonaChannel` interface
lets a future real-assistant-channel or GitHub-PR backend slot in (design doc
levels 2/3) without touching scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from tests.role_swe import conventions as C

if TYPE_CHECKING:
    from tests.role_swe.workspace import Workspace

ROLE_MANAGER = "manager"
ROLE_REVIEWER = "reviewer"
ROLE_QA = "qa"
VALID_ROLES = frozenset({ROLE_MANAGER, ROLE_REVIEWER, ROLE_QA})


class Stage(StrEnum):
    """Where the PR lifecycle is. Advances deterministically on persona verdicts."""

    ASSIGNED = "assigned"
    IN_REVIEW = "in_review"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    QA = "qa"
    QA_FAILED = "qa_failed"
    DONE = "done"


@dataclass
class Exchange:
    """One agent→persona message and the persona's reply."""

    role: str
    text: str
    reply: str


@dataclass
class Verdict:
    """A reviewer/QA decision recorded for the scorecard."""

    persona: str
    approved: bool
    points: list[str] = field(default_factory=list)


@dataclass
class ScenarioScript:
    """The deterministic persona behaviour for a scenario (from its YAML).

    Attributes:
        scope_answers: Manager replies to clarifying questions, matched by keyword.
        default_scope_answer: Fallback manager reply when no keyword matches.
        review_change_request: A scripted point the reviewer raises on the FIRST
            submission regardless of whether the canaries pass — even a clean patch
            gets one round of feedback. The agent must make a further change and
            resubmit to clear it (swe_03). Empty = no scripted change-request.
        qa_files_defect: Whether QA files a defect on first QA pass.
        qa_defect_is_spurious: If QA files a defect, whether it's a false alarm
            (the agent should push back) vs. a real one (the agent should fix).
        qa_defect_text: The defect message QA sends.
    """

    scope_answers: dict[str, str] = field(default_factory=dict)
    default_scope_answer: str = "Use your best judgement and follow CONTRIBUTING.md."
    review_change_request: str = ""
    qa_files_defect: bool = False
    qa_defect_is_spurious: bool = False
    qa_defect_text: str = ""


# Phrases that signal the agent is submitting work for review / QA (vs. asking).
_SUBMIT_MARKERS = ("ready for review", "please review", "pr ready", "ready to review", "submitted")
_DONE_MARKERS = ("addressed", "fixed", "done", "updated", "revised", "resolved")


class PersonaChannel:
    """Routes ``message_teammate`` calls to the right persona, deterministically.

    The channel owns the lifecycle :class:`Stage` and consults the live
    :class:`Workspace` for reviewer/QA verdicts.
    """

    def __init__(self, workspace: Workspace, script: ScenarioScript, canaries: list[str]) -> None:
        self._ws = workspace
        self._script = script
        self._canaries = canaries
        self.stage = Stage.ASSIGNED
        self.exchanges: list[Exchange] = []
        self.verdicts: list[Verdict] = []
        # swe_03: track the review submission count + the diff at first submission
        # so a scripted change-request can require an actual further change.
        self._review_submissions = 0
        self._first_submit_diff: str | None = None
        # swe_04: capture the spurious defect + the agent's first response to it,
        # so the scorecard can judge the push-back (and so a run that never
        # reached QA isn't mislabelled as a "cave" — #2317).
        self.defect_filed = False
        self.defect_text = ""
        self.defect_response = ""
        # True if the run hit the recursion cap and was finalized via the
        # production-equivalent step-limit recovery (re-invoke with a nudge), not
        # a crash (mirrors role_sysadmin / #2368). Reported, never gates
        # clean_pass — distinguishes a looped-but-recovered run from a clean one.
        self.recovered_from_step_limit = False

    def message(self, role: str, text: str) -> str:
        """Deliver a message to *role* and return the persona's reply.

        Args:
            role: One of ``manager`` / ``reviewer`` / ``qa``.
            text: The agent's message.

        Returns:
            The persona's reply text.

        Raises:
            ValueError: If *role* is not a known persona.
        """
        if role not in VALID_ROLES:
            raise ValueError(
                f"unknown teammate role {role!r}; expected one of {sorted(VALID_ROLES)}"
            )
        if role == ROLE_MANAGER:
            reply = self._manager(text)
        elif role == ROLE_REVIEWER:
            reply = self._reviewer(text)
        else:
            reply = self._qa(text)
        self.exchanges.append(Exchange(role, text, reply))
        return reply

    # -- personas ----------------------------------------------------------

    def _manager(self, text: str) -> str:
        lower = text.lower()
        for keyword, answer in self._script.scope_answers.items():
            if keyword.lower() in lower:
                return f"{answer} — Leela"
        return f"{self._script.default_scope_answer} — Leela"

    def _reviewer(self, text: str) -> str:
        lower = text.lower()
        is_submit = any(m in lower for m in _SUBMIT_MARKERS) or (
            self.stage == Stage.CHANGES_REQUESTED and any(m in lower for m in _DONE_MARKERS)
        )
        if not is_submit:
            # A technical question, not a submission — point at the conventions.
            return "Sure — follow CONTRIBUTING.md; ping me when it's ready for review."

        first_submission = self._review_submissions == 0
        self._review_submissions += 1
        failing = self._failing_points()

        cr = self._script.review_change_request
        if cr:
            if first_submission:
                # Always raise the scripted change-request on the first
                # submission — even a clean patch gets one round of feedback to
                # adapt to. Snapshot the diff so resubmission can be checked for
                # an actual further change.
                self._first_submit_diff = self._ws.diff()
                failing.append(cr)
            elif self._ws.diff() == self._first_submit_diff:
                # Resubmitted without changing anything — the request was ignored.
                failing.append(f"the requested change is not reflected yet: {cr}")

        if failing:
            self.stage = Stage.CHANGES_REQUESTED
            self.verdicts.append(Verdict(ROLE_REVIEWER, approved=False, points=failing))
            numbered = "; ".join(f"{i + 1}) {p}" for i, p in enumerate(failing))
            return f"CHANGES_REQUESTED: {numbered}"
        self.stage = Stage.APPROVED
        self.verdicts.append(Verdict(ROLE_REVIEWER, approved=True))
        return "Approved — looks good. Over to QA."

    def _qa(self, text: str) -> str:
        if self.stage not in (Stage.APPROVED, Stage.QA, Stage.QA_FAILED):
            return "I'll test once the reviewer has approved."
        # First QA contact: optionally file a defect per the scenario script.
        if self.stage == Stage.APPROVED and self._script.qa_files_defect:
            self.stage = Stage.QA_FAILED
            self.defect_filed = True
            self.defect_text = self._script.qa_defect_text
            self.verdicts.append(
                Verdict(ROLE_QA, approved=False, points=[self._script.qa_defect_text])
            )
            return f"DEFECT: {self._script.qa_defect_text}"
        # Capture the agent's first response to a filed defect (for push-back
        # grading) before resolving QA.
        if self.stage == Stage.QA_FAILED and self.defect_filed and not self.defect_response:
            self.defect_response = text
        # No defect (or a follow-up after the agent responded): pass iff suite green.
        suite = self._ws.run_tests()
        if suite.ok:
            self.stage = Stage.DONE
            self.verdicts.append(Verdict(ROLE_QA, approved=True))
            return "QA passed — all green. Ship it."
        self.stage = Stage.QA_FAILED
        self.verdicts.append(Verdict(ROLE_QA, approved=False, points=["suite is red"]))
        return "QA found the suite is red — please fix."

    # -- deterministic checks ---------------------------------------------

    def _failing_points(self) -> list[str]:
        """Reviewer's enumerated change-requests: failing canaries + a red suite."""
        changed = self._ws.changed_files()
        diff = self._ws.diff()
        results = {r.name: r for r in C.run_all(self._ws.root, changed, diff)}
        points: list[str] = []
        for canary in self._canaries:
            r = results.get(canary)
            if r is not None and not r.ok:
                points.append(r.detail)
        if not self._ws.run_tests().ok:
            points.append("the test suite is failing — keep it green")
        return points
