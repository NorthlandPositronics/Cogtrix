"""Regression test for the recovery-cascade budget reset (#2055 / #2014).

The #1960 kill switch (`_MAX_RECOVERY_FIRINGS_PER_TURN`) resets its per-turn
counter whenever the most-recent-HumanMessage index advances. Recovery nodes
inject their nudges AS HumanMessages (tagged `_RECOVERY_INJECTED_MARKER`), so
counting them made the turn marker advance every recovery round → the budget
reset → the kill switch never reached its cap → a no-progress loop ran to the
LangGraph recursion limit (observed in Gate-2: empty response, tools=0,
"Recursion limit of 60 reached").

`_last_human_msg_idx` must therefore ignore recovery-injected nudges and only
report the index of a *genuine user* message, so the turn marker is stable
across a recovery cascade and the budget accumulates to the cap.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from cogtrix_core.orchestration.graph import _last_human_msg_idx
from cogtrix_core.orchestration.nodes.recovery import _RECOVERY_INJECTED_MARKER


def _nudge(text: str = "[recovery nudge]") -> HumanMessage:
    return HumanMessage(content=text, additional_kwargs={_RECOVERY_INJECTED_MARKER: True})


def test_recovery_injected_nudges_are_skipped() -> None:
    """The turn marker must point at the genuine user message, NOT a later
    recovery nudge — otherwise the cascade budget resets every round."""
    msgs = [
        HumanMessage(content="real user question"),  # idx 0 — the genuine turn
        AIMessage(content=""),  # no-progress round
        _nudge(),  # idx 2 — recovery nudge (must be skipped)
        AIMessage(content=""),
        _nudge(),  # idx 4 — another recovery nudge (must be skipped)
    ]
    assert _last_human_msg_idx(msgs) == 0


def test_stable_marker_across_growing_cascade() -> None:
    """As more recovery nudges accumulate, the reported index stays put (so
    route_after_model sees the same turn and the budget is NOT reset)."""
    base = [HumanMessage(content="q"), AIMessage(content="")]
    assert _last_human_msg_idx(base) == 0
    base += [_nudge(), AIMessage(content="")]
    assert _last_human_msg_idx(base) == 0
    base += [_nudge(), AIMessage(content="")]
    assert _last_human_msg_idx(base) == 0


def test_genuine_followup_user_message_advances_marker() -> None:
    """A real new user message (no recovery marker) DOES advance the marker."""
    msgs = [
        HumanMessage(content="first"),
        AIMessage(content="answer"),
        HumanMessage(content="second real turn"),  # idx 2 — genuine
    ]
    assert _last_human_msg_idx(msgs) == 2


def test_no_human_returns_minus_one() -> None:
    assert _last_human_msg_idx([AIMessage(content="x")]) == -1


def test_only_recovery_nudges_returns_minus_one() -> None:
    assert _last_human_msg_idx([_nudge(), AIMessage(content=""), _nudge()]) == -1
