"""Regression tests for the ``_repair_tool_message_pairs`` identity-subset
invariant (#2276 / #2444).

``call_model`` writes repaired state back to LangGraph via ``RemoveMessage``
only (see ``cogtrix_core/orchestration/nodes/call_model.py``'s
``repaired_state_ids`` / ``repair_removals`` handling) — it never re-adds a
*modified* message. That means every one of ``_repair_tool_message_pairs``'s
three passes (displaced-contiguity, duplicate-answer, redundant-declaration)
MUST either:

  1. pass an input message through UNCHANGED (same object, ``is`` identity), or
  2. emit a brand-new synthetic ``ToolMessage`` explicitly tagged with
     ``additional_kwargs["cogtrix.kind"] == "synthetic_tool_repair"``.

A repair that instead built a *merged* replacement object would silently
disappear from state on write-back (the object never existed in the input,
so nothing ``is`` it, and it isn't tagged synthetic either) — the exact bug
class the #2276 dedup fix closed. These tests pin that invariant across a
representative set of malformed histories so a future "helpful" rewrite of
one of the passes can't reintroduce it.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cogtrix_core.orchestration.message_repair import _repair_tool_message_pairs

_SYNTHETIC_KIND = "synthetic_tool_repair"


def _tc(name: str, tcid: str) -> dict:
    return {"name": name, "args": {}, "id": tcid, "type": "tool_call"}


def _is_synthetic(msg: object) -> bool:
    return (
        isinstance(msg, ToolMessage)
        and (getattr(msg, "additional_kwargs", None) or {}).get("cogtrix.kind") == _SYNTHETIC_KIND
    )


def _assert_identity_subset_or_synthetic(original: list, repaired: list) -> None:
    """Every repaired message must be one of the ORIGINAL objects (by
    identity) or a marked synthetic placeholder — never a merged copy."""
    for msg in repaired:
        is_original = any(msg is orig for orig in original)
        assert is_original or _is_synthetic(msg), (
            f"Repaired message (type={type(msg).__name__}, "
            f"content={getattr(msg, 'content', None)!r}) is neither an "
            "original input object nor a marked synthetic placeholder — "
            "the removal-only write-back contract (#2276) was violated."
        )


# ── Scenario fixtures — one per repair pass, plus a combined-chaos case ────


def _scenario_unanswered_tool_call() -> list:
    """Pass 3 (#2238): a declared tool_call with no following ToolMessage
    gets a synthetic placeholder injected."""
    return [
        HumanMessage(content="do X"),
        AIMessage(content="", tool_calls=[_tc("search", "tc1")]),
    ]


def _scenario_duplicate_answer() -> list:
    """Pass 5 (#2276): two ToolMessages answer the same tool_call_id; the
    extra must be dropped, not merged into the first."""
    return [
        HumanMessage(content="do X"),
        AIMessage(content="", tool_calls=[_tc("search", "tc1")]),
        ToolMessage(content="first answer", tool_call_id="tc1", name="search"),
        ToolMessage(content="duplicate answer", tool_call_id="tc1", name="search"),
    ]


def _scenario_displaced_answer() -> list:
    """Pass 4: a foreign message wedged between the declaring AIMessage and
    its answering ToolMessages breaks contiguity; the answers are relocated
    (as the SAME objects) to sit right after the declaration."""
    return [
        HumanMessage(content="do X and Y"),
        AIMessage(content="", tool_calls=[_tc("search", "tc1"), _tc("search", "tc2")]),
        SystemMessage(content="[nudge] please continue"),
        ToolMessage(content="result 1", tool_call_id="tc1", name="search"),
        ToolMessage(content="result 2", tool_call_id="tc2", name="search"),
    ]


def _scenario_redundant_duplicate_declaration() -> list:
    """Pass 3b (#2365 defect B): a content-free AIMessage re-declares an
    already-answered tool_call_id (retry/compression churn); the whole
    re-declaring message must be dropped outright, not rewritten."""
    ai1 = AIMessage(content="", tool_calls=[_tc("search", "tc1")])
    tm1 = ToolMessage(content="the answer", tool_call_id="tc1", name="search")
    ai2 = AIMessage(content="", tool_calls=[_tc("search", "tc1")])  # redundant re-declare
    return [HumanMessage(content="do X"), ai1, tm1, ai2]


def _scenario_orphaned_tool_message() -> list:
    """A ToolMessage answering a tool_call_id no AIMessage ever declared
    must be dropped outright (never rewritten into something else)."""
    return [
        HumanMessage(content="do X"),
        ToolMessage(content="orphaned result", tool_call_id="ghost", name="search"),
    ]


def _scenario_misordered_tool_message() -> list:
    """A ToolMessage appearing BEFORE its declaring AIMessage is dropped as
    misordered — and since the declaration is now unanswered, a synthetic
    placeholder is injected for it."""
    return [
        ToolMessage(content="premature result", tool_call_id="tc1", name="search"),
        HumanMessage(content="do X"),
        AIMessage(content="", tool_calls=[_tc("search", "tc1")]),
    ]


def _scenario_combined_chaos() -> list:
    """Multiple faults in a single turn — unanswered + duplicate + displaced
    — stress-testing all passes together."""
    return [
        HumanMessage(content="multi-step task"),
        AIMessage(content="", tool_calls=[_tc("a", "tc1"), _tc("b", "tc2")]),
        ToolMessage(content="a result", tool_call_id="tc1", name="a"),
        ToolMessage(content="a result dup", tool_call_id="tc1", name="a"),
        SystemMessage(content="[nudge]"),
        ToolMessage(content="b result", tool_call_id="tc2", name="b"),
        AIMessage(content="", tool_calls=[_tc("c", "tc3")]),  # unanswered
    ]


_SCENARIOS = {
    "unanswered_tool_call": _scenario_unanswered_tool_call,
    "duplicate_answer": _scenario_duplicate_answer,
    "displaced_answer": _scenario_displaced_answer,
    "redundant_duplicate_declaration": _scenario_redundant_duplicate_declaration,
    "orphaned_tool_message": _scenario_orphaned_tool_message,
    "misordered_tool_message": _scenario_misordered_tool_message,
    "combined_chaos": _scenario_combined_chaos,
}


class TestRepairIdentitySubsetInvariant:
    """Parametrized property test: for every malformed-history shape the
    three repair passes handle, the repaired output only ever contains
    original objects or marked synthetic placeholders."""

    @pytest.mark.parametrize("scenario_name", sorted(_SCENARIOS))
    def test_repaired_output_is_identity_subset_or_marked_synthetic(
        self, scenario_name: str
    ) -> None:
        messages = _SCENARIOS[scenario_name]()
        repaired = _repair_tool_message_pairs(messages)
        _assert_identity_subset_or_synthetic(messages, repaired)

    def test_well_formed_history_returned_unchanged_by_identity(self) -> None:
        """Sanity check for both the invariant helper and the fast-path
        short-circuit (``return messages`` when nothing needs repair)."""
        messages = [
            HumanMessage(content="do X"),
            AIMessage(content="", tool_calls=[_tc("search", "tc1")]),
            ToolMessage(content="the answer", tool_call_id="tc1", name="search"),
        ]
        repaired = _repair_tool_message_pairs(messages)
        assert repaired is messages
        _assert_identity_subset_or_synthetic(messages, repaired)

    def test_unanswered_tool_call_produces_exactly_one_marked_synthetic(self) -> None:
        """Focused check that the ONE genuinely new object introduced by
        this scenario really does carry the sentinel marker — proving the
        parametrized invariant isn't vacuously true because no synthetic
        message was produced at all."""
        messages = _scenario_unanswered_tool_call()
        repaired = _repair_tool_message_pairs(messages)
        new_objects = [m for m in repaired if not any(m is orig for orig in messages)]
        assert len(new_objects) == 1
        assert _is_synthetic(new_objects[0]), (
            f"Expected the one new object to be a marked synthetic placeholder; "
            f"got additional_kwargs={getattr(new_objects[0], 'additional_kwargs', None)!r}"
        )

    def test_duplicate_answer_extra_is_dropped_not_merged(self) -> None:
        """The dropped duplicate must not resurface as a merged/modified
        message — only the FIRST original ToolMessage object survives."""
        messages = _scenario_duplicate_answer()
        repaired = _repair_tool_message_pairs(messages)
        tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0] is messages[2]  # the FIRST answer, by identity
        assert tool_msgs[0].content == "first answer"

    def test_redundant_duplicate_declaration_is_dropped_not_merged(self) -> None:
        """The re-declaring AIMessage must be dropped whole — not rewritten
        with its tool_calls stripped (that would be a merged copy)."""
        messages = _scenario_redundant_duplicate_declaration()
        repaired = _repair_tool_message_pairs(messages)
        ai_msgs = [m for m in repaired if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 1
        assert ai_msgs[0] is messages[1]  # the original declaring AIMessage

    def test_displaced_answers_are_relocated_by_identity_not_copied(self) -> None:
        """Contiguity repair must move the SAME ToolMessage objects next to
        their declaration, not synthesize replacements for them."""
        messages = _scenario_displaced_answer()
        repaired = _repair_tool_message_pairs(messages)
        tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        assert tool_msgs[0] is messages[3]
        assert tool_msgs[1] is messages[4]
        # And they must now be contiguous with the declaring AIMessage.
        ai_idx = next(i for i, m in enumerate(repaired) if m is messages[1])
        assert repaired[ai_idx + 1] is messages[3]
        assert repaired[ai_idx + 2] is messages[4]
