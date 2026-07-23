"""#2276 — _repair_tool_message_pairs must enforce tool-block contiguity.

OpenAI/Azure reject a request unless every ``tool`` message *immediately* follows
its declaring ``assistant(tool_calls)`` (or a sibling ``tool`` message in the same
block). A directive/nudge/compression step can wedge a non-tool message between an
``AIMessage(tool_calls)`` and its answering ``ToolMessage(s)`` — declaration and
order stay valid, so the orphaned/misordered/unanswered passes leave it, but the
provider 400s ("messages with role 'tool' must be a response to a preceeding
message with 'tool_calls'") and the turn crashes.

The repair now relocates each AIMessage's answered ToolMessages to sit immediately
after it (foreign wedged messages move past the block), without dropping any tool
result, and is a strict no-op for already-contiguous histories.

Found via the PM role-test harness (gpt-4o, role_pm_06) on 2026-06-26.
"""

from __future__ import annotations

import os

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from cogtrix_core.orchestration.message_repair import _repair_tool_message_pairs  # noqa: E402


def _ai_tc(*ids: str, content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[{"id": i, "name": "kb", "args": {}} for i in ids],
    )


def _contiguity_violation_index(messages: list) -> int | None:
    """Return the index of the first ToolMessage that does NOT immediately follow
    an AIMessage with tool_calls or a sibling ToolMessage; ``None`` if valid."""
    for i, m in enumerate(messages):
        if isinstance(m, ToolMessage):
            prev = messages[i - 1] if i > 0 else None
            ok = (isinstance(prev, AIMessage) and (getattr(prev, "tool_calls", None) or [])) or (
                isinstance(prev, ToolMessage)
            )
            if not ok:
                return i
    return None


class TestContiguityRepair:
    def test_wedged_foreign_message_relocated(self) -> None:
        """A SystemMessage wedged between AI(tool_calls) and its ToolMessage is
        moved past the tool block, restoring contiguity (the gpt-4o 400)."""
        ai = _ai_tc("tc1")
        advisory = SystemMessage(content="[ADVISORY] you are looping; think first")
        tm = ToolMessage(content="result", tool_call_id="tc1", name="kb")
        final = AIMessage(content="final answer")
        out = _repair_tool_message_pairs([HumanMessage(content="hi"), ai, advisory, tm, final])

        assert _contiguity_violation_index(out) is None
        # No data lost: the tool result and the advisory both survive.
        assert tm in out
        assert advisory in out
        # The tool message now sits immediately after its declaring AIMessage.
        assert out[out.index(ai) + 1] is tm

    def test_parallel_batch_split_by_advisory(self) -> None:
        """A two-call batch whose answers are split by a wedged message is made
        contiguous, both results preserved."""
        ai = _ai_tc("t1", "t2")
        t1 = ToolMessage(content="r1", tool_call_id="t1")
        nudge = SystemMessage(content="nudge")
        t2 = ToolMessage(content="r2", tool_call_id="t2")
        out = _repair_tool_message_pairs([ai, t1, nudge, t2, AIMessage(content="done")])

        assert _contiguity_violation_index(out) is None
        assert t1 in out and t2 in out and nudge in out
        # Both answers are pulled directly behind the declaring AIMessage.
        i = out.index(ai)
        assert out[i + 1] is t1
        assert out[i + 2] is t2

    def test_already_contiguous_is_noop(self) -> None:
        """A valid, already-contiguous pair is returned unchanged (same object)."""
        ai = _ai_tc("x")
        tm = ToolMessage(content="r", tool_call_id="x")
        src = [HumanMessage(content="hi"), ai, tm, AIMessage(content="ok")]
        out = _repair_tool_message_pairs(src)
        assert out is src  # no-op early return preserves identity

    def test_contiguity_repair_preserves_unanswered_injection(self) -> None:
        """Contiguity relocation composes with #2238 synthetic injection: an
        answered + an unanswered tool_call from the same AIMessage both end up in
        a contiguous block after it."""
        ai = _ai_tc("ans", "missing")
        wedge = SystemMessage(content="wedge")
        answered = ToolMessage(content="r", tool_call_id="ans")
        out = _repair_tool_message_pairs([ai, wedge, answered, AIMessage(content="end")])

        assert _contiguity_violation_index(out) is None
        i = out.index(ai)
        # Real answer first, synthetic placeholder for the missing one next.
        assert out[i + 1] is answered
        assert isinstance(out[i + 2], ToolMessage)
        assert out[i + 2].tool_call_id == "missing"
        assert out[i + 2].content == "[tool call not completed]"

    def test_orphan_and_misordered_still_dropped_with_contiguity(self) -> None:
        """The new pass doesn't regress orphan/misordered dropping."""
        ai = _ai_tc("real")
        wedge = SystemMessage(content="wedge")
        real = ToolMessage(content="r", tool_call_id="real")
        orphan = ToolMessage(content="x", tool_call_id="ghost")
        out = _repair_tool_message_pairs([ai, wedge, real, orphan])

        assert orphan not in out
        assert real in out
        assert _contiguity_violation_index(out) is None


def _tool_call_ids(messages: list) -> list[str]:
    return [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]


class TestDuplicateToolMessageDedup:
    """The actual gpt-4o role_pm_06 crash: a tool node emitted TWO results for one
    declared tool_call. OpenAI/Azure require exactly one tool response per
    tool_call_id — the second is unmatched → 400. The repair keeps the first answer
    and drops the rest (removal-only, so it's compatible with call_model's
    RemoveMessage write-back). Distinct-content duplicates are prevented at the
    source instead (process_tools merges its identical-error hint into the error
    message rather than emitting a second message with the same id)."""

    def test_duplicate_answer_for_same_tool_call_dropped(self) -> None:
        """Exact sc06 shape: AIMessage declares 2 calls, but one is answered twice."""
        ai = _ai_tc("call_a", "call_b")
        out = _repair_tool_message_pairs(
            [
                SystemMessage(content="sys"),
                HumanMessage(content="q"),
                ai,
                ToolMessage(content="r-a", tool_call_id="call_a"),
                ToolMessage(content="r-b", tool_call_id="call_b"),
                ToolMessage(content="r-b-dup", tool_call_id="call_b"),  # duplicate
                SystemMessage(content="[ADVISORY]"),
            ]
        )
        # Exactly one answer per declared tool_call_id; no leftover duplicate.
        assert _tool_call_ids(out) == ["call_a", "call_b"]
        assert _contiguity_violation_index(out) is None

    def test_first_duplicate_answer_kept(self) -> None:
        """The first answer is the one retained (its content survives)."""
        ai = _ai_tc("x")
        first = ToolMessage(content="keep-me", tool_call_id="x")
        dup = ToolMessage(content="drop-me", tool_call_id="x")
        out = _repair_tool_message_pairs([ai, first, dup, AIMessage(content="done")])
        kept = [m for m in out if isinstance(m, ToolMessage)]
        assert len(kept) == 1
        assert kept[0] is first

    def test_duplicate_split_by_wedge_dedups_and_contiguous(self) -> None:
        """Dedup composes with contiguity relocation."""
        ai = _ai_tc("x")
        out = _repair_tool_message_pairs(
            [
                ai,
                ToolMessage(content="r1", tool_call_id="x"),
                SystemMessage(content="wedge"),
                ToolMessage(content="r1-dup", tool_call_id="x"),
                AIMessage(content="done"),
            ]
        )
        assert _tool_call_ids(out) == ["x"]
        assert _contiguity_violation_index(out) is None

    def test_single_answer_per_call_is_noop(self) -> None:
        """No duplicates → unchanged (identity preserved)."""
        ai = _ai_tc("a", "b")
        src = [
            ai,
            ToolMessage(content="ra", tool_call_id="a"),
            ToolMessage(content="rb", tool_call_id="b"),
        ]
        assert _repair_tool_message_pairs(src) is src
