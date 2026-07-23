"""#2365 (defect B) — occurrence-aware duplicate tool_call declarations.

The pre-existing repair keyed on SET math: ``unanswered_ids = declared_ids -
answered_ids``. When heavy compression / retry churn re-emits a tool-calling
AIMessage, the SAME ``tool_call_id`` ends up declared by TWO AIMessages but
answered once. The set view says "answered" (the first occurrence was), so
nothing is repaired — but OpenAI-compatible providers validate PER
assistant-message occurrence and 400 on the second, unanswered declaration
("tool_call_ids did not have response messages: …").

A ``tool_call_id`` can be answered only once, so injecting a second placeholder
answer would itself 400 (duplicate answer). The correct, provider-valid repair
is to DROP the redundant re-declaration. It must be removal-only so it composes
with call_model's id()-based ``RemoveMessage`` state write-back (a modifying
repair would be deleted from state and never re-added — the #2276 coupling), so
the guard drops only a re-declaring AIMessage that carries no unique text (a pure
re-emitted tool-call block); content-bearing duplicates / id-form mismatches are
left for the shipped #2365 diagnostic to characterise.
"""

from __future__ import annotations

import os

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from cogtrix_core.orchestration.message_repair import _repair_tool_message_pairs  # noqa: E402


def _ai(tool_calls, content: str = ""):
    return AIMessage(content=content, tool_calls=tool_calls)


def _tc(name: str, tcid: str, **args):
    return {"name": name, "args": args, "id": tcid}


def _ai_ids(messages: list) -> list[str]:
    """Every tool_call id still declared by an AIMessage, in order (with repeats)."""
    out: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                if tc.get("id"):
                    out.append(tc["id"])
    return out


def _tool_ids(messages: list) -> list[str]:
    return [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]


class TestDuplicateDeclarationDropped:
    def test_issue_repro_redeclared_after_answer(self) -> None:
        """The #2365 defect-B shape: X declared, answered, then RE-declared by a
        pure tool-call AIMessage. The redundant re-declaration is dropped so the
        provider never sees an unanswered second occurrence."""
        first = _ai([_tc("shell", "shell:11", i=1)])
        answer = ToolMessage(content="out-1", tool_call_id="shell:11")
        dup = _ai([_tc("shell", "shell:11", i=2)])  # re-emitted declaration, no answer
        repaired = _repair_tool_message_pairs([HumanMessage(content="go"), first, answer, dup])

        # The duplicate declaration is gone; the id is declared exactly once…
        assert _ai_ids(repaired) == ["shell:11"]
        assert dup not in repaired
        # …the original pair survives untouched, and no bogus 2nd answer was added.
        assert first in repaired and answer in repaired
        assert _tool_ids(repaired) == ["shell:11"]

    def test_declared_once_per_message_after_repair(self) -> None:
        """Every surviving declared id appears in exactly one AIMessage (the
        provider's per-occurrence contract)."""
        first = _ai([_tc("q", "a"), _tc("r", "b")])
        answers = [
            ToolMessage(content="1", tool_call_id="a"),
            ToolMessage(content="2", tool_call_id="b"),
        ]
        dup = _ai([_tc("q", "a"), _tc("r", "b")])  # full re-emission of the batch
        repaired = _repair_tool_message_pairs([first, *answers, dup])

        ids = _ai_ids(repaired)
        assert sorted(ids) == ["a", "b"]  # each id declared once, no duplicates
        # Exactly one tool-calling AIMessage survives (first == dup by value here,
        # so assert by count, not membership).
        assert sum(1 for m in repaired if isinstance(m, AIMessage) and m.tool_calls) == 1

    def test_content_bearing_duplicate_is_left_alone(self) -> None:
        """A re-declaring AIMessage that ALSO carries text is NOT dropped (removal
        would lose the text; modifying it would corrupt the removal-only state
        write-back). Left for the shipped #2365 diagnostic to characterise."""
        first = _ai([_tc("shell", "s:1")])
        answer = ToolMessage(content="out", tool_call_id="s:1")
        dup = _ai([_tc("shell", "s:1")], content="Here is a summary of what I did.")
        repaired = _repair_tool_message_pairs([first, answer, dup])

        assert dup in repaired  # content-bearing → untouched by this guard

    def test_no_duplicate_is_behaviour_preserving(self) -> None:
        """Healthy single-declaration conversations are returned unchanged."""
        ai = _ai([_tc("f", "ok")])
        tm = ToolMessage(content="done", tool_call_id="ok")
        msgs = [HumanMessage(content="hi"), ai, tm]
        assert _repair_tool_message_pairs(msgs) == msgs

    def test_idempotent(self) -> None:
        first = _ai([_tc("shell", "x:9")])
        answer = ToolMessage(content="r", tool_call_id="x:9")
        dup = _ai([_tc("shell", "x:9")])
        once = _repair_tool_message_pairs([first, answer, dup])
        twice = _repair_tool_message_pairs(once)
        assert once == twice
        assert _ai_ids(twice) == ["x:9"]
