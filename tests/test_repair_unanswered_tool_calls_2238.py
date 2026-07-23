"""#2238 — _repair_tool_message_pairs must answer declared-but-unanswered tool_calls.

An AIMessage whose declared ``tool_call_id``s are not all followed by ToolMessages
makes OpenAI-compatible providers reject the request with a 400 ("insufficient
tool messages following tool_calls"), killing the turn. The guard now injects a
synthetic placeholder ToolMessage for each unanswered call so the pairing holds.
"""

from __future__ import annotations

import os

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from src.orchestration.message_repair import _repair_tool_message_pairs  # noqa: E402


def _declared_ids(messages: list) -> set[str]:
    ids: set[str] = set()
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                if tc.get("id"):
                    ids.add(tc["id"])
    return ids


def _answered_ids(messages: list) -> set[str]:
    return {
        m.tool_call_id
        for m in messages
        if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None)
    }


def _ai(tool_calls):
    return AIMessage(content="", tool_calls=tool_calls)


class TestUnansweredInjection:
    def test_issue_repro_partial_batch(self) -> None:
        """The exact #2238 repro: 2 declared calls, 1 answered → the other gets a
        synthetic answer so declared ⊆ answered."""
        msgs = [
            HumanMessage(content="list everything"),
            _ai(
                [
                    {"name": "read_file", "args": {"path": "a"}, "id": "call_a"},
                    {"name": "read_file", "args": {"path": "b"}, "id": "call_b"},
                ]
            ),
            ToolMessage(content="contents of a", tool_call_id="call_a"),
        ]
        repaired = _repair_tool_message_pairs(msgs)

        assert _declared_ids(repaired) <= _answered_ids(repaired)
        assert _answered_ids(repaired) == {"call_a", "call_b"}
        # The original answer is preserved; only the missing one is synthesised.
        assert any(
            isinstance(m, ToolMessage)
            and m.tool_call_id == "call_b"
            and m.content == "[tool call not completed]"
            for m in repaired
        )

    def test_synthetic_carries_name_and_kind(self) -> None:
        msgs = [
            _ai([{"name": "search_web", "args": {}, "id": "c1"}]),
        ]
        repaired = _repair_tool_message_pairs(msgs)
        synth = [m for m in repaired if isinstance(m, ToolMessage)]
        assert len(synth) == 1
        assert synth[0].tool_call_id == "c1"
        assert synth[0].name == "search_web"
        assert synth[0].additional_kwargs.get("cogtrix.kind") == "synthetic_tool_repair"

    def test_synthetic_immediately_follows_declaring_ai(self) -> None:
        ai = _ai([{"name": "f", "args": {}, "id": "x1"}])
        repaired = _repair_tool_message_pairs(
            [HumanMessage(content="go"), ai, HumanMessage(content="next")]
        )
        i_ai = repaired.index(ai)
        assert isinstance(repaired[i_ai + 1], ToolMessage)
        assert repaired[i_ai + 1].tool_call_id == "x1"

    def test_fully_unanswered_batch_all_synthesised(self) -> None:
        ai = _ai(
            [
                {"name": "f", "args": {}, "id": "a"},
                {"name": "g", "args": {}, "id": "b"},
                {"name": "h", "args": {}, "id": "c"},
            ]
        )
        repaired = _repair_tool_message_pairs([ai])
        assert _answered_ids(repaired) == {"a", "b", "c"}

    def test_misordered_then_unanswered_gets_synthetic(self) -> None:
        """A ToolMessage before its declaring AIMessage is dropped (misordered),
        leaving the call unanswered — which is then synthesised."""
        ai = _ai([{"name": "calc", "args": {}, "id": "m1"}])
        before = ToolMessage(content="42", tool_call_id="m1")
        repaired = _repair_tool_message_pairs([before, ai])
        assert before not in repaired  # misordered dropped
        assert _declared_ids(repaired) <= _answered_ids(repaired)

    def test_complete_pair_unchanged(self) -> None:
        ai = _ai([{"name": "f", "args": {}, "id": "ok"}])
        tm = ToolMessage(content="done", tool_call_id="ok")
        repaired = _repair_tool_message_pairs([ai, tm])
        assert repaired == [ai, tm]  # identity preserved, no injection

    def test_idempotent(self) -> None:
        msgs = [
            _ai(
                [
                    {"name": "f", "args": {}, "id": "p"},
                    {"name": "g", "args": {}, "id": "q"},
                ]
            ),
            ToolMessage(content="p-done", tool_call_id="p"),
        ]
        once = _repair_tool_message_pairs(msgs)
        twice = _repair_tool_message_pairs(once)
        # Second pass finds everything answered → no further change.
        assert _answered_ids(twice) == {"p", "q"}
        assert len(twice) == len(once)
