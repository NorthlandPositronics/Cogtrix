"""Tests for _repair_tool_message_pairs — orphaned ToolMessage cleanup.

Covers:
- No orphans: list returned unchanged
- Fully orphaned ToolMessage (no matching AIMessage tool_call): dropped
- Mixed: only orphans dropped, valid pairs preserved
- Empty AIMessage dropped alongside orphan (repair-triggered cleanup)
- Non-empty AIMessage with only string content preserved
- Non-empty AIMessage with list content (Anthropic-style) preserved
- AIMessage with content blocks containing tool_use entries: ids are harvested
- AIMessage with additional_kwargs["tool_calls"] (OpenAI style): ids harvested
- Partial orphan: one orphan and one valid ToolMessage in same history
- Limit: PlanLimitSnapshot.within_limit with cap=0 and large current always True
"""

from __future__ import annotations

import os

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from src.orchestration.graph import _repair_tool_message_pairs  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ai(content=None, tool_calls=None, additional_kwargs=None):
    kwargs: dict = {}
    if tool_calls is not None:
        kwargs["tool_calls"] = tool_calls
    if additional_kwargs is not None:
        kwargs["additional_kwargs"] = additional_kwargs
    return AIMessage(content=content if content is not None else "", **kwargs)


def _tool(tool_call_id: str, content: str = "result"):
    return ToolMessage(content=content, tool_call_id=tool_call_id)


def _human(content="hi"):
    return HumanMessage(content=content)


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------


class TestNoOrphans:
    def test_empty_list(self):
        assert _repair_tool_message_pairs([]) == []

    def test_no_tool_messages(self):
        msgs = [_human("hello"), _ai("sure")]
        result = _repair_tool_message_pairs(msgs)
        assert result == msgs

    def test_valid_openai_pair_preserved(self):
        ai = _ai(tool_calls=[{"id": "call_1", "name": "calc", "args": {}}])
        tm = _tool("call_1", "42")
        result = _repair_tool_message_pairs([ai, tm])
        assert len(result) == 2
        assert result[0] is ai
        assert result[1] is tm

    def test_valid_additional_kwargs_pair_preserved(self):
        """OpenAI-style tool calls in additional_kwargs are recognised."""
        ai = _ai(additional_kwargs={"tool_calls": [{"id": "call_ak", "name": "x", "args": {}}]})
        tm = _tool("call_ak")
        result = _repair_tool_message_pairs([ai, tm])
        assert len(result) == 2

    def test_valid_anthropic_content_block_pair_preserved(self):
        """Anthropic-style tool_use content blocks are recognised."""
        ai = _ai(content=[{"type": "tool_use", "id": "tu_1", "name": "search", "input": {}}])
        tm = _tool("tu_1", "found it")
        result = _repair_tool_message_pairs([ai, tm])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Orphan removal
# ---------------------------------------------------------------------------


class TestOrphanDropped:
    def test_fully_orphaned_tool_message_dropped(self):
        orphan = _tool("ghost_id")
        result = _repair_tool_message_pairs([_human("q"), orphan])
        assert orphan not in result

    def test_orphan_dropped_valid_pair_kept(self):
        ai = _ai(tool_calls=[{"id": "real", "name": "f", "args": {}}])
        valid_tm = _tool("real")
        orphan_tm = _tool("ghost")
        result = _repair_tool_message_pairs([ai, valid_tm, orphan_tm])
        assert valid_tm in result
        assert orphan_tm not in result
        assert ai in result

    def test_multiple_orphans_all_dropped(self):
        orphan_a = _tool("ghost_a")
        orphan_b = _tool("ghost_b")
        result = _repair_tool_message_pairs([orphan_a, orphan_b])
        assert result == []

    def test_empty_ai_message_dropped_alongside_orphan(self):
        """Empty AIMessage (no content, no tool_calls) dropped when orphan repair runs."""
        empty_ai = _ai("")  # no content, no tool_calls
        orphan = _tool("ghost")
        result = _repair_tool_message_pairs([empty_ai, orphan])
        assert result == []

    def test_human_messages_not_dropped(self):
        human = _human("question")
        orphan = _tool("ghost")
        result = _repair_tool_message_pairs([human, orphan])
        assert human in result
        assert orphan not in result

    def test_tool_message_before_declaring_ai_dropped(self):
        """A ToolMessage must appear after the AIMessage that declares it."""
        ai = _ai(tool_calls=[{"id": "call_1", "name": "calc", "args": {}}])
        tool_msg = _tool("call_1", "42")
        result = _repair_tool_message_pairs([tool_msg, ai])
        assert tool_msg not in result
        assert ai in result


# ---------------------------------------------------------------------------
# AIMessage preservation edge cases
# ---------------------------------------------------------------------------


class TestAIMessagePreservation:
    def test_ai_with_string_content_preserved(self):
        ai = _ai("I found the answer.")
        result = _repair_tool_message_pairs([ai])
        assert ai in result

    def test_ai_with_list_content_preserved(self):
        """AIMessage with non-string (list) content must NOT be dropped."""
        ai = _ai(content=[{"type": "text", "text": "here is my analysis"}])
        result = _repair_tool_message_pairs([ai])
        assert ai in result

    def test_ai_with_anthropic_tool_use_block_preserved(self):
        """AIMessage whose only content is a tool_use block must be preserved."""
        ai = _ai(content=[{"type": "tool_use", "id": "tu_2", "name": "q", "input": {}}])
        tm = _tool("tu_2")
        result = _repair_tool_message_pairs([ai, tm])
        assert ai in result
        assert tm in result

    def test_completely_empty_ai_dropped_when_orphan_present(self):
        """Empty AIMessage dropped only when orphan repair is triggered."""
        empty = _ai("")
        orphan = _tool("ghost")
        result = _repair_tool_message_pairs([empty, orphan])
        assert result == []

    def test_ai_with_whitespace_only_content_dropped_when_orphan_present(self):
        ws = _ai("   \n\t  ")
        orphan = _tool("ghost")
        result = _repair_tool_message_pairs([ws, orphan])
        assert result == []

    def test_completely_empty_ai_alone_not_dropped(self):
        """Empty AIMessage is NOT dropped when there are no orphans (no-op path)."""
        empty = _ai("")
        result = _repair_tool_message_pairs([empty])
        # No orphans → function returns the list unchanged
        assert len(result) == 1


# ---------------------------------------------------------------------------
# PlanLimitSnapshot.within_limit regression (cap=0 always unlimited)
# ---------------------------------------------------------------------------


class TestWithinLimitUnlimited:
    def test_cap_zero_with_zero_current(self):
        from src.api.plan_enforcement import PlanLimitSnapshot

        snap = PlanLimitSnapshot("free", 0, 0, 0, 0, 0, 0, 0)
        assert snap.within_limit(0, 0) is True

    def test_cap_zero_with_large_current(self):
        from src.api.plan_enforcement import PlanLimitSnapshot

        snap = PlanLimitSnapshot("free", 0, 0, 0, 0, 50000, 0, 0)
        assert snap.within_limit(0, 50000) is True

    def test_cap_zero_can_always_add_user(self):
        from src.api.plan_enforcement import PlanLimitSnapshot

        snap = PlanLimitSnapshot("free", 0, 0, 0, 0, 99999, 0, 0)
        assert snap.can_add_user is True
