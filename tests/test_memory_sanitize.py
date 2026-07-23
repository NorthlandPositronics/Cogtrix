"""Tests for sanitize_history BUG-033 and BUG-064 fixes.

BUG-033: Back-scanning after removing a bad AI also removes the triggering HumanMessage
         to prevent an orphaned HumanMessage with no following response.

BUG-064: The is_human branch scans forward past tool chains to find the terminal AI
         response, so [human, ai-tc, tool, bad-ai] is handled as a single bad chain.
"""

from __future__ import annotations


def _make_messages(spec: list[tuple[str, str, bool]]):
    """Build a message list from a compact spec.

    Each tuple is (type, content, has_tool_calls).
    type: 'human' | 'ai' | 'tool'
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    msgs = []
    tc_counter = [0]
    for msg_type, content, has_tc in spec:
        if msg_type == "human":
            msgs.append(HumanMessage(content=content))
        elif msg_type == "ai":
            if has_tc:
                tc_counter[0] += 1
                msgs.append(
                    AIMessage(
                        content=content,
                        tool_calls=[
                            {
                                "name": f"t{tc_counter[0]}",
                                "args": {},
                                "id": f"tc{tc_counter[0]}",
                            }
                        ],
                    )
                )
            else:
                msgs.append(AIMessage(content=content))
        elif msg_type == "tool":
            msgs.append(ToolMessage(content=content, tool_call_id=f"tc{tc_counter[0]}"))
    return msgs


class TestSanitizeHistoryBug033:
    """BUG-033: Orphaned HumanMessage after back-scan must be removed."""

    def test_human_ai_tc_tool_bad_ai_all_removed(self):
        """[human, ai-tc, tool, bad-ai] — all four must be removed."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        messages = _make_messages(
            [
                ("human", "query", False),
                ("ai", "", True),  # ai-tc
                ("tool", "result", False),
                ("ai", "An error occurred: boom", False),  # bad
            ]
        )
        cleaned = BaseMemoryManager.sanitize_history(messages)
        assert len(cleaned) == 0

    def test_human_bad_ai_both_removed(self):
        """[human, bad-ai] — both must be removed."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        messages = _make_messages(
            [
                ("human", "query", False),
                ("ai", "An error occurred: something", False),  # bad
            ]
        )
        cleaned = BaseMemoryManager.sanitize_history(messages)
        assert len(cleaned) == 0

    def test_good_pair_then_bad_chain_only_bad_removed(self):
        """[human, good-ai, human, ai-tc, tool, bad-ai] — last four removed, first two preserved."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        messages = _make_messages(
            [
                ("human", "first question", False),
                ("ai", "Good answer", False),  # good
                ("human", "second question", False),
                ("ai", "", True),  # ai-tc
                ("tool", "result", False),
                ("ai", "", False),  # bad (empty)
            ]
        )
        cleaned = BaseMemoryManager.sanitize_history(messages)
        assert len(cleaned) == 2
        assert cleaned[0].content == "first question"
        assert cleaned[1].content == "Good answer"

    def test_standalone_bad_ai_without_human_removed(self):
        """A standalone bad AI with no preceding human is removed."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        messages = _make_messages(
            [
                ("ai", "An error occurred: oops", False),  # bad, no human before it
            ]
        )
        cleaned = BaseMemoryManager.sanitize_history(messages)
        assert len(cleaned) == 0

    def test_valid_tool_chain_preserved(self):
        """[human, ai-tc, tool, good-ai] — nothing removed."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        messages = _make_messages(
            [
                ("human", "search cats", False),
                ("ai", "", True),  # ai-tc
                ("tool", "3 cats found", False),
                ("ai", "Found 3 cats!", False),  # good
            ]
        )
        cleaned = BaseMemoryManager.sanitize_history(messages)
        assert len(cleaned) == 4


class TestSanitizeHistoryBug064:
    """BUG-064: is_human branch must scan past tool chains to find terminal bad AI."""

    def test_human_then_ai_tc_then_tool_then_bad_ai_all_removed(self):
        """[human, ai-tc, tool, bad-ai] detected in the is_human forward-scan."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        messages = _make_messages(
            [
                ("human", "do work", False),
                ("ai", "", True),  # ai-tc
                ("tool", "result", False),
                ("ai", "An error occurred: failed", False),  # bad terminal
            ]
        )
        cleaned = BaseMemoryManager.sanitize_history(messages)
        assert len(cleaned) == 0

    def test_human_then_multiple_tool_chain_steps_then_bad_ai_all_removed(self):
        """[human, ai-tc1, tool1, ai-tc2, tool2, bad-ai] — all six removed."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        messages = _make_messages(
            [
                ("human", "do work", False),
                ("ai", "", True),  # ai-tc1
                ("tool", "r1", False),
                ("ai", "", True),  # ai-tc2
                ("tool", "r2", False),
                ("ai", "", False),  # bad (empty)
            ]
        )
        cleaned = BaseMemoryManager.sanitize_history(messages)
        assert len(cleaned) == 0

    def test_good_human_pair_then_bad_chain_selective(self):
        """First pair preserved; second chain with tool steps is removed."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        messages = _make_messages(
            [
                ("human", "q1", False),
                ("ai", "Good response", False),  # good
                ("human", "q2", False),
                ("ai", "", True),  # ai-tc
                ("tool", "result", False),
                ("ai", "An error occurred: failure", False),  # bad
            ]
        )
        cleaned = BaseMemoryManager.sanitize_history(messages)
        assert len(cleaned) == 2
        assert cleaned[0].content == "q1"
        assert cleaned[1].content == "Good response"


class TestSanitizeHistoryParity:
    """Property-style tests to ensure sanitize_history maintains message structure."""

    def test_result_length_even_for_human_ai_pairs(self):
        """When all bad messages are removed, only good human+AI pairs remain (even count)."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        messages = _make_messages(
            [
                ("human", "q1", False),
                ("ai", "Good a1", False),
                ("human", "q2", False),
                ("ai", "Good a2", False),
                ("human", "q3", False),
                ("ai", "", False),  # bad — removes q3+bad-ai
            ]
        )
        cleaned = BaseMemoryManager.sanitize_history(messages)
        # 4 messages remain (q1/a1 + q2/a2); q3 and its bad AI are both removed
        assert len(cleaned) % 2 == 0
        assert len(cleaned) == 4

    def test_empty_history_returns_empty(self):
        """Empty input returns empty list."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        assert BaseMemoryManager.sanitize_history([]) == []

    def test_all_good_messages_unchanged(self):
        """History with no bad messages is returned unchanged."""
        from cogtrix_core.memory.manager import BaseMemoryManager

        messages = _make_messages(
            [
                ("human", "hi", False),
                ("ai", "hello", False),
                ("human", "bye", False),
                ("ai", "farewell", False),
            ]
        )
        cleaned = BaseMemoryManager.sanitize_history(messages)
        assert len(cleaned) == 4
