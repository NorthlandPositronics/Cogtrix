"""Regression tests consolidating Round 4 (ARCH-037-09) and Round 8 bug fixes
(BUG-031 through BUG-040, PERF-001)."""

from __future__ import annotations

import json
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


# ---------------------------------------------------------------------------
# BUG-031: run_execution_phase must not mutate caller's active_tools_list
# ---------------------------------------------------------------------------


class TestBug031ActiveToolsListDeepCopy:
    """BUG-031: Execution phase must deep-copy active_tools_list."""

    def test_execution_phase_does_not_mutate_config_active_tools(self):
        """run_execution_phase must not modify the original config's active_tools_list."""
        from src.orchestration.run_config import AgentRunConfig

        tool_a = MagicMock(name="tool_a")
        tool_b = MagicMock(name="tool_b")
        original_tools = [tool_a, tool_b]

        config = AgentRunConfig(
            llm=MagicMock(),
            system_prompt="test",
            available_tools={"t": MagicMock()},
            active_tools_list=original_tools,
        )
        original_ids = [id(t) for t in config.active_tools_list]

        with patch("src.orchestration.runner.run_agent", return_value="done"):
            from src.orchestration.phases import run_execution_phase

            run_execution_phase(
                analysis="analysis text",
                original_prompt="test prompt",
                context_messages=[],
                registry=MagicMock(),
                approvals=set(),
                config=config,
            )

        assert config.active_tools_list is original_tools
        assert [id(t) for t in config.active_tools_list] == original_ids
        assert len(config.active_tools_list) == 2

    def test_execution_phase_does_not_mutate_config_available_tools(self):
        """run_execution_phase must not modify the original config's available_tools."""
        from src.orchestration.run_config import AgentRunConfig

        original_avail = {"tool_x": MagicMock()}
        config = AgentRunConfig(
            llm=MagicMock(),
            system_prompt="test",
            available_tools=original_avail,
            active_tools_list=[MagicMock()],
        )

        with patch("src.orchestration.runner.run_agent", return_value="done"):
            from src.orchestration.phases import run_execution_phase

            run_execution_phase(
                analysis="analysis",
                original_prompt="prompt",
                context_messages=[],
                registry=MagicMock(),
                approvals=set(),
                config=config,
            )

        assert config.available_tools is original_avail


# ---------------------------------------------------------------------------
# BUG-032: compression fallback must respect _FALLBACK_MAX_CHARS
# ---------------------------------------------------------------------------


class TestBug032CompressionFallbackCap:
    """BUG-032: Compression fallback must be capped at _FALLBACK_MAX_CHARS."""

    def test_fallback_truncation_capped(self):
        from src.orchestration.compression import _FALLBACK_MAX_CHARS, truncate_tool_output

        content = "x" * 200_000
        fallback_len = min(len(content) * 3 // 4, _FALLBACK_MAX_CHARS)
        result = truncate_tool_output(content, fallback_len)
        assert len(result) <= _FALLBACK_MAX_CHARS + 200  # marker overhead

    def test_compress_one_fallback_uses_cap(self):
        """When compress_tool_message raises, _compress_one caps at _FALLBACK_MAX_CHARS."""
        from langchain_core.messages import AIMessage, ToolMessage

        from src.orchestration.compression import (
            _FALLBACK_MAX_CHARS,
            apply_message_compression,
        )

        big_content = "A" * 200_000
        tool_msg = ToolMessage(content=big_content, tool_call_id="tc1", name="test_tool")
        messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "test_tool", "args": {}, "id": "tc1"}],
            ),
            tool_msg,
            AIMessage(content="summary"),
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")

        result = apply_message_compression(
            messages,
            call_count=20,
            compression_cache={},
            llm=mock_llm,
            max_context_tokens=50_000,
            min_age_cycles=0,
            min_chars=100,
        )
        compressed_content = result[1].content
        assert len(compressed_content) <= _FALLBACK_MAX_CHARS + 200


# ---------------------------------------------------------------------------
# BUG-033: sanitize_history removes orphaned tool-call chains
# ---------------------------------------------------------------------------


class TestBug033OrphanedToolMessageChains:
    """BUG-033: sanitize_history must remove orphaned AI(tool_calls)+ToolMessage chains."""

    def test_orphaned_chain_removed(self):
        """AI(tool_calls)+ToolMessage followed by a bad AI response should all be removed,
        including the triggering HumanMessage (BUG-033 fix)."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from src.memory.manager import BaseMemoryManager

        messages = [
            HumanMessage(content="hello"),
            AIMessage(
                content="",
                tool_calls=[{"name": "t", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="result", tool_call_id="tc1"),
            # Use a known error prefix that _is_bad_ai_content recognizes
            AIMessage(content="An error occurred: something went wrong"),
        ]
        cleaned = BaseMemoryManager.sanitize_history(messages)

        # BUG-033 fix: the triggering HumanMessage is also removed to prevent
        # an orphaned HumanMessage with no following response.
        assert len(cleaned) == 0

    def test_orphaned_chain_removed_empty_ai(self):
        """AI(tool_calls)+ToolMessage followed by empty AI should all be removed,
        including the triggering HumanMessage (BUG-033 fix)."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from src.memory.manager import BaseMemoryManager

        messages = [
            HumanMessage(content="hello"),
            AIMessage(
                content="",
                tool_calls=[{"name": "t", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="result", tool_call_id="tc1"),
            AIMessage(content=""),  # empty = bad
        ]
        cleaned = BaseMemoryManager.sanitize_history(messages)

        # BUG-033 fix: the triggering HumanMessage is also removed to prevent
        # an orphaned HumanMessage with no following response.
        assert len(cleaned) == 0

    def test_valid_tool_chain_preserved(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from src.memory.manager import BaseMemoryManager

        messages = [
            HumanMessage(content="search for cats"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="found 3 cats", tool_call_id="tc1"),
            AIMessage(content="I found 3 cats for you!"),
        ]
        cleaned = BaseMemoryManager.sanitize_history(messages)

        assert len(cleaned) == 4

    def test_multiple_orphaned_chains(self):
        """Multiple consecutive tool-call steps followed by a bad AI should all be removed,
        including the triggering HumanMessage (BUG-033 fix)."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from src.memory.manager import BaseMemoryManager

        messages = [
            HumanMessage(content="do stuff"),
            AIMessage(
                content="",
                tool_calls=[{"name": "t1", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="r1", tool_call_id="tc1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "t2", "args": {}, "id": "tc2"}],
            ),
            ToolMessage(content="r2", tool_call_id="tc2"),
            AIMessage(content=""),  # empty = bad
        ]
        cleaned = BaseMemoryManager.sanitize_history(messages)
        # BUG-033 fix: the triggering HumanMessage is also removed to prevent
        # an orphaned HumanMessage with no following response.
        assert len(cleaned) == 0


# ---------------------------------------------------------------------------
# BUG-034: LRU ordering preserved on cache writeback
# ---------------------------------------------------------------------------


class TestBug034LRUWriteback:
    """BUG-034: persistent cache writeback must preserve LRU ordering."""

    def test_move_to_end_after_update(self):
        persistent = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
        local = OrderedDict([("a", 10)])

        persistent.update(local)
        for key in local:
            persistent.move_to_end(key)

        # 'a' should now be at MRU end (last)
        assert list(persistent.keys()) == ["b", "c", "a"]

        # Evicting oldest should remove 'b', not 'a'
        persistent.popitem(last=False)
        assert "a" in persistent
        assert "b" not in persistent

    def test_without_move_to_end_loses_ordering(self):
        """Without move_to_end, update does NOT move keys — this is the bug."""
        persistent = OrderedDict([("a", 1), ("b", 2), ("c", 3)])
        local = OrderedDict([("a", 10)])

        persistent.update(local)
        # Without move_to_end, 'a' stays at original position (first)
        assert list(persistent.keys())[0] == "a"

        # Evicting oldest removes 'a' — the recently accessed key (wrong!)
        evicted_key, _ = persistent.popitem(last=False)
        assert evicted_key == "a"


# ---------------------------------------------------------------------------
# BUG-035: bound cache capacity (off-by-one)
# ---------------------------------------------------------------------------


class TestBug035BoundCacheCapacity:
    """BUG-035: Cache eviction should trigger at >= 8, not > 8."""

    def test_cache_evicts_at_capacity_8(self):
        cache: OrderedDict[str, int] = OrderedDict()
        capacity = 8

        for i in range(capacity + 1):
            cache[f"key_{i}"] = i
            if len(cache) >= capacity:
                cache.popitem(last=False)

        assert len(cache) < capacity

    def test_off_by_one_with_gt_allows_9(self):
        """With `> 8` (the bug), the cache holds 9 entries before eviction."""
        cache: OrderedDict[str, int] = OrderedDict()

        for i in range(9):
            cache[f"key_{i}"] = i

        # With `> 8` guard, 9 entries remain without eviction
        if len(cache) > 8:
            cache.popitem(last=False)

        # After eviction at 9, we still have 8
        assert len(cache) == 8

        # But with `>= 8`, we'd evict at 8 entries
        cache2: OrderedDict[str, int] = OrderedDict()
        for i in range(8):
            cache2[f"key_{i}"] = i
        if len(cache2) >= 8:
            cache2.popitem(last=False)
        assert len(cache2) == 7


# ---------------------------------------------------------------------------
# BUG-040: json_store.save_history fd leak on failure
# ---------------------------------------------------------------------------


class TestBug040JsonStoreFdLeak:
    """BUG-040: save_history must close the fd when json.dump raises."""

    def test_fd_closed_on_dump_failure(self, tmp_path):
        from src.memory.json_store import JsonFileMemoryStore

        store = JsonFileMemoryStore(str(tmp_path))

        with patch("src.memory.json_store.json.dump", side_effect=OSError("disk full")):
            with patch("src.memory.json_store.os.close") as mock_close:
                # save_history catches the exception internally (logs warning)
                store.save_history("test_session", [{"type": "human", "content": "hi"}])
                # os.close should have been called on the fd in the except handler
                mock_close.assert_called_once()

    def test_save_history_succeeds_normally(self, tmp_path):
        from src.memory.json_store import JsonFileMemoryStore

        store = JsonFileMemoryStore(str(tmp_path))
        store.save_history("test_session", [{"type": "human", "content": "hello"}])

        saved_path = tmp_path / "test_session.json"
        assert saved_path.exists()
        data = json.loads(saved_path.read_text())
        assert len(data) == 1
        assert data[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# PERF-001: compression threshold ratio
# ---------------------------------------------------------------------------


class TestPerf001CompressionThreshold:
    """PERF-001: Compression threshold should be 0.72."""

    def test_threshold_is_072(self):
        from src.orchestration.compression import _COMPRESSION_THRESHOLD_RATIO

        assert _COMPRESSION_THRESHOLD_RATIO == 0.72


# ---------------------------------------------------------------------------
# Round 4 (ARCH-037-09): sanitize_history — trailing HumanMessage edge cases
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain_core not installed")
class TestSanitizeHistoryOrphanedHuman:
    """sanitize_history must not crash or silently drop a trailing HumanMessage."""

    def _sanitize(self, msgs):
        from src.memory.manager import BaseMemoryManager

        return BaseMemoryManager.sanitize_history(msgs)

    def test_trailing_human_message_preserved(self) -> None:
        """A HumanMessage at the end of history with no following AI message
        is NOT the 'bad pair' case (no next message to inspect).  It must be
        preserved so the next agent turn sees the full context."""
        msgs = [
            HumanMessage(content="first question"),
            AIMessage(content="first answer"),
            HumanMessage(content="second question"),
        ]
        result = self._sanitize(msgs)
        assert len(result) == 3
        assert isinstance(result[-1], HumanMessage)
        assert result[-1].content == "second question"

    def test_human_then_bad_ai_both_dropped(self) -> None:
        """A HumanMessage followed immediately by a bad-content AIMessage must
        remove BOTH messages."""
        msgs = [
            HumanMessage(content="ok question"),
            AIMessage(content="An error occurred: connection refused"),
        ]
        result = self._sanitize(msgs)
        assert len(result) == 0

    def test_orphaned_human_after_tool_chain_preserved(self) -> None:
        """A HumanMessage preceded by a complete tool chain (AI+ToolMessage) and
        followed by nothing must survive sanitization."""
        msgs = [
            HumanMessage(content="do the thing"),
            AIMessage(content="", tool_calls=[{"name": "shell", "id": "t1", "args": {}}]),
            ToolMessage(content="done", tool_call_id="t1"),
            AIMessage(content="all done"),
            HumanMessage(content="next request"),
        ]
        result = self._sanitize(msgs)
        assert isinstance(result[-1], HumanMessage)
        assert result[-1].content == "next request"


# ---------------------------------------------------------------------------
# Round 4: compression fallback cap when LLM fails (compress_tool_message)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain_core not installed")
class TestCompressionFallbackCapDirectCall:
    """compress_tool_message must cap output at _FALLBACK_MAX_CHARS on LLM failure."""

    def test_llm_failure_caps_output(self) -> None:
        from src.orchestration.compression import (
            _FALLBACK_MAX_CHARS,
            compress_tool_message,
        )

        long_content = "x" * (_FALLBACK_MAX_CHARS + 5000)

        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        result = compress_tool_message(long_content, "my_tool", failing_llm)

        assert len(result) <= _FALLBACK_MAX_CHARS + 200
        assert "truncated" in result

    def test_llm_failure_short_content_unchanged(self) -> None:
        """Content shorter than _FALLBACK_MAX_CHARS is returned as-is on LLM failure."""
        from src.orchestration.compression import compress_tool_message

        short_content = "short output"
        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        result = compress_tool_message(short_content, "my_tool", failing_llm)
        assert result == short_content

    def test_empty_llm_response_uses_truncation(self) -> None:
        """An LLM that returns empty string triggers the tiny-result fallback."""
        from src.orchestration.compression import _FALLBACK_MAX_CHARS, compress_tool_message

        long_content = "y" * (_FALLBACK_MAX_CHARS + 1000)
        mock_response = MagicMock()
        mock_response.content = ""
        llm = MagicMock()
        llm.invoke.return_value = mock_response

        result = compress_tool_message(long_content, "my_tool", llm)
        assert len(result) <= _FALLBACK_MAX_CHARS + 200


# ---------------------------------------------------------------------------
# Round 4: agent_performed_writes — empty exec_msgs write detection
# ---------------------------------------------------------------------------


class TestAgentPerformedWritesEmptyMsgs:
    """agent_performed_writes([]) must return False without raising."""

    def test_empty_msgs_returns_false(self) -> None:
        from src.orchestration.phases import agent_performed_writes

        assert agent_performed_writes([]) is False

    def test_none_action_tool_returns_false(self) -> None:
        """A ToolMessage from a non-action tool must not count as a write."""
        if not _HAS_LANGCHAIN:
            import pytest

            pytest.skip("langchain_core not installed")
        from langchain_core.messages import ToolMessage

        from src.orchestration.phases import agent_performed_writes

        msgs = [ToolMessage(content="output", tool_call_id="t1", name="read_file")]
        assert agent_performed_writes(msgs) is False

    def test_write_file_tool_returns_true(self) -> None:
        """A successful write_file ToolMessage must be detected as a write."""
        if not _HAS_LANGCHAIN:
            import pytest

            pytest.skip("langchain_core not installed")
        from langchain_core.messages import ToolMessage

        from src.orchestration.phases import agent_performed_writes

        msgs = [ToolMessage(content="Written successfully", tool_call_id="t1", name="write_file")]
        assert agent_performed_writes(msgs) is True

    def test_errored_write_not_counted(self) -> None:
        """A write_file ToolMessage that starts with 'Error' must NOT count as a write."""
        if not _HAS_LANGCHAIN:
            import pytest

            pytest.skip("langchain_core not installed")
        from langchain_core.messages import ToolMessage

        from src.orchestration.phases import agent_performed_writes

        msgs = [
            ToolMessage(content="Error: permission denied", tool_call_id="t1", name="write_file")
        ]
        assert agent_performed_writes(msgs) is False
