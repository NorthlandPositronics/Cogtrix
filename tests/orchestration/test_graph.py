"""Tests for src/orchestration/graph.py helper functions."""

from types import SimpleNamespace

from src.orchestration.graph import _detect_tool_request


class TestDetectToolRequest:
    """Unit tests for _detect_tool_request."""

    def _ai_msg(self, tool_calls):
        """Create a fake AIMessage with given tool_calls."""
        return SimpleNamespace(tool_calls=tool_calls)

    def _tool_msg(self, content="ok", name="request_tools", tool_call_id="tc1"):
        """Create a fake ToolMessage (no tool_calls attribute)."""
        return SimpleNamespace(content=content, name=name, tool_call_id=tool_call_id)

    def test_returns_none_for_empty_messages(self):
        assert _detect_tool_request([], start_idx=0) is None

    def test_extracts_add_from_ai_message(self):
        ai = self._ai_msg(
            [{"name": "request_tools", "args": {"add": ["search_web", "http_get"]}, "id": "tc1"}]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["search_web", "http_get"]
        assert result.remove == []

    def test_extracts_remove_from_ai_message(self):
        ai = self._ai_msg([{"name": "request_tools", "args": {"remove": ["shell"]}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.remove == ["shell"]

    def test_legacy_names_fallback(self):
        ai = self._ai_msg(
            [{"name": "request_tools", "args": {"names": ["calculator"]}, "id": "tc1"}]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["calculator"]

    def test_ignores_tool_messages(self):
        """Regression test for BUG-076: ToolMessages have no tool_calls attribute.

        _detect_tool_request must be called with the AIMessage that contains
        tool_calls, NOT with ToolMessage results.  If only ToolMessages are
        passed, the function must return None.
        """
        tool_msg = self._tool_msg(content="Tools loaded: search_web. They are now active.")
        result = _detect_tool_request([tool_msg], start_idx=0)
        assert result is None

    def test_ai_message_with_non_request_tools_ignored(self):
        ai = self._ai_msg([{"name": "search_web", "args": {"query": "test"}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        assert result is None

    def test_mixed_add_and_remove(self):
        ai = self._ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": ["http_get"], "remove": ["calculator"]},
                    "id": "tc1",
                }
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["http_get"]
        assert result.remove == ["calculator"]
        assert result.has_changes is True

    def test_start_idx_skips_earlier_messages(self):
        ai1 = self._ai_msg([{"name": "request_tools", "args": {"add": ["shell"]}, "id": "tc1"}])
        ai2 = self._ai_msg(
            [{"name": "request_tools", "args": {"add": ["calculator"]}, "id": "tc2"}]
        )
        result = _detect_tool_request([ai1, ai2], start_idx=1)
        assert result is not None
        assert result.add == ["calculator"]

    def test_multiple_request_tools_calls_in_single_message(self):
        """GAP-5: Multiple parallel request_tools calls are aggregated."""
        ai = self._ai_msg(
            [
                {"name": "request_tools", "args": {"add": ["tool_a"]}, "id": "tc1"},
                {"name": "request_tools", "args": {"add": ["tool_b"]}, "id": "tc2"},
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["tool_a", "tool_b"]
        assert result.has_changes is True

    def test_mixed_request_tools_and_regular_calls(self):
        """Only request_tools calls are extracted; regular tool calls are ignored."""
        ai = self._ai_msg(
            [
                {"name": "search_web", "args": {"query": "test"}, "id": "tc1"},
                {"name": "request_tools", "args": {"add": ["calculator"]}, "id": "tc2"},
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["calculator"]

    def test_empty_add_and_remove_returns_none(self):
        """request_tools with empty lists returns None (no changes)."""
        ai = self._ai_msg(
            [{"name": "request_tools", "args": {"add": [], "remove": []}, "id": "tc1"}]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is None
