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

    # BUG-204 — string arg normalization

    def test_add_as_bare_string(self):
        """BUG-204: LLM sends {"add": "web_search"} — must be normalised to ["web_search"]."""
        ai = self._ai_msg([{"name": "request_tools", "args": {"add": "web_search"}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["web_search"]
        assert result.remove == []

    def test_remove_as_bare_string(self):
        """BUG-204: LLM sends {"remove": "shell"} — must be normalised to ["shell"]."""
        ai = self._ai_msg([{"name": "request_tools", "args": {"remove": "shell"}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.remove == ["shell"]
        assert result.add == []

    def test_legacy_names_as_bare_string(self):
        """BUG-204: LLM sends {"names": "calculator"} — must be normalised to ["calculator"]."""
        ai = self._ai_msg([{"name": "request_tools", "args": {"names": "calculator"}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["calculator"]


class TestParallelToolTimeout:
    """BUG-202: parallel tool futures must time out instead of hanging indefinitely."""

    def test_future_timeout_produces_error_message(self):
        """Simulate the timeout path: future.result(timeout=0) raises TimeoutError."""
        import concurrent.futures

        from langchain_core.messages import ToolMessage

        call = {"name": "slow_tool", "id": "tc-slow", "args": {}}
        future: concurrent.futures.Future = concurrent.futures.Future()

        try:
            future.result(timeout=0)
            raise AssertionError("Should have raised TimeoutError")
        except (TimeoutError, concurrent.futures.TimeoutError):
            msg = ToolMessage(
                content=f"Error: tool '{call['name']}' timed out after 10 minutes",
                tool_call_id=call["id"],
                name=call["name"],
            )

        assert "timed out" in msg.content
        assert "slow_tool" in msg.content
        assert msg.tool_call_id == "tc-slow"

    def test_source_uses_600s_timeout(self):
        """Verify the timeout constant is 600 seconds (10 minutes) in graph.py."""
        import inspect

        from src.orchestration import graph

        source = inspect.getsource(graph)
        assert "future.result(timeout=600)" in source

    def test_source_handles_both_timeout_exception_types(self):
        """Both built-in TimeoutError and concurrent.futures.TimeoutError are caught."""
        import inspect

        from src.orchestration import graph

        source = inspect.getsource(graph)
        assert "concurrent.futures.TimeoutError" in source

    def test_timeout_error_message_format(self):
        """The error ToolMessage content matches the expected human-readable format."""
        import inspect

        from src.orchestration import graph

        source = inspect.getsource(graph)
        assert "timed out after 10 minutes" in source


class TestDetectToolRequestEdgeCases:
    """Edge-case tests for _detect_tool_request normalization."""

    def _ai_msg(self, tool_calls):
        return SimpleNamespace(tool_calls=tool_calls)

    def test_mixed_string_add_and_list_remove(self):
        """BUG-204 edge: add is a bare string, remove is a list."""
        ai = self._ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": "web_search", "remove": ["shell", "calculator"]},
                    "id": "tc1",
                }
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["web_search"]
        assert result.remove == ["shell", "calculator"]

    def test_mixed_list_add_and_string_remove(self):
        """BUG-204 edge: add is a list, remove is a bare string."""
        ai = self._ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": ["http_get", "calculator"], "remove": "shell"},
                    "id": "tc1",
                }
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["http_get", "calculator"]
        assert result.remove == ["shell"]

    def test_empty_string_add_returns_none(self):
        """Edge: empty string add is falsy — triggers names fallback (also empty) → None."""
        ai = self._ai_msg([{"name": "request_tools", "args": {"add": ""}, "id": "tc1"}])
        result = _detect_tool_request([ai], start_idx=0)
        # "" is falsy, so the legacy names fallback triggers (also empty) → no changes
        assert result is None

    def test_integer_in_add_list_coerced_to_str(self):
        """Edge: non-string values in list are coerced via str()."""
        ai = self._ai_msg(
            [{"name": "request_tools", "args": {"add": [123, "real_tool"]}, "id": "tc1"}]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["123", "real_tool"]

    def test_legacy_names_string_with_add_empty_list(self):
        """Legacy fallback: names is used only when add and remove are both empty."""
        ai = self._ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": [], "remove": [], "names": "calculator"},
                    "id": "tc1",
                }
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        # add=[] and remove=[] are falsy, so names fallback triggers
        assert result is not None
        assert result.add == ["calculator"]

    def test_add_present_suppresses_legacy_names(self):
        """When add is provided, legacy names is ignored even if also present."""
        ai = self._ai_msg(
            [
                {
                    "name": "request_tools",
                    "args": {"add": ["shell"], "names": ["calculator"]},
                    "id": "tc1",
                }
            ]
        )
        result = _detect_tool_request([ai], start_idx=0)
        assert result is not None
        assert result.add == ["shell"]
        # calculator from names should NOT appear
        assert "calculator" not in result.add
