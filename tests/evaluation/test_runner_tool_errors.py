"""Unit tests for the Bug L follow-up: tool errors must fail eval scenarios.

Before this change, the runner's pass calculation only checked
``task_completion`` (required tools called) and ``per_turn_failed``
(success_criteria match). A scenario where ``http_get`` raised a
pydantic ValidationError and the model produced a graceful "could not
find" answer was being reported as passed because the failure was
invisible to text-based success_criteria.

The fix introduces ``EvalResult.tool_errors`` and the
``_collect_tool_errors`` helper, and folds the result into the pass
calculation. Gate 2's ``_final_passed`` honours the new field as a
hard veto.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.evaluation.runner import EvalResult, _collect_tool_errors


class TestCollectToolErrors:
    def test_empty_messages_returns_empty(self) -> None:
        assert _collect_tool_errors([]) == []

    def test_clean_run_returns_empty(self) -> None:
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"q": "x"}, "id": "c1"}],
            ),
            ToolMessage(content="result 1\nresult 2", tool_call_id="c1", name="search_web"),
            AIMessage(content="found it"),
        ]
        assert _collect_tool_errors(msgs) == []

    def test_detects_error_executing_wrap(self) -> None:
        # Canonical shape emitted by orchestration/graph.py when a tool
        # raises (e.g. a pydantic ValidationError on http_get headers).
        msgs = [
            HumanMessage(content="fetch X"),
            AIMessage(
                content="",
                tool_calls=[{"name": "http_get", "args": {"url": "https://x"}, "id": "c1"}],
            ),
            ToolMessage(
                content="Error executing http_get: 1 validation error for HttpGetInput",
                tool_call_id="c1",
                name="http_get",
            ),
            AIMessage(content="could not find it"),
        ]
        errors = _collect_tool_errors(msgs)
        assert len(errors) == 1
        assert "http_get" in errors[0]
        assert "validation error" in errors[0].lower()

    def test_detects_tool_returned_error_prefix(self) -> None:
        # Some tools handle their own errors and return "Error: ..."
        # rather than raising — e.g. http_get on a timeout. Still a
        # failure from the scenario's perspective.
        msgs = [
            HumanMessage(content="fetch"),
            AIMessage(
                content="",
                tool_calls=[{"name": "http_get", "args": {"url": "https://x"}, "id": "c1"}],
            ),
            ToolMessage(
                content="Error: Request timed out after 30 seconds",
                tool_call_id="c1",
                name="http_get",
            ),
        ]
        errors = _collect_tool_errors(msgs)
        assert len(errors) == 1
        assert errors[0].startswith("http_get:")

    def test_detects_error_inside_duplicate_cache_wrap(self) -> None:
        # When the orchestration caches a prior error and the model
        # retries, the cached content gets wrapped in a "[Duplicate
        # call ...]" prefix. The error is still real.
        msgs = [
            ToolMessage(
                content=(
                    "[Duplicate call — returning cached result. Do NOT repeat this call.]"
                    "\n\nError executing http_get: validation error"
                ),
                tool_call_id="c1",
                name="http_get",
            ),
        ]
        errors = _collect_tool_errors(msgs)
        assert len(errors) == 1

    def test_does_not_match_legitimate_error_in_payload(self) -> None:
        # A tool that legitimately returns content mentioning "error"
        # without an error prefix must NOT trigger. Example: a search
        # result that quotes a doc title containing the word "error".
        msgs = [
            ToolMessage(
                content="Top result: 'How to debug a 500 error in production'",
                tool_call_id="c1",
                name="search_web",
            ),
        ]
        assert _collect_tool_errors(msgs) == []

    def test_handles_non_string_content_gracefully(self) -> None:
        # Multi-part content shapes are rare but possible. The collector
        # must not crash on them and must not produce false positives.
        msg = ToolMessage(content=[], tool_call_id="c1", name="some_tool")  # type: ignore[arg-type]
        assert _collect_tool_errors([msg]) == []

    def test_multiple_errors_collected_in_order(self) -> None:
        msgs = [
            ToolMessage(
                content="Error executing http_get: bad headers",
                tool_call_id="c1",
                name="http_get",
            ),
            ToolMessage(
                content="Error: Request timed out",
                tool_call_id="c2",
                name="http_post",
            ),
        ]
        errors = _collect_tool_errors(msgs)
        assert len(errors) == 2
        assert "http_get" in errors[0]
        assert "http_post" in errors[1]

    def test_tool_no_longer_active_counts_as_error(self) -> None:
        msgs = [
            ToolMessage(
                content="Tool 'unloaded_tool' is no longer active.",
                tool_call_id="c1",
                name="unloaded_tool",
            ),
        ]
        assert len(_collect_tool_errors(msgs)) == 1


class TestEvalResultToolErrorsField:
    def test_default_empty_list(self) -> None:
        r = EvalResult(
            scenario_id="x",
            model_id="m",
            model_display_name="M",
            passed=True,
            tool_calls_made=[],
            tool_calls_required=[],
            turns_used=0,
            elapsed_seconds=0.0,
            final_response="",
        )
        assert r.tool_errors == []

    def test_serialised_in_to_dict(self) -> None:
        r = EvalResult(
            scenario_id="x",
            model_id="m",
            model_display_name="M",
            passed=False,
            tool_calls_made=[],
            tool_calls_required=[],
            turns_used=0,
            elapsed_seconds=0.0,
            final_response="",
            tool_errors=["http_get: validation error"],
        )
        out = r.to_dict()
        assert out["tool_errors"] == ["http_get: validation error"]
