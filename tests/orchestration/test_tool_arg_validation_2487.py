"""Regression tests for #2487.

When the model emits tool-call arguments that fail the tool's pydantic schema,
``DedupedToolInvoker.invoke_one`` used to log a raw ``ValidationError`` stack
trace (`exc_info=True`) — noise that buries genuine tool crashes during triage.
Such a failure is the model's mistake (recoverable: it retries with corrected
args), not a Cogtrix bug, so it should log at WARNING WITHOUT a traceback and
return a clear "invalid arguments" steer. Genuine tool exceptions keep the full
traceback.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from pydantic import BaseModel, ValidationError

from tests.orchestration.test_deduped_tool_invoker import _make_invoker


def _a_validation_error() -> ValidationError:
    class _Schema(BaseModel):
        n: int

    try:
        _Schema(n="not-an-int")  # type: ignore[arg-type]
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


def _invoke_raising(exc: BaseException) -> tuple[ToolMessage, MagicMock]:
    """Drive one invoke_one() whose tool raises ``exc``; return (result, fake_log)."""
    tool = MagicMock()
    tool.name = "x"
    tool.invoke.side_effect = exc
    invoker, _, _, _ = _make_invoker(tool=tool, tool_name="x")
    fake_log = MagicMock()
    with patch(
        "cogtrix_core.orchestration.deduped_tool_invoker.get_logger",
        return_value=fake_log,
    ):
        result = invoker.invoke_one({"name": "x", "args": {"a": 1}, "id": "c1"}, None)
    assert isinstance(result, ToolMessage)
    return result, fake_log


class TestToolArgValidationNoise:
    def test_validation_error_returns_invalid_args_without_traceback(self) -> None:
        result, log = _invoke_raising(_a_validation_error())
        content = str(result.content)
        assert "Invalid arguments for x" in content
        assert "retry" in content.lower()
        # a model-side arg error must NOT emit a stack trace
        _, kwargs = log.warning.call_args
        assert kwargs.get("exc_info") in (None, False)

    def test_tool_exception_also_treated_as_invalid_args(self) -> None:
        result, log = _invoke_raising(ToolException("bad tool input"))
        assert "Invalid arguments for x" in result.content
        _, kwargs = log.warning.call_args
        assert kwargs.get("exc_info") in (None, False)

    def test_generic_exception_keeps_error_message_and_traceback(self) -> None:
        result, log = _invoke_raising(RuntimeError("boom"))
        assert "Error executing x" in result.content
        assert "Invalid arguments" not in result.content
        # a genuine tool crash keeps the full traceback for triage
        _, kwargs = log.warning.call_args
        assert kwargs.get("exc_info") is True

    def test_no_exception_propagates_for_any_arm(self) -> None:
        for exc in (_a_validation_error(), ToolException("x"), RuntimeError("y")):
            result, _ = _invoke_raising(exc)
            assert isinstance(result, ToolMessage)
