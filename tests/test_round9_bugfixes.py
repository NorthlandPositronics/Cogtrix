"""Regression tests for Forge run 9 bug fixes.

BUG-120: run_message_turn() never updated session.last_activity after a turn.
BUG-121: Dead code in _run_agent() — identical return in both branches of
         a conditional guard.
BUG-122: run_message_turn() mode parameter silently ignored.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
import time
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mock_session() -> MagicMock:
    """Build a minimal ApiSession-like mock."""
    session = MagicMock()
    session.id = "sess-120"
    session.agent_state = "idle"
    session.memory_manager = None
    session.run_config = None
    session.session_state = None
    session.last_activity = time.time() - 7200.0
    session.token_counts = {"input_tokens": 0, "output_tokens": 0}
    session.ws_queue = asyncio.Queue()
    return session


def _patched_run(mode: str, session: MagicMock | None = None) -> tuple[MagicMock, list[str]]:
    """Run run_message_turn with all internal imports patched.

    Returns (session, list_of_warning_messages).
    """
    if session is None:
        session = _build_mock_session()

    captured_warnings: list[str] = []

    # Capture the log instance that turn_runner uses.
    import src.api.turn_runner as _tr_mod

    def _capture(msg: str, *args: object, **kwargs: object) -> None:
        captured_warnings.append(msg % args if args else msg)

    fake_ws_cb = MagicMock(input_tokens=5, output_tokens=3, tool_call_count=0)

    async def _run_inner() -> None:
        with (
            # Patch classes imported inside the function body
            patch("src.api.callbacks.WebSocketCallbackHandler", return_value=fake_ws_cb),
            patch("src.api.confirmation.ApiConfirmationUI", return_value=MagicMock()),
            # Patch run_agent at its definition site so the import inside
            # run_message_turn gets the patched version.
            patch("src.orchestration.runner.run_agent", return_value="ok"),
            patch.object(_tr_mod.log, "warning", side_effect=_capture),
        ):
            from src.api.turn_runner import run_message_turn

            await run_message_turn(session=session, text="hello", mode=mode)

    asyncio.run(_run_inner())
    return session, captured_warnings


# ---------------------------------------------------------------------------
# BUG-120 — session.last_activity updated after successful turn
# ---------------------------------------------------------------------------


def test_last_activity_updated_after_successful_turn() -> None:
    """session.last_activity must be refreshed at end of run_message_turn()."""
    session = _build_mock_session()
    before = session.last_activity

    _patched_run("normal", session)

    assert (
        session.last_activity > before
    ), "session.last_activity must be updated to a time after the turn started"
    assert (
        time.time() - session.last_activity < 10.0
    ), "session.last_activity must be close to the current wall-clock time"


# ---------------------------------------------------------------------------
# BUG-121 — dead code removed from _run_agent()
# ---------------------------------------------------------------------------


def test_run_agent_no_dead_code_comment() -> None:
    """The dead-code comment 'BUG-NEW' must not appear in handler.py."""
    src_text = pathlib.Path("src/assistant/handler.py").read_text()
    assert "BUG-NEW" not in src_text, (
        "The dead-code comment 'BUG-NEW' must not appear in handler.py "
        "after the fix was applied."
    )


def test_run_agent_has_exactly_two_returns() -> None:
    """_run_agent() must have exactly 2 return statements (except + normal path)."""
    src_text = pathlib.Path("src/assistant/handler.py").read_text()
    tree = ast.parse(src_text)

    method_body: list[ast.stmt] | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MessageHandler":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_run_agent":
                    method_body = item.body
                    break
            break

    assert method_body is not None, "_run_agent method not found in MessageHandler"

    all_returns: list[ast.Return] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Return(self, node: ast.Return) -> None:
            all_returns.append(node)
            self.generic_visit(node)

    for stmt in method_body:
        _Visitor().visit(stmt)

    assert len(all_returns) == 2, (
        f"Expected 2 return statements in _run_agent (except + normal), "
        f"got {len(all_returns)}. Dead code conditional may be present."
    )


# ---------------------------------------------------------------------------
# BUG-122 — mode parameter triggers warnings for unsupported values
# ---------------------------------------------------------------------------


def test_unknown_mode_logs_warning() -> None:
    """run_message_turn() must warn when an unknown mode string is passed."""
    _, warnings = _patched_run("badmode")
    assert any(
        "unknown mode" in w.lower() for w in warnings
    ), f"Expected 'unknown mode' warning, got: {warnings}"


def test_think_mode_invokes_pipeline() -> None:
    """run_message_turn() mode='think' must invoke the deep-think pipeline."""
    import src.api.turn_runner as _tr_mod

    with patch.object(_tr_mod, "_run_think_pipeline", return_value="deep result") as mock_think:
        _patched_run("think")
        mock_think.assert_called_once()


def test_delegate_mode_invokes_pipeline() -> None:
    """run_message_turn() mode='delegate' must invoke the delegation pipeline."""
    import src.api.turn_runner as _tr_mod

    with patch.object(
        _tr_mod, "_run_delegate_pipeline", return_value="delegated result"
    ) as mock_del:
        _patched_run("delegate")
        mock_del.assert_called_once()


def test_normal_mode_no_mode_warning() -> None:
    """run_message_turn() must NOT log any mode-related warning for mode='normal'."""
    _, warnings = _patched_run("normal")
    mode_warnings = [w for w in warnings if "mode" in w.lower()]
    assert (
        not mode_warnings
    ), f"No mode-related warnings expected for mode='normal', got: {mode_warnings}"
