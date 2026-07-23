"""#2267 — extend_run tool messages must not induce announce-and-stop.

The original false-promise recurrence (#2258 → #2267) was driven by the tool's
own return text telling the agent that work would happen "after this run", so the
agent registered an extension, announced "results coming later", and ended the
turn. The messages must instead make clear that both modes complete in THIS turn,
and an empty-subtasks delegate must clearly NOT register an extension.
"""

from __future__ import annotations

from cogtrix_core.tools.extend_run import ExtendRunState, create_extend_run_tool


def _invoke(state: ExtendRunState, **kwargs) -> str:
    tool = create_extend_run_tool(state)
    assert tool is not None
    return tool.invoke(kwargs)


class TestExtendRunMessaging:
    def test_continue_message_promises_this_turn_not_later(self) -> None:
        state = ExtendRunState()
        msg = _invoke(state, mode="continue", reason="need more steps")
        low = msg.lower()
        assert "this turn" in low
        assert "no separate later run" in low  # honest framing
        assert "after this run" not in low  # the old misleading phrase is gone
        assert state.requested is True and state.mode == "continue"

    def test_delegate_message_promises_this_turn_not_later(self) -> None:
        state = ExtendRunState()
        msg = _invoke(state, mode="delegate", subtasks=["check A", "check B"], reason="r")
        low = msg.lower()
        assert "this turn" in low
        assert "no separate later run" in low  # honest framing
        assert "after this run" not in low  # the old misleading phrase is gone
        assert state.requested is True and state.subtasks == ["check A", "check B"]

    def test_empty_subtasks_delegate_does_not_register(self) -> None:
        # The malformed delegate must NOT leave the agent thinking work is queued.
        state = ExtendRunState()
        msg = _invoke(state, mode="delegate", subtasks=[], reason="r")
        assert msg.lower().startswith("error")
        assert "no extension" in msg.lower()
        assert state.requested is False
