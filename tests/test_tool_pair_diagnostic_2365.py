"""#2365 — tool-pair provider-400 diagnostic capture (diagnostic-first, no repro yet).

The set-based repair (`_repair_tool_message_pairs`) can report a message list clean
while the provider still 400s on an unanswered ``tool_call_id`` — because the repair's
view is per-set and the provider's is per-occurrence, and because native-Kimi colon ids
(``name:idx``) vs Cogtrix underscore ids (``name_idx``) can diverge between a declaration
and its answer. Before rewriting that critical path, we capture the exact structure:

* ``_is_tool_pair_mismatch_error`` — narrowly recognise the provider 400 so the dump only
  fires on the real failure.
* ``_format_tool_pair_diagnostic`` — an ordered, content-free structure dump that surfaces
  the two candidate mechanisms (duplicate declaration; id-form mismatch) directly.
"""

from __future__ import annotations

import os

os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from cogtrix_core.orchestration.graph import _is_tool_pair_mismatch_error  # noqa: E402
from cogtrix_core.orchestration.message_repair import (  # noqa: E402
    _format_tool_pair_diagnostic,
)


def _ai(tool_calls, content: str = ""):
    return AIMessage(content=content, tool_calls=tool_calls)


class TestIsToolPairMismatchError:
    def test_native_kimi_phrasing_matches(self) -> None:
        exc = Exception(
            "Error code: 400 - {'error': {'message': \"Invalid request: an assistant "
            "message with 'tool_calls' must be followed by tool messages responding to "
            "each 'tool_call_id'. The following tool_call_ids did not have response "
            "messages: execute_shell_command:11\", 'type': 'invalid_request_error'}}"
        )
        assert _is_tool_pair_mismatch_error(exc) is True

    def test_openai_role_tool_phrasing_matches(self) -> None:
        exc = Exception(
            "messages with role 'tool' must be a response to a preceeding message "
            "with 'tool_calls'"
        )
        assert _is_tool_pair_mismatch_error(exc) is True

    def test_generic_400_does_not_match(self) -> None:
        assert _is_tool_pair_mismatch_error(Exception("400 Bad Request: invalid model")) is False

    def test_context_overflow_does_not_match(self) -> None:
        assert (
            _is_tool_pair_mismatch_error(Exception("This model's maximum context length is 8192"))
            is False
        )


class TestFormatToolPairDiagnostic:
    def test_duplicate_declaration_answered_once_is_surfaced(self) -> None:
        """A tool_call_id declared by TWO AIMessages but answered ONCE — the exact
        set-vs-occurrence gap. A plain declared−answered set difference reports it
        answered; the per-occurrence view must flag it."""
        msgs = [
            HumanMessage(content="go"),
            _ai([{"name": "execute_shell_command", "args": {}, "id": "esc:11"}]),
            ToolMessage(content="ok", tool_call_id="esc:11"),
            _ai([{"name": "execute_shell_command", "args": {}, "id": "esc:11"}]),
        ]
        dump = _format_tool_pair_diagnostic(msgs)

        assert "duplicate_declarations=['esc:11']" in dump
        # The 2nd occurrence has no following answer → per-occurrence unanswered.
        assert "unanswered_by_occurrence=['esc:11']" in dump
        # Structure only — no message content leaked into the dump.
        assert "go" not in dump and "ok" not in dump

    def test_id_form_mismatch_is_surfaced(self) -> None:
        """Declared with a colon (native Kimi), answered with an underscore
        (Cogtrix-synth): the declaration reads unanswered and the answer reads
        orphaned — the id-form hazard, made visible."""
        msgs = [
            _ai([{"name": "execute_shell_command", "args": {}, "id": "execute_shell_command:11"}]),
            ToolMessage(content="result", tool_call_id="execute_shell_command_11"),
        ]
        dump = _format_tool_pair_diagnostic(msgs)

        assert "unanswered_by_occurrence=['execute_shell_command:11']" in dump
        assert "orphan_answers=['execute_shell_command_11']" in dump

    def test_clean_history_reports_no_issues(self) -> None:
        msgs = [
            _ai([{"name": "read_file", "args": {}, "id": "call_a"}]),
            ToolMessage(content="contents", tool_call_id="call_a"),
        ]
        dump = _format_tool_pair_diagnostic(msgs)

        assert "unanswered_by_occurrence=none" in dump
        assert "duplicate_declarations=none" in dump
        assert "orphan_answers=none" in dump

    def test_dump_lists_every_message_in_order(self) -> None:
        msgs = [
            HumanMessage(content="x"),
            _ai([{"name": "f", "args": {}, "id": "c1"}]),
            ToolMessage(content="y", tool_call_id="c1"),
        ]
        dump = _format_tool_pair_diagnostic(msgs)
        lines = dump.splitlines()
        # header + one line per message
        assert len(lines) == 1 + len(msgs)
        assert "[0] HumanMessage" in dump
        assert "[1] AIMessage declares=['c1']" in dump
        assert "[2] ToolMessage answers=c1" in dump
