"""Unit tests for src/assistant/handler.py — MessageHandler."""

from __future__ import annotations

import time
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from src.assistant.channel import IncomingMessage, SendResult
from src.assistant.handler import _DEFAULT_EXCLUDED, MessageHandler
from src.memory.context import MemoryContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(text: str = "Hello") -> IncomingMessage:
    return IncomingMessage(
        channel="telegram",
        chat_id="42",
        message_id="m1",
        sender_id="u1",
        sender_name="Alice",
        text=text,
        timestamp=time.time(),
    )


def _make_session(context_prefix: str | None = None) -> MagicMock:
    session = MagicMock()
    session.session_key = "telegram::42"
    session.lock = MagicMock()
    session.lock.__enter__ = MagicMock(return_value=None)
    session.lock.__exit__ = MagicMock(return_value=False)
    session.guardrail_violations = 0
    session.last_sent_message_id = None
    session.memory_manager.prepare_context.return_value = MemoryContext(
        messages=[],
        context_prefix=context_prefix,
    )
    return session


def _make_handler(
    config: dict | None = None,
    knowledge_store: MagicMock | None = None,
    available_tools: dict | None = None,
    active_tools: list | None = None,
    agent_runner: Callable | None = None,
) -> tuple[MessageHandler, MagicMock]:
    """Return (handler, mock_session_mgr)."""
    session = _make_session()
    session_mgr = MagicMock()
    session_mgr.get_or_create.return_value = session

    if agent_runner is None:
        agent_runner = MagicMock(return_value="")

    handler = MessageHandler(
        session_mgr=session_mgr,
        config=config or {},
        llm=MagicMock(),
        system_prompt="You are helpful.",
        registry=MagicMock(),
        approvals={"*"},
        available_tools=available_tools or {},
        active_tools=active_tools or [],
        knowledge_store=knowledge_store,
        agent_runner=agent_runner,
    )
    return handler, session_mgr


# ---------------------------------------------------------------------------
# TestToolExclusion
# ---------------------------------------------------------------------------


class TestToolExclusion:
    """Tests for messaging tool filtering in MessageHandler.__init__."""

    def test_default_excluded_tools_removed_from_available(self):
        """Default excluded tools are not present in _available_tools."""
        excluded_tool = MagicMock()
        excluded_tool.name = "whatsapp_send"
        safe_tool = MagicMock()
        safe_tool.name = "web_search"

        handler, _ = _make_handler(
            available_tools={
                "whatsapp_send": excluded_tool,
                "web_search": safe_tool,
            }
        )

        assert "whatsapp_send" not in handler._available_tools
        assert "web_search" in handler._available_tools

    def test_all_default_excluded_tools_filtered(self):
        """All tools in _DEFAULT_EXCLUDED are removed."""
        available = {name: MagicMock() for name in _DEFAULT_EXCLUDED}
        available["safe_tool"] = MagicMock()

        handler, _ = _make_handler(available_tools=available)

        for name in _DEFAULT_EXCLUDED:
            assert name not in handler._available_tools
        assert "safe_tool" in handler._available_tools

    def test_default_excluded_tools_removed_from_active(self):
        """Default excluded tools are removed from _active_tools."""
        excluded = MagicMock()
        excluded.name = "execute_shell_command"
        safe = MagicMock()
        safe.name = "web_search"

        handler, _ = _make_handler(active_tools=[excluded, safe])

        active_names = [getattr(t, "name", None) for t in handler._active_tools]
        assert "execute_shell_command" not in active_names
        assert "web_search" in active_names

    def test_custom_exclusions_applied(self):
        """Custom excluded_tools from config are also filtered out."""
        custom_tool = MagicMock()
        custom_tool.name = "dangerous_tool"
        other_tool = MagicMock()
        other_tool.name = "ok_tool"

        handler, _ = _make_handler(
            config={"excluded_tools": ["dangerous_tool"]},
            available_tools={
                "dangerous_tool": custom_tool,
                "ok_tool": other_tool,
            },
        )

        assert "dangerous_tool" not in handler._available_tools
        assert "ok_tool" in handler._available_tools

    @pytest.mark.parametrize(
        "tool_name",
        [
            "whatsapp_send",
            "whatsapp_check",
            "whatsapp_send_image",
            "whatsapp_contacts",
            "telegram_send",
            "telegram_check",
            "telegram_send_photo",
            "telegram_contacts",
            "execute_shell_command",
            "execute_python",
            "write_file",
            "append_file",
            "read_file",
            "list_directory",
            "file_info",
            "read_pdf",
        ],
    )
    def test_default_excluded_tools_membership(self, tool_name: str):
        assert tool_name in _DEFAULT_EXCLUDED


# ---------------------------------------------------------------------------
# TestResponseTruncation
# ---------------------------------------------------------------------------


class TestResponseTruncation:
    """Tests for max_response_length truncation."""

    def test_response_within_limit_not_truncated(self):
        """Responses shorter than max_response_length are sent as-is."""
        channel = MagicMock()
        mock_runner = MagicMock(return_value="Short reply")
        handler, _ = _make_handler(config={"max_response_length": 100}, agent_runner=mock_runner)

        handler.handle(_make_msg(), channel)

        channel.send.assert_called_once_with("42", "Short reply")

    def test_response_exceeding_limit_is_truncated(self):
        """Responses longer than max_response_length are cut and appended with '...'."""
        channel = MagicMock()
        long_response = "A" * 50
        mock_runner = MagicMock(return_value=long_response)
        handler, _ = _make_handler(config={"max_response_length": 20}, agent_runner=mock_runner)

        handler.handle(_make_msg(), channel)

        sent_text = channel.send.call_args[0][1]
        assert len(sent_text) == 20
        assert sent_text.endswith("...")

    def test_truncation_keeps_correct_prefix(self):
        """Truncated response uses first (max_length - 3) characters plus '...'."""
        channel = MagicMock()
        limit = 30
        long_response = "X" * 100
        mock_runner = MagicMock(return_value=long_response)
        handler, _ = _make_handler(config={"max_response_length": limit}, agent_runner=mock_runner)

        handler.handle(_make_msg(), channel)

        sent_text = channel.send.call_args[0][1]
        assert sent_text == "X" * (limit - 3) + "..."

    def test_default_max_response_length_is_4000(self):
        """Default max_response_length is 4000 characters."""
        handler, _ = _make_handler()
        assert handler._max_response_length == 4000


# ---------------------------------------------------------------------------
# TestKnowledgeRecall
# ---------------------------------------------------------------------------


class TestKnowledgeRecall:
    """Tests for knowledge injection and extraction."""

    def test_knowledge_injected_into_context_prefix(self):
        """When knowledge is available, it is prepended to context_prefix."""
        knowledge_store = MagicMock()
        knowledge_store.recall.return_value = "- Alice: Is a vet"

        channel = MagicMock()
        session = _make_session(context_prefix="History here")
        session_mgr = MagicMock()
        session_mgr.get_or_create.return_value = session

        captured_prefix: list[str | None] = []

        def _fake_run_agent(**kwargs: object) -> str:
            captured_prefix.append(kwargs.get("context_prefix"))
            return "OK"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            knowledge_store=knowledge_store,
            agent_runner=_fake_run_agent,
        )

        handler.handle(_make_msg(), channel)

        prefix = captured_prefix[0]
        assert prefix is not None
        assert "Known facts" in prefix
        assert "Alice: Is a vet" in prefix
        assert "History here" in prefix

    def test_knowledge_injected_when_no_prior_prefix(self):
        """Knowledge section is set even when context_prefix was None."""
        knowledge_store = MagicMock()
        knowledge_store.recall.return_value = "- Bob: Likes Python"

        channel = MagicMock()
        session = _make_session(context_prefix=None)
        session_mgr = MagicMock()
        session_mgr.get_or_create.return_value = session

        captured_prefix: list[str | None] = []

        def _fake_run_agent(**kwargs: object) -> str:
            captured_prefix.append(kwargs.get("context_prefix"))
            return "OK"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            knowledge_store=knowledge_store,
            agent_runner=_fake_run_agent,
        )

        handler.handle(_make_msg(), channel)

        prefix = captured_prefix[0]
        assert prefix is not None
        assert "Bob: Likes Python" in prefix

    def test_knowledge_extraction_called_after_agent(self):
        """extract_and_store is called with user input and agent response."""
        knowledge_store = MagicMock()
        knowledge_store.recall.return_value = None

        channel = MagicMock()
        mock_runner = MagicMock(return_value="Agent says hi")
        handler, _ = _make_handler(knowledge_store=knowledge_store, agent_runner=mock_runner)

        handler.handle(_make_msg(text="User says hello"), channel)

        knowledge_store.extract_and_store.assert_called_once_with(
            "User says hello", "Agent says hi"
        )

    def test_no_knowledge_store_runs_without_error(self):
        """handler.handle() works fine when knowledge_store is None."""
        channel = MagicMock()
        mock_runner = MagicMock(return_value="Fine")
        handler, _ = _make_handler(knowledge_store=None, agent_runner=mock_runner)

        handler.handle(_make_msg(), channel)

        channel.send.assert_called_once()

    def test_recall_none_leaves_prefix_unchanged(self):
        """When recall() returns None, the context_prefix from memory is used unchanged."""
        knowledge_store = MagicMock()
        knowledge_store.recall.return_value = None

        channel = MagicMock()
        session = _make_session(context_prefix="Memory prefix")
        session_mgr = MagicMock()
        session_mgr.get_or_create.return_value = session

        captured_prefix: list[str | None] = []

        def _fake_run_agent(**kwargs: object) -> str:
            captured_prefix.append(kwargs.get("context_prefix"))
            return "OK"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            knowledge_store=knowledge_store,
            agent_runner=_fake_run_agent,
        )

        handler.handle(_make_msg(), channel)

        assert captured_prefix[0] == "Memory prefix"


# ---------------------------------------------------------------------------
# TestAgentErrorHandling
# ---------------------------------------------------------------------------


class TestAgentErrorHandling:
    """Tests for agent error recovery and channel.send() call."""

    def test_agent_error_sends_fallback_message(self):
        """When run_agent raises, a fallback error message is sent via channel."""
        channel = MagicMock()
        mock_runner = MagicMock(side_effect=RuntimeError("LLM down"))
        handler, _ = _make_handler(agent_runner=mock_runner)

        handler.handle(_make_msg(), channel)

        channel.send.assert_called_once()
        sent_text = channel.send.call_args[0][1]
        assert "error" in sent_text.lower()

    def test_successful_response_sent_via_channel(self):
        """channel.send() is called with the agent's response."""
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)
        mock_runner = MagicMock(return_value="Here is the answer")
        handler, _ = _make_handler(agent_runner=mock_runner)

        handler.handle(_make_msg(), channel)

        channel.send.assert_called_once_with("42", "Here is the answer")

    def test_failed_send_logged_but_does_not_raise(self):
        """When channel.send() returns False, handle() does not raise."""
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=False)
        mock_runner = MagicMock(return_value="Reply")
        handler, _ = _make_handler(agent_runner=mock_runner)

        handler.handle(_make_msg(), channel)

        channel.send.assert_called_once()

    def test_memory_update_called_after_agent(self):
        """session.memory_manager.update() is called with user input and response."""
        channel = MagicMock()
        session = _make_session()
        session_mgr = MagicMock()
        session_mgr.get_or_create.return_value = session

        mock_runner = MagicMock(return_value="Resp")
        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=mock_runner,
        )

        handler.handle(_make_msg(text="Query"), channel)

        session.memory_manager.update.assert_called_once_with("Query", "Resp")


# ---------------------------------------------------------------------------
# TestSendBeforeMemoryUpdate (BUG-038)
# ---------------------------------------------------------------------------


class TestSendBeforeMemoryUpdate:
    """BUG-038: channel.send() must be called before memory_manager.update()."""

    def test_send_called_before_memory_update(self):
        """channel.send is called before memory_manager.update in handle()."""
        call_order: list[str] = []

        channel = MagicMock()
        channel.send.side_effect = lambda *_: call_order.append("send") or SendResult(ok=True)

        session = _make_session()
        session.memory_manager.update.side_effect = lambda *_: call_order.append("update")
        session_mgr = MagicMock()
        session_mgr.get_or_create.return_value = session

        mock_runner = MagicMock(return_value="Hello")
        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=mock_runner,
        )

        handler.handle(_make_msg(), channel)

        assert call_order.index("send") < call_order.index("update")

    def test_send_called_before_knowledge_store_extract(self):
        """channel.send is called before knowledge_store.extract_and_store()."""
        call_order: list[str] = []

        channel = MagicMock()
        channel.send.side_effect = lambda *_: call_order.append("send") or SendResult(ok=True)

        knowledge_store = MagicMock()
        knowledge_store.recall.return_value = None
        knowledge_store.extract_and_store.side_effect = lambda *_: call_order.append("extract")

        mock_runner = MagicMock(return_value="Response")
        handler, _ = _make_handler(knowledge_store=knowledge_store, agent_runner=mock_runner)

        channel.send.side_effect = lambda *_: call_order.append("send") or SendResult(ok=True)

        handler.handle(_make_msg(), channel)

        assert call_order.index("send") < call_order.index("extract")


# ---------------------------------------------------------------------------
# TestGuardrailViolationBlocksKnowledgeExtraction (BUG-060)
# ---------------------------------------------------------------------------


class TestGuardrailViolationBlocksKnowledgeExtraction:
    """BUG-060: Knowledge extraction must not run for sessions with guardrail violations.

    The fix snapshots ``guardrail_violations == 0`` inside the session lock so that
    a concurrent violation recorded between the check and the extract call cannot
    allow a violating user's content into the knowledge base.
    """

    def test_no_violation_knowledge_extracted(self):
        """When guardrail_violations == 0, extract_and_store is called normally."""
        knowledge_store = MagicMock()
        knowledge_store.recall.return_value = None

        channel = MagicMock()
        mock_runner = MagicMock(return_value="Good response")
        handler, _ = _make_handler(knowledge_store=knowledge_store, agent_runner=mock_runner)

        handler.handle(_make_msg(text="Clean input"), channel)

        knowledge_store.extract_and_store.assert_called_once()

    def test_guardrail_violation_blocks_knowledge_extraction(self):
        """When guardrail_violations > 0, extract_and_store must NOT be called."""
        knowledge_store = MagicMock()
        knowledge_store.recall.return_value = None

        channel = MagicMock()
        mock_runner = MagicMock(return_value="Response")

        session = _make_session()
        session.guardrail_violations = 1  # pre-existing violation
        session_mgr = MagicMock()
        session_mgr.get_or_create.return_value = session

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            knowledge_store=knowledge_store,
            agent_runner=mock_runner,
        )

        handler.handle(_make_msg(text="Violating input"), channel)

        knowledge_store.extract_and_store.assert_not_called()

    def test_sanitized_input_passed_to_extraction(self):
        """sanitize_output is called on the user input before knowledge extraction.

        The handler calls sanitize_output(msg.text) inside the session lock to
        compute sanitized_for_knowledge, and that sanitized value is passed to
        extract_and_store as the first argument (BUG-060 fix).
        """
        from unittest.mock import patch

        knowledge_store = MagicMock()
        knowledge_store.recall.return_value = None

        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)
        mock_runner = MagicMock(return_value="Response")
        handler, _ = _make_handler(knowledge_store=knowledge_store, agent_runner=mock_runner)

        raw_input = "raw user input"

        sanitize_calls: list[str] = []

        def _tracking_sanitize(text: str) -> str:
            sanitize_calls.append(text)
            return f"sanitized({text})"

        with patch.object(handler._guardrails, "sanitize_output", side_effect=_tracking_sanitize):
            handler.handle(_make_msg(text=raw_input), channel)

        # sanitize_output must have been called with the raw user input
        assert (
            raw_input in sanitize_calls
        ), f"sanitize_output was not called with raw input; calls: {sanitize_calls}"
        # extract_and_store must have received the sanitized version as first arg
        first_arg = knowledge_store.extract_and_store.call_args[0][0]
        assert (
            first_arg == f"sanitized({raw_input})"
        ), f"extract_and_store received '{first_arg}', expected 'sanitized({raw_input})'"


# ---------------------------------------------------------------------------
# TestBug110EditFailFallback (BUG-110)
# ---------------------------------------------------------------------------


class TestBug110EditFailFallback:
    """BUG-110: When edit_last_reply's channel.edit_message() returns ok=False and
    there is no schedule/queue activity, the handler must fall back to channel.send()
    so the user receives a reply rather than silence."""

    def _make_handler_with_edit_state(self) -> tuple:
        """Return (handler, session, channel) wired for edit_last_reply scenarios."""
        session = _make_session()
        session.last_sent_message_id = "msg-prev-1"
        session_mgr = MagicMock()
        session_mgr.get_or_create.return_value = session

        # Agent runner that sets edit_state.was_called via the closure pattern
        edit_state_holder: list = []

        def _fake_runner(**kwargs):  # type: ignore[override]
            # Simulate the edit_last_reply tool being called during the turn.
            # We reach into the active_tools list to find the EditReplyState.
            # In real code the tool is a closure; here we simply expose edit_state
            # via the holder so the test can set it directly.
            return "New text from agent"

        handler = MessageHandler(
            session_mgr=session_mgr,
            config={},
            llm=MagicMock(),
            system_prompt="sys",
            registry=MagicMock(),
            approvals=set(),
            available_tools={},
            active_tools=[],
            agent_runner=_fake_runner,
        )
        channel = MagicMock()
        return handler, session, channel, edit_state_holder

    def test_failed_edit_falls_back_to_send(self):
        """When edit_message returns ok=False and no schedule/queue, channel.send() is called."""
        from src.assistant.scheduler import EditReplyState, QueueReplyState, ScheduleReplyState

        handler, session, channel, _ = self._make_handler_with_edit_state()

        msg = _make_msg()
        edit_state = EditReplyState(was_called=True, new_text="Edited text")
        schedule_state = ScheduleReplyState()
        queue_state = QueueReplyState()

        channel.edit_message.return_value = SendResult(ok=False, error="not supported")
        channel.send.return_value = SendResult(ok=True, message_id="msg-new-1")

        result = handler._route_response(
            msg, channel, "original response", schedule_state, edit_state, queue_state, session
        )

        channel.send.assert_called_once_with("42", "Edited text")
        assert result == "Edited text"

    def test_failed_edit_updates_last_sent_message_id(self):
        """Fallback send updates session.last_sent_message_id with the new message ID."""
        from src.assistant.scheduler import EditReplyState, QueueReplyState, ScheduleReplyState

        handler, session, channel, _ = self._make_handler_with_edit_state()

        msg = _make_msg()
        edit_state = EditReplyState(was_called=True, new_text="Fallback text")
        schedule_state = ScheduleReplyState()
        queue_state = QueueReplyState()

        channel.edit_message.return_value = SendResult(ok=False)
        channel.send.return_value = SendResult(ok=True, message_id="msg-fallback-99")

        handler._route_response(
            msg, channel, "original", schedule_state, edit_state, queue_state, session
        )

        assert session.last_sent_message_id == "msg-fallback-99"

    def test_successful_edit_does_not_call_send(self):
        """When edit_message succeeds, channel.send() must NOT be called."""
        from src.assistant.scheduler import EditReplyState, QueueReplyState, ScheduleReplyState

        handler, session, channel, _ = self._make_handler_with_edit_state()

        msg = _make_msg()
        edit_state = EditReplyState(was_called=True, new_text="Successfully edited")
        schedule_state = ScheduleReplyState()
        queue_state = QueueReplyState()

        channel.edit_message.return_value = SendResult(ok=True)

        result = handler._route_response(
            msg, channel, "original", schedule_state, edit_state, queue_state, session
        )

        channel.send.assert_not_called()
        assert result == "Successfully edited"

    def test_failed_edit_with_schedule_does_not_call_send(self):
        """When edit fails but schedule_reply was also called, send() is NOT triggered
        because a scheduled delivery already covers the response."""
        from src.assistant.scheduler import EditReplyState, QueueReplyState, ScheduleReplyState

        handler, session, channel, _ = self._make_handler_with_edit_state()
        # Wire a scheduler so schedule_reply can be processed
        handler._scheduler = MagicMock()

        msg = _make_msg()
        edit_state = EditReplyState(was_called=True, new_text="Edit failed text")
        schedule_state = ScheduleReplyState(
            was_called=True, scheduled_text="Scheduled text", delay_minutes=5
        )
        queue_state = QueueReplyState()

        channel.edit_message.return_value = SendResult(ok=False)

        handler._route_response(
            msg, channel, "original", schedule_state, edit_state, queue_state, session
        )

        # schedule_reply was called — the response is handled by the scheduler, not send()
        channel.send.assert_not_called()

    def test_failed_edit_fallback_send_failure_is_logged(self):
        """When the fallback send also fails, the method still returns without raising."""
        from src.assistant.scheduler import EditReplyState, QueueReplyState, ScheduleReplyState

        handler, session, channel, _ = self._make_handler_with_edit_state()

        msg = _make_msg()
        edit_state = EditReplyState(was_called=True, new_text="Text")
        schedule_state = ScheduleReplyState()
        queue_state = QueueReplyState()

        channel.edit_message.return_value = SendResult(ok=False)
        channel.send.return_value = SendResult(ok=False, error="network error")

        # Should not raise even when fallback send fails;
        # returns None so undelivered text is not recorded in memory (BUG-138)
        result = handler._route_response(
            msg, channel, "original", schedule_state, edit_state, queue_state, session
        )
        assert result is None
