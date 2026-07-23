"""Unit tests for message debounce (MessageBuffer), SendResult, edit tool, and handler routing."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

from src.assistant.channel import Channel, IncomingMessage, SendResult
from src.assistant.poller import MessageBuffer
from src.assistant.scheduler import (
    EditReplyState,
    MessageScheduler,
    ScheduledMessage,
    ScheduleReplyState,
    _merge_phonebooks,
    _resolve_message_id,
    create_cancel_scheduled_tool,
    create_edit_reply_tool,
    create_edit_scheduled_tool,
    create_list_scheduled_tool,
    create_schedule_reply_tool,
)
from src.memory.context import MemoryContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(
    text: str = "Hello",
    chat_id: str = "42",
    channel: str = "telegram",
    message_id: str = "m1",
) -> IncomingMessage:
    return IncomingMessage(
        channel=channel,
        chat_id=chat_id,
        message_id=message_id,
        sender_id="u1",
        sender_name="Alice",
        text=text,
        timestamp=time.time(),
    )


def _make_session(last_sent_message_id: str | None = None) -> MagicMock:
    session = MagicMock()
    session.session_key = "telegram::42"
    session.lock = MagicMock()
    session.lock.__enter__ = MagicMock(return_value=None)
    session.lock.__exit__ = MagicMock(return_value=False)
    session.guardrail_violations = 0
    session.last_sent_message_id = last_sent_message_id
    session.memory_manager.prepare_context.return_value = MemoryContext(
        messages=[],
        context_prefix=None,
    )
    return session


def _make_handler(
    config: dict | None = None,
    agent_runner: Callable[..., Any] | None = None,
    session: MagicMock | None = None,
) -> tuple[Any, MagicMock, MagicMock]:
    """Return (handler, session_mgr, session)."""
    from src.assistant.handler import MessageHandler

    if session is None:
        session = _make_session()
    session_mgr = MagicMock()
    session_mgr.get_or_create.return_value = session

    if agent_runner is None:
        agent_runner = MagicMock(return_value="reply")

    handler = MessageHandler(
        session_mgr=session_mgr,
        config=config or {},
        llm=MagicMock(),
        system_prompt="You are helpful.",
        registry=MagicMock(),
        approvals=set(),
        available_tools={},
        active_tools=[],
        agent_runner=agent_runner,
    )
    return handler, session_mgr, session


# ---------------------------------------------------------------------------
# TestSendResult
# ---------------------------------------------------------------------------


class TestSendResult:
    """SendResult dataclass defaults and Channel.edit_message default."""

    def test_send_result_defaults(self):
        result = SendResult(ok=True)
        assert result.message_id is None
        assert result.error is None

    def test_send_result_ok_false_defaults(self):
        result = SendResult(ok=False)
        assert result.message_id is None
        assert result.error is None

    def test_send_result_stores_message_id(self):
        result = SendResult(ok=True, message_id="abc123")
        assert result.message_id == "abc123"

    def test_send_result_stores_error(self):
        result = SendResult(ok=False, error="timeout")
        assert result.error == "timeout"

    def test_channel_edit_message_default_returns_ok_false(self):
        """Channel subclass that does not override edit_message returns ok=False."""

        class _ConcreteChannel(Channel):
            @property
            def name(self) -> str:
                return "test"

            def poll(self) -> list[IncomingMessage]:
                return []

            def send(self, chat_id: str, text: str) -> SendResult:
                return SendResult(ok=True)

            def is_ready(self) -> bool:
                return True

        ch = _ConcreteChannel()
        result = ch.edit_message("chat1", "msg1", "new text")
        assert result.ok is False

    def test_channel_edit_message_default_error_text(self):
        """Default edit_message error field is not None."""

        class _MinimalChannel(Channel):
            @property
            def name(self) -> str:
                return "minimal"

            def poll(self) -> list[IncomingMessage]:
                return []

            def send(self, chat_id: str, text: str) -> SendResult:
                return SendResult(ok=True)

            def is_ready(self) -> bool:
                return True

        ch = _MinimalChannel()
        result = ch.edit_message("c", "m", "t")
        assert result.error is not None


# ---------------------------------------------------------------------------
# TestEditReplyState
# ---------------------------------------------------------------------------


class TestEditReplyState:
    """EditReplyState dataclass defaults and tool-closure mutation."""

    def test_edit_state_defaults(self):
        state = EditReplyState()
        assert state.was_called is False
        assert state.new_text == ""

    def test_edit_tool_mutates_state_was_called(self):
        state = EditReplyState()
        tool = create_edit_reply_tool(state)
        tool.invoke({"new_text": "corrected reply"})
        assert state.was_called is True

    def test_edit_tool_mutates_state_new_text(self):
        state = EditReplyState()
        tool = create_edit_reply_tool(state)
        tool.invoke({"new_text": "updated message"})
        assert state.new_text == "updated message"

    def test_edit_tool_returns_confirmation(self):
        state = EditReplyState()
        tool = create_edit_reply_tool(state)
        result = tool.invoke({"new_text": "any text"})
        assert "Last reply will be updated" in result

    def test_edit_tool_name(self):
        state = EditReplyState()
        tool = create_edit_reply_tool(state)
        assert tool.name == "edit_last_reply"

    def test_edit_tool_closure_isolated_per_state(self):
        """Two separate tool instances do not share state."""
        state_a = EditReplyState()
        state_b = EditReplyState()
        tool_a = create_edit_reply_tool(state_a)
        _tool_b = create_edit_reply_tool(state_b)
        tool_a.invoke({"new_text": "only A"})
        assert state_a.was_called is True
        assert state_b.was_called is False

    def test_edit_tool_idempotent_first_call_wins(self):
        """Second call is ignored — only the first edit per turn takes effect."""
        state = EditReplyState()
        tool = create_edit_reply_tool(state)
        tool.invoke({"new_text": "first"})
        result = tool.invoke({"new_text": "second"})
        assert state.new_text == "first"
        assert "already queued" in result


# ---------------------------------------------------------------------------
# TestMessageBuffer
# ---------------------------------------------------------------------------


class TestMessageBuffer:
    """MessageBuffer debounce and batch dispatch behaviour."""

    def _make_buffer(self, debounce: float = 0.1) -> tuple[MessageBuffer, MagicMock, MagicMock]:
        """Return (buffer, mock_handler, mock_executor)."""
        mock_handler = MagicMock()
        mock_executor = MagicMock()
        buf = MessageBuffer(mock_handler, mock_executor, debounce_seconds=debounce)
        return buf, mock_handler, mock_executor

    def _make_channel(self) -> MagicMock:
        ch = MagicMock(spec=Channel)
        ch.name = "telegram"
        return ch

    def test_single_message_dispatched(self):
        """A single message is dispatched after the debounce window expires."""
        buf, handler, executor = self._make_buffer(debounce=0.1)
        ch = self._make_channel()
        msg = _make_msg()

        buf.add(msg, ch)
        time.sleep(0.25)

        executor.submit.assert_called_once()
        _fn, dispatched_msgs, dispatched_ch = executor.submit.call_args[0]
        assert dispatched_msgs == [msg]
        assert dispatched_ch is ch

    def test_rapid_messages_batched_single_call(self):
        """Multiple messages added before debounce expires produce a single dispatch."""
        buf, handler, executor = self._make_buffer(debounce=0.2)
        ch = self._make_channel()
        msg1 = _make_msg(text="a", message_id="1")
        msg2 = _make_msg(text="b", message_id="2")
        msg3 = _make_msg(text="c", message_id="3")

        buf.add(msg1, ch)
        buf.add(msg2, ch)
        buf.add(msg3, ch)
        time.sleep(0.4)

        assert executor.submit.call_count == 1

    def test_rapid_messages_batched_contains_all(self):
        """All messages are included in the single batch dispatch."""
        buf, handler, executor = self._make_buffer(debounce=0.2)
        ch = self._make_channel()
        msg1 = _make_msg(text="a", message_id="1")
        msg2 = _make_msg(text="b", message_id="2")
        msg3 = _make_msg(text="c", message_id="3")

        buf.add(msg1, ch)
        buf.add(msg2, ch)
        buf.add(msg3, ch)
        time.sleep(0.4)

        _fn, dispatched_msgs, _ch = executor.submit.call_args[0]
        assert dispatched_msgs == [msg1, msg2, msg3]

    def test_different_chats_dispatch_separately(self):
        """Messages for two distinct chats produce two separate dispatch calls."""
        buf, handler, executor = self._make_buffer(debounce=0.1)
        ch = self._make_channel()
        msg_a = _make_msg(chat_id="42", channel="telegram")
        msg_b = _make_msg(chat_id="99", channel="telegram")

        buf.add(msg_a, ch)
        buf.add(msg_b, ch)
        time.sleep(0.3)

        assert executor.submit.call_count == 2

    def test_different_chats_messages_correct(self):
        """Each chat receives only its own messages."""
        buf, handler, executor = self._make_buffer(debounce=0.1)
        ch = self._make_channel()
        msg_a = _make_msg(chat_id="42", text="for_42")
        msg_b = _make_msg(chat_id="99", text="for_99")

        buf.add(msg_a, ch)
        buf.add(msg_b, ch)
        time.sleep(0.3)

        all_batches = [c[0][1] for c in executor.submit.call_args_list]
        assert [msg_a] in all_batches
        assert [msg_b] in all_batches

    def test_flush_all_dispatches_immediately(self):
        """flush_all() cancels pending timers and dispatches without waiting."""
        buf, handler, executor = self._make_buffer(debounce=10.0)
        ch = self._make_channel()
        msg = _make_msg()

        buf.add(msg, ch)
        assert executor.submit.call_count == 0

        buf.flush_all()

        executor.submit.assert_called_once()
        _fn, dispatched_msgs, _ch = executor.submit.call_args[0]
        assert dispatched_msgs == [msg]

    def test_flush_all_clears_buffers(self):
        """After flush_all(), internal buffers are empty."""
        buf, handler, executor = self._make_buffer(debounce=10.0)
        ch = self._make_channel()
        buf.add(_make_msg(), ch)
        buf.flush_all()
        assert buf._buffers == {}

    def test_flush_all_empty_is_noop(self):
        """flush_all() with no pending messages does not raise and makes no submit call."""
        buf, handler, executor = self._make_buffer()
        buf.flush_all()
        executor.submit.assert_not_called()

    def test_debounce_timer_reset_on_rapid_messages(self):
        """Adding a second message before the debounce fires resets the timer."""
        buf, handler, executor = self._make_buffer(debounce=0.2)
        ch = self._make_channel()

        buf.add(_make_msg(text="first"), ch)
        time.sleep(0.1)
        buf.add(_make_msg(text="second"), ch)
        # Check before the reset timer fires
        time.sleep(0.1)
        assert executor.submit.call_count == 0
        # Now wait for the reset timer to complete
        time.sleep(0.2)
        assert executor.submit.call_count == 1

    def test_submit_uses_handle_batch_method(self):
        """executor.submit is called with handler.handle_batch as the function."""
        buf, handler, executor = self._make_buffer(debounce=0.1)
        ch = self._make_channel()

        buf.add(_make_msg(), ch)
        time.sleep(0.25)

        fn_arg = executor.submit.call_args[0][0]
        assert fn_arg == handler.handle_batch


# ---------------------------------------------------------------------------
# TestHandleBatch
# ---------------------------------------------------------------------------


class TestHandleBatch:
    """MessageHandler.handle_batch() routing logic."""

    def test_empty_list_is_noop(self):
        """handle_batch([]) returns without calling handle."""
        handler, _, _ = _make_handler()
        channel = MagicMock()
        handler.handle = MagicMock()

        handler.handle_batch([], channel)

        handler.handle.assert_not_called()

    def test_single_message_delegates_to_handle(self):
        """handle_batch([msg]) calls handle(msg, channel) exactly once."""
        handler, _, _ = _make_handler()
        channel = MagicMock()
        handler.handle = MagicMock()
        msg = _make_msg()

        handler.handle_batch([msg], channel)

        handler.handle.assert_called_once_with(msg, channel)

    def test_multiple_messages_concatenated(self):
        """handle_batch with N>1 passes a combined message whose text is newline-joined."""
        captured: list[IncomingMessage] = []

        def _capture_handle(msg: IncomingMessage, _ch: Any) -> None:
            captured.append(msg)

        handler, _, _ = _make_handler()
        channel = MagicMock()
        handler.handle = _capture_handle

        msg1 = _make_msg(text="text1")
        msg2 = _make_msg(text="text2")
        msg3 = _make_msg(text="text3")

        handler.handle_batch([msg1, msg2, msg3], channel)

        assert len(captured) == 1
        assert captured[0].text == "text1\ntext2\ntext3"

    def test_multiple_messages_uses_last_as_primary(self):
        """Combined message uses metadata from the last message in the batch."""
        captured: list[IncomingMessage] = []

        def _capture_handle(msg: IncomingMessage, _ch: Any) -> None:
            captured.append(msg)

        handler, _, _ = _make_handler()
        channel = MagicMock()
        handler.handle = _capture_handle

        msg1 = _make_msg(text="first", message_id="m1")
        msg2 = _make_msg(text="last", message_id="m99")

        handler.handle_batch([msg1, msg2], channel)

        assert captured[0].message_id == "m99"

    def test_multiple_messages_single_handle_call(self):
        """handle is invoked exactly once even for many messages."""
        handle_count = [0]

        def _count_handle(_msg: IncomingMessage, _ch: Any) -> None:
            handle_count[0] += 1

        handler, _, _ = _make_handler()
        channel = MagicMock()
        handler.handle = _count_handle

        msgs = [_make_msg(text=f"msg{i}", message_id=str(i)) for i in range(5)]
        handler.handle_batch(msgs, channel)

        assert handle_count[0] == 1


# ---------------------------------------------------------------------------
# TestHandlerEditFlow
# ---------------------------------------------------------------------------


class TestHandlerEditFlow:
    """_route_response edit path: edit_state.was_called routing."""

    def _make_handler_with_session(
        self,
        agent_runner: Callable[..., Any],
        last_sent_message_id: str | None = None,
    ) -> tuple[Any, MagicMock, MagicMock]:
        session = _make_session(last_sent_message_id=last_sent_message_id)
        handler, session_mgr, session = _make_handler(agent_runner=agent_runner, session=session)
        return handler, session_mgr, session

    def test_edit_routes_to_channel_edit(self):
        """When edit_state.was_called=True and last_sent_message_id is set, channel.edit_message is called."""

        def _runner_that_edits(**kwargs: Any) -> str:
            tools: list = (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            for t in tools:
                if getattr(t, "name", None) == "edit_last_reply":
                    t.invoke({"new_text": "corrected text"})
                    break
            return "corrected text"

        handler, session_mgr, session = self._make_handler_with_session(
            agent_runner=_runner_that_edits,
            last_sent_message_id="prev_msg_id",
        )
        channel = MagicMock()
        channel.edit_message.return_value = SendResult(ok=True)
        msg = _make_msg()

        handler.handle(msg, channel)

        channel.edit_message.assert_called_once_with("42", "prev_msg_id", "corrected text")

    def test_edit_skipped_without_message_id(self):
        """When last_sent_message_id is None, edit_last_reply tool is not injected; falls through to send."""

        def _runner_noedit(**kwargs: Any) -> str:
            tools: list = (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            # edit_last_reply should not be present since no prior message id
            for t in tools:
                if getattr(t, "name", None) == "edit_last_reply":
                    t.invoke({"new_text": "should not edit"})
                    break
            return "regular reply"

        handler, session_mgr, session = self._make_handler_with_session(
            agent_runner=_runner_noedit,
            last_sent_message_id=None,
        )
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)
        msg = _make_msg()

        handler.handle(msg, channel)

        channel.edit_message.assert_not_called()
        channel.send.assert_called_once()

    def test_edit_tool_not_in_active_tools_when_no_prior_message(self):
        """edit_last_reply tool is not injected when session.last_sent_message_id is None."""
        captured_tools: list = []

        def _capture(**kwargs: Any) -> str:
            captured_tools.extend(
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            return "ok"

        handler, _, _ = self._make_handler_with_session(
            agent_runner=_capture,
            last_sent_message_id=None,
        )
        channel = MagicMock()
        handler.handle(_make_msg(), channel)

        tool_names = [getattr(t, "name", None) for t in captured_tools]
        assert "edit_last_reply" not in tool_names

    def test_edit_tool_injected_when_prior_message_exists(self):
        """edit_last_reply tool IS injected when session.last_sent_message_id is set."""
        captured_tools: list = []

        def _capture(**kwargs: Any) -> str:
            captured_tools.extend(
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            return "ok"

        handler, _, _ = self._make_handler_with_session(
            agent_runner=_capture,
            last_sent_message_id="some_id",
        )
        channel = MagicMock()
        channel.edit_message.return_value = SendResult(ok=True)
        handler.handle(_make_msg(), channel)

        tool_names = [getattr(t, "name", None) for t in captured_tools]
        assert "edit_last_reply" in tool_names

    def test_edit_not_called_when_edit_state_false(self):
        """Even if last_sent_message_id is set, channel.edit_message is NOT called when agent does not invoke the tool."""
        handler, _, _ = self._make_handler_with_session(
            agent_runner=MagicMock(return_value="normal reply"),
            last_sent_message_id="prev_id",
        )
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True)
        channel.edit_message.return_value = SendResult(ok=True)

        handler.handle(_make_msg(), channel)

        channel.edit_message.assert_not_called()
        channel.send.assert_called_once()

    def test_edit_uses_guardrail_sanitized_text(self):
        """Text passed to channel.edit_message is the sanitized version."""

        def _runner_edits(**kwargs: Any) -> str:
            for t in (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            ):
                if getattr(t, "name", None) == "edit_last_reply":
                    t.invoke({"new_text": "updated content"})
                    break
            return "updated content"

        handler, session_mgr, session = self._make_handler_with_session(
            agent_runner=_runner_edits,
            last_sent_message_id="mid_abc",
        )
        channel = MagicMock()
        channel.edit_message.return_value = SendResult(ok=True)

        handler.handle(_make_msg(), channel)

        assert channel.edit_message.called
        _chat_id, _msg_id, edit_text = channel.edit_message.call_args[0]
        assert edit_text == "updated content"


# ---------------------------------------------------------------------------
# TestSendResultTracking
# ---------------------------------------------------------------------------


class TestSendResultTracking:
    """session.last_sent_message_id is updated from channel.send() result."""

    def test_send_captures_message_id(self):
        """When channel.send returns SendResult with message_id, session.last_sent_message_id is updated."""
        session = _make_session()
        handler, _, _ = _make_handler(
            agent_runner=MagicMock(return_value="hello"),
            session=session,
        )
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True, message_id="msg123")

        handler.handle(_make_msg(), channel)

        assert session.last_sent_message_id == "msg123"

    def test_send_without_message_id_does_not_update(self):
        """When SendResult.message_id is None, session.last_sent_message_id stays None."""
        session = _make_session()
        handler, _, _ = _make_handler(
            agent_runner=MagicMock(return_value="hello"),
            session=session,
        )
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True, message_id=None)

        handler.handle(_make_msg(), channel)

        assert session.last_sent_message_id is None

    def test_failed_send_does_not_update_message_id(self):
        """When send fails (ok=False), session.last_sent_message_id is not set."""
        session = _make_session()
        handler, _, _ = _make_handler(
            agent_runner=MagicMock(return_value="hello"),
            session=session,
        )
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=False, message_id="should_not_set")

        handler.handle(_make_msg(), channel)

        assert session.last_sent_message_id is None

    def test_message_id_updated_each_successful_send(self):
        """Each successful send updates last_sent_message_id to the newest id."""
        session = _make_session()
        handler, _, _ = _make_handler(
            agent_runner=MagicMock(return_value="reply"),
            session=session,
        )
        channel = MagicMock()

        channel.send.return_value = SendResult(ok=True, message_id="first_id")
        handler.handle(_make_msg(), channel)
        assert session.last_sent_message_id == "first_id"

        channel.send.return_value = SendResult(ok=True, message_id="second_id")
        handler.handle(_make_msg(), channel)
        assert session.last_sent_message_id == "second_id"


# ---------------------------------------------------------------------------
# TestScheduledMessageRecipient
# ---------------------------------------------------------------------------


class TestScheduledMessageRecipient:
    """ScheduledMessage.recipient field serialization and backward compat."""

    def _make_scheduled(self, recipient: str | None = None) -> ScheduledMessage:
        return ScheduledMessage(
            id="test-id-1234",
            channel="whatsapp",
            chat_id="42@c.us",
            text="Hello",
            send_at=time.time() + 3600,
            created_at=time.time(),
            recipient=recipient,
        )

    def test_recipient_serializes(self):
        msg = ScheduledMessage(
            id="abc",
            channel="whatsapp",
            chat_id="42@c.us",
            text="Hi",
            send_at=time.time() + 60,
            created_at=time.time(),
            recipient="John",
        )
        d = msg.to_dict()
        assert "recipient" in d
        assert d["recipient"] == "John"

    def test_recipient_deserializes(self):
        data = {
            "id": "abc",
            "channel": "whatsapp",
            "chat_id": "42@c.us",
            "text": "Hi",
            "send_at": time.time() + 60,
            "created_at": time.time(),
            "recipient": "Alice",
            "status": "pending",
            "attempts": 0,
            "max_attempts": 3,
        }
        msg = ScheduledMessage.from_dict(data)
        assert msg.recipient == "Alice"

    def test_missing_recipient_defaults_none(self):
        data = {
            "id": "abc",
            "channel": "telegram",
            "chat_id": "99",
            "text": "Hey",
            "send_at": time.time() + 60,
            "created_at": time.time(),
            "status": "pending",
            "attempts": 0,
            "max_attempts": 3,
        }
        msg = ScheduledMessage.from_dict(data)
        assert msg.recipient is None


# ---------------------------------------------------------------------------
# TestSchedulerGetPending
# ---------------------------------------------------------------------------


class TestSchedulerGetPending:
    """MessageScheduler.get_pending() filtering behaviour."""

    def _make_scheduler(self, tmp_path: Any) -> MessageScheduler:
        return MessageScheduler({}, tmp_path / "sched.json")

    def test_get_pending_returns_all(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        now = time.time()
        scheduler.schedule("whatsapp", "chat1", "msg1", now + 100, recipient="Alice")
        scheduler.schedule("whatsapp", "chat2", "msg2", now + 200, recipient="Bob")
        scheduler.schedule("telegram", "chat3", "msg3", now + 50, recipient="Carol")

        pending = scheduler.get_pending()
        assert len(pending) == 3
        # Should be sorted by send_at ascending
        assert pending[0].text == "msg3"
        assert pending[1].text == "msg1"
        assert pending[2].text == "msg2"

    def test_get_pending_filters_by_recipient(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        now = time.time()
        scheduler.schedule("whatsapp", "chat1", "for John", now + 100, recipient="John")
        scheduler.schedule("whatsapp", "chat2", "for Alice", now + 200, recipient="Alice")

        result = scheduler.get_pending(recipient="john")
        assert len(result) == 1
        assert result[0].text == "for John"

    def test_get_pending_filters_by_chat_id(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        now = time.time()
        scheduler.schedule("whatsapp", "chatA", "for A", now + 100)
        scheduler.schedule("whatsapp", "chatB", "for B", now + 200)

        result = scheduler.get_pending(chat_id="chatA")
        assert len(result) == 1
        assert result[0].chat_id == "chatA"

    def test_get_pending_excludes_non_pending(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        now = time.time()
        msg_id = scheduler.schedule("whatsapp", "chat1", "already sent", now + 100)

        with scheduler._lock:
            scheduler._queue[msg_id].status = "sent"

        assert scheduler.get_pending() == []
        all_msgs = scheduler.get_pending(include_all=True)
        assert len(all_msgs) == 1
        assert all_msgs[0].status == "sent"

    def test_get_pending_fuzzy_match_phone(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        now = time.time()
        scheduler.schedule(
            "whatsapp",
            "971503308667@c.us",
            "phone message",
            now + 100,
            recipient="+971503308667",
        )

        result = scheduler.get_pending(recipient="971503308667")
        assert len(result) == 1
        assert result[0].text == "phone message"


# ---------------------------------------------------------------------------
# TestSchedulerEditMessage
# ---------------------------------------------------------------------------


class TestSchedulerEditMessage:
    """MessageScheduler.edit_message() behaviour."""

    def _make_scheduler(self, tmp_path: Any) -> MessageScheduler:
        return MessageScheduler({}, tmp_path / "sched.json")

    def test_edit_text(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        msg_id = scheduler.schedule("whatsapp", "chat1", "original text", time.time() + 3600)

        ok = scheduler.edit_message(msg_id, new_text="updated")
        assert ok is True
        assert scheduler._queue[msg_id].text == "updated"

    def test_edit_send_at(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        future = time.time() + 7200
        msg_id = scheduler.schedule("whatsapp", "chat1", "text", time.time() + 3600)

        ok = scheduler.edit_message(msg_id, new_send_at=future)
        assert ok is True
        assert abs(scheduler._queue[msg_id].send_at - future) < 1.0

    def test_edit_both(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        future = time.time() + 7200
        msg_id = scheduler.schedule("whatsapp", "chat1", "old", time.time() + 3600)

        ok = scheduler.edit_message(msg_id, new_text="new text", new_send_at=future)
        assert ok is True
        assert scheduler._queue[msg_id].text == "new text"
        assert abs(scheduler._queue[msg_id].send_at - future) < 1.0

    def test_edit_nonexistent_returns_false(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        ok = scheduler.edit_message("nonexistent-id", new_text="anything")
        assert ok is False

    def test_edit_sent_returns_false(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        msg_id = scheduler.schedule("whatsapp", "chat1", "text", time.time() + 3600)

        with scheduler._lock:
            scheduler._queue[msg_id].status = "sent"

        ok = scheduler.edit_message(msg_id, new_text="updated")
        assert ok is False


# ---------------------------------------------------------------------------
# TestSchedulerCancelMessage
# ---------------------------------------------------------------------------


class TestSchedulerCancelMessage:
    """MessageScheduler.cancel_message() behaviour."""

    def _make_scheduler(self, tmp_path: Any) -> MessageScheduler:
        return MessageScheduler({}, tmp_path / "sched.json")

    def test_cancel_pending(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        msg_id = scheduler.schedule("whatsapp", "chat1", "text", time.time() + 3600)

        ok = scheduler.cancel_message(msg_id)
        assert ok is True
        assert scheduler._queue[msg_id].status == "cancelled"

    def test_cancel_nonexistent(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        ok = scheduler.cancel_message("no-such-id")
        assert ok is False

    def test_cancel_already_cancelled(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        msg_id = scheduler.schedule("whatsapp", "chat1", "text", time.time() + 3600)

        first = scheduler.cancel_message(msg_id)
        assert first is True
        second = scheduler.cancel_message(msg_id)
        assert second is False


# ---------------------------------------------------------------------------
# TestResolveMessageId
# ---------------------------------------------------------------------------


class TestResolveMessageId:
    """_resolve_message_id() prefix matching."""

    def _make_scheduler(self, tmp_path: Any) -> MessageScheduler:
        return MessageScheduler({}, tmp_path / "sched.json")

    def test_full_uuid_resolves(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        msg_id = scheduler.schedule("whatsapp", "chat1", "hello", time.time() + 3600)

        resolved = _resolve_message_id(scheduler, msg_id)
        assert resolved == msg_id

    def test_short_prefix_resolves(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        msg_id = scheduler.schedule("whatsapp", "chat1", "hello", time.time() + 3600)

        resolved = _resolve_message_id(scheduler, msg_id[:8])
        assert resolved == msg_id

    def test_nonexistent_returns_none(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        scheduler.schedule("whatsapp", "chat1", "hello", time.time() + 3600)

        resolved = _resolve_message_id(scheduler, "xxxxxxxx")
        assert resolved is None

    def test_non_pending_not_resolved(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        msg_id = scheduler.schedule("whatsapp", "chat1", "hello", time.time() + 3600)

        with scheduler._lock:
            scheduler._queue[msg_id].status = "sent"

        resolved = _resolve_message_id(scheduler, msg_id[:8])
        assert resolved is None


# ---------------------------------------------------------------------------
# TestListScheduledTool
# ---------------------------------------------------------------------------


class TestListScheduledTool:
    """create_list_scheduled_tool() factory and output format."""

    def _make_scheduler(self, tmp_path: Any) -> MessageScheduler:
        return MessageScheduler({}, tmp_path / "sched.json")

    def test_empty_queue(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        tool = create_list_scheduled_tool(scheduler)

        result = tool.invoke({"recipient": ""})
        assert "No pending scheduled messages" in result

    def test_lists_pending_with_details(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        id1 = scheduler.schedule(
            "whatsapp", "chat1", "First message text", time.time() + 3600, recipient="Alice"
        )
        id2 = scheduler.schedule(
            "telegram", "chat2", "Second message text", time.time() + 7200, recipient="Bob"
        )
        tool = create_list_scheduled_tool(scheduler)

        result = tool.invoke({"recipient": ""})
        assert "2 pending scheduled message(s)" in result
        assert id1[:8] in result
        assert id2[:8] in result
        assert "Alice" in result
        assert "Bob" in result
        assert "First message text" in result
        assert "Second message text" in result

    def test_recipient_filter(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        scheduler.schedule("whatsapp", "chat1", "for John", time.time() + 3600, recipient="John")
        alice_id = scheduler.schedule(
            "whatsapp", "chat2", "for Alice", time.time() + 3600, recipient="Alice"
        )
        tool = create_list_scheduled_tool(scheduler)

        result = tool.invoke({"recipient": "alice"})
        assert "for Alice" in result
        assert "for John" not in result
        assert alice_id[:8] in result

    def test_chat_id_exact_filter(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        a_id = scheduler.schedule("whatsapp", "chatA@c.us", "msg for A", time.time() + 3600)
        scheduler.schedule("whatsapp", "chatB@c.us", "msg for B", time.time() + 3600)
        tool = create_list_scheduled_tool(scheduler)

        result = tool.invoke({"recipient": "", "chat_id": "chatA@c.us", "contact_name": ""})
        assert "msg for A" in result
        assert "msg for B" not in result
        assert a_id[:8] in result

    def test_contact_name_phonebook_resolve(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        services_config = {"whatsapp": {"phonebook": {"shraddha": "+971504069790"}}}
        msg_id = scheduler.schedule(
            "whatsapp",
            "971504069790@c.us",
            "hello shraddha",
            time.time() + 3600,
            recipient="+971504069790",
        )
        tool = create_list_scheduled_tool(scheduler, services_config=services_config)

        result = tool.invoke({"recipient": "", "chat_id": "", "contact_name": "shraddha"})
        assert "hello shraddha" in result
        assert msg_id[:8] in result

    def test_contact_name_no_phonebook_fallback(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        msg_id = scheduler.schedule(
            "whatsapp", "chat1", "message for Alice", time.time() + 3600, recipient="Alice"
        )
        tool = create_list_scheduled_tool(scheduler)

        result = tool.invoke({"recipient": "", "chat_id": "", "contact_name": "Alice"})
        assert "message for Alice" in result
        assert msg_id[:8] in result

    def test_contact_name_case_insensitive(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        services_config = {"whatsapp": {"phonebook": {"shraddha": "+971504069790"}}}
        msg_id = scheduler.schedule(
            "whatsapp",
            "971504069790@c.us",
            "case insensitive msg",
            time.time() + 3600,
            recipient="+971504069790",
        )
        tool = create_list_scheduled_tool(scheduler, services_config=services_config)

        result = tool.invoke({"recipient": "", "chat_id": "", "contact_name": "Shraddha"})
        assert "case insensitive msg" in result
        assert msg_id[:8] in result

    def test_combined_contact_name_and_chat_id(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        services_config = {"whatsapp": {"phonebook": {"alice": "+123"}}}
        first_id = scheduler.schedule(
            "whatsapp", "123@c.us", "first msg", time.time() + 3600, recipient="+123"
        )
        scheduler.schedule(
            "whatsapp", "456@c.us", "second msg", time.time() + 3600, recipient="+123"
        )
        tool = create_list_scheduled_tool(scheduler, services_config=services_config)

        result = tool.invoke({"recipient": "", "chat_id": "123@c.us", "contact_name": "alice"})
        assert "first msg" in result
        assert "second msg" not in result
        assert first_id[:8] in result

    def test_all_filters_empty_returns_all(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        scheduler.schedule("whatsapp", "chat1", "msg one", time.time() + 3600, recipient="Alice")
        scheduler.schedule("telegram", "chat2", "msg two", time.time() + 3600, recipient="Bob")
        scheduler.schedule("whatsapp", "chat3", "msg three", time.time() + 3600, recipient="Carol")
        tool = create_list_scheduled_tool(scheduler)

        result = tool.invoke({"recipient": "", "chat_id": "", "contact_name": ""})
        assert "3 pending scheduled message(s)" in result

    def test_no_results_with_filters_shows_labels(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        tool = create_list_scheduled_tool(scheduler)

        result = tool.invoke({"recipient": "nobody", "chat_id": "xxx", "contact_name": "ghost"})
        assert "No pending scheduled messages" in result
        assert "nobody" in result
        assert "xxx" in result
        assert "ghost" in result


# ---------------------------------------------------------------------------
# TestMergePhonebooks
# ---------------------------------------------------------------------------


class TestMergePhonebooks:
    """_merge_phonebooks() helper: contact name resolution across channels."""

    def test_merge_single_channel(self) -> None:
        config = {"whatsapp": {"phonebook": {"alice": "+123"}}}
        result = _merge_phonebooks(config)
        assert "alice" in result
        assert "123" in result["alice"]

    def test_merge_multiple_channels(self) -> None:
        config = {
            "whatsapp": {"phonebook": {"alice": "+123"}},
            "telegram": {"phonebook": {"bob": "456"}},
        }
        result = _merge_phonebooks(config)
        assert "alice" in result
        assert "bob" in result
        assert "123" in result["alice"]
        assert "456" in result["bob"]

    def test_merge_same_name_different_channels(self) -> None:
        config = {
            "whatsapp": {"phonebook": {"alice": "+123"}},
            "telegram": {"phonebook": {"alice": "789"}},
        }
        result = _merge_phonebooks(config)
        assert "alice" in result
        assert "123" in result["alice"]
        assert "789" in result["alice"]
        assert len(result["alice"]) == 2

    def test_merge_normalizes_identifiers(self) -> None:
        config = {"whatsapp": {"phonebook": {"alice": "+123@c.us"}}}
        result = _merge_phonebooks(config)
        assert "alice" in result
        assert result["alice"] == ["123"]

    def test_merge_empty_config(self) -> None:
        result = _merge_phonebooks({})
        assert result == {}

    def test_merge_case_insensitive_keys(self) -> None:
        config = {"whatsapp": {"phonebook": {"Alice": "+123"}}}
        result = _merge_phonebooks(config)
        assert "alice" in result
        assert "Alice" not in result

    def test_merge_skips_non_dict_values(self) -> None:
        config = {
            "whatsapp": {"phonebook": {"alice": "+123"}},
            "debug": True,
        }
        result = _merge_phonebooks(config)
        assert "alice" in result
        assert "123" in result["alice"]


# ---------------------------------------------------------------------------
# TestEditScheduledTool
# ---------------------------------------------------------------------------


class TestEditScheduledTool:
    """create_edit_scheduled_tool() factory."""

    def _make_scheduler(self, tmp_path: Any) -> MessageScheduler:
        return MessageScheduler({}, tmp_path / "sched.json")

    def test_edit_text_via_tool(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        msg_id = scheduler.schedule("whatsapp", "chat1", "original", time.time() + 3600)
        tool = create_edit_scheduled_tool(scheduler)

        tool.invoke({"message_id": msg_id[:8], "new_text": "updated via tool"})
        assert scheduler._queue[msg_id].text == "updated via tool"

    def test_edit_nonexistent_returns_error(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        tool = create_edit_scheduled_tool(scheduler)

        result = tool.invoke({"message_id": "badprefix"})
        assert "No pending message found" in result


# ---------------------------------------------------------------------------
# TestCancelScheduledTool
# ---------------------------------------------------------------------------


class TestCancelScheduledTool:
    """create_cancel_scheduled_tool() factory."""

    def _make_scheduler(self, tmp_path: Any) -> MessageScheduler:
        return MessageScheduler({}, tmp_path / "sched.json")

    def test_cancel_via_tool(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        msg_id = scheduler.schedule("whatsapp", "chat1", "to cancel", time.time() + 3600)
        tool = create_cancel_scheduled_tool(scheduler)

        tool.invoke({"message_id": msg_id[:8]})
        assert scheduler._queue[msg_id].status == "cancelled"

    def test_cancel_nonexistent_returns_error(self, tmp_path: Any) -> None:
        scheduler = self._make_scheduler(tmp_path)
        tool = create_cancel_scheduled_tool(scheduler)

        result = tool.invoke({"message_id": "badprefix"})
        assert "No pending message found" in result


# ---------------------------------------------------------------------------
# TestHandlerResolveRecipient
# ---------------------------------------------------------------------------


class TestHandlerResolveRecipient:
    """MessageHandler._resolve_recipient() priority logic."""

    def _make_msg_full(
        self,
        resolved_phone: str | None = None,
        sender_name: str | None = None,
        chat_id: str = "42",
    ) -> IncomingMessage:
        return IncomingMessage(
            channel="whatsapp",
            chat_id=chat_id,
            message_id="m1",
            sender_id="u1",
            sender_name=sender_name,
            text="Hello",
            timestamp=time.time(),
            resolved_phone=resolved_phone,
        )

    def test_prefers_resolved_phone(self) -> None:
        handler, _, _ = _make_handler()
        msg = self._make_msg_full(resolved_phone="+123", sender_name="John")
        assert handler._resolve_recipient(msg) == "+123"

    def test_falls_back_to_sender_name(self) -> None:
        handler, _, _ = _make_handler()
        msg = self._make_msg_full(resolved_phone=None, sender_name="John")
        assert handler._resolve_recipient(msg) == "John"

    def test_falls_back_to_chat_id(self) -> None:
        handler, _, _ = _make_handler()
        msg = self._make_msg_full(resolved_phone=None, sender_name=None, chat_id="chat99")
        assert handler._resolve_recipient(msg) == "chat99"


# ---------------------------------------------------------------------------
# TestScheduleReplyIdempotency
# ---------------------------------------------------------------------------


class TestScheduleReplyIdempotency:
    """BUG-029: ScheduleReplyState lock and idempotency guard."""

    def test_second_call_returns_already_scheduled(self):
        """Second call to schedule_reply returns guard message without overwriting state."""
        state = ScheduleReplyState()
        tool = create_schedule_reply_tool(state)
        result1 = tool.invoke({"text": "first message", "delay_minutes": 30})
        result2 = tool.invoke({"text": "second message", "delay_minutes": 60})
        assert "scheduled for delivery" in result1
        assert "already scheduled" in result2.lower()
        assert state.scheduled_text == "first message"
        assert state.delay_minutes == 30

    def test_concurrent_calls_only_one_wins(self):
        """Under concurrent invocation, only the first call's values are stored."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        state = ScheduleReplyState()
        tool = create_schedule_reply_tool(state)
        results = []

        def _invoke(text: str, delay: int) -> str:
            return tool.invoke({"text": text, "delay_minutes": delay})

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_invoke, "msg_a", 10),
                pool.submit(_invoke, "msg_b", 20),
            ]
            for f in as_completed(futures):
                results.append(f.result())

        assert state.was_called is True
        success = [r for r in results if "already scheduled" not in r.lower()]
        rejected = [r for r in results if "already scheduled" in r.lower()]
        assert len(success) == 1
        assert len(rejected) == 1
        assert state.scheduled_text in ("msg_a", "msg_b")


# ---------------------------------------------------------------------------
# TestEditReplyIdempotency
# ---------------------------------------------------------------------------


class TestEditReplyIdempotency:
    """BUG-042: EditReplyState lock and idempotency guard."""

    def test_second_call_returns_already_queued(self):
        """Second call to edit_last_reply returns guard message without overwriting."""
        state = EditReplyState()
        tool = create_edit_reply_tool(state)
        result1 = tool.invoke({"new_text": "first edit"})
        result2 = tool.invoke({"new_text": "second edit"})
        assert "will be updated" in result1
        assert "already queued" in result2.lower()
        assert state.new_text == "first edit"

    def test_concurrent_calls_only_one_wins(self):
        """Under concurrent invocation, only the first call's text is stored."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        state = EditReplyState()
        tool = create_edit_reply_tool(state)
        results = []

        def _invoke(text: str) -> str:
            return tool.invoke({"new_text": text})

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_invoke, "edit_a"),
                pool.submit(_invoke, "edit_b"),
            ]
            for f in as_completed(futures):
                results.append(f.result())

        assert state.was_called is True
        success = [r for r in results if "already queued" not in r.lower()]
        rejected = [r for r in results if "already queued" in r.lower()]
        assert len(success) == 1
        assert len(rejected) == 1
        assert state.new_text in ("edit_a", "edit_b")


# ---------------------------------------------------------------------------
# TestScheduledMessageFromDictCoercion
# ---------------------------------------------------------------------------


class TestScheduledMessageFromDictCoercion:
    """BUG-045: from_dict type coercion for numeric fields."""

    def _base_data(self) -> dict:
        return {
            "id": "test-id",
            "channel": "whatsapp",
            "chat_id": "42@c.us",
            "text": "Hello",
            "send_at": 1700000000.5,
            "created_at": 1700000000.0,
            "status": "pending",
            "attempts": 0,
            "max_attempts": 3,
        }

    def test_string_send_at_coerced_to_float(self):
        data = self._base_data()
        data["send_at"] = "1700000000.5"
        msg = ScheduledMessage.from_dict(data)
        assert msg.send_at == 1700000000.5
        assert isinstance(msg.send_at, float)

    def test_int_send_at_coerced_to_float(self):
        data = self._base_data()
        data["send_at"] = 1700000000
        msg = ScheduledMessage.from_dict(data)
        assert isinstance(msg.send_at, float)

    def test_string_created_at_coerced_to_float(self):
        data = self._base_data()
        data["created_at"] = "1700000000.0"
        msg = ScheduledMessage.from_dict(data)
        assert isinstance(msg.created_at, float)

    def test_float_attempts_coerced_to_int(self):
        data = self._base_data()
        data["attempts"] = 3.0
        msg = ScheduledMessage.from_dict(data)
        assert msg.attempts == 3
        assert isinstance(msg.attempts, int)

    def test_unparseable_send_at_raises(self):
        import pytest

        data = self._base_data()
        data["send_at"] = "not-a-number"
        with pytest.raises(ValueError):
            ScheduledMessage.from_dict(data)


# ---------------------------------------------------------------------------
# TestRouteResponseEditAndSchedule
# ---------------------------------------------------------------------------


class TestRouteResponseEditAndSchedule:
    """BUG-048: edit + schedule should both fire, not early-return after edit."""

    def test_edit_and_schedule_both_fire(self):
        """When both edit and schedule are called, both channel.edit_message and scheduler.schedule are invoked."""

        def _runner_both(**kwargs: Any) -> str:
            tools: list = (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            for t in tools:
                name = getattr(t, "name", None)
                if name == "edit_last_reply":
                    t.invoke({"new_text": "edited text"})
                elif name == "schedule_reply":
                    t.invoke({"text": "scheduled text", "delay_minutes": 60})
            return "agent response"

        session = _make_session(last_sent_message_id="prev_msg")
        handler, _, session = _make_handler(
            agent_runner=_runner_both,
            session=session,
            config={},
        )
        handler._scheduler = MagicMock()
        handler._scheduler.schedule = MagicMock()

        channel = MagicMock()
        channel.edit_message.return_value = SendResult(ok=True)

        handler.handle(_make_msg(), channel)

        channel.edit_message.assert_called_once()
        handler._scheduler.schedule.assert_called_once()

    def test_edit_and_schedule_returns_scheduled_text_for_memory(self):
        """When both fire, the return value (for memory) is the scheduled text."""

        def _runner_both(**kwargs: Any) -> str:
            tools: list = (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            )
            for t in tools:
                name = getattr(t, "name", None)
                if name == "edit_last_reply":
                    t.invoke({"new_text": "edited"})
                elif name == "schedule_reply":
                    t.invoke({"text": "scheduled for later", "delay_minutes": 30})
            return "agent response"

        session = _make_session(last_sent_message_id="prev_msg")
        handler, _, session = _make_handler(
            agent_runner=_runner_both,
            session=session,
        )
        handler._scheduler = MagicMock()
        handler._scheduler.schedule = MagicMock()

        channel = MagicMock()
        channel.edit_message.return_value = SendResult(ok=True)

        handler.handle(_make_msg(), channel)

        session.memory_manager.update.assert_called_once()
        _user_text, response_for_memory = session.memory_manager.update.call_args[0]
        assert response_for_memory == "scheduled for later"

    def test_edit_only_no_send(self):
        """When only edit fires (no schedule), channel.send is NOT called."""

        def _runner_edit_only(**kwargs: Any) -> str:
            for t in (
                kwargs["config"].active_tools_list
                if kwargs.get("config") and kwargs["config"].active_tools_list
                else []
            ):
                if getattr(t, "name", None) == "edit_last_reply":
                    t.invoke({"new_text": "corrected"})
                    break
            return "agent response"

        session = _make_session(last_sent_message_id="prev_msg")
        handler, _, session = _make_handler(
            agent_runner=_runner_edit_only,
            session=session,
        )
        channel = MagicMock()
        channel.edit_message.return_value = SendResult(ok=True)

        handler.handle(_make_msg(), channel)

        channel.edit_message.assert_called_once()
        channel.send.assert_not_called()

    def test_neither_called_immediate_send(self):
        """When neither edit nor schedule is called, channel.send fires."""
        session = _make_session()
        handler, _, session = _make_handler(
            agent_runner=MagicMock(return_value="hello"),
            session=session,
        )
        channel = MagicMock()
        channel.send.return_value = SendResult(ok=True, message_id="new_id")

        handler.handle(_make_msg(), channel)

        channel.send.assert_called_once()
        channel.edit_message.assert_not_called()
