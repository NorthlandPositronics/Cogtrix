"""Unit tests for src/assistant/handler.py — MessageHandler."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.assistant.channel import IncomingMessage
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
) -> tuple[MessageHandler, MagicMock]:
    """Return (handler, mock_session_mgr)."""
    session = _make_session()
    session_mgr = MagicMock()
    session_mgr.get_or_create.return_value = session

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
        safe_tool.name = "read_file"

        handler, _ = _make_handler(
            available_tools={
                "whatsapp_send": excluded_tool,
                "read_file": safe_tool,
            }
        )

        assert "whatsapp_send" not in handler._available_tools
        assert "read_file" in handler._available_tools

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
        excluded.name = "shell"
        safe = MagicMock()
        safe.name = "web_search"

        handler, _ = _make_handler(active_tools=[excluded, safe])

        active_names = [getattr(t, "name", None) for t in handler._active_tools]
        assert "shell" not in active_names
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

    def test_whatsapp_and_telegram_tools_all_excluded(self):
        """Both whatsapp_* and telegram_* tools are excluded by default."""
        messaging_tools = [
            "whatsapp_send",
            "whatsapp_check",
            "whatsapp_send_image",
            "whatsapp_contacts",
            "telegram_send",
            "telegram_check",
            "telegram_send_photo",
            "telegram_contacts",
        ]
        for tool_name in messaging_tools:
            assert tool_name in _DEFAULT_EXCLUDED, f"{tool_name} should be in _DEFAULT_EXCLUDED"

    def test_write_and_shell_tools_excluded(self):
        """shell, write_file, append_file are excluded by default."""
        for name in ("shell", "write_file", "append_file"):
            assert name in _DEFAULT_EXCLUDED


# ---------------------------------------------------------------------------
# TestResponseTruncation
# ---------------------------------------------------------------------------


class TestResponseTruncation:
    """Tests for max_response_length truncation."""

    def test_response_within_limit_not_truncated(self):
        """Responses shorter than max_response_length are sent as-is."""
        channel = MagicMock()
        handler, _ = _make_handler(config={"max_response_length": 100})

        with patch("cogtrix.run_agent", return_value="Short reply"):
            handler.handle(_make_msg(), channel)

        channel.send.assert_called_once_with("42", "Short reply")

    def test_response_exceeding_limit_is_truncated(self):
        """Responses longer than max_response_length are cut and appended with '...'."""
        channel = MagicMock()
        handler, _ = _make_handler(config={"max_response_length": 20})

        long_response = "A" * 50
        with patch("cogtrix.run_agent", return_value=long_response):
            handler.handle(_make_msg(), channel)

        sent_text = channel.send.call_args[0][1]
        assert len(sent_text) == 20
        assert sent_text.endswith("...")

    def test_truncation_keeps_correct_prefix(self):
        """Truncated response uses first (max_length - 3) characters plus '...'."""
        channel = MagicMock()
        limit = 30
        handler, _ = _make_handler(config={"max_response_length": limit})

        long_response = "X" * 100
        with patch("cogtrix.run_agent", return_value=long_response):
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
        )

        captured_prefix: list[str | None] = []

        def _fake_run_agent(**kwargs: object) -> str:
            captured_prefix.append(kwargs.get("context_prefix"))
            return "OK"

        with patch("cogtrix.run_agent", side_effect=_fake_run_agent):
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
        )

        captured_prefix: list[str | None] = []

        def _fake_run_agent(**kwargs: object) -> str:
            captured_prefix.append(kwargs.get("context_prefix"))
            return "OK"

        with patch("cogtrix.run_agent", side_effect=_fake_run_agent):
            handler.handle(_make_msg(), channel)

        prefix = captured_prefix[0]
        assert prefix is not None
        assert "Bob: Likes Python" in prefix

    def test_knowledge_extraction_called_after_agent(self):
        """extract_and_store is called with user input and agent response."""
        knowledge_store = MagicMock()
        knowledge_store.recall.return_value = None

        channel = MagicMock()
        handler, _ = _make_handler(knowledge_store=knowledge_store)

        with patch("cogtrix.run_agent", return_value="Agent says hi"):
            handler.handle(_make_msg(text="User says hello"), channel)

        knowledge_store.extract_and_store.assert_called_once_with(
            "User says hello", "Agent says hi"
        )

    def test_no_knowledge_store_runs_without_error(self):
        """handler.handle() works fine when knowledge_store is None."""
        channel = MagicMock()
        handler, _ = _make_handler(knowledge_store=None)

        with patch("cogtrix.run_agent", return_value="Fine"):
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
        )

        captured_prefix: list[str | None] = []

        def _fake_run_agent(**kwargs: object) -> str:
            captured_prefix.append(kwargs.get("context_prefix"))
            return "OK"

        with patch("cogtrix.run_agent", side_effect=_fake_run_agent):
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
        handler, _ = _make_handler()

        with patch("cogtrix.run_agent", side_effect=RuntimeError("LLM down")):
            handler.handle(_make_msg(), channel)

        channel.send.assert_called_once()
        sent_text = channel.send.call_args[0][1]
        assert "error" in sent_text.lower()

    def test_successful_response_sent_via_channel(self):
        """channel.send() is called with the agent's response."""
        channel = MagicMock()
        channel.send.return_value = True
        handler, _ = _make_handler()

        with patch("cogtrix.run_agent", return_value="Here is the answer"):
            handler.handle(_make_msg(), channel)

        channel.send.assert_called_once_with("42", "Here is the answer")

    def test_failed_send_logged_but_does_not_raise(self):
        """When channel.send() returns False, handle() does not raise."""
        channel = MagicMock()
        channel.send.return_value = False
        handler, _ = _make_handler()

        with patch("cogtrix.run_agent", return_value="Reply"):
            handler.handle(_make_msg(), channel)

        channel.send.assert_called_once()

    def test_memory_update_called_after_agent(self):
        """session.memory_manager.update() is called with user input and response."""
        channel = MagicMock()
        session = _make_session()
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
        )

        with patch("cogtrix.run_agent", return_value="Resp"):
            handler.handle(_make_msg(text="Query"), channel)

        session.memory_manager.update.assert_called_once_with("Query", "Resp")
