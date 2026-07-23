"""#2417 fix #2 — don't silently suppress a reply on an operator-initiated thread.

The persona's spam/sales-restraint correctly stays silent on cold unsolicited
messages, but mis-fires on high-intent openers (e.g. a Property-Finder-forwarded
template + greeting) that are really a lead responding to OUR own outreach — and
the message then ages out, losing the lead with no operator signal. When the
thread already contains prior outbound from us (operator-initiated), a
model-suppressed turn is re-prompted once to actually reply and that reply is
delivered. Fail-safe: an empty/failed re-prompt keeps the suppression.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from cogtrix_core.assistant.channel import IncomingMessage
from cogtrix_core.assistant.handler import MessageHandler
from cogtrix_core.memory.context import MemoryContext


class TestOperatorInitiatedDetection:
    def test_history_with_prior_outbound_is_operator_initiated(self) -> None:
        assert MessageHandler._thread_is_operator_initiated([AIMessage(content="our outreach")])

    def test_cold_thread_is_not_operator_initiated(self) -> None:
        assert not MessageHandler._thread_is_operator_initiated([HumanMessage(content="Hi")])
        assert not MessageHandler._thread_is_operator_initiated([])


class TestEngageReprompt:
    def _handler(self, llm) -> MessageHandler:
        h = MessageHandler.__new__(MessageHandler)
        h._llm = llm
        return h

    def test_reprompt_returns_reply(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="Sure — which unit did you mean?")
        h = self._handler(llm)
        assert (
            h._rewrite_to_engage_operator_initiated("sys", [], "PF template + Hi")
            == "Sure — which unit did you mean?"
        )

    def test_reprompt_failure_returns_empty(self) -> None:
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("boom")
        h = self._handler(llm)
        assert h._rewrite_to_engage_operator_initiated("sys", [], "x") == ""


# --- Integration: suppress override through handle() -----------------------


def _make_msg(text: str = "Property Finder passed on your enquiry … Hi") -> IncomingMessage:
    return IncomingMessage(
        channel="whatsapp",
        chat_id="971564016789@c.us",
        message_id="m1",
        sender_id="u1",
        sender_name="Mahnoor",
        text=text,
        timestamp=time.time(),
    )


def _make_session(history: list) -> MagicMock:
    session = MagicMock()
    session.session_key = "whatsapp::971564016789"
    session.lock.__enter__ = MagicMock(return_value=None)
    session.lock.__exit__ = MagicMock(return_value=False)
    session.memory_manager.prepare_context.return_value = MemoryContext(
        messages=history, context_prefix=None
    )
    return session


def _make_handler(history: list, engaged_reply: str):
    session = _make_session(history)
    session_mgr = MagicMock()
    session_mgr.get_or_create.return_value = session

    # fake agent runner: invoke the injected suppress_reply tool, then return "".
    def _suppressing_runner(**kwargs):
        cfg = kwargs.get("config")
        for tool in getattr(cfg, "active_tools_list", None) or []:
            if getattr(tool, "name", "") == "suppress_reply":
                tool.invoke({})
        return ""

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=engaged_reply)

    handler = MessageHandler(
        session_mgr=session_mgr,
        config={},
        llm=llm,
        system_prompt="You are a real-estate assistant.",
        registry=MagicMock(),
        approvals=set(),
        available_tools={},
        active_tools=[],
        agent_runner=_suppressing_runner,
    )
    return handler, session


class TestSuppressOverrideIntegration:
    def test_operator_initiated_thread_engages_instead_of_suppressing(self) -> None:
        channel = MagicMock()
        channel.send.return_value = MagicMock(ok=True, message_id="x")
        handler, _session = _make_handler(
            history=[AIMessage(content="[our prior outreach to this lead]")],
            engaged_reply="Thanks for getting back — which unit were you asking about?",
        )
        handler.handle(_make_msg(), channel)
        # Suppression overridden: the engaged reply was delivered.
        assert channel.send.called
        sent = channel.send.call_args[0][1]
        assert "which unit" in sent

    def test_cold_thread_still_suppresses(self) -> None:
        channel = MagicMock()
        handler, session = _make_handler(
            history=[],  # cold: no prior outbound from us
            engaged_reply="(should not be used)",
        )
        handler.handle(_make_msg(), channel)
        assert not channel.send.called
        session.memory_manager.discard_prerecord.assert_called()
