"""Tests for shared-dict isolation in MessageHandler.handle().

BUG-027: handle() must not mutate self._available_tools or self._active_tools
when the runner (or graph process_tools node) pops from / modifies the passed
copies.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

from cogtrix_core.assistant.channel import IncomingMessage
from cogtrix_core.assistant.handler import MessageHandler
from cogtrix_core.assistant.scheduler import EditReplyState, QueueReplyState, ScheduleReplyState
from cogtrix_core.memory.context import MemoryContext

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


def _make_session() -> MagicMock:
    session = MagicMock()
    session.session_key = "telegram::42"
    session.lock = MagicMock()
    session.lock.__enter__ = MagicMock(return_value=None)
    session.lock.__exit__ = MagicMock(return_value=False)
    session.memory_manager.prepare_context.return_value = MemoryContext(
        messages=[],
        context_prefix=None,
    )
    return session


def _make_handler(
    available_tools: dict | None = None,
    active_tools: list | None = None,
    agent_runner: Callable | None = None,
    config: dict | None = None,
    llm: Any | None = None,
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
        llm=llm if llm is not None else MagicMock(),
        system_prompt="sys",
        registry=MagicMock(),
        approvals=set(),
        available_tools=available_tools or {},
        active_tools=active_tools or [],
        agent_runner=agent_runner,
    )
    return handler, session_mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAvailableToolsIsolation:
    """handle() passes a copy of _available_tools to the runner via config."""

    def test_runner_pop_does_not_mutate_available_tools(self):
        """When the runner pops a key from the passed dict, the original is unchanged."""
        tool_a = MagicMock()
        tool_a.name = "tool_a"
        tool_b = MagicMock()
        tool_b.name = "tool_b"

        def _runner_that_pops(**kwargs: object) -> str:
            cfg = kwargs.get("config")
            if cfg is not None and cfg.available_tools:
                cfg.available_tools.pop("tool_a", None)
            return "done"

        handler, _ = _make_handler(
            available_tools={"tool_a": tool_a, "tool_b": tool_b},
            agent_runner=_runner_that_pops,
        )
        original_keys = set(handler._available_tools.keys())

        handler.handle(_make_msg(), MagicMock())

        assert set(handler._available_tools.keys()) == original_keys

    def test_runner_receives_all_available_tools(self):
        """The runner receives a dict that contains all entries from _available_tools."""
        tool_x = MagicMock()
        tool_x.name = "tool_x"

        captured: list[dict] = []

        def _runner_capture(**kwargs: object) -> str:
            cfg = kwargs.get("config")
            captured.append(dict(cfg.available_tools) if cfg and cfg.available_tools else {})
            return "ok"

        handler, _ = _make_handler(
            available_tools={"tool_x": tool_x},
            agent_runner=_runner_capture,
        )

        handler.handle(_make_msg(), MagicMock())

        assert "tool_x" in captured[0]

    def test_available_tools_passed_is_a_different_object(self):
        """The dict object passed to the runner is not the same as self._available_tools."""
        tool = MagicMock()
        tool.name = "some_tool"

        captured_id: list[int] = []

        def _runner_capture(**kwargs: object) -> str:
            cfg = kwargs.get("config")
            if cfg is not None:
                captured_id.append(id(cfg.available_tools))
            return "ok"

        handler, _ = _make_handler(
            available_tools={"some_tool": tool},
            agent_runner=_runner_capture,
        )

        handler.handle(_make_msg(), MagicMock())

        assert captured_id[0] != id(handler._available_tools)


class TestActiveToolsIsolation:
    """handle() passes a copy of _active_tools to the runner via config."""

    def test_runner_pop_does_not_mutate_active_tools(self):
        """When the runner removes an item from the passed list, the original is unchanged."""
        tool_1 = MagicMock()
        tool_1.name = "tool_1"
        tool_2 = MagicMock()
        tool_2.name = "tool_2"

        def _runner_that_pops(**kwargs: object) -> str:
            cfg = kwargs.get("config")
            if cfg is not None and cfg.active_tools_list:
                cfg.active_tools_list.pop()
            return "done"

        handler, _ = _make_handler(
            active_tools=[tool_1, tool_2],
            agent_runner=_runner_that_pops,
        )
        original_length = len(handler._active_tools)

        handler.handle(_make_msg(), MagicMock())

        assert len(handler._active_tools) == original_length

    def test_runner_receives_all_active_tools(self):
        """The runner receives a list that contains all entries from _active_tools."""
        tool_a = MagicMock()
        tool_a.name = "active_tool_a"

        captured: list[list] = []

        def _runner_capture(**kwargs: object) -> str:
            cfg = kwargs.get("config")
            captured.append(list(cfg.active_tools_list) if cfg and cfg.active_tools_list else [])
            return "ok"

        handler, _ = _make_handler(
            active_tools=[tool_a],
            agent_runner=_runner_capture,
        )

        handler.handle(_make_msg(), MagicMock())

        assert tool_a in captured[0]

    def test_active_tools_passed_is_a_different_object(self):
        """The list object passed to the runner is not the same as self._active_tools."""
        tool = MagicMock()
        tool.name = "t"

        captured_id: list[int] = []

        def _runner_capture(**kwargs: object) -> str:
            cfg = kwargs.get("config")
            if cfg is not None:
                captured_id.append(id(cfg.active_tools_list))
            return "ok"

        handler, _ = _make_handler(
            active_tools=[tool],
            agent_runner=_runner_capture,
        )

        handler.handle(_make_msg(), MagicMock())

        assert captured_id[0] != id(handler._active_tools)


class TestNonDeliverableSilence:
    """#2052: internal control/error messages must never be sent to a contact.

    When the agent returns an internal sentinel (empty output, the
    recovery-failed message, an agent-error string, or a provider auth
    failure), the handler must stay silent — no channel.send, no schedule.
    """

    def _silent_check(self, runner_response: str) -> bool:
        """Run handler with a runner that returns *runner_response*; return
        True if NOTHING was delivered (silent)."""
        channel = MagicMock()
        channel.send.return_value = MagicMock(ok=True, message_id="x")

        handler, _ = _make_handler(agent_runner=MagicMock(return_value=runner_response))
        handler.handle(_make_msg(), channel)
        return not channel.send.called

    def test_empty_response_is_silent(self):
        assert self._silent_check("")
        assert self._silent_check("   \n  ")

    def test_recovery_failed_sentinel_is_silent(self):
        from cogtrix_core.orchestration.phases import RECOVERY_FAILED_MESSAGE

        assert self._silent_check(RECOVERY_FAILED_MESSAGE)

    def test_agent_error_string_is_silent(self):
        assert self._silent_check("I encountered a TimeoutError error. Please try again.")

    def test_auth_failure_is_silent(self):
        assert self._silent_check("**Authentication failed:** check your API key.")

    def test_real_reply_is_delivered(self):
        channel = MagicMock()
        channel.send.return_value = MagicMock(ok=True, message_id="x")
        handler, _ = _make_handler(
            agent_runner=MagicMock(return_value="Sure, what's the building name?")
        )
        handler.handle(_make_msg(), channel)
        assert channel.send.called


class TestInternalReportSuppression:
    """#2364: an internal operator-style status/summary recap must never reach a
    contact — it breaks the human persona and leaks strategy. Fail safe by
    staying silent. But legitimate persona replies (incl. bulleted property
    lists) MUST still be delivered — the guard has to be precise, not blunt.
    """

    def _delivered(self, runner_response: str) -> bool:
        """Run the handler with a runner returning *runner_response*; return
        True if it was actually sent to the contact."""
        channel = MagicMock()
        channel.send.return_value = MagicMock(ok=True, message_id="x")
        handler, _ = _make_handler(agent_runner=MagicMock(return_value=runner_response))
        handler.handle(_make_msg(), channel)
        return channel.send.called

    # ── leaks that MUST be suppressed ─────────────────────────────────────
    def test_where_things_stand_recap_is_suppressed(self):
        leak = (
            "Here's a summary of where things stand with this agent (Property "
            "Finder contact):\n---\n**Buildings discussed:**\n"
            "- Canal Residence — Dead for now.\n"
            "- Marina Height 2 — **Best lead.** Price: 95K achievable.\n"
            "---\n**Current status:** Waiting on the agent to confirm the floor."
        )
        assert not self._delivered(leak)

    def test_echoed_session_summary_marker_is_suppressed(self):
        assert not self._delivered(
            "[Session context summary]\nThe user is negotiating for Marina Height 2."
        )

    def test_hardened_marker_format_is_suppressed(self):
        # The emitters now use "[Session context summary — …]" / "[Operator
        # instruction — …]" (no "]" right after the word). The markers are
        # prefixes so the hardened forms are still caught (forge audit).
        assert not self._delivered(
            "[Session context summary — internal background for YOUR reference "
            "only. Do NOT relay this block to the user.]\nBest lead: Marina Height 2."
        )
        assert not self._delivered(
            "[Operator instruction — initiate conversation with the agent]\nAsk about the floor."
        )

    def test_other_internal_markers_suppressed(self):
        for marker_text in (
            "[Progress checkpoints]\n1. Called the agent.",
            "[Conversation so far]\nUser asked about the flat.",
            "[Operator instruction] Do not reveal the reserve price.",
        ):
            assert not self._delivered(marker_text), marker_text

    # ── legitimate persona replies that MUST still be delivered ───────────
    def test_bare_status_line_without_third_person_recap_is_delivered(self):
        # A bare "Current status:" line is NOT reliably a leak — a real person may
        # send "Current status: your offer was accepted!". The full operator recap
        # (third-person "with this agent") is what's suppressed, not this. The old
        # blanket "Current status:" rule dropped legitimate replies (forge audit).
        assert self._delivered("**Current status:** waiting on the agent for the floor plan.")
        assert self._delivered("Current status: your offer was accepted! 🎉")

    def test_where_things_stand_with_your_offer_is_delivered(self):
        # Second-person framing = a genuine client-facing update, not the leak.
        assert self._delivered(
            "Here's a summary of where things stand with your offer: the owner "
            "accepted 95K and we're booking the viewing for Saturday."
        )

    def test_property_list_reply_is_delivered(self):
        # A persona DOES send bulleted option lists — must not be a false positive.
        assert self._delivered(
            "Here are the 2BR options I found:\n"
            "- Marina Height 2 — 95K, sea view\n"
            "- RDK — 88K, no balcony\n"
            "Which would you like to view first?"
        )

    def test_casual_status_question_is_delivered(self):
        assert self._delivered("Any update on the price? Keen to lock in the viewing.")

    def test_summary_of_options_phrasing_is_delivered(self):
        # "summary of the options" is legitimate — only "…of where things stand" trips.
        assert self._delivered(
            "Quick summary of the options: Marina Height 2 is the strongest on price."
        )


class TestPersonaHardConstraintGuard:
    """#2384: a persona reply must never COMMIT to an operator-declared excluded
    term; on a violation the model is re-prompted once to decline, and if it still
    commits the turn is suppressed. Generic — the terms come from config, never
    from this codebase (synthetic placeholders used here)."""

    _CFG = {"guardrails": {"persona_constraints": {"excluded_terms": ["Building Z"]}}}

    def _run(self, agent_reply: str, rewrite_reply: str) -> tuple[MagicMock, MagicMock]:
        channel = MagicMock()
        channel.send.return_value = MagicMock(ok=True, message_id="x")
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=rewrite_reply)
        handler, _ = _make_handler(
            agent_runner=MagicMock(return_value=agent_reply),
            config=self._CFG,
            llm=llm,
        )
        handler.handle(_make_msg(), channel)
        return channel, llm

    def test_commit_is_reprompted_and_corrected_reply_sent(self):
        # Agent commits to the excluded term → re-prompt rewrites to a decline →
        # the corrected reply is delivered (not the committing one).
        channel, llm = self._run(
            agent_reply="We're all set for Saturday at Building Z.",
            rewrite_reply="I'll pass on Building Z — let's keep looking at the others.",
        )
        assert llm.invoke.called  # re-prompted
        assert channel.send.called
        sent = channel.send.call_args[0][1]
        assert "pass on Building Z" in sent
        assert "all set" not in sent.lower()

    def test_persistent_commit_is_suppressed(self):
        # If the re-prompt STILL commits to the excluded term, suppress (no send).
        channel, llm = self._run(
            agent_reply="We're all set for Saturday at Building Z.",
            rewrite_reply="Sure — Building Z it is, see you Saturday!",
        )
        assert llm.invoke.called
        assert not channel.send.called

    def test_decline_of_excluded_delivered_without_reprompt(self):
        # A reply that DECLINES the excluded term is not a violation — sent as-is.
        channel, llm = self._run(
            agent_reply="I'll steer clear of Building Z. Let's focus on the others.",
            rewrite_reply="(unused)",
        )
        assert not llm.invoke.called  # no re-prompt
        assert channel.send.called

    def test_no_constraints_configured_is_noop(self):
        # With no persona_constraints configured, even a committing reply is sent.
        channel = MagicMock()
        channel.send.return_value = MagicMock(ok=True, message_id="x")
        handler, _ = _make_handler(
            agent_runner=MagicMock(return_value="We're all set for Saturday at Building Z.")
        )
        handler.handle(_make_msg(), channel)
        assert channel.send.called


class TestPersonaConstraintSecondaryOutbounds:
    """#2386 (gap 3): the #2384 commitment guard is extended to the
    schedule_reply / queue_reply / edit_reply outbound paths (which bypass the
    main-reply guard). A secondary outbound committing to an operator-excluded
    term is neutralized (never delivered). Synthetic placeholder term only."""

    _CFG = {"guardrails": {"persona_constraints": {"excluded_terms": ["Building Z"]}}}

    def _handler(self, *, config: dict | None = None):
        handler, _ = _make_handler(config=config if config is not None else self._CFG)
        handler._scheduler = MagicMock()
        return handler

    def test_scheduled_reply_committing_to_excluded_is_suppressed(self):
        handler = self._handler()
        channel = MagicMock()
        channel.send.return_value = MagicMock(ok=True, message_id="x")
        schedule_state = ScheduleReplyState(
            was_called=True, scheduled_text="See you at Building Z on Saturday!", delay_minutes=10
        )
        handler._route_response(
            _make_msg(),
            channel,
            "ok",
            schedule_state,
            EditReplyState(),
            QueueReplyState(),
            _make_session(),
        )
        handler._scheduler.schedule.assert_not_called()

    def test_benign_scheduled_reply_is_delivered(self):
        handler = self._handler()
        channel = MagicMock()
        schedule_state = ScheduleReplyState(
            was_called=True, scheduled_text="Talk soon!", delay_minutes=10
        )
        handler._route_response(
            _make_msg(),
            channel,
            "ok",
            schedule_state,
            EditReplyState(),
            QueueReplyState(),
            _make_session(),
        )
        handler._scheduler.schedule.assert_called_once()

    def test_edit_reply_committing_to_excluded_is_suppressed(self):
        handler = self._handler()
        channel = MagicMock()
        channel.send.return_value = MagicMock(ok=True, message_id="x")
        channel.edit_message.return_value = MagicMock(ok=True)
        session = _make_session()
        session.last_sent_message_id = "m0"
        edit_state = EditReplyState(was_called=True, new_text="Confirmed — Building Z it is.")
        handler._route_response(
            _make_msg(),
            channel,
            "Here is the info you asked for.",
            ScheduleReplyState(),
            edit_state,
            QueueReplyState(),
            session,
        )
        # The offending edit is not applied; the guarded main reply is sent instead.
        channel.edit_message.assert_not_called()
        channel.send.assert_called_once()

    def test_queue_items_committing_to_excluded_are_dropped(self):
        handler = self._handler()
        channel = MagicMock()
        qs = QueueReplyState(
            items=[
                QueueReplyState.Item(text="Confirmed for Building Z.", gap_minutes=1),
                QueueReplyState.Item(text="Looking forward to it!", gap_minutes=2),
            ]
        )
        handler._route_response(
            _make_msg(), channel, "ok", ScheduleReplyState(), EditReplyState(), qs, _make_session()
        )
        # Only the benign item is queued; the committing one is dropped.
        assert handler._scheduler.queue_after_tail.call_count == 1

    def test_no_excluded_terms_configured_is_noop(self):
        handler = self._handler(config={})  # no persona_constraints
        channel = MagicMock()
        schedule_state = ScheduleReplyState(
            was_called=True, scheduled_text="See you at Building Z!", delay_minutes=5
        )
        handler._route_response(
            _make_msg(),
            channel,
            "ok",
            schedule_state,
            EditReplyState(),
            QueueReplyState(),
            _make_session(),
        )
        handler._scheduler.schedule.assert_called_once()  # not suppressed when unconfigured


class TestAbandonsPreferredDetection:
    """#2386 gap 2 — the `_abandons_preferred` predicate (pure, deterministic)."""

    _TERMS = ["Horizon"]

    def test_decline_of_preferred_fires(self):
        from cogtrix_core.assistant.handler import _abandons_preferred

        assert (
            _abandons_preferred("Sure, I'll steer clear of Horizon then.", self._TERMS) == "Horizon"
        )

    def test_retention_phrase_suppresses(self):
        from cogtrix_core.assistant.handler import _abandons_preferred

        # Declines something else but explicitly keeps the preferred option.
        assert (
            _abandons_preferred(
                "I'll avoid the pricey units, but let's keep Horizon in play.", self._TERMS
            )
            is None
        )

    def test_preferred_mentioned_positively_is_not_abandonment(self):
        from cogtrix_core.assistant.handler import _abandons_preferred

        assert _abandons_preferred("Horizon looks like a great fit!", self._TERMS) is None

    def test_no_preferred_terms_is_noop(self):
        from cogtrix_core.assistant.handler import _abandons_preferred

        assert _abandons_preferred("I'll steer clear of Horizon.", []) is None


class TestPersonaPreferredOptionGuard:
    """#2386 gap 2: a persona reply must not ABANDON an operator-declared preferred
    term on a bare negative claim without pushing back. On a violation the model is
    re-prompted once to defend the option — but the reply is NEVER suppressed (a
    soft preference miss must not drop an otherwise-deliverable message). Generic —
    the term comes from config (synthetic placeholder here)."""

    _CFG = {"guardrails": {"persona_constraints": {"preferred_terms": ["Horizon"]}}}

    def _run(self, agent_reply: str, rewrite_reply: str) -> tuple[MagicMock, MagicMock]:
        channel = MagicMock()
        channel.send.return_value = MagicMock(ok=True, message_id="x")
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=rewrite_reply)
        handler, _ = _make_handler(
            agent_runner=MagicMock(return_value=agent_reply),
            config=self._CFG,
            llm=llm,
        )
        handler.handle(_make_msg(), channel)
        return channel, llm

    def test_abandonment_is_reprompted_and_defended_reply_sent(self):
        channel, llm = self._run(
            agent_reply="You're right, I'll steer clear of Horizon then.",
            rewrite_reply="What's the concern with Horizon? It fits your brief — worth a look.",
        )
        assert llm.invoke.called  # re-prompted to push back
        assert channel.send.called
        sent = channel.send.call_args[0][1]
        assert "What's the concern with Horizon" in sent

    def test_persistent_abandonment_is_still_sent_not_suppressed(self):
        # Even if the re-prompt STILL abandons the preferred term, the reply is
        # delivered — gap 2 never suppresses (unlike the excluded-term guard).
        channel, llm = self._run(
            agent_reply="Okay, I'll avoid Horizon.",
            rewrite_reply="Alright, dropping Horizon.",
        )
        assert llm.invoke.called
        assert channel.send.called

    def test_positive_mention_not_reprompted(self):
        channel, llm = self._run(
            agent_reply="Horizon is a great fit — want me to book a viewing?",
            rewrite_reply="(unused)",
        )
        assert not llm.invoke.called
        assert channel.send.called

    def test_no_preferred_terms_configured_is_noop(self):
        channel = MagicMock()
        channel.send.return_value = MagicMock(ok=True, message_id="x")
        handler, _ = _make_handler(
            agent_runner=MagicMock(return_value="You're right, I'll steer clear of Horizon."),
        )
        handler.handle(_make_msg(), channel)
        assert channel.send.called  # unconfigured → guard is a no-op, reply sent
