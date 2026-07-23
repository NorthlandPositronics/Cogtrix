"""Unit tests for extracted orchestration recovery nodes."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.modifier import RemoveMessage

from cogtrix_core.orchestration.nodes.recovery import (
    _RECOVERY_INJECTED_MARKER,
    _find_current_turn_start,
    build_handle_action_intent_node,
    build_handle_phantom_node,
)
from cogtrix_core.orchestration.nodes.recovery import HumanMessage as _RecoveryHumanMessage


class _DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[object, ...]] = []
        self.infos: list[tuple[object, ...]] = []

    def warning(self, *args: object) -> None:
        self.warnings.append(args)

    def info(self, *args: object) -> None:
        self.infos.append(args)


def _state(message_id: str = "msg-1") -> dict:
    return {"messages": [AIMessage(content="stub", id=message_id)]}


class TestHandlePhantomNode:
    def test_injects_parse_retry_hint_before_exhaustion(self) -> None:
        phantom_count = [0]
        logger = _DummyLogger()
        node = build_handle_phantom_node(phantom_count, max_retries=3, logger=lambda: logger)

        result = node(_state("p1"))

        assert phantom_count[0] == 1
        assert len(result["messages"]) == 2
        assert result["messages"][0] == RemoveMessage(id="p1")
        assert isinstance(result["messages"][1], HumanMessage)
        assert "could not be parsed" in result["messages"][1].content
        assert logger.warnings

    def test_falls_back_after_retry_budget_is_exceeded(self) -> None:
        phantom_count = [3]
        logger = _DummyLogger()
        node = build_handle_phantom_node(phantom_count, max_retries=3, logger=lambda: logger)

        result = node(_state("p2"))

        assert phantom_count[0] == 4
        assert len(result["messages"]) == 2
        assert result["messages"][0] == RemoveMessage(id="p2")
        # The give-up branch now synthesizes from accumulated state instead
        # of returning the hard-coded "persistent formatting issues" message.
        # With no accumulated checkpoints / tool results, the fallback is the
        # polite "rephrase the question" prompt.
        assert isinstance(result["messages"][1], AIMessage)
        assert "rephrase" in result["messages"][1].content.lower()


class TestHandleActionIntentNode:
    def test_injects_nudge_before_exhaustion(self) -> None:
        action_intent_count = [0]
        logger = _DummyLogger()
        node = build_handle_action_intent_node(
            action_intent_count,
            max_retries=3,
            logger=lambda: logger,
        )

        result = node(_state("a1"))

        assert action_intent_count[0] == 1
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert "did not call any tools" in result["messages"][0].content
        assert logger.warnings

    def test_gives_up_after_retry_budget_is_exceeded(self) -> None:
        action_intent_count = [3]
        logger = _DummyLogger()
        node = build_handle_action_intent_node(
            action_intent_count,
            max_retries=3,
            logger=lambda: logger,
        )

        result = node(_state("a2"))

        assert action_intent_count[0] == 4
        # Previously returned [] which let the model's stuck-thinking output
        # stand as the user-facing response.  After the May 2026 user
        # disaster fix, the give-up branch synthesizes a clean answer from
        # accumulated state; with no checkpoints the polite fallback fires.
        assert len(result["messages"]) == 2
        assert result["messages"][0] == RemoveMessage(id="a2")
        assert isinstance(result["messages"][1], AIMessage)
        assert "rephrase" in result["messages"][1].content.lower()


class TestRecoveryInjectedHumanMessageMarker:
    """Recovery-injected ``HumanMessage``s must carry the marker so
    ``_find_current_turn_start`` can distinguish them from a real user
    turn.  Without the marker, detectors that read ``sources.user_prompt``
    latch onto nudge text and fire a cascade of false positives — see
    the CI failure on ``regression_persist_before_refusing × kimi-k2-5``
    where the fabricated-action nudge ("no ToolMessage appears … do NOT
    repeat …") triggered ``detect_topic_substitution`` to extract
    ``ToolMessage`` and ``NOT`` as missing user subjects.
    """

    def test_action_intent_nudge_is_marked_recovery_injected(self) -> None:
        logger = _DummyLogger()
        node = build_handle_action_intent_node(
            action_intent_count=[0],
            max_retries=3,
            logger=lambda: logger,
        )

        result = node(_state("a1"))

        nudge = result["messages"][0]
        assert isinstance(nudge, HumanMessage)
        # The marker MUST be present; otherwise downstream detectors
        # cannot tell a nudge from a real user turn.
        assert (nudge.additional_kwargs or {}).get(_RECOVERY_INJECTED_MARKER) is True, (
            "Recovery-injected nudges must be tagged so _find_current_turn_start "
            f"can skip them; got additional_kwargs={nudge.additional_kwargs!r}"
        )

    def test_recovery_humanmessage_factory_tags_message(self) -> None:
        """The module-local ``HumanMessage`` factory must tag every
        message it produces with the recovery marker — that is the
        whole point of shadowing the langchain import.
        """
        msg = _RecoveryHumanMessage(content="hello")

        assert isinstance(msg, HumanMessage)
        assert msg.content == "hello"
        assert (msg.additional_kwargs or {}).get(_RECOVERY_INJECTED_MARKER) is True

    def test_recovery_humanmessage_factory_preserves_extra_kwargs(self) -> None:
        """Callers may supply their own ``additional_kwargs`` — the
        recovery marker must be merged with them, not clobber them.
        """
        msg = _RecoveryHumanMessage(
            content="hello",
            additional_kwargs={"diagnostic_key": "value"},
        )

        kwargs = msg.additional_kwargs or {}
        assert kwargs.get(_RECOVERY_INJECTED_MARKER) is True
        assert kwargs.get("diagnostic_key") == "value"


class TestFindCurrentTurnStart:
    """``_find_current_turn_start`` must return the index of the real
    user turn, skipping any recovery-injected ``HumanMessage``s ahead
    of it in the message list.  Pre-fix the function returned the
    most recent HumanMessage regardless of provenance, which let
    nudge text bleed into ``sources.user_prompt`` for the grounding
    detectors.
    """

    def test_returns_real_user_turn_when_no_recovery_messages(self) -> None:
        msgs = [
            HumanMessage(content="What is the status of INV-2026-0428?", id="u1"),
            AIMessage(content="Let me check.", id="a1"),
        ]

        assert _find_current_turn_start(msgs) == 0

    def test_skips_recovery_nudges_when_finding_user_turn(self) -> None:
        """The CI repro: a real user turn, an assistant fabrication,
        then a recovery nudge.  The detector pipeline needs the index
        of the *real* user turn, not the recovery nudge.
        """
        msgs = [
            HumanMessage(content="Find me an open-source port of FooBar.", id="u1"),
            AIMessage(content="Sure, I'll search the web…", id="a1"),
            _RecoveryHumanMessage(
                content="Your response claims … no ToolMessage appears … do NOT repeat …",
                id="nudge-1",
            ),
        ]

        # Without the fix, this would return 2 (the recovery nudge),
        # causing the detector to read nudge text as the user prompt.
        assert _find_current_turn_start(msgs) == 0

    def test_skips_multiple_stacked_recovery_nudges(self) -> None:
        """When the recovery cascade fires more than once in a row
        (action-intent → fabricated-action → topic-substitution …),
        every injected nudge must be skipped until the real user
        turn is found.
        """
        msgs = [
            HumanMessage(content="Show me the Q3 supplier report.", id="u1"),
            AIMessage(content="Pulling the report now…", id="a1"),
            _RecoveryHumanMessage(content="action-intent nudge", id="nudge-1"),
            AIMessage(content="Here is what I think…", id="a2"),
            _RecoveryHumanMessage(content="fabricated-action nudge", id="nudge-2"),
            AIMessage(content="Let me try again…", id="a3"),
            _RecoveryHumanMessage(content="topic-substitution nudge", id="nudge-3"),
        ]

        assert _find_current_turn_start(msgs) == 0

    def test_falls_back_to_zero_when_only_recovery_nudges_exist(self) -> None:
        """Defensive: if for some reason every HumanMessage in the
        stream is a recovery nudge (no real user turn), fall back to
        index 0 rather than raising.  Matches the pre-fix behaviour
        for a message list with no HumanMessage at all.
        """
        msgs = [
            _RecoveryHumanMessage(content="nudge A", id="nudge-1"),
            _RecoveryHumanMessage(content="nudge B", id="nudge-2"),
        ]

        assert _find_current_turn_start(msgs) == 0

    def test_real_humanmessage_after_recovery_nudge_takes_precedence(self) -> None:
        """If a real user follow-up arrives after a recovery nudge
        (multi-turn scenario), the real turn is the one that
        ``_find_current_turn_start`` must point at.
        """
        msgs = [
            HumanMessage(content="First question.", id="u1"),
            AIMessage(content="first answer", id="a1"),
            _RecoveryHumanMessage(content="action-intent nudge", id="nudge-1"),
            AIMessage(content="revised answer", id="a2"),
            HumanMessage(content="Second question.", id="u2"),
        ]

        assert _find_current_turn_start(msgs) == 4


class TestTopicSubstitutionDoesNotFireOnRecoveryNudge:
    """End-to-end reproducer for the CI failure: a real user turn with
    no CamelCase, followed by an assistant fabrication, followed by a
    recovery nudge containing 'ToolMessage' and 'NOT'.  Pre-fix the
    topic-substitution detector latched onto those tokens and flagged
    the next assistant response as off-topic; post-fix the recovery
    nudge is skipped and the detector sees the real (plain-text) user
    prompt, which contains no distinctive subjects, so the detector
    correctly returns no missing subjects.
    """

    def test_topic_substitution_skips_recovery_nudge(self) -> None:
        from cogtrix_core.orchestration.verification import (
            collect_grounded_sources,
            detect_topic_substitution,
        )

        # Real user prompt: a plain question with NO CamelCase / acronyms
        # that the detector would key on.
        real_user_prompt = (
            "Find me a good open-source reimplementation or community "
            "port of the imaginary tool — i would like the github repo "
            "url, the language it is written in, and the build status."
        )

        # The fabricated-action recovery nudge text the CI failure
        # latched onto.  Critically: contains the literal substrings
        # 'ToolMessage' (CamelCase compound, 11 chars) and 'NOT'
        # (3-char all-caps acronym not in _GENERIC_ACRONYMS) which the
        # distinctive-subject extractor would pick up if it ever read
        # this message as ``user_prompt``.
        nudge_text = (
            "Your response claims a file or system change was completed, "
            "but you did not invoke any tool in this turn — no tool call "
            "was issued and no ToolMessage appears in the conversation. "
            "You cannot have performed the action. Choose exactly one "
            "path and answer accordingly — do NOT repeat the false-"
            "completion claim."
        )

        msgs = [
            HumanMessage(content=real_user_prompt, id="u1"),
            AIMessage(content="I'll search the web for this.", id="a1"),
            _RecoveryHumanMessage(content=nudge_text, id="nudge-1"),
        ]

        turn_start = _find_current_turn_start(msgs)
        sources = collect_grounded_sources(msgs, turn_start)

        # Sanity check: the user_prompt the detector sees is the REAL
        # one, not the nudge.  Pre-fix this assertion would fail —
        # ``sources.user_prompt`` would equal ``nudge_text``.
        assert sources.user_prompt == real_user_prompt, (
            "Recovery nudge leaked into sources.user_prompt — the fix "
            f"is not in place.  Got: {sources.user_prompt!r}"
        )

        # The agent's next response is unrelated to the nudge tokens
        # and is on-topic for the real user prompt.
        agent_response = (
            "I searched the web but could not find any open-source "
            "reimplementation of the tool you mentioned.  No "
            "github.com repository surfaced for that name in any of the "
            "queries I ran, and I did not find any matching projects "
            "covered in the available knowledge sources for this run."
        )

        missing = detect_topic_substitution(agent_response, sources=sources)

        # Pre-fix: would return ['ToolMessage', 'NOT'] from the nudge.
        # Post-fix: real user prompt has no distinctive subjects, so
        # the detector returns no missing subjects.
        assert missing == [], (
            "Topic-substitution detector misfired on recovery-nudge "
            f"tokens; missing subjects = {missing!r}"
        )
