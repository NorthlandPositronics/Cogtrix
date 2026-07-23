"""Tests for the #2015 corpus-aware attribution-mismatch recovery node.

Covers ``build_handle_corpus_attribution_mismatch_node`` from
``src/orchestration/nodes/recovery.py``.  The node is corpus-agnostic:
it consumes a caller-supplied ``corpus_attribution_detector`` closure
that returns a list of human-readable mismatch strings.  Tests inject
fake detectors so the cases are deterministic and don't depend on the
PM corpus.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.modifier import RemoveMessage

from src.orchestration.nodes.recovery import (
    _format_corpus_attribution_mismatch_nudge,
    build_handle_corpus_attribution_mismatch_node,
)

# ── Test fixtures ───────────────────────────────────────────────────


class _DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[object, ...]] = []
        self.infos: list[tuple[object, ...]] = []

    def warning(self, *args: object) -> None:
        self.warnings.append(args)

    def info(self, *args: object) -> None:
        self.infos.append(args)


def _state(response: str, msg_id: str = "a1") -> dict:
    """Build a minimal CogtrixState shape: one AIMessage with the given content."""
    return {"messages": [AIMessage(content=response, id=msg_id)]}


# ── Nudge formatter ────────────────────────────────────────────────


class TestFormatNudge:
    def test_single_mismatch_renders_clean_message(self) -> None:
        nudge = _format_corpus_attribution_mismatch_nudge(
            ["R-13 attributed to 'Hyeon-Jin Park' but corpus owners are {Tomislav Hessford}"]
        )
        # The mismatch text appears verbatim.
        assert (
            "R-13 attributed to 'Hyeon-Jin Park' but corpus owners are {Tomislav Hessford}" in nudge
        )
        # And both imperative revision options are spelled out.  Wording
        # tightened in #2006 cycle-10 post-mortem (REPLACE / DROP — the
        # softer "Correct" / "Omit" got ignored by the model after the
        # nudge).
        assert "REPLACE" in nudge
        assert "DROP" in nudge

    def test_count_and_plural_render_correctly(self) -> None:
        # Multi-mismatch case — counts shown, plural noun used.
        nudge = _format_corpus_attribution_mismatch_nudge(
            [
                "R-12 attributed to 'X' but corpus owners are {A}",
                "R-13 attributed to 'Y' but corpus owners are {B}",
            ]
        )
        assert "2 entity-owner attribution mismatches" in nudge

        # Single-mismatch case — no plural "s".
        single = _format_corpus_attribution_mismatch_nudge(
            ["R-12 attributed to 'X' but corpus owners are {A}"]
        )
        assert "1 entity-owner attribution mismatch versus" in single

    def test_references_tracking_issues(self) -> None:
        nudge = _format_corpus_attribution_mismatch_nudge(
            ["R-12 attributed to 'X' but corpus owners are {A}"]
        )
        # Provenance for future readers tracing why this exists.
        assert "#2015" in nudge


# ── Recovery node — happy path ──────────────────────────────────────


class TestHandleCorpusAttributionMismatchHappyPath:
    def test_emits_remove_plus_nudge_when_detector_reports_mismatch(self) -> None:
        counter = [0]
        logger = _DummyLogger()

        # Fake detector: always returns one mismatch.
        def fake_detector(_response: str) -> list[str]:
            return ["R-13 attributed to 'Hyeon-Jin Park' but corpus owners are {Tomislav Hessford}"]

        node = build_handle_corpus_attribution_mismatch_node(
            corpus_attribution_mismatch_count=counter,
            max_retries=2,
            corpus_attribution_detector=fake_detector,
            logger=lambda: logger,
        )

        result = node(_state("bad attribution", msg_id="msg-1"))

        # Counter advanced.
        assert counter[0] == 1
        # Two messages: RemoveMessage for the bad response + HumanMessage nudge.
        assert len(result["messages"]) == 2
        assert result["messages"][0] == RemoveMessage(id="msg-1")
        assert isinstance(result["messages"][1], HumanMessage)
        assert "R-13" in result["messages"][1].content
        assert logger.warnings, "warning log expected on every detector firing"


# ── Recovery node — defensive paths ────────────────────────────────


class TestHandleCorpusAttributionMismatchDefensivePaths:
    def test_returns_empty_when_detector_reports_no_mismatches(self) -> None:
        counter = [0]

        def clean_detector(_response: str) -> list[str]:
            return []

        node = build_handle_corpus_attribution_mismatch_node(
            corpus_attribution_mismatch_count=counter,
            max_retries=2,
            corpus_attribution_detector=clean_detector,
            logger=_DummyLogger,
        )

        result = node(_state("clean response", msg_id="msg-1"))

        # Counter still advances (the node was called) but no
        # messages are returned — caller treats this as "no action
        # required, route forward normally".
        assert counter[0] == 1
        assert result["messages"] == []

    def test_returns_empty_when_detector_raises(self) -> None:
        counter = [0]
        logger = _DummyLogger()

        def broken_detector(_response: str) -> list[str]:
            raise RuntimeError("simulated detector failure")

        node = build_handle_corpus_attribution_mismatch_node(
            corpus_attribution_mismatch_count=counter,
            max_retries=2,
            corpus_attribution_detector=broken_detector,
            logger=lambda: logger,
        )

        result = node(_state("any response", msg_id="msg-1"))

        # Detector crash is contained — node returns no-op rather than
        # propagating the exception (which would kill the whole graph run).
        assert result["messages"] == []
        # And a warning was logged for triage visibility.
        assert any("corpus_attribution_detector raised" in str(w) for w in logger.warnings)

    def test_returns_empty_for_non_string_response_content(self) -> None:
        """Some assistant messages carry list-of-blocks content (e.g.
        Anthropic's content-block format).  The detector takes a str;
        the node should short-circuit on non-str rather than blow up.
        """
        counter = [0]

        def fake_detector(_response: str) -> list[str]:
            return ["should never fire"]

        node = build_handle_corpus_attribution_mismatch_node(
            corpus_attribution_mismatch_count=counter,
            max_retries=2,
            corpus_attribution_detector=fake_detector,
            logger=_DummyLogger,
        )

        # AIMessage with non-str content.
        state = {"messages": [AIMessage(content=[{"type": "text", "text": "hi"}], id="m")]}
        result = node(state)

        assert result["messages"] == []

    def test_returns_empty_when_message_list_is_empty(self) -> None:
        counter = [0]

        def fake_detector(_response: str) -> list[str]:
            return ["should never fire"]

        node = build_handle_corpus_attribution_mismatch_node(
            corpus_attribution_mismatch_count=counter,
            max_retries=2,
            corpus_attribution_detector=fake_detector,
            logger=_DummyLogger,
        )

        result = node({"messages": []})
        assert result["messages"] == []


# ── Recovery node — retry budget ────────────────────────────────────


class TestHandleCorpusAttributionMismatchRetryBudget:
    def test_accepts_response_after_budget_exhausted(self) -> None:
        """After ``max_retries`` attempts the response ships as-is —
        the model demonstrably can't self-correct and we'd rather emit
        the bad response than spin forever.  Same shape as the other
        prose-fidelity recovery nodes.
        """
        # Counter already at the budget — the next call exceeds it.
        counter = [2]
        logger = _DummyLogger()

        def fake_detector(_response: str) -> list[str]:
            return ["R-13 attributed to 'X' but corpus owners are {Y}"]

        node = build_handle_corpus_attribution_mismatch_node(
            corpus_attribution_mismatch_count=counter,
            max_retries=2,
            corpus_attribution_detector=fake_detector,
            logger=lambda: logger,
        )

        result = node(_state("bad attribution again", msg_id="msg-1"))

        # Counter still ticked.
        assert counter[0] == 3
        # No messages returned (no RemoveMessage, no nudge) — response ships.
        assert result["messages"] == []
        # An info log records the give-up for triage visibility.
        assert any("retries exhausted" in str(i) for i in logger.infos)

    def test_first_retry_still_revises(self) -> None:
        """Counter at max_retries means we're on the LAST allowed retry
        — the node should still emit RemoveMessage + nudge.  Only when
        counter exceeds max_retries do we accept the response.
        """
        counter = [1]  # at max_retries=2, this is attempt 2/2
        logger = _DummyLogger()

        def fake_detector(_response: str) -> list[str]:
            return ["R-13 attributed to 'X' but corpus owners are {Y}"]

        node = build_handle_corpus_attribution_mismatch_node(
            corpus_attribution_mismatch_count=counter,
            max_retries=2,
            corpus_attribution_detector=fake_detector,
            logger=lambda: logger,
        )

        result = node(_state("bad attribution", msg_id="msg-1"))

        # Counter advances to 2 — STILL within budget.
        assert counter[0] == 2
        # Nudge is still emitted (RemoveMessage + HumanMessage).
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][1], HumanMessage)
