"""Regression tests for the production step-limit recovery cascade (#2444).

``recover_from_step_limit`` (cogtrix_core/orchestration/phases.py) fires whenever the
agent graph exhausts ``recursion_limit`` without producing a usable answer.
Prior to this file it had ZERO direct tests — the only coverage was via
full-graph integration paths that never actually forced the RecursionError
branch.  These tests pin the real cascade order:

  1. Retry with a tight nudge (``recursion_limit=4``, "answer now, no tools").
  2. If the retry ALSO raises RecursionError, or its answer is itself a
     step-limit apology, fall through to ``build_tool_results_response``.
  3. If no tool results exist, fall through to ``extract_partial_results``.
  4. If nothing usable was found anywhere, return the fixed
     ``RECOVERY_FAILED_MESSAGE`` sentinel.

Also covers ``is_step_limit_apology`` (the gate that keeps step 1 from
short-circuiting on a disguised give-up) and ``extract_partial_results``
(the structured last-mile extractor) directly, since neither had unit
coverage either.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cogtrix_core.orchestration.phases import (
    RECOVERY_FAILED_MESSAGE,
    extract_partial_results,
    is_step_limit_apology,
    recover_from_step_limit,
)


class _DummyLogger:
    """Minimal logger stub matching the call sites used in phases.py."""

    def __init__(self) -> None:
        self.infos: list[tuple[object, ...]] = []
        self.warnings: list[tuple[object, ...]] = []
        self.errors: list[tuple[object, ...]] = []

    def info(self, *args: object) -> None:
        self.infos.append(args)

    def warning(self, *args: object) -> None:
        self.warnings.append(args)

    def error(self, *args: object) -> None:
        self.errors.append(args)


class _StreamStub:
    """Fake ``agent_executor`` exposing only the ``.stream()`` surface that
    ``recover_from_step_limit`` calls.

    ``chunks_or_error`` is a list — one entry per expected call to
    ``.stream()``.  Each entry is either a list of chunk dicts to yield, or
    the ``RecursionError`` exception class/instance to raise instead.
    """

    def __init__(self, chunks_or_error: list[Any]) -> None:
        self._plan = chunks_or_error
        self.calls: list[dict[str, Any]] = []

    def stream(self, inputs: dict, config: dict, stream_mode: str) -> Any:
        self.calls.append({"inputs": inputs, "config": config, "stream_mode": stream_mode})
        plan = self._plan[len(self.calls) - 1]
        if isinstance(plan, type) and issubclass(plan, BaseException):
            raise plan("stream raised")
        if isinstance(plan, BaseException):
            raise plan
        yield from plan


# ─────────────────────────────────────────────────────────────────────────────
# is_step_limit_apology
# ─────────────────────────────────────────────────────────────────────────────


class TestIsStepLimitApology:
    def test_short_give_up_phrase_is_detected(self) -> None:
        assert is_step_limit_apology("Sorry, I need more steps to process this request.")

    def test_long_legitimate_answer_mentioning_steps_is_not_flagged(self) -> None:
        long_answer = (
            "Here are the deployment steps: 1) build the image, 2) push to the "
            "registry, 3) update the manifest. " + ("Additional detail. " * 20)
        )
        assert len(long_answer) > 300
        assert not is_step_limit_apology(long_answer)

    def test_negated_need_is_not_flagged(self) -> None:
        assert not is_step_limit_apology("I don't need more steps — here is the answer: 42.")

    def test_empty_text_is_not_flagged(self) -> None:
        assert not is_step_limit_apology("")
        assert not is_step_limit_apology(None)  # type: ignore[arg-type]

    def test_iteration_limit_phrase_is_detected(self) -> None:
        assert is_step_limit_apology("I've hit the iteration limit before finishing.")


# ─────────────────────────────────────────────────────────────────────────────
# extract_partial_results
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractPartialResults:
    def test_returns_none_for_empty_messages(self) -> None:
        assert extract_partial_results([]) is None

    def test_returns_none_when_nothing_usable(self) -> None:
        # Only errors and short AI content — nothing worth surfacing.
        msgs = [
            HumanMessage(content="do the thing"),
            ToolMessage(content="Error: not found", tool_call_id="tc1", name="search_tool"),
            AIMessage(content="ok"),
        ]
        assert extract_partial_results(msgs) is None

    def test_pulls_real_tool_results_and_last_long_ai_content(self) -> None:
        msgs = [
            HumanMessage(content="research the topic"),
            ToolMessage(
                content="Company X reported $4.2B revenue in FY25.",
                tool_call_id="tc1",
                name="search_tool",
            ),
            AIMessage(
                content=(
                    "Based on the search results so far, Company X appears to be "
                    "growing steadily year over year."
                )
            ),
        ]
        result = extract_partial_results(msgs)
        assert result is not None
        assert "Company X reported $4.2B revenue" in result
        assert "search_tool" in result
        assert "growing steadily" in result
        # Must not start with an error prefix (so the caller treats the
        # turn as a valid, savable response — see the Ralph Loop note in
        # the source docstring).
        assert not result.startswith("Error")

    def test_error_tool_results_are_excluded(self) -> None:
        msgs = [
            ToolMessage(content="Error: timeout", tool_call_id="tc1", name="search_tool"),
            ToolMessage(
                content="Real finding: the price is $19.99.",
                tool_call_id="tc2",
                name="search_tool",
            ),
        ]
        result = extract_partial_results(msgs)
        assert result is not None
        assert "Error: timeout" not in result
        assert "Real finding" in result

    def test_caps_tool_results_at_ten(self) -> None:
        msgs = [
            ToolMessage(content=f"Finding number {i}", tool_call_id=f"tc{i}", name="t")
            for i in range(15)
        ]
        result = extract_partial_results(msgs)
        assert result is not None
        assert "Finding number 0" in result
        assert "Finding number 9" in result
        assert "Finding number 10" not in result


# ─────────────────────────────────────────────────────────────────────────────
# recover_from_step_limit — full cascade
# ─────────────────────────────────────────────────────────────────────────────


class TestRecoverFromStepLimitCascade:
    def test_retry_with_nudge_succeeds_and_uses_tight_recursion_limit(self) -> None:
        """Step 1: the retry re-invokes with recursion_limit=4 and the
        "answer now, no tools" nudge; a real answer on retry short-circuits
        the rest of the cascade."""
        input_messages = [HumanMessage(content="What is the exchange rate?")]
        result = {
            "messages": [
                *input_messages,
                AIMessage(content="", tool_calls=[{"name": "fx", "args": {}, "id": "tc1"}]),
            ]
        }
        recovered = AIMessage(content="The current exchange rate is 1.08 USD/EUR.")
        stub = _StreamStub([[{"messages": [*result["messages"], recovered]}]])
        log = _DummyLogger()

        response = recover_from_step_limit(stub, result, input_messages, {}, log)

        assert response == "The current exchange rate is 1.08 USD/EUR."
        assert len(stub.calls) == 1, "Only the retry step should have run — no fallback needed"
        call = stub.calls[0]
        assert call["config"]["recursion_limit"] == 4, (
            "Retry must use the tight recursion_limit=4 ceiling so a model "
            "that ignores the nudge can't burn the whole recovery budget"
        )
        nudge_texts = [m.content for m in call["inputs"]["messages"] if isinstance(m, HumanMessage)]
        assert any("Do NOT call any more tools" in t for t in nudge_texts), (
            "Retry must inject the 'answer now, no tools' nudge as the "
            f"final message; got nudges: {nudge_texts!r}"
        )

    def test_apology_on_retry_falls_through_to_tool_results(self) -> None:
        """Step 1 producing a disguised give-up ("need more steps") must be
        gated by is_step_limit_apology and NOT accepted as the final answer
        — the cascade must fall through to step 2."""
        input_messages = [HumanMessage(content="Summarize the findings")]
        original_tool_msg = ToolMessage(
            content="The report shows Q3 revenue grew 12% year over year.",
            tool_call_id="tc1",
            name="report_tool",
        )
        result = {"messages": [*input_messages, original_tool_msg]}
        apology = AIMessage(content="Sorry, I need more steps to finish this.")
        assert is_step_limit_apology(apology.content)
        stub = _StreamStub([[{"messages": [*result["messages"], apology]}]])
        log = _DummyLogger()

        response = recover_from_step_limit(stub, result, input_messages, {}, log)

        assert response is not None
        assert "need more steps" not in response.lower(), (
            "The step-limit apology must never be returned as the recovered "
            f"answer; got: {response!r}"
        )
        assert "Q3 revenue grew 12%" in response, (
            "Step 2 (build_tool_results_response) must surface the real tool "
            f"finding once the apology is rejected; got: {response!r}"
        )

    def test_retry_also_recursing_does_not_crash_and_degrades_gracefully(self) -> None:
        """Step 1 itself raising RecursionError must be swallowed, logged,
        and degrade to the next cascade step rather than propagating."""
        input_messages = [HumanMessage(content="Deep research task")]
        original_tool_msg = ToolMessage(
            content="Partial finding: three vendors matched the criteria.",
            tool_call_id="tc1",
            name="search_tool",
        )
        result = {"messages": [*input_messages, original_tool_msg]}
        stub = _StreamStub([RecursionError])
        log = _DummyLogger()

        response = recover_from_step_limit(stub, result, input_messages, {}, log)

        assert response is not None
        assert "three vendors matched" in response
        assert any(
            "recursion" in str(args).lower() for args in log.warnings
        ), f"A RecursionError on retry must be logged as a warning; got: {log.warnings!r}"

    def test_last_resort_returns_recovery_failed_message(self) -> None:
        """When retry recurses AND there are no tool results AND no partial
        content exists anywhere, the fixed RECOVERY_FAILED_MESSAGE sentinel
        must be returned so callers (e.g. the assistant handler) can
        recognize it and decide whether to deliver it (#2052)."""
        input_messages = [HumanMessage(content="Impossible task")]
        result = {"messages": list(input_messages)}
        stub = _StreamStub([RecursionError])
        log = _DummyLogger()

        response = recover_from_step_limit(stub, result, input_messages, {}, log)

        assert response == RECOVERY_FAILED_MESSAGE
        assert log.errors, "Last-resort path must log an error for observability"

    def test_recovered_answer_from_retry_is_preferred_over_tool_results(self) -> None:
        """Sanity: when the retry genuinely succeeds, its answer wins even
        though tool results (which would also satisfy step 2) exist —
        the cascade must short-circuit at the earliest successful step."""
        input_messages = [HumanMessage(content="q")]
        result = {
            "messages": [
                *input_messages,
                ToolMessage(
                    content="Some tool finding that step 2 would also surface.",
                    tool_call_id="tc1",
                    name="t",
                ),
            ]
        }
        recovered = AIMessage(content="Direct recovered answer from the retry.")
        stub = _StreamStub([[{"messages": [*result["messages"], recovered]}]])
        log = _DummyLogger()

        response = recover_from_step_limit(stub, result, input_messages, {}, log)

        assert response == "Direct recovered answer from the retry."
        assert len(stub.calls) == 1
