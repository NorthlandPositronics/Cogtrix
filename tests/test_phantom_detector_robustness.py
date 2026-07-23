"""Holistic regression bed for the markdown-phantom detector.

The cogtrix45.log incident (Bug K) demonstrated that
``_looks_like_markdown_phantom_report`` could misclassify any
well-structured technical answer as a fabricated tool report,
triggering a recovery loop that eventually caused the LLM to
topic-drift to stale conversation history. PR #1695 fixed the
detector by requiring a claim-of-action content signal alongside
the structural markers (markdown table + numbered section header).

This file pins that fix against the wider class of false positives.
Not just the specific agent-memory answer from cogtrix45 — every
common shape of legitimate structured technical content that AI
assistants routinely produce: architecture comparisons, code review
findings, API documentation, runbooks, schema comparisons, tradeoff
analyses, etc.

Also covers the symmetric side: explicit fabrication patterns
across multiple shapes should still be caught — so we don't
accidentally weaken the detector in a future refactor.

And one integration-level test that simulates the exact cogtrix45
multi-turn shape: a Vienna-shopping question, an ML question, a
systems-design question. The answer to question 3 must be about
question 3, not about question 1.

If a future change relaxes the detector and re-introduces the
false-positive class, these tests are the tripwire.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cogtrix_core.orchestration.graph import _looks_like_markdown_phantom_report

# ── Domain false-positives — must NOT be flagged as phantom ──────────


class TestDomainFalsePositives:
    """Each scenario is a domain where AI assistants routinely produce
    well-structured technical answers (markdown table + numbered
    section header) without claiming to have retrieved data.

    The pre-PR-#1695 detector flagged every one of these. The fixed
    detector must let them through — none of them claim a tool action,
    so none should match ``_FAKE_TOOL_OUTPUT_SIGNAL_RE``."""

    def test_agent_memory_systems_design(self) -> None:
        """Exact shape of the cogtrix45.log turn-3 false positive
        (the canonical regression case)."""
        msg = AIMessage(
            content=(
                "You're asking about long-context management.\n\n"
                "### Core Principles\n\n"
                "| Goal | Technique | Why it works |\n"
                "|------|-----------|--------------|\n"
                "| Prevent bloat | Summarization | Reduces tokens |\n"
                "| Avoid drift | Subagents | Fresh context per phase |\n\n"
                "#### 1. Progressive Summarization\n"
                "After each 5-7 turns, run a structured summary call.\n\n"
                "#### 2. Tool-Aware State Management\n"
                "Store only actionable state in external KV store.\n\n"
                "#### 3. Hierarchical Sub-agents\n"
                "Break the task into phases.\n"
            ),
            id="d1",
        )
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_architecture_comparison(self) -> None:
        """Comparing architectural patterns — common AI-assistant output
        for design questions. Tables compare trade-offs; numbered
        sections enumerate approaches."""
        msg = AIMessage(
            content=(
                "Choosing between event-driven and request-response\n"
                "depends on your throughput vs. latency profile.\n\n"
                "### Trade-off Matrix\n\n"
                "| Property | Event-driven | Request-response |\n"
                "|----------|--------------|------------------|\n"
                "| Throughput | High | Medium |\n"
                "| Latency | Variable | Predictable |\n"
                "| Debugging | Harder (traces) | Easier (stack) |\n\n"
                "#### 1. Use event-driven when\n"
                "Your write rate exceeds 10k/sec.\n\n"
                "#### 2. Use request-response when\n"
                "Per-call latency must be bounded.\n"
            ),
            id="d2",
        )
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_code_review_recommendations(self) -> None:
        """Code review answer — numbered findings + impact table. No
        claim of having scanned a repo."""
        msg = AIMessage(
            content=(
                "Reviewing the patch, here are improvements to consider.\n\n"
                "### Impact summary\n\n"
                "| Severity | Category | Recommendation |\n"
                "|----------|----------|----------------|\n"
                "| HIGH | Memory | Reuse buffers in hot loop |\n"
                "| LOW | Style | Rename `x` to `request_id` |\n\n"
                "#### 1. Memory allocation in the hot loop\n"
                "Allocating a new list per iteration costs ~80% of CPU.\n\n"
                "#### 2. Variable naming\n"
                "Short names hurt readability for new contributors.\n"
            ),
            id="d3",
        )
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_api_documentation_answer(self) -> None:
        """Tutorial-style API answer. Parameter table + numbered
        usage steps. No retrieval claim."""
        msg = AIMessage(
            content=(
                "To call the `/v1/sessions` endpoint:\n\n"
                "### Parameters\n\n"
                "| Name | Type | Required | Description |\n"
                "|------|------|----------|-------------|\n"
                "| user_id | string | yes | The owning user |\n"
                "| name | string | no | Session label |\n\n"
                "#### 1. Build the request\n"
                "Construct a POST with JSON body containing the params.\n\n"
                "#### 2. Authenticate\n"
                "Include a bearer token in the Authorization header.\n\n"
                "#### 3. Handle the response\n"
                "On success you get a 201 with the session ID.\n"
            ),
            id="d4",
        )
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_runbook_incident_response(self) -> None:
        """Ops runbook content. Symptom/cause table + numbered steps.
        Describes a procedure; doesn't claim to have executed it."""
        msg = AIMessage(
            content=(
                "If the queue worker stalls, walk through these steps.\n\n"
                "### Symptom matrix\n\n"
                "| Symptom | Likely cause | First-aid |\n"
                "|---------|--------------|-----------|\n"
                "| No new jobs picked up | Stale lock | Restart worker |\n"
                "| OOM after N hours | Memory leak | Roll the process |\n\n"
                "#### 1. Check the lock holder\n"
                "Inspect the lock table for stale entries.\n\n"
                "#### 2. Restart the worker\n"
                "Use the systemd unit; don't kill -9.\n"
            ),
            id="d5",
        )
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_database_schema_comparison(self) -> None:
        """Educational schema-design answer. Field-comparison table +
        numbered design considerations. No data retrieval claim."""
        msg = AIMessage(
            content=(
                "Choosing between a JSON column and a side table.\n\n"
                "### Comparison\n\n"
                "| Approach | Pros | Cons |\n"
                "|----------|------|------|\n"
                "| JSON column | Schema flex | No indexes |\n"
                "| Side table | Indexable | Joins required |\n\n"
                "#### 1. Use a JSON column when\n"
                "Fields evolve frequently and reads dominate.\n\n"
                "#### 2. Use a side table when\n"
                "You need to filter or aggregate on the fields.\n"
            ),
            id="d6",
        )
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_security_threat_analysis(self) -> None:
        """Security-style answer comparing attack vectors. Threat
        matrix + numbered mitigations. Describes vulnerabilities in
        the abstract; doesn't claim to have audited a system."""
        msg = AIMessage(
            content=(
                "Common attack surface for a multi-tenant API.\n\n"
                "### Threat matrix\n\n"
                "| Threat | Likelihood | Mitigation |\n"
                "|--------|------------|------------|\n"
                "| Token leak via logs | Medium | Redact at sink |\n"
                "| Cross-tenant read | Low | Row-level RBAC |\n\n"
                "#### 1. Token redaction\n"
                "Strip Authorization headers before structured logging.\n\n"
                "#### 2. Tenant isolation\n"
                "Enforce row filters at the ORM layer, not just route.\n"
            ),
            id="d7",
        )
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_machine_learning_training_recipe(self) -> None:
        """ML training methodology answer. Hyperparameter table +
        numbered procedure. Describes methodology; doesn't claim to
        have run any experiments."""
        msg = AIMessage(
            content=(
                "A typical recipe for instruction-tuning a 7B model.\n\n"
                "### Hyperparameters\n\n"
                "| Parameter | Value | Why |\n"
                "|-----------|-------|-----|\n"
                "| Learning rate | 2e-5 | Stable for AdamW |\n"
                "| Batch size | 32 | Fits 4x A100 |\n"
                "| Epochs | 3 | Avoids overfit |\n\n"
                "#### 1. Data preparation\n"
                "Filter for response length 100-2000 tokens.\n\n"
                "#### 2. Training loop\n"
                "Use gradient accumulation if VRAM is tight.\n"
            ),
            id="d8",
        )
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_kubernetes_deployment_options(self) -> None:
        """K8s-specific answer comparing deployment strategies."""
        msg = AIMessage(
            content=(
                "Choosing between Deployment and StatefulSet.\n\n"
                "### Suitability\n\n"
                "| Workload | Pick | Rationale |\n"
                "|----------|------|-----------|\n"
                "| Stateless API | Deployment | Fast rollout |\n"
                "| DB primary | StatefulSet | Stable hostname |\n\n"
                "#### 1. When Deployment fits\n"
                "Pods are interchangeable; no per-pod identity.\n\n"
                "#### 2. When StatefulSet fits\n"
                "Pods need ordered start, stable storage, or hostname.\n"
            ),
            id="d9",
        )
        assert _looks_like_markdown_phantom_report(msg) is False


# ── Domain true-positives — must STILL be flagged as phantom ─────────


class TestDomainTruePositives:
    """The detector's purpose is real. These scenarios are how
    fabricated tool reports actually look. The fix must not weaken
    the detector against them."""

    def test_fabricated_slack_check(self) -> None:
        """The shape that motivated the original detector (PR #170).
        Claims to have retrieved Slack messages without calling any
        Slack tool. Bullet-point past-tense claim."""
        msg = AIMessage(
            content=(
                "### 1. Slack Check — #cogtrix-project-discussions\n"
                "- Retrieved last 8 messages. No new mentions.\n\n"
                "### 2. Open Issues\n"
                "| Issue | Title | Updated |\n"
                "|-------|-------|---------|\n"
                "| #42 | Fix memory leak | 10 min ago |\n"
                "| #39 | Add API docs | 1 hour ago |\n"
            ),
            id="p1",
        )
        assert _looks_like_markdown_phantom_report(msg) is True

    def test_fabricated_search_with_first_person_claim(self) -> None:
        """LLM narrates a search it didn't run. First-person past-tense
        verb + object phrase: ``I searched the documentation``."""
        msg = AIMessage(
            content=(
                "### Findings\n\n"
                "| Source | Date | Verdict |\n"
                "|--------|------|---------|\n"
                "| RFC 9110 | 2022 | Confirms behaviour |\n"
                "| MDN docs | 2024 | Same |\n\n"
                "#### 1. Primary source\n"
                "I searched the documentation and found definitive guidance.\n\n"
                "#### 2. Cross-reference\n"
                "Both sources align.\n"
            ),
            id="p2",
        )
        assert _looks_like_markdown_phantom_report(msg) is True

    def test_fabricated_with_sources_section(self) -> None:
        """Report-style ``Sources:`` header. Common in models that
        hallucinate citations."""
        msg = AIMessage(
            content=(
                "### Summary\n\n"
                "| Claim | Confidence |\n"
                "|-------|-----------|\n"
                "| X is true | High |\n"
                "| Y depends on Z | Medium |\n\n"
                "#### 1. Detail one\n"
                "Lorem ipsum dolor sit amet.\n\n"
                "Sources:\n"
                "- https://example.com/a\n"
                "- https://example.com/b\n"
            ),
            id="p3",
        )
        assert _looks_like_markdown_phantom_report(msg) is True

    def test_fabricated_results_show_pattern(self) -> None:
        """Reporting-verb structure where ``results show`` acts as
        the subject — typical of LLM-hallucinated reports."""
        msg = AIMessage(
            content=(
                "### Analysis\n\n"
                "| Metric | Value |\n"
                "|--------|-------|\n"
                "| Avg latency | 84 ms |\n"
                "| p99 latency | 312 ms |\n\n"
                "#### 1. Latency breakdown\n"
                "The results show a clear bimodal distribution.\n\n"
                "#### 2. Recommendation\n"
                "Consider caching the slow path.\n"
            ),
            id="p4",
        )
        assert _looks_like_markdown_phantom_report(msg) is True

    def test_fabricated_with_according_to(self) -> None:
        """The ``According to my search`` claim pattern."""
        msg = AIMessage(
            content=(
                "### Documents reviewed\n\n"
                "| Doc | Section | Says |\n"
                "|-----|---------|------|\n"
                "| Style guide | 4.2 | Use kebab-case |\n"
                "| Internal RFC | 7 | Confirms |\n\n"
                "#### 1. Primary guidance\n"
                "According to my search, the team uses kebab-case for URLs.\n\n"
                "#### 2. Edge cases\n"
                "Snake_case is OK in JSON bodies.\n"
            ),
            id="p5",
        )
        assert _looks_like_markdown_phantom_report(msg) is True


# ── Structural edge cases ────────────────────────────────────────────


class TestStructuralEdgeCases:
    """Inputs that exercise the detector's preconditions: too-short
    bodies, missing structural markers, tool_calls present, etc."""

    def test_short_body_below_80_char_threshold(self) -> None:
        """Detector skips bodies under 80 chars even with all signals
        present — short responses are not the fabrication shape."""
        msg = AIMessage(
            content=("| a | b |\n|---|---|\n| 1 | 2 |\n#### 1. step\n" "I retrieved the data."),
            id="e1",
        )
        # Length check first — this content might be exactly at the
        # threshold so be specific.
        assert len(str(msg.content)) < 80
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_tool_calls_present_short_circuits(self) -> None:
        """If the AIMessage has actual tool_calls, it can't be a
        phantom — by definition. Detector returns False immediately."""
        tool_call = {
            "name": "web_search",
            "id": "tc1",
            "args": {"query": "x"},
            "type": "tool_call",
        }
        msg = AIMessage(
            content=(
                "### Search Results\n\n"
                "| Title | URL |\n|-------|-----|\n| X | y |\n\n"
                "#### 1. Top result\n"
                "I retrieved the documents.\n"
            ),
            tool_calls=[tool_call],
            id="e2",
        )
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_table_without_numbered_header_passes(self) -> None:
        """Just a table — not a fabrication shape."""
        msg = AIMessage(
            content=(
                "Here's a comparison of widgets:\n\n"
                "| Name | Size | Color |\n"
                "|------|------|-------|\n"
                "| A | small | red |\n"
                "| B | large | blue |\n\n"
                "Hope this helps. I retrieved the catalog for you.\n"
            ),
            id="e3",
        )
        # Has a claim-of-action ("I retrieved the catalog") and a table
        # but NO numbered section header — structural precondition
        # fails so the detector returns False.
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_numbered_header_without_table_passes(self) -> None:
        """Just numbered sections — not a fabrication shape."""
        msg = AIMessage(
            content=(
                "Steps to reproduce:\n\n"
                "#### 1. Set up the environment\n"
                "Install the venv and dependencies.\n\n"
                "#### 2. Run the suite\n"
                "I retrieved the latest test results just now.\n"
            ),
            id="e4",
        )
        assert _looks_like_markdown_phantom_report(msg) is False

    def test_non_string_content_returns_false(self) -> None:
        """LangChain sometimes carries content as a list of parts.
        Detector must not crash on those."""
        msg = AIMessage(
            content=[{"type": "text", "text": "hello"}],  # type: ignore[arg-type]
            id="e5",
        )
        assert _looks_like_markdown_phantom_report(msg) is False


# ── Multi-turn topic-stability integration test ──────────────────────


def _make_mock_llm(responses: list[AIMessage]) -> MagicMock:
    """Mock LLM that yields *responses* in order across .invoke() calls."""
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.side_effect = responses
    return llm


def _make_registry() -> MagicMock:
    reg = MagicMock()
    reg.requires_confirmation.return_value = False
    return reg


class TestMultiTurnTopicStability:
    """Mirror the cogtrix45.log shape: a multi-turn conversation
    where an earlier turn used tools (Vienna shopping) and a later
    turn's question is unrelated structured prose (systems design).
    The detector + recovery loop must not let stale tool context
    bleed into the later turn's response.

    This is the integration-level safety net. Even if the
    *detector* somehow misfires in a future refactor, the response
    to the current turn must be about the current turn's question."""

    def test_structured_answer_after_prior_tool_turn(self) -> None:
        """Conversation shape from cogtrix45:

          [0] User: "How many Soudal sealants for $100 NZD?" (turn 1)
          [1] AI: (tool call to web_search)
          [2] Tool: search results
          [3] AI: answer about Soudal
          [4] User: structured-prose question (turn 2)

        Turn 2's structured-prose answer must flow straight through
        without phantom recovery, even though the assistant has a
        full tool-using turn-1 history right above the current turn."""
        from cogtrix import _build_agent_graph

        # The structured answer to the new question. No claim-of-action
        # signal — the detector must return False even though the
        # context above contains tool-using messages.
        structured_answer = AIMessage(
            content=(
                "You're asking about long-context management.\n\n"
                "### Core Principles\n\n"
                "| Goal | Technique |\n"
                "|------|-----------|\n"
                "| Prevent bloat | Summarization |\n"
                "| Avoid drift | Subagents |\n\n"
                "#### 1. Progressive Summarization\n"
                "After each 5-7 turns, run a structured summary call.\n\n"
                "#### 2. Hierarchical Sub-agents\n"
                "Break the 45-minute task into phases.\n"
            ),
            id="t2-ai",
        )
        mock_llm = _make_mock_llm([structured_answer])

        graph = _build_agent_graph(
            llm=mock_llm,
            system_prompt="",
            active_tools_list=[],
            available_tools={},
            registry=_make_registry(),
            approvals=set(),
        )

        # Multi-turn history: prior turn used tools, current turn is
        # an unrelated structured-prose question.
        history_plus_new_question = [
            HumanMessage(
                content="How many Soudal Fix All Silirub I can buy for $100 NZD in Vienna?"
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "id": "tc-prior",
                        "args": {"query": "Soudal Fix All Vienna price"},
                        "type": "tool_call",
                    }
                ],
                id="t1-ai-toolcall",
            ),
            ToolMessage(
                content="(canned snippet: Soudal Fix All sealants…)",
                tool_call_id="tc-prior",
                id="t1-tool",
            ),
            AIMessage(
                content="I couldn't find the exact product code at Vienna retailers.",
                id="t1-ai-final",
            ),
            HumanMessage(
                content=(
                    "How do you handle memory when an agent needs to perform "
                    "a complex 45-minute task without blowing out its maximum "
                    "context window or degrading in reasoning?"
                )
            ),
        ]
        result = graph.invoke({"messages": history_plus_new_question})

        # Strict assertion: exactly one LLM invocation. If recovery
        # had triggered, we'd need additional mock responses and the
        # mock would have raised StopIteration.
        assert mock_llm.invoke.call_count == 1, (
            f"phantom recovery fired on a multi-turn structured-prose answer — "
            f"LLM called {mock_llm.invoke.call_count} times (expected 1). "
            "This is the cogtrix45 regression."
        )

        # Topic-stability assertion: the agent's final AIMessage
        # contains turn-2 content, not turn-1 stale topic.
        final_ai = next(
            m
            for m in reversed(result["messages"])
            if isinstance(m, AIMessage) and m.content and m.id != "t1-ai-final"
        )
        content = str(final_ai.content)
        assert (
            "Summarization" in content or "context" in content.lower()
        ), f"final AI response is missing turn-2 topic markers; got: {content[:200]}"
        assert (
            "Soudal" not in content
        ), f"final AI response leaked stale turn-1 topic (Soudal); got: {content[:200]}"


# ── Property-style detector invariants ───────────────────────────────


class TestDetectorInvariants:
    """Properties the detector must satisfy regardless of fixture
    specifics. Together with the domain coverage above, these guard
    against future refactors that quietly change behaviour."""

    def test_no_claim_of_action_means_no_phantom(self) -> None:
        """For 8 representative legitimate structured answers, none
        should be flagged. This is the parametrised counterpart of the
        per-domain tests above — same property, condensed."""
        legitimate_corpus = [
            "comparing X and Y",
            "trade-offs between A and B",
            "design patterns for long-running tasks",
            "incident response procedure",
            "API parameter reference",
            "schema design considerations",
            "deployment strategy comparison",
            "ML training recipe",
        ]
        # Build a structural shell with NO claim-of-action signal.
        for topic in legitimate_corpus:
            msg = AIMessage(
                content=(
                    f"Discussion of {topic}.\n\n"
                    "### Comparison\n\n"
                    "| Option | Property A | Property B |\n"
                    "|--------|-----------|-----------|\n"
                    "| X | yes | no |\n"
                    "| Y | no | yes |\n\n"
                    "#### 1. Use X when\n"
                    "Property A matters more.\n\n"
                    "#### 2. Use Y when\n"
                    "Property B matters more.\n"
                ),
                id=f"inv-{topic[:10]}",
            )
            assert _looks_like_markdown_phantom_report(msg) is False, (
                f"false positive on topic={topic!r}; detector flagged a "
                "legitimate comparison answer that has no claim-of-action signal"
            )

    @pytest.mark.parametrize(
        "claim_phrase",
        [
            "\n- Retrieved last 5 records.\n",
            "\nI retrieved the top 10 search hits.\n",
            "\nI fetched the most recent log entries.\n",
            "\nI searched the documentation.\n",
            "\nSources:\n- https://example.com/a\n",
            "\nAccording to my search, the value is 42.\n",
            "\nThe results show a clear trend.\n",
            "\nThe search returned several matches.\n",
        ],
    )
    def test_any_claim_of_action_signal_flips_detector(self, claim_phrase: str) -> None:
        """Adding any one of the five claim-of-action patterns to the
        same structural shell flips the detector to True. Pins each
        pattern individually so we know which ones contribute."""
        shell = (
            "### Comparison\n\n"
            "| Option | A | B |\n"
            "|--------|---|---|\n"
            "| X | 1 | 0 |\n\n"
            "#### 1. Use X when A\n"
            "Reason one.\n\n"
            "#### 2. Use X when B\n"
            "Reason two.\n"
        )
        msg = AIMessage(content=shell + claim_phrase, id="inv-claim")
        assert (
            _looks_like_markdown_phantom_report(msg) is True
        ), f"detector failed to flag a fabrication pattern: {claim_phrase!r}"

    def test_imperative_retrieve_is_not_a_claim(self) -> None:
        """The verb ``retrieve`` in imperative or infinitive form is
        not a claim of action. This is the most common false-positive
        trap — both the cogtrix45 fixture and many tutorials describe
        a *technique* that involves retrieval ("retrieve top-3 chunks")
        without making a retrieval claim themselves."""
        msg = AIMessage(
            content=(
                "### Retrieval design\n\n"
                "| Step | Operation |\n"
                "|------|-----------|\n"
                "| 1 | Retrieve top-K |\n"
                "| 2 | Rerank |\n\n"
                "#### 1. Retrieval phase\n"
                "Retrieve top-3 relevant chunks using cosine similarity.\n\n"
                "#### 2. Rerank phase\n"
                "Apply a cross-encoder for relevance scoring.\n"
            ),
            id="inv-imperative",
        )
        assert _looks_like_markdown_phantom_report(msg) is False
