"""Tests for the topic-substitution detector (#1989 / #1987 Cluster C).

Pins the behaviour of ``detect_topic_substitution`` against the
silent question-reframing failure mode catalogued in the PM role-test
cycle-2 post-mortem (#1987): user asks about subject X that's absent
from the corpus, agent silently retitles its response to in-corpus
subject Y and answers Y instead of acknowledging out-of-scope.
"""

from __future__ import annotations

from src.orchestration.verification import (
    GroundedSources,
    _extract_distinctive_subjects,
    detect_topic_substitution,
    format_topic_substitution_nudge,
)

# ── Direct cycle-2 reproducer (bug B13 / B16) ──────────────────────────


class TestCycle2Reproducer:
    """Exact wording from #1987's scenario-05 evidence: user asked about
    *CompactSync codebase tech debt*; agent retitled to *Project Nimbus
    Technical Debt Risks* and answered the wrong question."""

    _USER_PROMPT = (
        "What's the biggest tech-debt risk we're carrying on the CompactSync "
        "codebase right now? Give me your top three items."
    )

    _SUBSTITUTED_RESPONSE = (
        "Based on my previous queries and the information I've gathered, I can "
        "now provide a comprehensive response about the biggest tech-debt risks "
        "for the codebase.\n\n"
        "## Project Nimbus Technical Debt Risks\n\n"
        "Based on the available Project Nimbus documentation, here are the top "
        "three tech-debt risks we're currently carrying:\n\n"
        "### 1. Replication Lag (R-12)\n\n"
        "- **Probability:** High\n- **Impact:** High\n"
        "- **Owner:** Beatriz Cazadora-Olesen (Platform Squad Lead)\n\n"
        "### 2. APAC Region Capacity (R-13)\n\n"
        "Detailed risk content here.\n\n"
        "### 3. Vendor TAM Rotation (R-17)\n\n"
        "More risk content here."
    )

    def test_substitution_flagged(self) -> None:
        sources = GroundedSources(
            tool_results=("Project Nimbus risk register: R-12, R-13, R-17 detailed entries.",),
            user_prompt=self._USER_PROMPT,
            system_prompt="You are the Project Nimbus PM.",
        )
        missing = detect_topic_substitution(self._SUBSTITUTED_RESPONSE, sources=sources)
        assert missing, "Topic substitution must be flagged"
        assert "CompactSync" in missing

    def test_correctly_addressed_does_not_fire(self) -> None:
        """When the response DOES name the user's subject (even to defer
        on it), no substitution occurred."""
        defer_response = (
            "I don't have information about CompactSync codebase in the Project "
            "Nimbus corpus.  CompactSync appears to be outside the scope of the "
            "current project.  Recommend escalating this question to the engineering "
            "lead or CTO for a code-level architectural assessment.  In the meantime "
            "I can give you the Project Nimbus risk register if that would help "
            "frame the conversation."
        )
        sources = GroundedSources(
            tool_results=("Project Nimbus risk register: R-12 ...",),
            user_prompt=self._USER_PROMPT,
            system_prompt="You are the Project Nimbus PM.",
        )
        assert detect_topic_substitution(defer_response, sources=sources) == []


# ── Distinctive-subject extractor ──────────────────────────────────────


class TestDistinctiveSubjectExtractor:
    def test_camelcase_compound(self) -> None:
        result = _extract_distinctive_subjects("Tell me about CompactSync, AcmeCloud, and AcmeDB.")
        assert "CompactSync" in result
        assert "AcmeCloud" in result
        assert "AcmeDB" in result

    def test_titlecase_multi_word(self) -> None:
        result = _extract_distinctive_subjects(
            "What's happening with Project Nimbus and New York operations?"
        )
        # 'Project' is a structural-stopword but the phrase has 'Nimbus' too,
        # so 'Project Nimbus' should be kept.
        assert "Project Nimbus" in result
        assert "New York" in result

    def test_acronyms(self) -> None:
        result = _extract_distinctive_subjects(
            "We need PMBOK references and AWS integration. UTC timestamps OK?"
        )
        assert "PMBOK" in result
        assert "AWS" in result
        # Generic acronyms filtered.
        assert "UTC" not in result
        assert "OK" not in result

    def test_filters_short_camelcase(self) -> None:
        """4-char CamelCase tokens are below the min-chars threshold."""
        result = _extract_distinctive_subjects("This is McDo (4-char) test.")
        assert "McDo" not in result

    def test_filters_pure_structural_phrase(self) -> None:
        """'Risk Register' / 'Status Update' are pure structural — filtered."""
        result = _extract_distinctive_subjects("Give me the Risk Register and the Status Update.")
        assert "Risk Register" not in result
        assert "Status Update" not in result

    def test_empty_input(self) -> None:
        assert _extract_distinctive_subjects("") == []
        assert _extract_distinctive_subjects("   ") == []

    def test_no_capitalised_content(self) -> None:
        """A prompt with no proper-noun / camelcase / acronym content
        produces no subjects."""
        result = _extract_distinctive_subjects("give me a summary of the budget")
        assert result == []

    def test_titlecase_phrase_does_not_span_newlines(self) -> None:
        """TitleCase phrases must NOT match across paragraph breaks.

        Regression for PR #1999 CI failure on
        ``procurement_supplier_registration × kimi-k2-5``: the user
        prompt contained ``"Primary product category: Electronics\\n
        Please register…"`` and the original ``\\s+`` inter-token
        whitespace produced ``"Electronics\\nPlease"`` as a missing
        subject, kicking off an infinite recovery cascade that timed
        out the scenario.  A TitleCase topic-subject phrase that spans
        a line break is almost always a regex false positive.
        """
        prompt = (
            "I need to onboard a new supplier.  Primary product "
            "category: Electronics\nPlease register this supplier "
            "and validate the information."
        )
        result = _extract_distinctive_subjects(prompt)
        # The line-spanning false positive must not appear.
        assert not any(
            "\n" in s for s in result
        ), f"TitleCase phrase must not span newlines; got {result!r}"
        assert "Electronics\nPlease" not in result
        # And the literal "Electronics Please" (without newline) must
        # also not appear, since neither "Electronics" alone nor
        # "Please" alone is a phrase, and the two are NOT contiguous
        # on a single line.
        assert "Electronics Please" not in result

    def test_titlecase_phrase_still_matches_within_a_line(self) -> None:
        """The newline fix must not break the legitimate same-line case."""
        result = _extract_distinctive_subjects(
            "We need updates on the Nimbus Project and the Acme Migration soon."
        )
        # ``Nimbus Project`` has a structural stopword (Project) but
        # also a non-stopword token (Nimbus), so the helper keeps it.
        assert "Nimbus Project" in result
        assert "Acme Migration" in result


# ── Substantive-response gate ──────────────────────────────────────────


class TestSubstantiveResponseGate:
    def test_short_response_not_flagged(self) -> None:
        """Short responses (< min_response_chars) are usually refusals or
        clarifications — NOT substitution."""
        short_response = "I don't have data on CompactSync."  # 35 chars
        sources = GroundedSources(
            tool_results=("Project Nimbus context.",),
            user_prompt="Tell me about CompactSync.",
        )
        assert detect_topic_substitution(short_response, sources=sources) == []

    def test_long_response_with_subject_absent_flagged(self) -> None:
        long_response = (
            "Here is a comprehensive analysis of the project. " * 10
            + "We have evaluated multiple risk dimensions. " * 5
        )
        sources = GroundedSources(
            tool_results=("Project Nimbus.",),
            user_prompt="What's the status of CompactSync?",
        )
        missing = detect_topic_substitution(long_response, sources=sources)
        assert "CompactSync" in missing


# ── Refusal short-circuit ──────────────────────────────────────────────


class TestRefusalShortCircuit:
    def test_refusal_not_flagged(self) -> None:
        """Refusal responses MUST short-circuit — the agent is correctly
        declining; we don't punish that."""
        refusal = (
            "I cannot fulfill this request without proper authorization.  "
            "CompactSync is not something I have access to in my current "
            "scope.  Please escalate to the appropriate technical authority "
            "for a code-level architectural review of the CompactSync codebase."
        ) * 3  # padded for substantive gate
        sources = GroundedSources(
            tool_results=(),
            user_prompt="Tell me about CompactSync tech debt.",
        )
        assert detect_topic_substitution(refusal, sources=sources) == []


# ── False-positive guards ──────────────────────────────────────────────


class TestNoFalsePositives:
    def test_subject_in_response_no_flag(self) -> None:
        """If the response NAMES the user's subject (even to defer), no flag."""
        sources = GroundedSources(
            tool_results=(),
            user_prompt="What's the status of CompactSync?",
        )
        substantive_with_subject = (
            "CompactSync is a customer-side codebase that is not part of the "
            "Project Nimbus scope I have visibility into.  Based on what I CAN "
            "see in the Project Nimbus corpus, the deliverables that COULD "
            "interface with CompactSync are R-13 (AcmeCloud capacity) and "
            "R-17 (vendor TAM).  However I cannot speak to tech debt within "
            "CompactSync itself."
        )
        assert detect_topic_substitution(substantive_with_subject, sources=sources) == []

    def test_all_tool_results_empty_no_flag(self) -> None:
        """#1992 follow-up: when every tool result is empty (e.g. the
        ``regression_persist_before_refusing`` scenario where every
        ``search_web`` returns zero hits), the agent's response not
        naming the user's subject is the honest *'I searched, found
        nothing'* shape — NOT silent topic substitution.

        Substitution requires the agent to PIVOT to a different
        topic with content; if no content was retrieved at all,
        there's no pivot.  The detector must not fire here, otherwise
        it triggers an extra recovery cycle that runs up cost on the
        already-persistent scenario.
        """
        sources = GroundedSources(
            tool_results=("", "", ""),  # 3 search calls, all empty
            user_prompt="Find me a reimplementation of Captain Claw",
            system_prompt="You are a research assistant.",
        )
        # Long response describing the empty-search outcome, but
        # without naming "Captain Claw" in the part we measure
        # (representative of Sonnet's response shape on this scenario).
        response = (
            "It looks like the search tool has been completely "
            "unresponsive — returning zero results for every single "
            "query, including completely generic ones. " * 8
        )
        assert detect_topic_substitution(response, sources=sources) == [], (
            "Empty-tool-results case must NOT flag as substitution — "
            "the agent honestly couldn't find anything"
        )

    def test_subject_in_tool_results_no_flag(self) -> None:
        """If the tool results return the user's subject (i.e. it IS in
        scope after all), the response addresses it grounded — no flag.
        Use a longer subject so it's not coincidentally in the response."""
        sources = GroundedSources(
            tool_results=("OperaSidekick subsystem analysis: full document content.",),
            user_prompt="Walk me through OperaSidekick subsystem.",
        )
        # Substantive response that happens to NOT mention OperaSidekick
        # by name — but the tool retrieved it, so substitution isn't the
        # right diagnosis.
        response_without_subject_name = (
            "Based on the retrieved documentation, the subsystem you asked "
            "about has the following architectural properties: " + ("blah blah " * 50)
        )
        # OperaSidekick is in tool_results → no flag.
        assert detect_topic_substitution(response_without_subject_name, sources=sources) == []

    def test_no_distinctive_subjects_no_flag(self) -> None:
        """User prompt with only common words → no distinctive subjects → no flag."""
        sources = GroundedSources(
            tool_results=("budget data",),
            user_prompt="give me a summary of the budget for the project",
        )
        # Substantive response about something completely unrelated, but the
        # user prompt has no distinctive subjects to track.
        response = "Here is a detailed analysis: " + ("text " * 200)
        assert detect_topic_substitution(response, sources=sources) == []

    def test_empty_user_prompt_no_flag(self) -> None:
        sources = GroundedSources(
            tool_results=("content",),
            user_prompt="",
        )
        response = "Detailed response: " + ("padding " * 100)
        assert detect_topic_substitution(response, sources=sources) == []


# ── System-prompt grounding ────────────────────────────────────────────


class TestSystemPromptGrounding:
    def test_subject_in_system_prompt_no_flag(self) -> None:
        """When the user's subject appears in the system prompt (e.g. the
        agent's persona scope), addressing it via persona-knowledge is
        not substitution."""
        sources = GroundedSources(
            tool_results=(),
            user_prompt="What about Helmsdale?",
            system_prompt=(
                "You manage Project Nimbus, which includes a Helmsdale "
                "Logistics integration sub-programme."
            ),
        )
        response = (
            "Helmsdale Logistics is one of the integration sub-programmes; here "
            "is a summary of what I can share: " + ("details " * 60)
        )
        # 'Helmsdale' appears in both response AND system_prompt → no flag.
        assert detect_topic_substitution(response, sources=sources) == []


# ── Nudge rendering ────────────────────────────────────────────────────


class TestNudgeRendering:
    def test_singular(self) -> None:
        n = format_topic_substitution_nudge(["CompactSync"])
        assert "one distinctive subject" in n
        assert "``CompactSync``" in n

    def test_plural(self) -> None:
        n = format_topic_substitution_nudge(["CompactSync", "AcmeDB"])
        assert "two distinctive subject" in n
        assert "``CompactSync``" in n
        assert "``AcmeDB``" in n


# ── Protocol conformance ───────────────────────────────────────────────


class TestProtocolConformance:
    def test_registered(self) -> None:
        from src.orchestration.verification import GROUNDED_DETECTORS

        names = [spec.name for spec in GROUNDED_DETECTORS]
        assert "topic_substitution" in names

    def test_handler_node_convention(self) -> None:
        from src.orchestration.verification import GROUNDED_DETECTORS

        spec = next(s for s in GROUNDED_DETECTORS if s.name == "topic_substitution")
        assert spec.handler_node == "handle_topic_substitution"

    def test_protocol_runtime_check(self) -> None:
        from src.orchestration.verification import GROUNDED_DETECTORS, GroundedDetector

        spec = next(s for s in GROUNDED_DETECTORS if s.name == "topic_substitution")
        assert isinstance(spec.detect, GroundedDetector)

    def test_consumes_user_prompt_false(self) -> None:
        """Per the spec, user_prompt is the SOURCE of candidates, not
        verification grounding."""
        from src.orchestration.verification import GROUNDED_DETECTORS

        spec = next(s for s in GROUNDED_DETECTORS if s.name == "topic_substitution")
        assert spec.consumes_user_prompt is False
        assert spec.extracts_candidates_from_user_prompt is True
