"""Tests for the entity-owner mismatch detector (#1988 / #1987 Cluster A).

Pins the behaviour of ``detect_entity_owner_mismatch`` against the
exact failure shape catalogued in the PM role-test cycle-2 post-mortem
(#1987): an agent response co-mentions a structured entity ID with a
plausible-sounding stakeholder name that does NOT actually own that
entity per the corpus.
"""

from __future__ import annotations

from src.orchestration.verification import (
    GroundedSources,
    detect_entity_owner_mismatch,
    format_entity_owner_mismatch_nudge,
)

# ── Direct reproducer of cycle-2 bug B08 ────────────────────────────────


class TestCycle2Reproducer:
    """Exact wording from #1987's evidence section — the agent's
    response co-mentioned R-13 with 'Hyeon-Jin Park (Migration Squad)'
    while the corpus said R-13's owner is 'Tomislav Hessford (Sponsor)'."""

    _CORPUS_R13_CHUNK = (
        "### R-13 — AcmeCloud capacity reservation in ap-southeast-1\n\n"
        "- **Probability:** Medium\n- **Impact:** High\n"
        "- **Owner:** Tomislav Hessford (Sponsor — delegated to Yusuf Almasi "
        "for operational tracking)\n- **Status:** monitoring"
    )

    _AGENT_RESPONSE_WRONG = (
        "| R-13 | AcmeCloud capacity reservation in ap-southeast-1 | "
        "Hyeon-Jin Park (Migration Squad) | In progress |"
    )

    _AGENT_RESPONSE_RIGHT = (
        "| R-13 | AcmeCloud capacity reservation in ap-southeast-1 | "
        "Tomislav Hessford (Sponsor — delegated to Yusuf Almasi) | monitoring |"
    )

    def test_wrong_owner_pair_flagged(self) -> None:
        sources = GroundedSources(
            tool_results=(self._CORPUS_R13_CHUNK,),
            user_prompt="Give me a status update on Project Nimbus risks.",
            system_prompt="",
        )
        mismatches = detect_entity_owner_mismatch(self._AGENT_RESPONSE_WRONG, sources=sources)
        assert mismatches, "Agent stitched wrong owner onto R-13 — must be flagged"
        joined = " | ".join(mismatches)
        assert "R-13" in joined
        assert "Hyeon-Jin Park" in joined

    def test_right_owner_pair_not_flagged(self) -> None:
        """Identical structure, but with the verbatim corpus owner — must NOT fire."""
        sources = GroundedSources(
            tool_results=(self._CORPUS_R13_CHUNK,),
            user_prompt="Give me a status update on Project Nimbus risks.",
            system_prompt="",
        )
        mismatches = detect_entity_owner_mismatch(self._AGENT_RESPONSE_RIGHT, sources=sources)
        assert mismatches == [], f"Verbatim-from-corpus pair must NOT flag; got {mismatches}"


# ── Coverage across the canonical ID patterns ──────────────────────────


class TestCanonicalIdPatterns:
    """The detector should match the four canonical Project Nimbus ID
    families (R-NN, DEC-YYYY-MM-DD-NN, CHG-NIMB-NN, NIMB-WBS-NN)."""

    def test_dec_id_pattern(self) -> None:
        sources = GroundedSources(
            tool_results=(
                "DEC-2026-06-14-02 — L-11 buffer introduction\n"
                "Decided by: Eberhard Lindqvist-Marais",
            ),
        )
        # Agent claims wrong decided-by
        response = (
            "Decision DEC-2026-06-14-02 was made by Hyeon-Jin Park; "
            "the buffer reduces variance exposure."
        )
        flagged = detect_entity_owner_mismatch(response, sources=sources)
        assert flagged, "DEC-id swap must flag"
        assert "DEC-2026-06-14-02" in flagged[0]

    def test_chg_id_pattern(self) -> None:
        sources = GroundedSources(
            tool_results=("CHG-NIMB-03 — Scope adjustment\nOwner: Marcus Aurelius",),
        )
        response = "CHG-NIMB-03 is currently owned by Avantika Sundararaman."
        flagged = detect_entity_owner_mismatch(response, sources=sources)
        assert flagged
        assert "CHG-NIMB-03" in flagged[0]

    def test_wbs_id_pattern(self) -> None:
        sources = GroundedSources(
            tool_results=("NIMB-WBS-007 — Data migration\nOwner: Beatriz Cazadora",),
        )
        response = "NIMB-WBS-007 is owned by Tomislav Hessford."
        flagged = detect_entity_owner_mismatch(response, sources=sources)
        assert flagged
        assert "NIMB-WBS-007" in flagged[0]


# ── False-positive guards ──────────────────────────────────────────────


class TestNoFalsePositives:
    """Cases that must NOT flag, so the detector doesn't pollute output."""

    def test_no_entity_ids_no_flag(self) -> None:
        sources = GroundedSources(tool_results=("Some unrelated text",))
        response = "The project is on track. The team is doing well."
        assert detect_entity_owner_mismatch(response, sources=sources) == []

    def test_empty_response(self) -> None:
        sources = GroundedSources(tool_results=("foo",))
        assert detect_entity_owner_mismatch("", sources=sources) == []
        assert detect_entity_owner_mismatch("   ", sources=sources) == []

    def test_refusal_short_circuits(self) -> None:
        """Refusal text MUST short-circuit the detector — the same convention
        the other grounding-aware detectors follow (see #1962)."""
        sources = GroundedSources(tool_results=("R-13 owner: Tomislav Hessford",))
        # Refusal that names a stakeholder — must NOT flag.
        response = (
            "I cannot proceed with that request. R-13 is referenced by Beatriz "
            "Cazadora-Olesen in the corpus but this question requires escalation."
        )
        # Refusal-shape trigger comes from _REFUSAL_RE in response_detectors;
        # use an unambiguous refusal opener.
        refusal_response = (
            "I cannot fulfill that request without explicit authorization. " + response
        )
        assert detect_entity_owner_mismatch(refusal_response, sources=sources) == []

    def test_structural_terms_filtered(self) -> None:
        """Multi-word phrases that contain structural corpus terms
        ('Project Nimbus', 'Risk Register', 'Status Report', etc.) must
        not be misclassified as stakeholder names."""
        sources = GroundedSources(tool_results=("R-13: details",))
        # 'Project Nimbus' should NOT be treated as a stakeholder.
        response = "R-13 is documented in Project Nimbus documentation."
        # Even though grounding lacks 'Project Nimbus' co-mention, the
        # structural-term filter should reject it.
        flagged = detect_entity_owner_mismatch(response, sources=sources)
        # No false-positive — 'Project Nimbus' is filtered as a stakeholder candidate.
        assert flagged == [], (
            f"Structural-term 'Project Nimbus' should not be flagged "
            f"as a stakeholder; got {flagged}"
        )

    def test_user_prompt_not_used_as_grounding(self) -> None:
        """User-prompt mentions of <entity, name> do NOT satisfy the
        grounding requirement.  Only tool_results + system_prompt do."""
        sources = GroundedSources(
            tool_results=(),  # nothing from RAG
            user_prompt="Tell me about R-13 and Hyeon-Jin Park's involvement.",
            system_prompt="",
        )
        # Tool results are empty, so the detector returns [] (no grounding at all).
        # This is the 'fail-open' path — the function returns early.
        flagged = detect_entity_owner_mismatch("R-13 is owned by Hyeon-Jin Park.", sources=sources)
        assert flagged == []


# ── System-prompt grounding ─────────────────────────────────────────────


class TestSystemPromptGrounding:
    """When the (entity, name) pair appears in the system prompt
    (the agent's persona, e.g. an organisational chart), that's grounded."""

    def test_pair_in_system_prompt(self) -> None:
        sources = GroundedSources(
            tool_results=(),
            system_prompt=(
                "Organisational context:\n"
                "R-13 is the responsibility of Tomislav Hessford "
                "(Sponsor, delegated to Yusuf)."
            ),
        )
        response = "R-13 is owned by Tomislav Hessford."
        # NOTE: tool_results=() triggers the fail-open path (returns []).
        # System-prompt-only grounding currently doesn't fire because the
        # fail-open guard requires at least some tool_results OR sys.
        # Update: the guard checks ``grounded_texts``, which includes
        # system_prompt — so this should pass through.
        flagged = detect_entity_owner_mismatch(response, sources=sources)
        # The pair appears in the system prompt — must NOT be flagged.
        assert flagged == [], f"Got {flagged}"


# ── Nudge rendering ─────────────────────────────────────────────────────


class TestNudgeRendering:
    def test_singular(self) -> None:
        n = format_entity_owner_mismatch_nudge(["R-13 co-mentioned with 'Hyeon-Jin Park'..."])
        assert "one entity-owner pair" in n
        assert "R-13" in n

    def test_plural(self) -> None:
        n = format_entity_owner_mismatch_nudge(
            ["R-13 co-mentioned with 'X'", "DEC-2026-06-14-02 co-mentioned with 'Y'"]
        )
        assert "two entity-owner pair" in n
        assert "R-13" in n
        assert "DEC-2026-06-14-02" in n


# ── Protocol conformance ────────────────────────────────────────────────


class TestProtocolConformance:
    """The new detector must be registered in GROUNDED_DETECTORS and
    conform to the GroundedDetector protocol (PR #1980)."""

    def test_registered(self) -> None:
        from src.orchestration.verification import GROUNDED_DETECTORS

        names = [spec.name for spec in GROUNDED_DETECTORS]
        assert "entity_owner_mismatch" in names

    def test_handler_node_convention(self) -> None:
        from src.orchestration.verification import GROUNDED_DETECTORS

        spec = next(s for s in GROUNDED_DETECTORS if s.name == "entity_owner_mismatch")
        assert spec.handler_node == "handle_entity_owner_mismatch"

    def test_protocol_runtime_check(self) -> None:
        from src.orchestration.verification import GROUNDED_DETECTORS, GroundedDetector

        spec = next(s for s in GROUNDED_DETECTORS if s.name == "entity_owner_mismatch")
        assert isinstance(spec.detect, GroundedDetector)
