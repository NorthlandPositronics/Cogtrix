"""Tests for the corpus attribution index + mismatch detector.

Issue: #1948 cycle-2 item #4.  Motivating bug from run-2 of the PM
role-test harness — the agent attributed R-12 to *"Hyeon-Jin Park
(Migration Squad Lead)"*, but R-12 belongs to Beatriz Cazadora-Olesen
(Data Squad).  The detector exists to catch that exact pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.role_pm.attribution_index import (
    AttributionIndex,
    AttributionMismatch,
    build_attribution_index,
    detect_attribution_mismatches,
)

_CORPUS_DIR = Path(__file__).parent / "corpus"


@pytest.fixture(scope="module")
def index() -> AttributionIndex:
    return build_attribution_index(_CORPUS_DIR)


class TestBuildAttributionIndex:
    def test_indexes_every_risk(self, index: AttributionIndex) -> None:
        """R-12 through R-23 — all 12 risks must be in the index."""
        for risk_id in (
            "R-12",
            "R-13",
            "R-14",
            "R-15",
            "R-16",
            "R-17",
            "R-18",
            "R-19",
            "R-20",
            "R-21",
            "R-22",
            "R-23",
        ):
            assert risk_id in index.owners, (
                f"{risk_id} missing from attribution index — risk register parse " "regression"
            )

    def test_r12_owner_is_beatriz(self, index: AttributionIndex) -> None:
        """The motivating bug's reference — R-12's owner must parse
        to ``Beatriz Cazadora-Olesen``."""
        assert index.owners["R-12"] == frozenset({"Beatriz Cazadora-Olesen"})

    def test_r19_owner_is_linnaea(self, index: AttributionIndex) -> None:
        """The other High-High risk — owner Linnaea Korhonen."""
        assert index.owners["R-19"] == frozenset({"Linnaea Korhonen"})

    def test_indexes_decision_log(self, index: AttributionIndex) -> None:
        """Decision-log entries use plain ``**Owner:**`` (no bullet);
        the regex must still pick them up."""
        assert "DEC-2026-04-02-01" in index.owners
        assert "Tomislav Hessford" in index.owners["DEC-2026-04-02-01"]

    def test_known_stakeholders_includes_expected_names(self, index: AttributionIndex) -> None:
        """The flat stakeholder set is what the detector scans against;
        it must include every name that appears as an owner anywhere."""
        for name in (
            "Beatriz Cazadora-Olesen",
            "Tomislav Hessford",
            "Linnaea Korhonen",
            "Avantika Sundararaman",
            "Hyeon-Jin Park",
            "Aldous Pemberton-Riggs",
            "Bartholomew Okafor-Sing",
        ):
            assert name in index.known_stakeholders, f"{name!r} expected in known_stakeholders"


class TestDetectAttributionMismatches:
    def test_motivating_bug_r12_swap(self, index: AttributionIndex) -> None:
        """The exact failure observed in run-2 of #1948.  The response
        attributed R-12 to Hyeon-Jin Park; corpus says Beatriz
        Cazadora-Olesen.  This test pins the detector to that
        specific bug — if it ever stops firing, regression."""
        response = (
            "Based on the risk register, the top risks are:\n"
            "1. R-12 — AcmeDB cross-region replication lag\n"
            "   Owner: Hyeon-Jin Park (Migration Squad Lead)\n"
            "   Status: escalated\n"
        )
        hits = detect_attribution_mismatches(response, index)
        assert len(hits) == 1
        h = hits[0]
        assert h.entity_id == "R-12"
        assert h.claimed_owner == "Hyeon-Jin Park"
        assert h.valid_owners == frozenset({"Beatriz Cazadora-Olesen"})

    def test_correct_attribution_does_not_fire(self, index: AttributionIndex) -> None:
        """A response that names the CORRECT owner must not trip the
        detector."""
        response = (
            "R-12 — AcmeDB cross-region replication lag.  "
            "Owner: Beatriz Cazadora-Olesen.  Mitigation in flight."
        )
        hits = detect_attribution_mismatches(response, index)
        assert hits == []

    def test_no_owner_attribution_does_not_fire(self, index: AttributionIndex) -> None:
        """If the response mentions an entity_id but no stakeholder
        name in the window after it, no mismatch — the agent
        legitimately referenced the risk without claiming ownership."""
        response = "We should monitor R-12 closely this week."
        hits = detect_attribution_mismatches(response, index)
        assert hits == []

    def test_unknown_entity_id_does_not_fire(self, index: AttributionIndex) -> None:
        """An invented entity id (R-99 doesn't exist in the corpus)
        is OUT OF SCOPE for this detector — the negative-canary checks
        handle invented IDs.  This detector is specifically about
        swapping a known entity's owner."""
        response = "R-99 — Invented Risk.  Owner: Tomislav Hessford.  " "Status: open."
        hits = detect_attribution_mismatches(response, index)
        assert hits == []

    def test_multiple_mismatches_collected_independently(self, index: AttributionIndex) -> None:
        """A response that mis-attributes TWO different risks should
        produce TWO findings — they each count toward bug_count."""
        response = (
            "1. R-12 — AcmeDB replication.  Owner: Hyeon-Jin Park.\n"
            "2. R-19 — Helmsdale window.  Owner: Tomislav Hessford.\n"
        )
        hits = detect_attribution_mismatches(response, index)
        assert len(hits) == 2
        entity_ids = {h.entity_id for h in hits}
        assert entity_ids == {"R-12", "R-19"}

    def test_mismatch_describe_is_human_readable(self, index: AttributionIndex) -> None:
        """The describe() output is what surfaces in the scorecard's
        bug list — must name the entity_id, the claimed owner, and
        the valid owner set."""
        m = AttributionMismatch(
            entity_id="R-12",
            claimed_owner="Hyeon-Jin Park",
            valid_owners=frozenset({"Beatriz Cazadora-Olesen"}),
        )
        text = m.describe()
        assert "R-12" in text
        assert "Hyeon-Jin Park" in text
        assert "Beatriz Cazadora-Olesen" in text

    def test_empty_response_returns_empty(self, index: AttributionIndex) -> None:
        assert detect_attribution_mismatches("", index) == []
        assert detect_attribution_mismatches("   ", index) == []


# ── Shared-owner parsing (#2006 cycle-6 post-mortem) ─────────────────


class TestSharedOwnerSplitting:
    """Cycle-6 post-mortem (#2006) found that shared owner strings like
    ``"Tomislav Hessford + Avantika Sundararaman"`` were stored
    atomically in the index, causing partial-correct attributions to
    be flagged as mismatches: the model writing ``"Tomislav Hessford"``
    for an entity whose corpus owners are ``{"Tomislav Hessford +
    Avantika Sundararaman"}`` (one element) was rejected because the
    individual name wasn't in the valid set.

    The fix splits on ``+`` / ``,`` / ``and`` / ``&`` so individual
    names land in the set.  These tests pin that behaviour against
    the actual corpus AND against synthetic inputs.
    """

    def test_split_helper_handles_plus_separator(self) -> None:
        from tests.role_pm.attribution_index import _split_shared_owners

        result = _split_shared_owners("Tomislav Hessford + Avantika Sundararaman")
        assert result == ["Tomislav Hessford", "Avantika Sundararaman"]

    def test_split_helper_handles_comma_separator(self) -> None:
        from tests.role_pm.attribution_index import _split_shared_owners

        result = _split_shared_owners("PM, Customer Success")
        assert result == ["PM", "Customer Success"]

    def test_split_helper_handles_and_separator(self) -> None:
        from tests.role_pm.attribution_index import _split_shared_owners

        result = _split_shared_owners("Tomislav Hessford and CTO")
        assert result == ["Tomislav Hessford", "CTO"]

    def test_split_helper_handles_ampersand_separator(self) -> None:
        from tests.role_pm.attribution_index import _split_shared_owners

        result = _split_shared_owners("PM & Customer Success")
        assert result == ["PM", "Customer Success"]

    def test_split_helper_passes_single_owner_through(self) -> None:
        from tests.role_pm.attribution_index import _split_shared_owners

        result = _split_shared_owners("Beatriz Cazadora-Olesen")
        assert result == ["Beatriz Cazadora-Olesen"]

    def test_split_helper_drops_empty_fragments(self) -> None:
        from tests.role_pm.attribution_index import _split_shared_owners

        # Trailing separator or repeated separators produce empty
        # fragments — must be dropped, not stored as "" owners.
        assert _split_shared_owners("X + ") == ["X"]
        assert _split_shared_owners(" + X") == ["X"]
        assert _split_shared_owners("X + + Y") == ["X", "Y"]

    def test_split_helper_handles_empty_or_whitespace_input(self) -> None:
        from tests.role_pm.attribution_index import _split_shared_owners

        assert _split_shared_owners("") == []
        assert _split_shared_owners("   ") == []

    def test_corpus_shared_owners_are_split(self, index: AttributionIndex) -> None:
        """Cycle-6 partial-match false positives must clear: each
        co-owner of a shared-ownership entity in the corpus appears as
        its OWN element in the entity's valid set.
        """
        # DEC-2026-07-09-01 — corpus has "Tomislav Hessford + Avantika
        # Sundararaman" as a single Decided-by line.
        decided_by = index.owners.get("DEC-2026-07-09-01", frozenset())
        assert (
            "Tomislav Hessford" in decided_by
        ), f"Tomislav must be a recognised co-owner; got {sorted(decided_by)}"
        assert (
            "Avantika Sundararaman" in decided_by
        ), f"Avantika must be a recognised co-owner; got {sorted(decided_by)}"

        # CHG-NIMB-003 — corpus has "Tomislav Hessford + CTO".
        chg_003 = index.owners.get("CHG-NIMB-003", frozenset())
        assert "Tomislav Hessford" in chg_003
        assert "CTO" in chg_003

        # CHG-NIMB-004 — corpus has "PM + Customer Success".
        chg_004 = index.owners.get("CHG-NIMB-004", frozenset())
        assert "PM" in chg_004
        assert "Customer Success" in chg_004

    def test_corpus_known_stakeholders_include_split_person_names(
        self, index: AttributionIndex
    ) -> None:
        """The flat ``known_stakeholders`` set must include the
        post-split individual *person* names so the detector can
        match on the response side as well.

        Cycle-7 post-mortem (#2006): role tokens like ``CTO`` /
        ``PM`` / ``Customer Success`` are deliberately EXCLUDED
        from this set — see
        ``test_role_tokens_excluded_from_known_stakeholders``.
        """
        for name in ["Tomislav Hessford", "Avantika Sundararaman"]:
            assert name in index.known_stakeholders, (
                f"{name!r} must appear in known_stakeholders after the cycle-6 "
                f"shared-owner split fix; otherwise the detector cannot match "
                f"the model's reference to {name!r} against the entity's owners."
            )

    def test_partial_co_owner_attribution_is_accepted(self, index: AttributionIndex) -> None:
        """Regression for the cycle-6 partial-match false positive:
        the model writing just one of the shared co-owners must NOT be
        flagged as a mismatch.  Three real cycle-6 cases:
        DEC-2026-07-09-01 → 'Tomislav Hessford' (corpus: Tom +
        Avantika); CHG-NIMB-003 → 'Tomislav Hessford' (corpus: Tom +
        CTO); CHG-NIMB-004 → 'PM' (corpus: PM + Customer Success).
        """
        # DEC-2026-07-09-01 with just Tomislav — should NOT be flagged.
        response = "Decision DEC-2026-07-09-01 was made by Tomislav Hessford."
        findings = detect_attribution_mismatches(response, index)
        assert not findings, (
            f"Naming Tomislav (one of two co-owners) for DEC-2026-07-09-01 "
            f"should not be a mismatch; got {[f.describe() for f in findings]}"
        )

        # CHG-NIMB-003 with just Tomislav.
        response = "Change CHG-NIMB-003 is owned by Tomislav Hessford."
        findings = detect_attribution_mismatches(response, index)
        assert not findings

        # CHG-NIMB-004 with just PM.
        response = "Change CHG-NIMB-004 routed to PM for approval."
        findings = detect_attribution_mismatches(response, index)
        assert not findings

    def test_wrong_attribution_still_caught_after_split(self, index: AttributionIndex) -> None:
        """The split fix must NOT silence real mismatches.  Naming a
        stakeholder who is NOT in the entity's owner set (even after
        split) still flags."""
        # DEC-2026-07-09-02 corpus owner: just Tomislav.  Linnaea is a
        # different real stakeholder and must still be flagged.
        response = "DEC-2026-07-09-02 was decided by Linnaea Korhonen."
        findings = detect_attribution_mismatches(response, index)
        assert any(
            "DEC-2026-07-09-02" in f.describe() and "Linnaea Korhonen" in f.describe()
            for f in findings
        ), f"genuine mismatch must still be flagged; got {[f.describe() for f in findings]}"


# ── Role-token filter (#2006 cycle-7 post-mortem) ─────────────────


class TestRoleTokenFilter:
    """Cycle-7 post-mortem (#2006) — the shared-owner split fix added
    role tokens ``CTO``, ``PM``, ``Customer Success`` to the global
    ``known_stakeholders`` set.  This caused 22-of-38 cluster A
    mismatches in cycle 7 because the model uses these tokens
    casually in prose ("approved by the CTO", "escalated to
    Customer Success") within 240 chars of unrelated entity IDs.

    Fix: role tokens are valid per-entity owners (so the model
    writing them for THAT entity passes) but are filtered out of
    the global ``known_stakeholders`` set (so the detector doesn't
    scan for them across the response).
    """

    def test_is_role_token_recognises_caps_abbreviations(self) -> None:
        from tests.role_pm.attribution_index import _is_role_token

        for token in ["CTO", "CEO", "COO", "PM", "VP", "CFO", "CIO", "HR", "QA", "IT"]:
            assert _is_role_token(token), f"{token!r} should be classified as a role token"

    def test_is_role_token_recognises_curated_phrases(self) -> None:
        from tests.role_pm.attribution_index import _is_role_token

        assert _is_role_token("Customer Success")
        assert _is_role_token("Engineering Manager")
        assert _is_role_token("Steering Committee")

    def test_is_role_token_rejects_person_names(self) -> None:
        from tests.role_pm.attribution_index import _is_role_token

        for name in [
            "Tomislav Hessford",
            "Beatriz Cazadora-Olesen",
            "Aldous Pemberton-Riggs",
            "Hyeon-Jin Park",
            "Avantika Sundararaman",
        ]:
            assert not _is_role_token(
                name
            ), f"{name!r} is a person name, must not be classified as a role token"

    def test_is_role_token_handles_empty_input(self) -> None:
        from tests.role_pm.attribution_index import _is_role_token

        assert not _is_role_token("")
        assert not _is_role_token("   ")

    def test_role_tokens_excluded_from_known_stakeholders(self, index: AttributionIndex) -> None:
        """Cycle-7 regression guard: CTO / PM / Customer Success must
        NOT appear in the global ``known_stakeholders`` set, even
        though they appear as co-owners in the corpus.  If they do,
        the model's casual prose mention of "CTO" will trip the
        mismatch detector against every nearby entity ID."""
        for token in ["CTO", "PM", "Customer Success"]:
            assert token not in index.known_stakeholders, (
                f"{token!r} is a role token and must be filtered out of "
                f"known_stakeholders; including it caused 22-of-38 false "
                f"positives in cycle 7 (see #2006 post-mortem)."
            )

    def test_role_token_still_accepted_for_its_own_entity(self, index: AttributionIndex) -> None:
        """The filter must NOT silence the cycle-6 partial-credit win:
        the model writing ``"PM"`` or ``"CTO"`` for the entity that
        actually has that role as a co-owner must still pass."""
        # CHG-NIMB-004 corpus owner: "PM + Customer Success".  Model
        # writing just "PM" for this entity must NOT be flagged.
        response = "Change CHG-NIMB-004 was routed to PM for sign-off."
        findings = detect_attribution_mismatches(response, index)
        assert not findings, (
            f"PM is a valid co-owner of CHG-NIMB-004; naming it must not "
            f"flag a mismatch; got {[f.describe() for f in findings]}"
        )

        # CHG-NIMB-003 corpus owner: "Tomislav Hessford + CTO".  Model
        # writing "CTO" for this entity must NOT be flagged either.
        response = "Change CHG-NIMB-003 was approved by the CTO."
        findings = detect_attribution_mismatches(response, index)
        assert not findings

    def test_role_token_in_prose_near_other_entity_not_flagged(
        self, index: AttributionIndex
    ) -> None:
        """The cycle-7 regression case: model writes a role token
        casually in prose near an entity ID whose corpus owner is a
        different person.  Before the filter, this produced false
        positives like ``R-12 attributed to 'CTO' but corpus owners
        are {Beatriz Cazadora-Olesen}``."""
        # R-12 corpus owner: Beatriz Cazadora-Olesen.  Model casually
        # mentions "the CTO" — must NOT be flagged.
        response = "R-12 was escalated to the CTO during steering."
        findings = detect_attribution_mismatches(response, index)
        cto_findings = [f for f in findings if f.claimed_owner == "CTO"]
        assert not cto_findings, (
            f"Casual prose mention of 'CTO' near R-12 must not be flagged "
            f"as a mismatch; got {[f.describe() for f in cto_findings]}"
        )

        # Similarly for "Customer Success" near R-19 (owner: Linnaea).
        response = "R-19 — Helmsdale window.  Escalation routed to Customer Success."
        findings = detect_attribution_mismatches(response, index)
        cs_findings = [f for f in findings if f.claimed_owner == "Customer Success"]
        assert not cs_findings


# ── Table-row scope + directional patterns (#2006 cycle-10 post-mortem) ─


class TestTableRowScope:
    """Cycle-10 post-mortem (#2006) — when an entity-id lives inside a
    markdown table row, the 240-char window walks across other cells
    in the same row.  A row that legitimately attributes ENTITY-A to
    person X can produce a false positive on ENTITY-B (mitigation
    cell) when X is not B's owner.

    Real C10 case (scenario 01 iter 3):

        | R-12 (Replication Lag) | High | ... CHG-NIMB-003 (parallelisation);
        Data Squad target: 3.5s lag by 2026-07-31 | Beatriz Cazadora-Olesen |

    R-12 → Beatriz is the row's intent; the detector also flagged
    CHG-NIMB-003 → Beatriz because the 240-char window from
    CHG-NIMB-003 picked up Beatriz from the same row.  Fix 1A caps
    the window at end of line when the entity sits inside a table
    row.
    """

    def test_helper_classifies_table_row_lines(self) -> None:
        from tests.role_pm.attribution_index import _is_table_row_line

        assert _is_table_row_line("| R-12 | High | Beatriz |")
        assert _is_table_row_line("  | leading whitespace | also a row |")
        # Header separator: bars + dashes only — NOT a content row.
        assert not _is_table_row_line("|---|---|---|")
        assert not _is_table_row_line("| :--- | ---: | :---: |")
        # Plain prose.
        assert not _is_table_row_line("R-12 was escalated to the CTO.")
        assert not _is_table_row_line("")

    def test_line_bounds_at_finds_correct_line(self) -> None:
        from tests.role_pm.attribution_index import _line_bounds_at

        text = "first\nsecond line here\nthird"
        # Position 10 is inside "second line here"
        start, end = _line_bounds_at(text, 10)
        assert text[start:end] == "second line here"
        # Position 2 is inside "first"
        start, end = _line_bounds_at(text, 2)
        assert text[start:end] == "first"
        # Position past end clamps.
        start, end = _line_bounds_at(text, 999)
        assert text[start:end] == "third"

    def test_table_row_does_not_flag_other_entitys_owner(self, index: AttributionIndex) -> None:
        """The real C10 case: R-12's row mentions CHG-NIMB-003 as a
        mitigation and Beatriz as R-12's owner.  Window from
        CHG-NIMB-003 must NOT capture Beatriz."""
        response = (
            "| Risk | Severity | Mitigation | Owner |\n"
            "|---|---|---|---|\n"
            "| R-12 (Replication Lag) | High | Approved CHG-NIMB-003 (parallelisation); "
            "Data Squad target: 3.5s lag by 2026-07-31 | Beatriz Cazadora-Olesen |\n"
        )
        findings = detect_attribution_mismatches(response, index)
        # The R-12 → Beatriz attribution is correct; CHG-NIMB-003 → Beatriz
        # would be a FP because CHG-NIMB-003's true owners are {CTO, Tomislav}.
        chg_findings = [f for f in findings if f.entity_id == "CHG-NIMB-003"]
        assert (
            not chg_findings
        ), f"Table-row scope must prevent CHG-NIMB-003 → Beatriz FP; got {[f.describe() for f in chg_findings]}"

    def test_prose_attribution_still_flagged_outside_tables(self, index: AttributionIndex) -> None:
        """Fix 1A must not weaken the detector for non-table prose."""
        response = (
            "Status update: R-12 — AcmeDB cross-region replication lag.  "
            "Owner: Hyeon-Jin Park (Migration Squad Lead)."
        )
        findings = detect_attribution_mismatches(response, index)
        assert any(
            f.entity_id == "R-12" and f.claimed_owner == "Hyeon-Jin Park" for f in findings
        ), "Genuine prose mismatch must still be flagged"


class TestForwardAttributionPatterns:
    """Cycle-10 post-mortem (#2006) — when prose says
    ``<PERSON>'s ownership of <ENTITY>`` the person attaches to the
    entity that comes AFTER in the sentence, not the entity that
    preceded.  The 240-char window from the preceding entity picks
    up the person and reports a false mismatch.

    Real C10 case (scenario 02 iter 2):

        "06_stakeholder_register.md — Yusuf Almasi's ownership of R-13;
        Hyeon-Jin Park's ownership of R-12"

    R-13's window captured "Hyeon-Jin Park" before reaching R-12,
    so the detector reported R-13 → Hyeon-Jin even though the
    prose attributed Hyeon-Jin to R-12.  Fix 1B detects the
    forward-attribution pattern and suppresses the (prev_entity,
    person) finding.
    """

    def test_person_attaches_forward_recognises_apostrophe_s_pattern(self) -> None:
        from tests.role_pm.attribution_index import _person_attaches_forward

        window = "Yusuf Almasi's ownership of R-13; Hyeon-Jin Park's ownership of R-12"
        assert _person_attaches_forward(window, "Yusuf Almasi")
        assert _person_attaches_forward(window, "Hyeon-Jin Park")
        assert not _person_attaches_forward(window, "Tomislav Hessford")

    def test_person_attaches_forward_recognises_owns_pattern(self) -> None:
        from tests.role_pm.attribution_index import _person_attaches_forward

        assert _person_attaches_forward("Hyeon-Jin Park owns R-12.", "Hyeon-Jin Park")

    def test_person_attaches_forward_recognises_is_the_owner_of_pattern(self) -> None:
        from tests.role_pm.attribution_index import _person_attaches_forward

        assert _person_attaches_forward("Hyeon-Jin Park is the owner of R-12.", "Hyeon-Jin Park")

    def test_forward_attribution_suppresses_false_positive(self, index: AttributionIndex) -> None:
        """The real C10 case (scenario 02 iter 2): R-13's window
        captured "Hyeon-Jin Park" before reaching R-12, but the
        prose ``Hyeon-Jin Park's ownership of R-12`` attaches
        Hyeon-Jin to R-12 — not R-13.  Fix 1B must suppress the
        (R-13, Hyeon-Jin) finding."""
        response = (
            "References: Yusuf Almasi's ownership of R-13; " "Hyeon-Jin Park's ownership of R-12."
        )
        findings = detect_attribution_mismatches(response, index)
        bad = [f for f in findings if f.entity_id == "R-13" and f.claimed_owner == "Hyeon-Jin Park"]
        assert not bad, (
            f"Forward-attribution suppression must clear R-13 → Hyeon-Jin FP; "
            f"got {[f.describe() for f in bad]}"
        )

    def test_genuine_mismatch_still_fires_when_forward_pattern_unrelated(
        self, index: AttributionIndex
    ) -> None:
        """Fix 1B must NOT silence a genuine mismatch in the same
        response.  Construct prose where the FA pattern targets one
        entity but a different entity has a real mis-attribution that
        does NOT match the FA suppression criterion."""
        # FA pattern attaches Yusuf to R-13 (suppresses R-12 → Yusuf if any).
        # Separate sentence then mis-attributes R-19 → Hyeon-Jin
        # using the standard "ENTITY ... Owner: PERSON" form.
        response = (
            "Yusuf Almasi's ownership of R-13 was confirmed.  "
            "Separately, R-19 — Helmsdale pricing window.  Owner: Hyeon-Jin Park."
        )
        findings = detect_attribution_mismatches(response, index)
        assert any(
            f.entity_id == "R-19" and f.claimed_owner == "Hyeon-Jin Park" for f in findings
        ), (
            f"R-19 → Hyeon-Jin (genuine) must still fire even when an unrelated "
            f"forward-attribution pattern exists elsewhere; got {[f.describe() for f in findings]}"
        )

    def test_normal_entity_owner_prose_still_flagged(self, index: AttributionIndex) -> None:
        """Fix 1B must NOT silence the standard ``ENTITY ... Owner:
        PERSON`` pattern — only the inverted ``PERSON's ownership of
        ENTITY`` form gets suppressed."""
        response = "R-12 (replication lag).  Owner: Hyeon-Jin Park."
        findings = detect_attribution_mismatches(response, index)
        assert any(f.entity_id == "R-12" and f.claimed_owner == "Hyeon-Jin Park" for f in findings)


# ── Disavowal-context suppression (#2026 cycle-13 generalisation) ───


class TestDisavowalContextSuppression:
    """#2026 (DeepSeek V4 Pro cycle-13 finding) — when prose contains
    English negation patterns around a stakeholder mention ("X cited
    only as ROLE", "X NOT the owner", "X in his capacity as <role>"),
    the (entity, X) co-mention is NOT an attribution claim and must
    not be flagged.  Generic English NLP — works in any deployment
    where the model writes careful disavowal prose.
    """

    def test_helper_recognises_cited_only_pattern(self) -> None:
        from tests.role_pm.attribution_index import _person_in_disavowal_context

        window = "Hyeon-Jin Park cited only as Migration Squad Lead"
        assert _person_in_disavowal_context(window, "Hyeon-Jin Park")

    def test_helper_recognises_capacity_pattern(self) -> None:
        from tests.role_pm.attribution_index import _person_in_disavowal_context

        window = "Tomislav Hessford in his capacity as Sponsor"
        assert _person_in_disavowal_context(window, "Tomislav Hessford")

    def test_helper_recognises_not_pattern(self) -> None:
        from tests.role_pm.attribution_index import _person_in_disavowal_context

        window = "Tomislav Hessford ... not the risk owner"
        assert _person_in_disavowal_context(window, "Tomislav Hessford")

    def test_helper_skips_when_pattern_too_far_from_person(self) -> None:
        from tests.role_pm.attribution_index import _person_in_disavowal_context

        # 200-char gap between person and disavowal — out of range.
        window = "Tomislav Hessford " + "." * 200 + " not the owner"
        assert not _person_in_disavowal_context(window, "Tomislav Hessford")

    def test_helper_passes_clean_prose_through(self) -> None:
        from tests.role_pm.attribution_index import _person_in_disavowal_context

        window = "Tomislav Hessford approved the budget at Q2 steering."
        assert not _person_in_disavowal_context(window, "Tomislav Hessford")

    def test_detector_suppresses_real_c13_disavowal_case(self, index: AttributionIndex) -> None:
        """The real C13 prose: R-19 attributed to Linnaea correctly,
        Tomislav explicitly disavowed.  Detector must not flag
        (R-19, Tomislav)."""
        response = (
            "Linnaea Korhonen is listed as the risk owner for R-19 in both "
            "the detailed section and the summary table, with Tomislav "
            "Hessford cited only in his capacity as Sponsor who formalised "
            "the contingency pricing decision (DEC-2026-07-09-02) at Q2 "
            "steering, not as the risk owner."
        )
        findings = detect_attribution_mismatches(response, index)
        bad = [
            f for f in findings if f.entity_id == "R-19" and f.claimed_owner == "Tomislav Hessford"
        ]
        assert not bad, (
            f"Disavowal suppression must clear R-19 → Tomislav FP; "
            f"got {[f.describe() for f in bad]}"
        )

    def test_genuine_attribution_still_flagged_without_disavowal(
        self, index: AttributionIndex
    ) -> None:
        """Affirmative attribution without negation MUST still flag."""
        response = "R-19 owner: Tomislav Hessford (Sponsor)."
        findings = detect_attribution_mismatches(response, index)
        assert any(
            f.entity_id == "R-19" and f.claimed_owner == "Tomislav Hessford" for f in findings
        )

    def test_long_self_correcting_response_disavowal_caught(self, index: AttributionIndex) -> None:
        """#2027 — DeepSeek V4 Pro cycle-20 prose put the disavowal
        ~400 chars from the first entity mention.  The previous
        ±200-char window missed it; widening to the full response
        catches it.  Real prose pattern (slightly compacted)."""
        response = (
            "Vendor-dependency risks summary:\n\n"
            "| R-ID | Risk | Owner | Status |\n"
            "|---|---|---|---|\n"
            "| R-13 | AcmeCloud capacity | Tomislav Hessford | Mitigated |\n"
            "| R-19 | Helmsdale acceptance-window risk | Linnaea Korhonen (Head of "
            "Customer Success) | Contingency approved, not yet exercised |\n"
            "\n---\n\n"
            "**Correction note:** The previous response incorrectly omitted "
            "Linnaea Korhonen as the R-19 risk owner and conflated the "
            "decision owner (Tomislav Hessford, for DEC-2026-07-09-02) with "
            "risk ownership. The corpus is clear: R-19 is owned by "
            "**Linnaea Korhonen**. The decision to authorise the pricing "
            "commitment was Tomislav Hessford's, but that is the mitigation "
            "decision — not the risk itself."
        )
        findings = detect_attribution_mismatches(response, index)
        bad = [
            f for f in findings if f.entity_id == "R-19" and f.claimed_owner == "Tomislav Hessford"
        ]
        assert not bad, (
            f"Long-response disavowal must clear R-19 → Tomislav FP; "
            f"got {[f.describe() for f in bad]}"
        )


# ── Role-paren context suppression (#2026 cycle-13 generalisation) ──


class TestRoleParenContextSuppression:
    """#2026 — when prose labels a person inside a parenthesised role
    qualifier (``Sponsor (Tomislav Hessford, COO)``) or with a
    trailing role suffix (``Tomislav Hessford, COO``), that person
    mention is a role-context clarification, not an attribution
    claim.  Generic English/markdown convention — works in any
    deployment that uses the ``Role (Name)`` labelling style.
    """

    def test_helper_recognises_role_paren_structure(self) -> None:
        from tests.role_pm.attribution_index import _person_in_role_paren_context

        window = "escalation to the CTO, Sponsor (Tomislav Hessford, COO), and ..."
        assert _person_in_role_paren_context(window, "Tomislav Hessford")

    def test_helper_recognises_trailing_role_suffix(self) -> None:
        from tests.role_pm.attribution_index import _person_in_role_paren_context

        window = "approved by Tomislav Hessford, COO of the company"
        assert _person_in_role_paren_context(window, "Tomislav Hessford")

    def test_helper_passes_clean_attribution_through(self) -> None:
        from tests.role_pm.attribution_index import _person_in_role_paren_context

        # No paren-role structure, no trailing role suffix → no suppression.
        window = "R-12 was approved by Tomislav Hessford on 2026-08-15."
        assert not _person_in_role_paren_context(window, "Tomislav Hessford")

    def test_helper_does_not_suppress_entity_paren_owner_form(self) -> None:
        """``ENTITY (PERSON)`` is a real attribution form (e.g. R-12
        (Beatriz Cazadora-Olesen) is a Data Squad risk).  The role-
        paren helper must not suppress when the ``role`` slot is
        actually an entity-id."""
        from tests.role_pm.attribution_index import _person_in_role_paren_context

        window = "R-12 (Tomislav Hessford) is open"
        # Should NOT match the role-paren pattern since R-12 isn't a
        # TitleCase role phrase.  The role-paren regex only matches
        # TitleCase phrases of 1-4 capitalised words.
        # R-12 doesn't match [A-Z][a-z]+ so this is not suppressed.
        assert not _person_in_role_paren_context(window, "Tomislav Hessford")

    def test_detector_suppresses_real_c13_role_paren_case(self, index: AttributionIndex) -> None:
        """The real C13 prose: R-16 attributed to Avantika correctly,
        Tomislav inside ``Sponsor (Tomislav Hessford, COO)`` role-context.
        Detector must not flag (R-16, Tomislav)."""
        response = (
            "R-16 reopens the moment work passes 2026-11-12, and "
            "escalation to the CTO, Sponsor (Tomislav Hessford, COO), "
            "and Avantika Sundararaman (R-16 owner) is the immediate "
            "next step."
        )
        findings = detect_attribution_mismatches(response, index)
        bad = [
            f for f in findings if f.entity_id == "R-16" and f.claimed_owner == "Tomislav Hessford"
        ]
        assert not bad, (
            f"Role-paren suppression must clear R-16 → Tomislav FP; "
            f"got {[f.describe() for f in bad]}"
        )

    def test_genuine_attribution_still_flagged_outside_role_paren(
        self, index: AttributionIndex
    ) -> None:
        """A standalone attribution sentence (no role-paren) must still
        flag."""
        response = (
            "R-16 — Helmsdale pricing window.  Owner: Tomislav Hessford.  " "Status: under review."
        )
        findings = detect_attribution_mismatches(response, index)
        assert any(
            f.entity_id == "R-16" and f.claimed_owner == "Tomislav Hessford" for f in findings
        )
