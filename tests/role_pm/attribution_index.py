"""Corpus attribution index for the PM role-test harness (cycle-2 item #4).

Builds an in-memory ``{entity_id → {valid_owner_names}}`` mapping from the
Project Nimbus corpus by parsing the ``## <ENTITY-ID> — ...`` headings and
the ``**Owner:** ...`` line that follows each.  Used by
:func:`detect_attribution_mismatches` to flag responses that stitch a real
stakeholder name onto the wrong entity.

The motivating bug — run-2 of #1948 — produced *"R-12 — AcmeDB cross-region
replication lag.  Owner: Hyeon-Jin Park (Migration Squad Lead)"*.  Hyeon-Jin
Park is the Migration Squad Lead per the stakeholder register, but R-12 is a
Data Squad risk owned by Beatriz Cazadora-Olesen.  The agent stitched a
correct name onto the wrong risk.  Today the harness flags it generically as
``hallucination_present=True`` via the negative-canary trip; this module
gives the operator a distinct ``attribution_mismatch`` signal that spells out
the right and wrong owners.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Risk register: "### R-12 — AcmeDB cross-region replication lag"
# Decision log:  "## DEC-2026-04-02-01 — Raise budget envelope..."
# Change log:    "### CHG-NIMB-001 — Budget envelope raise..."
_ENTITY_ID_RE = re.compile(
    r"#{2,3}\s*(?P<id>(?:R-\d+|DEC-\d{4}-\d{2}-\d{2}-\d+|CHG-NIMB-\d+))\s+—",
    re.IGNORECASE,
)

# "- **Owner:** Beatriz Cazadora-Olesen (Data Squad Lead)"  → "Beatriz Cazadora-Olesen"
# "- **Decided by:** Tomislav Hessford (with CTO concurrence)" → "Tomislav Hessford"
# "- **Owner:** Beatriz Cazadora-Olesen"                    → "Beatriz Cazadora-Olesen"
# Captures the FULL name span up to the first "(", ",", ".", newline, OR
# end-of-string so bare-name lines without a trailing role hint still parse.
# Owner / Decided-by line.  Two formats observed across the corpus:
# - Bulleted (risk register, change log): ``- **Owner:** <name>``
# - Plain    (decision log):              ``**Owner:** <name>``
# Make the leading ``- `` optional so both parse.
_OWNER_LINE_RE = re.compile(
    r"(?:-\s+)?\*\*(?:Owner|Decided\s+by)\s*:\s*\*\*\s+(?P<name>[^\n(,.]+?)(?:\s*\(|\s*,|\s*\.|\s*\n|\s*$)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class AttributionMismatch:
    """One mismatched ``<entity_id, claimed_owner, valid_owners>`` finding."""

    entity_id: str
    claimed_owner: str
    valid_owners: frozenset[str]

    def describe(self) -> str:
        valid_list = ", ".join(sorted(self.valid_owners)) if self.valid_owners else "<none>"
        return (
            f"{self.entity_id} attributed to '{self.claimed_owner}' but corpus "
            f"owners are {{{valid_list}}}"
        )


@dataclass(slots=True)
class AttributionIndex:
    """In-memory ``{entity_id → {valid_owner_names}}`` index built from the corpus."""

    # entity_id → set of canonical owner full names that the corpus declares
    # for that entity.  Multiple names per entity are valid when the corpus
    # explicitly assigns shared / delegated ownership.
    owners: dict[str, frozenset[str]] = field(default_factory=dict)
    # Flat set of every stakeholder name observed across the index.  Used by
    # the detector to recognise a "claimed_owner" candidate without having to
    # enumerate every possible name pattern.
    known_stakeholders: frozenset[str] = field(default_factory=frozenset)


def build_attribution_index(corpus_dir: Path) -> AttributionIndex:
    """Parse the corpus markdown to build an entity → owner mapping.

    Walks every ``*.md`` file under *corpus_dir*.  Within each file, finds
    headings of the form ``### R-NN — ...`` / ``### DEC-... — ...`` /
    ``### CHG-NIMB-NNN — ...`` and pairs each heading with the next
    ``- **Owner:** <name>`` or ``- **Decided by:** <name>`` line in that
    section.

    Returns an :class:`AttributionIndex` covering the whole corpus.
    """
    owners: dict[str, set[str]] = {}
    all_stakeholders: set[str] = set()

    for md_path in sorted(corpus_dir.glob("*.md")):
        text = md_path.read_text()
        _populate_from_file(text, owners, all_stakeholders)

    # Freeze for cheap hash-based lookup downstream.
    frozen_owners = {k: frozenset(v) for k, v in owners.items()}
    return AttributionIndex(
        owners=frozen_owners,
        known_stakeholders=frozenset(all_stakeholders),
    )


# Shared-ownership separators observed in the PM corpus.  When the
# ``Owner:`` / ``Decided by:`` field captures a string like
# ``Tomislav Hessford + Avantika Sundararaman`` (or ``X, Y`` or ``X and
# Y``), we split into individual owner names rather than storing the
# combined string as a single atomic owner.  Cycle-6 post-mortem
# (#2006) discovered that storing the combined string flagged
# partially-correct attributions as mismatches: model writes
# ``Tomislav Hessford`` for an entity whose corpus owners are
# ``{"Tomislav Hessford + Avantika Sundararaman"}`` (one element);
# detector sees ``"Tomislav Hessford" not in valid_set`` and flags it
# as a mismatch even though Tom IS a co-owner.  Splitting on these
# separators turns the valid set into ``{"Tomislav Hessford",
# "Avantika Sundararaman"}`` so either name (or both) is accepted.
_SHARED_OWNER_SEPARATOR_RE = re.compile(
    r"\s*(?:\+|,|\s+and\s+|\s+&\s+)\s*",
    re.IGNORECASE,
)


# Cycle-7 post-mortem (#2006): role tokens that show up as
# co-owners in the corpus ("Tomislav + CTO", "PM + Linnaea",
# "PM + Customer Success") must be registered against the
# specific entity so the model writing the role for THAT entity
# passes — but they must NOT join the global
# ``known_stakeholders`` set.  If they did, the model casually
# writing "approved by the CTO" or "escalated to Customer
# Success" in prose near ANOTHER entity would falsely trip the
# mismatch detector (cycle 7 saw 22 of 38 mismatches as exactly
# this class of false positive).  Roles are: any all-caps token
# 2-5 chars (CTO, CEO, COO, PM, VP, CFO, CIO, HR, QA, IT) plus a
# small curated phrase set seen in this corpus.
_ROLE_TOKEN_PHRASES: frozenset[str] = frozenset(
    {
        "Customer Success",
        "Engineering Manager",
        "Steering Committee",
    }
)
_ROLE_TOKEN_ABBREV_RE = re.compile(r"^[A-Z]{2,5}$")


def _is_role_token(name: str) -> bool:
    """Return True if *name* is a generic role label (not a person identity).

    Role tokens are valid co-owner annotations for a specific
    entity but unsafe to register as global stakeholders because
    the model uses them loosely in prose.
    """
    stripped = name.strip()
    if not stripped:
        return False
    if _ROLE_TOKEN_ABBREV_RE.match(stripped):
        return True
    if stripped in _ROLE_TOKEN_PHRASES:
        return True
    return False


def _split_shared_owners(raw_name: str) -> list[str]:
    """Split a raw owner field into individual canonical owner names.

    Examples (cycle-6 corpus shapes):

    * ``"Beatriz Cazadora-Olesen"`` → ``["Beatriz Cazadora-Olesen"]``
    * ``"Tomislav Hessford + Avantika Sundararaman"`` →
      ``["Tomislav Hessford", "Avantika Sundararaman"]``
    * ``"PM + Customer Success"`` → ``["PM", "Customer Success"]``
    * ``"Tomislav Hessford and CTO"`` →
      ``["Tomislav Hessford", "CTO"]``

    Empty fragments are dropped.  Leading/trailing whitespace is
    trimmed on every fragment.
    """
    if not raw_name or not raw_name.strip():
        return []
    return [part.strip() for part in _SHARED_OWNER_SEPARATOR_RE.split(raw_name) if part.strip()]


def _populate_from_file(
    text: str,
    owners: dict[str, set[str]],
    all_stakeholders: set[str],
) -> None:
    """Walk *text*, populating *owners* and *all_stakeholders* in place.

    Pairs each entity-id heading with the FIRST ``Owner:`` / ``Decided by:``
    line that appears before the next heading of the same level.

    Cycle-6 post-mortem (#2006): shared owner strings like
    ``"Tomislav Hessford + Avantika Sundararaman"`` are split into
    individual names so the detector treats either co-owner as a
    valid attribution.  See ``_split_shared_owners``.

    Cycle-7 post-mortem (#2006): role tokens like ``CTO`` / ``PM`` /
    ``Customer Success`` from shared-owner strings are added to
    THIS entity's owner set (so the model writing the role for
    this entity passes) but NOT to ``all_stakeholders`` (so the
    detector doesn't scan for those tokens across the response and
    flag casual prose mentions as mismatches).  See
    ``_is_role_token``.
    """
    # Index every heading position so we can compute section bounds.
    headings: list[tuple[int, str]] = []
    for m in _ENTITY_ID_RE.finditer(text):
        headings.append((m.start(), m.group("id").upper()))

    # Add an end-sentinel.
    bounds = headings + [(len(text), "")]

    for i, (start, entity_id) in enumerate(headings):
        end = bounds[i + 1][0]
        section = text[start:end]
        for owner_match in _OWNER_LINE_RE.finditer(section):
            raw_name = owner_match.group("name").strip()
            if not raw_name:
                continue
            for individual_name in _split_shared_owners(raw_name):
                owners.setdefault(entity_id, set()).add(individual_name)
                if not _is_role_token(individual_name):
                    all_stakeholders.add(individual_name)


# Window after an entity-ID mention in the response within which we look for
# an attributed owner.  Wider than a single line because the model often
# spreads "R-12" and "Owner: X" across a multi-line bullet block.
_ATTRIBUTION_WINDOW_CHARS: int = 240

# Cycle-10 post-mortem (#2006): two structural false-positive
# classes were observed in C10.  Both are addressed by tightening
# the window-shaping logic below; the loose 240-char scan stays in
# place for the common bullet-block case.
_ENTITY_ID_RAW = r"R-\d+|DEC-\d{4}-\d{2}-\d{2}-\d+|CHG-NIMB-\d+"
# Fix 1B — directional attribution patterns.
#
# When the model writes prose like ``Yusuf Almasi's ownership of
# R-13; Hyeon-Jin Park's ownership of R-12``, R-13's 240-char
# window legitimately captures "Hyeon-Jin Park" before reaching
# R-12.  But the prose attributes Hyeon-Jin to R-12 (the entity
# AFTER), not R-13.  Detect ``<PERSON>'s ownership of <NEXT>`` /
# ``<PERSON> owns <NEXT>`` / ``<PERSON> is the owner of <NEXT>`` /
# ``Owner of <NEXT> is <PERSON>`` patterns and skip the
# (prev_entity, person) finding when the person attaches forward.
_FORWARD_ATTRIBUTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?P<person>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)['’]s\s+(?:ownership|owner)\s+of\s+(?P<entity>"
        + _ENTITY_ID_RAW
        + r")",
    ),
    re.compile(
        r"(?P<person>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)\s+owns\s+(?P<entity>"
        + _ENTITY_ID_RAW
        + r")",
    ),
    re.compile(
        r"(?P<person>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)\s+is\s+the\s+owner\s+of\s+(?P<entity>"
        + _ENTITY_ID_RAW
        + r")",
    ),
)


def _person_attaches_forward(window: str, person: str) -> bool:
    """Return True when *person* appears in a forward-attribution pattern
    inside *window* (attaching to an entity that comes AFTER, not the
    one the window started from).
    """
    for pat in _FORWARD_ATTRIBUTION_PATTERNS:
        for m in pat.finditer(window):
            if m.group("person") == person:
                return True
    return False


def _line_bounds_at(text: str, position: int) -> tuple[int, int]:
    """Return ``(line_start, line_end)`` for the line containing *position*.

    ``line_end`` points just BEFORE the line's trailing newline (or to
    ``len(text)`` if the position is on the final unterminated line).
    """
    if position < 0:
        position = 0
    if position > len(text):
        position = len(text)
    line_start = text.rfind("\n", 0, position) + 1  # 0 if no prior newline
    next_newline = text.find("\n", position)
    line_end = next_newline if next_newline != -1 else len(text)
    return line_start, line_end


def _is_table_row_line(line: str) -> bool:
    """Return True if *line* (with newline stripped) is a markdown table row.

    A table row starts with optional leading whitespace then a pipe.
    Excludes the header-separator row (``|---|---|``) which also matches
    but never carries attribution prose.
    """
    stripped = line.lstrip()
    if not stripped.startswith("|"):
        return False
    # Header-separator: cells are only ``-``, ``:``, and ``|``.
    if set(stripped.replace(" ", "")) <= set("-:|"):
        return False
    return True


def _next_pipe_after(text: str, start: int, line_end: int) -> int:
    """Return the position of the next ``|`` after *start* (exclusive)
    within ``[start, line_end)``, or *line_end* if none found.

    Used to cap a table-row attribution window at the next cell
    boundary so the scan doesn't bleed into the next cell.
    """
    pipe_pos = text.find("|", start, line_end)
    return pipe_pos if pipe_pos != -1 else line_end


# #2006 cycle-13 (DeepSeek V4 Pro): two more structural false-positive
# classes were observed in C13.  Both are **generic English NLP**
# patterns — they're not tied to PM-corpus tokens.  Implementing them
# as pattern-shape recognition (not specific-token blocklists) means
# they help any future corpus deployment where the model uses similar
# English prose conventions, not just the PM harness.

# Disavowal patterns — prose that explicitly clarifies a person was
# named in some lesser capacity, NOT as the owner / decision-maker /
# attribution subject.  Real C13 case (scenario 02 iter 3):
#
#     "Linnaea Korhonen is listed as the risk owner for R-19 ...,
#     with Tomislav Hessford cited only in his capacity as Sponsor
#     ... not as the risk owner."
#
# The 240-char window from R-19 captures "Tomislav Hessford" and
# reports a (R-19, Tomislav) mismatch — but the prose EXPLICITLY
# disavows that very attribution.  Suppressing on disavowal preserves
# detector strength in the affirmative case while clearing the
# explicit-negation false positives.
#
# Pattern shapes (all generic English negation, no domain tokens):
#   * ``<PERSON> ... (cited|mentioned|named) only (as|in) ...``
#   * ``<PERSON> ... in (his|her|their) capacity as ...``
#   * ``<PERSON> ... NOT (as|the) <noun>``
#   * ``<PERSON> ... only (as|in) <noun>`` (compact form)
_DISAVOWAL_RE = re.compile(
    r"(?:cited|mentioned|named|referenced|appears?)\s+only\s+(?:as|in)"
    r"|in\s+(?:his|her|their|whose)\s+capacity\s+as"
    r"|not\s+(?:as|the)\s+(?:the\s+)?[a-z][\w-]*"
    r"|only\s+(?:as|in\s+the\s+role\s+of)\s+[a-z]",
    re.IGNORECASE,
)


def _person_in_disavowal_context(window: str, person: str, span: int = 80) -> bool:
    """Return True when *person* sits within ``span`` chars of an
    English-negation pattern that disavows an attribution claim.

    Generic English NLP — works in any domain where the model writes
    careful prose like ``X mentioned only as <role>`` or ``X NOT the
    <noun>``.  Does NOT depend on any corpus-specific role token.
    """
    person_starts = [m.start() for m in re.finditer(re.escape(person), window)]
    if not person_starts:
        return False
    for ds_match in _DISAVOWAL_RE.finditer(window):
        ds_pos = ds_match.start()
        for ps in person_starts:
            # Disavowal can appear either before or after the person
            # mention (e.g. ``Sponsor ... X ... not the owner`` vs
            # ``X ... only as Sponsor``).
            if abs(ds_pos - ps) <= span:
                return True
    return False


# Role-context paren patterns — prose that labels a person inside a
# parenthesised role qualifier, where the parenthesis content itself
# IS the role context not a claimed ownership relation.  Real C13
# case (scenario 03 iter 2):
#
#     "escalation to the CTO, Sponsor (Tomislav Hessford, COO), and
#     Avantika Sundararaman (R-16 owner) is the immediate next step."
#
# The 240-char window from R-16 captures "Tomislav Hessford" inside
# the ``Sponsor (Tomislav Hessford, COO)`` structure, even though the
# prose immediately afterwards explicitly attributes R-16 to Avantika.
# The pattern shape — a leading TitleCase word/phrase followed by a
# parenthesised person reference — is a generic English/markdown
# convention for *"role (name)"* labelling.  Treating the person
# inside such a structure as a role-context mention (not an attribution
# claim) clears the false positive cleanly.
#
# Pattern shapes (generic — TitleCase phrase + paren containing
# capitalised person name):
#   * ``<TitleCase_Phrase> (<PersonName>[, <suffix>]*)``  →  role-context
#   * ``<PersonName>, <TitleCase_Phrase>``  →  trailing role suffix
#
# Limit: requires the TitleCase phrase to NOT itself match the entity-
# ID regex (so we don't accidentally suppress a real ``ENTITY (PERSON)``
# attribution).
# Role-slot pattern: either an ALL-CAPS acronym (COO, CTO, CEO, PM,
# VP, ...) OR a TitleCase noun phrase ("Sponsor", "Head of Finance",
# "Migration Squad Lead", "Engineering Manager", "Counsel").  Lowercase
# joiners ("of", "the") accepted between TitleCase tokens.
_ROLE_SLOT = r"(?:[A-Z]{2,8}|[A-Z][a-z]+(?:[\s-](?:[A-Z][a-z]+|of|the|for|and)){0,4})"
_ROLE_PAREN_RE = re.compile(
    # "Sponsor (Tomislav Hessford, COO)" style — role prefix +
    # parenthesised name.  Allow trailing comma-separated qualifiers.
    r"(?<![A-Za-z0-9_-])"
    r"(?P<role>" + _ROLE_SLOT + r")"
    r"\s*\(\s*(?P<person>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)"
    r"(?:,\s*[A-Z][\w-]+){0,2}\s*\)",
)
_TRAILING_ROLE_RE = re.compile(
    # "Tomislav Hessford, COO" / "Smith, Head of Finance" style —
    # name followed by comma then role token (acronym or TitleCase).
    r"(?P<person>[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)+)"
    r",\s+(?P<role>" + _ROLE_SLOT + r")"
    r"(?=[\s,.;:)\]]|$)",
)


def _person_in_role_paren_context(window: str, person: str) -> bool:
    """Return True when *person* appears inside a role-context paren
    structure (``Role (Person, suffix)``) or a trailing role suffix
    (``Person, Role``).

    Generic prose structure — works in any deployment that uses the
    English convention of parenthesised role labelling.  Does NOT
    depend on any corpus-specific role token (the role slot matches
    any TitleCase phrase).
    """
    for m in _ROLE_PAREN_RE.finditer(window):
        if m.group("person") == person:
            # Defensive: skip when the "role" slot is itself an
            # entity-id (suppresses real ``ENTITY (PERSON)``).
            role_text = m.group("role")
            if re.match(_ENTITY_ID_RAW, role_text):
                continue
            return True
    for m in _TRAILING_ROLE_RE.finditer(window):
        if m.group("person") == person:
            return True
    return False


def detect_attribution_mismatches(
    response_text: str,
    index: AttributionIndex,
) -> list[AttributionMismatch]:
    """Scan *response_text* for ``<entity_id> ... <owner>`` patterns and return
    every mismatch against *index*.

    For each occurrence of a known entity id (``R-NN`` / ``DEC-...`` /
    ``CHG-NIMB-NNN``), looks within the next ``_ATTRIBUTION_WINDOW_CHARS`` for
    a stakeholder name from the index's ``known_stakeholders`` set.  If a name
    is found and is NOT in the entity's valid owner set, record an
    :class:`AttributionMismatch`.

    Returns an empty list when no mismatches are present (or when the
    response carries no entity-ID references at all).

    Cycle-10 post-mortem (#2006) tightens the window in two ways:

    * **Table-row scope (Fix 1A)** — when the entity-id mention lives
      inside a markdown table row, cap the window at the end of that
      row instead of 240 chars.  Prevents a row that legitimately
      attributes ENTITY-A to person X (in the row's owner cell)
      from also flagging ENTITY-B (mitigation cell of the same row,
      whose owner is in a different row).  Real C10 case:
      ``| R-12 ... CHG-NIMB-003 (mitigation); ... | Beatriz |`` —
      R-12 → Beatriz is the row's intent, but the 240-char window
      from CHG-NIMB-003 picked up Beatriz too.
    * **Forward-attribution prose (Fix 1B)** — when the window
      contains ``<PERSON>'s ownership of <NEXT_ENTITY>``, the person
      attaches FORWARD to the next entity, not backward to the one
      the window started from.  Real C10 case:
      ``Yusuf Almasi's ownership of R-13; Hyeon-Jin Park's
      ownership of R-12`` — R-13's window captured Hyeon-Jin who
      legitimately attached to R-12.
    """
    if not response_text or not index.owners:
        return []

    findings: list[AttributionMismatch] = []
    seen: set[tuple[str, str]] = set()

    # Find every entity-id mention in the response.  Loose pattern — we don't
    # require a heading marker because the agent writes prose, not markdown
    # headings.
    response_entity_re = re.compile(
        r"\b(" + _ENTITY_ID_RAW + r")\b",
        re.IGNORECASE,
    )

    all_entity_starts = [m.start() for m in response_entity_re.finditer(response_text)]

    for entity_match in response_entity_re.finditer(response_text):
        entity_id = entity_match.group(1).upper()
        valid = index.owners.get(entity_id)
        if not valid:
            # Entity isn't in the corpus index — out of scope for THIS detector
            # (the response might just be quoting an invented entity id, which
            # the negative-canary checks catch separately).
            continue

        # Look for a stakeholder name in the window after the entity mention.
        # Cap the window at:
        #   1. ``_ATTRIBUTION_WINDOW_CHARS`` characters out, AND
        #   2. the start of the NEXT entity-id mention (so R-12's window
        #      doesn't bleed into R-19's section in a multi-risk list).
        #   3. (Fix 1A) the end of the current line if the entity sits
        #      inside a markdown table row — table rows pack multiple
        #      entity/owner pairs and the 240-char window walks across
        #      them.
        window_start = entity_match.end()
        window_cap = window_start + _ATTRIBUTION_WINDOW_CHARS
        # Find the next entity start that's strictly after this one.
        next_entity_starts = [s for s in all_entity_starts if s > entity_match.start()]
        if next_entity_starts:
            window_cap = min(window_cap, next_entity_starts[0])
        # Fix 1A — when entity sits inside a markdown table row, cap
        # at the next cell boundary (``|``) within that row so the
        # scan doesn't bleed across cells.  Capping at end-of-line
        # is too lax — the entire table row IS one line, and rows
        # routinely pack multiple entities with their respective
        # owners in different cells.
        line_start, line_end = _line_bounds_at(response_text, entity_match.start())
        if _is_table_row_line(response_text[line_start:line_end]):
            window_cap = min(window_cap, _next_pipe_after(response_text, window_start, line_end))
        window_end = min(len(response_text), window_cap)
        window = response_text[window_start:window_end]

        # Fix 1B — for the forward-attribution suppression check, look
        # at a slightly extended window that includes the NEXT entity
        # ID.  The match-window above caps at the next entity start so
        # we don't pick stakeholders attributed forward — but the
        # forward-attribution pattern itself ends WITH the next entity,
        # so we need that entity inside the search slice.  Extend by
        # 32 chars past the next entity (long enough for the longest
        # entity-id form ``DEC-2026-07-09-09`` plus closing
        # punctuation).
        fa_window_end = min(len(response_text), window_end + 32)
        fa_window = response_text[window_start:fa_window_end]

        # #2027 (DeepSeek V4 Pro cycle-20): disavowal / role-paren
        # suppression context widened from a ±200-char entity slice to
        # the FULL response.  Long self-correcting responses (e.g.
        # ``**Correction note:** ... the corpus is clear: <entity> is
        # owned by <correct_person>. The decision was <wrong_person>'s,
        # but that is the mitigation decision — not the risk itself.``)
        # put the disavowal phrase several hundred chars away from the
        # first entity mention.  Widening lets the suppression catch
        # them in any response — costs a few CPU cycles per stakeholder
        # check, no semantic risk because the disavowal / role-paren
        # patterns are specific enough that random matches are rare.
        suppression_context = response_text

        for stakeholder in index.known_stakeholders:
            if stakeholder in window and stakeholder not in valid:
                # Fix 1B — if the stakeholder name appears as the
                # subject of a forward-attribution pattern targeting
                # an entity that comes AFTER our current one, the
                # prose attaches the person to that future entity,
                # not the one we're scanning from.  Suppress.
                if _person_attaches_forward(fa_window, stakeholder):
                    continue
                # #2006 cycle-13 — generic English NLP suppressions:
                # disavowal prose ("X cited only as ROLE, not the
                # owner") and role-context paren ("Sponsor (X, COO)")
                # both indicate the person mention is a role-context
                # clarification rather than an ownership attribution.
                if _person_in_disavowal_context(suppression_context, stakeholder):
                    continue
                if _person_in_role_paren_context(suppression_context, stakeholder):
                    continue
                key = (entity_id, stakeholder)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    AttributionMismatch(
                        entity_id=entity_id,
                        claimed_owner=stakeholder,
                        valid_owners=valid,
                    )
                )

    return findings
