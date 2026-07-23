"""Canonical fuzzy tool-name resolver.

Single source of truth for all tool-name resolution across Cogtrix.
Used by both the orchestration graph (process_tools node) and the
request_tools meta-tool factory.
"""

from __future__ import annotations

from typing import Any

FUZZY_MATCH_THRESHOLD = (
    0.65  # raised from 0.40; blocks read_file↔write_file (0.63), search_web↔search_news (0.63)
)

#: Length cutoff for the short-request guard (#1924 / #1919 Finding 1).
#: A normalised tool-name request of this length or shorter that is a
#: single token (no underscores after normalisation) is too ambiguous to
#: fuzzy-resolve: the Jaccard + bonus formula gives any 1-token candidate
#: a free score boost from ``_is_word_contained`` + ``issubset``, which
#: makes ``"run"`` score 1.10 against ``"extend_run"`` while
#: ``"execute_shell_command"`` scores 0.00 — the resolver picks the
#: coincidental-overlap candidate, not the semantic one.  The guard
#: forces such short single-token requests to either match exactly or
#: come through the curated alias map.
_SHORT_REQUEST_MAX_LEN = 4

#: Curated alias map for tool names the LLM commonly hallucinates from
#: training distribution (#1924 / #1919 Finding 4).  Consulted BEFORE
#: fuzzy matching so the resolver returns a deterministic, correct
#: mapping for well-known variants — fuzzy matching cannot reliably
#: catch them because ``run_shell_command ∩ execute_shell_command``
#: scores 0.5 (below threshold) and no qualifying bonus fires.
#:
#: Each entry maps a hallucinated name → the canonical tool name.  The
#: resolver verifies the canonical target exists in ``available_tools``
#: or ``active_tool_names`` before returning — an entry whose target
#: isn't registered is silently ignored (so removing a tool from the
#: catalog doesn't make the alias point at nothing).
#:
#: Extend over time as agent-test runs surface more variants.  Each
#: addition should be a *systematic* model-training-distribution
#: variant, not a one-off typo — typos are what the fuzzy matcher is
#: for.
_KNOWN_ALIASES: dict[str, str] = {
    # Shell execution family — Anthropic Claude / Cline / Aider / Aider's
    # "bash" tool, various LangChain templates all use these names.
    "run_shell_command": "execute_shell_command",
    "run_command": "execute_shell_command",
    "execute_command": "execute_shell_command",
    "bash": "execute_shell_command",
    "shell": "execute_shell_command",
    "run_bash": "execute_shell_command",
    "shell_exec": "execute_shell_command",
    "exec": "execute_shell_command",
}


def _is_word_contained(short: str, long: str) -> bool:
    """True when *short* appears in *long* on underscore word boundaries."""
    if short == long:
        return True
    return long.startswith(short + "_") or long.endswith("_" + short) or f"_{short}_" in long


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    # Standard dynamic programming implementation
    m, n = len(a), len(b)
    if m < n:
        a, b, m, n = b, a, n, m

    # Use two rows instead of full matrix
    prev = list(range(n + 1))

    for i, char_a in enumerate(a, 1):
        curr = [i] + [0] * n
        for j, char_b in enumerate(b, 1):
            if char_a == char_b:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    return prev[n]


def resolve_tool_name(
    requested: str,
    available_tools: dict[str, Any],
    active_tool_names: set[str] | None = None,
) -> tuple[str | None, str]:
    """Map *requested* (the name the LLM used) to an actual registered tool.

    Resolution order:
      1. Exact match in the on-demand pool  → ``("name", "available")``
      2. Exact match among active tools     → ``("name", "active")``
      3. Curated alias map (#1924)          → ``("canonical", "available"|"active")``
      4. Short-request guard (#1924) — refuse to fuzzy-resolve single-token
         requests of length ≤ ``_SHORT_REQUEST_MAX_LEN`` characters → ``(None, "none")``
      5. Fuzzy match (token overlap /
         substring containment)             → ``("name", "available"|"active")``
      6. No match                           → ``(None, "none")``

    Tie-breaking rules (applied when scores are equal):
      - Exact name match wins
      - Shorter Levenshtein distance wins
      - Alphabetically first wins

    Args:
        requested: The raw tool name produced by the LLM.
        available_tools: ``{name: tool}`` dict of on-demand (not yet loaded) tools.
        active_tool_names: Names of tools currently loaded in the agent.
            Pass ``None`` or an empty set when only searching the available pool.

    Returns:
        ``(resolved_name, source)`` where *source* is one of ``"available"``,
        ``"active"``, or ``"none"``.
    """
    _active: set[str] = active_tool_names or set()

    if requested in available_tools:
        return requested, "available"
    if requested in _active:
        return requested, "active"

    req_norm = requested.lower().replace("-", "_")
    req_tokens = set(req_norm.split("_"))

    # ── Step 3: curated alias map ─────────────────────────────────────
    # Consult BEFORE fuzzy matching: fuzzy can't reliably catch
    # model-training-distribution variants like ``run_shell_command``
    # → ``execute_shell_command`` (Jaccard 0.5, below threshold, no
    # qualifying bonus).  The alias map gives a deterministic answer
    # for known variants; everything else falls through to fuzzy.
    canonical = _KNOWN_ALIASES.get(req_norm)
    if canonical is not None:
        if canonical in available_tools:
            return canonical, "available"
        if canonical in _active:
            return canonical, "active"
        # Canonical target not registered — silently fall through to
        # fuzzy.  This keeps the alias map robust against a tool being
        # removed from the catalog without coordinated alias-map edits.

    # ── Step 4: short-request guard ───────────────────────────────────
    # Single-token requests of length ≤ _SHORT_REQUEST_MAX_LEN cannot
    # be safely fuzzy-resolved: the Jaccard + bonus formula always
    # gives any 1-token candidate a free boost (word_contained +
    # issubset both fire when req_tokens={req} ⊂ tn_tokens), so the
    # winner is whichever short candidate has the fewest other tokens
    # — coincidental overlap, not semantic match.  ``run`` matches
    # ``extend_run`` (1.10) over ``execute_shell_command`` (0.00)
    # exactly this way.  Refuse to guess; the dispatcher's
    # ``not a valid tool and could not be resolved`` message is a
    # clearer signal than a wrong fuzzy answer.
    if len(req_norm) <= _SHORT_REQUEST_MAX_LEN and len(req_tokens) == 1:
        return None, "none"

    # ── Step 5: fuzzy match ───────────────────────────────────────────
    candidates = _score_candidates(req_norm, req_tokens, available_tools, _active)
    if not candidates:
        return None, "none"
    # ``candidates`` is sorted by descending score (ties broken by
    # Levenshtein then alphabetical); the first entry is the best.
    best = candidates[0]
    if best[1] >= FUZZY_MATCH_THRESHOLD:
        return best[0], best[2]
    return None, "none"


def _score_candidates(
    req_norm: str,
    req_tokens: set[str],
    available_tools: dict[str, Any],
    active_tool_names: set[str],
) -> list[tuple[str, float, str, float]]:
    """Score every candidate in both pools, return a ranked list.

    Each entry is ``(tool_name, score, source, levenshtein_dist)``.
    Sorted by descending score; ties broken by lower Levenshtein
    distance, then alphabetical name.

    Shared by :func:`resolve_tool_name` (which takes the top entry and
    applies :data:`FUZZY_MATCH_THRESHOLD`) and :func:`top_k_candidates`
    (which returns up to K entries above a softer threshold for
    diagnostic surfacing).  Extracting the scoring loop here avoids
    double-scoring when both APIs are called for the same request and
    keeps the formula in one place.
    """
    scored: list[tuple[str, float, str, float]] = []
    pools: list[tuple[dict[str, Any] | set[str], str]] = [
        (available_tools, "available"),
        (active_tool_names, "active"),
    ]
    for pool, source in pools:
        for tool_name in pool:
            tn_norm = tool_name.lower().replace("-", "_")
            tn_tokens = set(tn_norm.split("_"))

            intersection = req_tokens & tn_tokens
            union = req_tokens | tn_tokens
            score = len(intersection) / len(union) if union else 0.0

            if _is_word_contained(req_norm, tn_norm) or _is_word_contained(tn_norm, req_norm):
                score += 0.40
            if req_tokens.issubset(tn_tokens) or tn_tokens.issubset(req_tokens):
                score += 0.20

            _prefix_hit = any(
                (len(a) >= 3 and len(b) >= 3 and a != b and (b.startswith(a) or a.startswith(b)))
                for a in req_tokens
                for b in tn_tokens
            )
            if _prefix_hit:
                score += 0.35

            levenshtein_dist = _levenshtein_distance(req_norm, tn_norm)
            scored.append((tool_name, score, source, levenshtein_dist))

    # Sort: higher score first, then lower Levenshtein, then alphabetical.
    scored.sort(key=lambda c: (-c[1], c[3], c[0]))
    return scored


def top_k_candidates(
    requested: str,
    available_tools: dict[str, Any],
    active_tool_names: set[str] | None = None,
    *,
    k: int = 3,
    min_score: float = 0.30,
) -> list[tuple[str, float, str]]:
    """Return up to ``k`` highest-scoring candidates for diagnostic surfacing.

    Each entry: ``(tool_name, score, source)``.  Sorted by descending
    score (ties broken by Levenshtein distance then alphabetical name —
    same tiebreak chain as :func:`resolve_tool_name`).

    Filtered by ``min_score`` (default 0.30) so trivially-zero
    candidates don't leak into diagnostic output.  This threshold is
    intentionally lower than :data:`FUZZY_MATCH_THRESHOLD` — the items
    returned here are *suggestions* for the agent (or the dispatcher's
    "Did you mean ..." message), not confident matches.  Callers that
    need confident matches should call :func:`resolve_tool_name`.

    Exact matches in either pool are intentionally included if their
    score clears ``min_score`` — but callers that have already checked
    for exact matches typically don't need to filter them out (an
    exact match scores 1.40+ and is unmistakable).

    The alias map (see :data:`_KNOWN_ALIASES`) and the short-request
    guard are NOT applied here — top-K is purely a scoring surface,
    not a resolution policy.  When the agent calls a name in the
    alias map the dispatcher's resolve-then-emit flow already mapped
    it to the canonical name; if the resolver returned None and the
    dispatcher needs suggestions, top-K shows the closest
    non-aliased candidates.

    Args:
        requested: The raw tool name produced by the LLM.
        available_tools: ``{name: tool}`` dict of on-demand tools.
        active_tool_names: Names of tools currently loaded.  Pass
            ``None`` or empty when only searching available.
        k: Maximum entries to return.  Must be ≥ 1.
        min_score: Drop candidates scoring below this value.

    Returns:
        List of ``(name, score, source)`` tuples, length ≤ ``k``.
        Empty when no candidate clears ``min_score``.
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")
    _active: set[str] = active_tool_names or set()
    req_norm = requested.lower().replace("-", "_")
    req_tokens = set(req_norm.split("_"))
    scored = _score_candidates(req_norm, req_tokens, available_tools, _active)
    # #1932: ``_score_candidates`` iterates both ``available_tools`` and
    # ``active_tool_names`` and emits one entry per pool sighting; a tool
    # present in both (transient mid-turn state) would otherwise surface
    # twice in the diagnostic output.  De-duplicate by name, keeping the
    # first sighting — pool iteration order is available-then-active,
    # which matches the source-precedence ``resolve_tool_name`` uses.
    seen: set[str] = set()
    deduped: list[tuple[str, float, str]] = []
    for name, score, source, _lev in scored:
        if score < min_score:
            continue
        if name in seen:
            continue
        seen.add(name)
        deduped.append((name, score, source))
        if len(deduped) >= k:
            break
    return deduped
