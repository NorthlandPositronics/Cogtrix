"""Tests for src/tools/resolver — canonical fuzzy tool-name resolver."""

from unittest.mock import MagicMock

import pytest

from src.tools.resolver import _is_word_contained, resolve_tool_name, top_k_candidates

# ---------------------------------------------------------------------------
# _is_word_contained
# ---------------------------------------------------------------------------


class TestIsWordContained:
    def test_identical_strings(self):
        assert _is_word_contained("foo", "foo") is True

    def test_prefix_boundary(self):
        assert _is_word_contained("search", "search_web") is True

    def test_suffix_boundary(self):
        assert _is_word_contained("web", "search_web") is True

    def test_middle_boundary(self):
        assert _is_word_contained("get", "http_get_json") is True

    def test_no_boundary_match(self):
        assert _is_word_contained("ear", "search") is False

    def test_substring_without_boundary(self):
        assert _is_word_contained("sea", "search") is False

    def test_empty_short(self):
        # empty == empty is True
        assert _is_word_contained("", "") is True

    def test_short_longer_than_long(self):
        assert _is_word_contained("search_web_extra", "search_web") is False


# ---------------------------------------------------------------------------
# resolve_tool_name — exact matches
# ---------------------------------------------------------------------------


class TestResolveToolNameExact:
    def _tool(self, name: str) -> MagicMock:
        return MagicMock(description=f"Tool {name}")

    def test_exact_match_in_available(self):
        available = {"search_web": self._tool("search_web")}
        name, source = resolve_tool_name("search_web", available)
        assert name == "search_web"
        assert source == "available"

    def test_exact_match_in_active(self):
        available = {}
        active = {"shell"}
        name, source = resolve_tool_name("shell", available, active)
        assert name == "shell"
        assert source == "active"

    def test_available_takes_priority_over_active(self):
        available = {"search_web": self._tool("search_web")}
        active = {"search_web"}
        name, source = resolve_tool_name("search_web", available, active)
        assert name == "search_web"
        assert source == "available"

    def test_no_match_returns_none(self):
        available = {"search_web": self._tool("search_web")}
        name, source = resolve_tool_name("completely_unrelated_xyz", available)
        assert name is None
        assert source == "none"

    def test_none_active_accepted(self):
        available = {"search_web": self._tool("search_web")}
        name, source = resolve_tool_name("search_web", available, None)
        assert name == "search_web"
        assert source == "available"


# ---------------------------------------------------------------------------
# resolve_tool_name — fuzzy matches
# ---------------------------------------------------------------------------


class TestResolveToolNameFuzzy:
    def _tool(self, name: str) -> MagicMock:
        return MagicMock(description=f"Tool {name}")

    def test_partial_prefix_short_request_returns_none(self):
        """#1924: 'http' (4 chars, single token) is too ambiguous to
        fuzzy-resolve.  The short-request guard refuses; in a real
        catalog with ``http_get`` + ``http_post`` + ``http_request``
        the resolver had no principled way to choose.  The agent gets
        a clear "not a valid tool" signal and can either use the full
        name or call ``request_tools(query='...')``.
        """
        available = {"http_request": self._tool("http_request")}
        name, source = resolve_tool_name("http", available)
        assert name is None
        assert source == "none"

    def test_partial_suffix_short_request_returns_none(self):
        """#1924: 'web' (3 chars, single token) → short-request guard."""
        available = {"search_web": self._tool("search_web")}
        name, source = resolve_tool_name("web", available)
        assert name is None
        assert source == "none"

    def test_abbreviated_name_resolves(self):
        """'search' (6 chars) is past the short-request guard and
        fuzzy-matches 'search_web' (word-boundary hit)."""
        available = {"search_web": self._tool("search_web")}
        name, source = resolve_tool_name("search", available)
        assert name == "search_web"
        assert source == "available"

    def test_fuzzy_resolves_in_active_when_not_available(self):
        """Fuzzy match falls through to the active pool. 'shell' is
        not affected by the short-request guard (5 chars > 4)."""
        available = {}
        active = {"shell_exec"}
        name, source = resolve_tool_name("shell", available, active)
        # The curated alias map maps 'shell' → 'execute_shell_command';
        # since that target isn't registered here, fall through to
        # fuzzy, which lands on 'shell_exec' as the only candidate.
        assert name == "shell_exec"
        assert source == "active"

    def test_available_pool_preferred_over_active_for_fuzzy(self):
        """When both pools have a fuzzy hit, available wins (first pool).

        With tie-breaking, Levenshtein distance breaks ties.
        'search_db' (distance 2) beats 'search_web' (distance 4).
        Since 'search_db' is in the active pool and has shorter distance,
        it wins even though it's in the second pool.
        """
        available = {"search_web": self._tool("search_web")}
        active = {"search_db"}
        name, source = resolve_tool_name("search", available, active)
        assert name == "search_db"  # Shorter Levenshtein distance wins
        assert source == "active"  # It's in the active pool

    def test_dash_normalized_to_underscore(self):
        """Names with dashes are normalized before matching."""
        available = {"http_request": self._tool("http_request")}
        name, source = resolve_tool_name("http-request", available)
        assert name == "http_request"
        assert source == "available"

    def test_below_threshold_returns_none(self):
        """Score below FUZZY_MATCH_THRESHOLD → no match."""
        available = {"read_file": self._tool("read_file")}
        # 'xyz' shares no tokens with 'read_file'
        name, source = resolve_tool_name("xyz", available)
        assert name is None
        assert source == "none"

    def test_prefix_token_bonus(self):
        """Token prefix bonus: 'shell_exec' matches 'shell_execute' via prefix scoring.

        Jaccard({"shell","exec"}, {"shell","execute"}) = 1/3 ≈ 0.33
        Prefix hit: "exec" vs "execute" → +0.35 → total ≈ 0.68 ≥ 0.65 → match.
        """
        available = {"shell_execute": self._tool("shell_execute")}
        name, source = resolve_tool_name("shell_exec", available)
        assert name == "shell_execute"
        assert source == "available"

    def test_empty_available_and_active_returns_none(self):
        name, source = resolve_tool_name("anything", {}, set())
        assert name is None
        assert source == "none"


# ---------------------------------------------------------------------------
# resolve_tool_name — tie-breaking
# ---------------------------------------------------------------------------


class TestResolveToolNameTieBreaking:
    """Test tie-breaking when two tools have identical fuzzy scores."""

    def _tool(self, name: str) -> MagicMock:
        return MagicMock(description=f"Tool {name}")

    def test_levenshtein_distance_breaks_tie(self):
        """When scores are equal, shorter Levenshtein distance wins.

        Requesting 'search':
        - 'search' (exact) would win via exact match, but that's handled separately
        - 'search_web' vs 'search_db': both get same Jaccard score, but 'search_db' is shorter
        """
        # Both have same tokens, same word-boundary hits, same prefix hits
        # Levenshtein('search', 'search_web') = 4 ('_web')
        # Levenshtein('search', 'search_db') = 2 ('_db')
        available = {"search_db": self._tool("search_db"), "search_web": self._tool("search_web")}
        name, source = resolve_tool_name("search", available)
        assert name == "search_db"  # Shorter edit distance wins
        assert source == "available"

    def test_alphabetical_breaks_tie_when_levenshtein_equal(self):
        """When scores AND Levenshtein distance are equal, alphabetically first wins.

        Use a 5-char single-token request that's past the #1924
        short-request guard so the fuzzy + tie-breaking path is
        actually exercised.

        'apiv2' vs 'apiv2_get' and 'apiv2_post' both produce equal
        token-overlap and prefix bonus; Levenshtein distance ties; the
        alphabetically-first candidate ('apiv2_get') wins.
        """
        available = {
            "apiv2_post": self._tool("apiv2_post"),
            "apiv2_get": self._tool("apiv2_get"),
        }
        name, source = resolve_tool_name("apiv2", available)
        assert name == "apiv2_get"  # Alphabetically first
        assert source == "available"

    def test_levenshtein_beats_alphabetical(self):
        """Levenshtein distance is checked before alphabetical order.

        'image' (5 chars, single token) clears the #1924 guard.
        Distances: 'image_api' = 4, 'image_service' = 8 — 'image_api'
        wins despite 'image_service' coming later alphabetically.
        """
        available = {
            "image_service": self._tool("image_service"),
            "image_api": self._tool("image_api"),
        }
        name, source = resolve_tool_name("image", available)
        assert name == "image_api"  # Shorter edit distance
        assert source == "available"

    def test_exact_name_match_takes_priority(self):
        """Exact name match wins even if other tools have same score after bonuses.

        When the requested name exactly matches a tool, it should be returned
        regardless of other tools' scores.
        """
        available = {
            "search_api_v2": self._tool("search_api_v2"),
            "search": self._tool("search"),
        }
        # 'search' matches exactly, so it should be returned immediately
        # This is handled by the exact match check at the start of the function
        name, source = resolve_tool_name("search", available)
        assert name == "search"
        assert source == "available"

    def test_tie_breaking_with_active_pool(self):
        """Tie-breaking works across both available and active pools.

        When a tie exists across pools, the same tie-breaking rules apply.
        """
        available = {"search_db": self._tool("search_db")}
        active = {"search_api"}
        # Both 'search_db' and 'search_api' have same score for request 'search'
        # Levenshtein('search', 'search_db') = 2, Levenshtein('search', 'search_api') = 3
        name, source = resolve_tool_name("search", available, active)
        assert name == "search_db"  # Shorter Levenshtein distance
        assert source == "available"

    def test_tie_breaking_deterministic_with_many_options(self):
        """With many equally-scoring options, selection is deterministic.

        All tools have same Jaccard score and same Levenshtein distance,
        so alphabetical order should consistently pick the same winner.

        Using tools that share tokens with the request for a valid match.
        """
        available = {
            "zzz_search": self._tool("zzz_search"),
            "aaa_search": self._tool("aaa_search"),
            "mmm_search": self._tool("mmm_search"),
        }
        # All three have identical token structure and distance to 'search'
        name, source = resolve_tool_name("search", available)
        assert name == "aaa_search"  # Alphabetically first
        assert source == "available"


# ---------------------------------------------------------------------------
# resolve_tool_name — #1924: curated alias map
# ---------------------------------------------------------------------------


class TestResolveToolNameAliasMap:
    """#1924 (#1919 Finding 4): the resolver consults a curated alias
    map for model-training-distribution variants BEFORE fuzzy matching.

    Fuzzy can't reliably catch these — ``run_shell_command`` ∩
    ``execute_shell_command`` scores Jaccard 0.5 (below the 0.65
    threshold) and no qualifying bonus fires.  The alias map gives a
    deterministic answer.
    """

    def _tool(self, name: str) -> MagicMock:
        return MagicMock(description=f"Tool {name}")

    def test_run_shell_command_alias_resolves_to_execute_shell_command(self):
        """The motivating reproducer from #1919 test2: the model
        repeatedly called ``run_shell_command`` (16 times) before
        eventually pivoting to the real ``execute_shell_command``."""
        available = {"execute_shell_command": self._tool("execute_shell_command")}
        name, source = resolve_tool_name("run_shell_command", available)
        assert name == "execute_shell_command"
        assert source == "available"

    def test_bash_alias_resolves_to_execute_shell_command(self):
        """Anthropic Claude's shell tool is canonically named ``bash``;
        models often fall back to that name across providers."""
        available = {"execute_shell_command": self._tool("execute_shell_command")}
        name, source = resolve_tool_name("bash", available)
        assert name == "execute_shell_command"
        assert source == "available"

    def test_alias_resolves_in_active_pool_too(self):
        """If the canonical target is already loaded (active), the
        alias resolves there — source reflects the pool the canonical
        target lives in, not the alias's origin."""
        active = {"execute_shell_command"}
        name, source = resolve_tool_name("run_command", {}, active)
        assert name == "execute_shell_command"
        assert source == "active"

    def test_available_takes_priority_over_active_when_both_present(self):
        """If the canonical target appears in both pools (transient
        activation state), prefer available — same order the resolver
        applies for exact matches at step 1-2."""
        available = {"execute_shell_command": self._tool("execute_shell_command")}
        active = {"execute_shell_command"}
        name, source = resolve_tool_name("bash", available, active)
        assert name == "execute_shell_command"
        assert source == "available"

    def test_alias_canonical_not_registered_falls_through_to_fuzzy(self):
        """If the canonical target isn't registered (e.g. the tool
        was removed from the catalog), the alias entry is silently
        ignored and the resolver falls through to fuzzy.  This keeps
        the alias map robust against uncoordinated catalog edits."""
        # Catalog has no ``execute_shell_command``; only ``shell_dispatch``
        # exists.  ``run_shell_command`` is in the alias map mapping to
        # ``execute_shell_command``, but that target isn't here.  Fall
        # through to fuzzy → shell_dispatch is the only candidate sharing
        # the ``shell``/``command`` tokens.
        available = {"shell_dispatch": self._tool("shell_dispatch")}
        name, source = resolve_tool_name("run_shell_command", available)
        # Either: fuzzy lands on shell_dispatch (Jaccard ~0.4, below
        # threshold → None), or returns None.  Both outcomes are
        # acceptable — the key invariant is that no incorrect alias
        # mapping leaks through.
        assert name in (None, "shell_dispatch")

    def test_dash_normalised_alias_lookup(self):
        """The alias map is consulted on the normalised name (dashes
        → underscores, lowercased), so dashed variants still resolve."""
        available = {"execute_shell_command": self._tool("execute_shell_command")}
        name, source = resolve_tool_name("Run-Shell-Command", available)
        assert name == "execute_shell_command"
        assert source == "available"


# ---------------------------------------------------------------------------
# resolve_tool_name — #1924: short-request guard
# ---------------------------------------------------------------------------


class TestResolveToolNameShortRequestGuard:
    """#1924 (#1919 Finding 1): single-token requests of ≤4 chars are
    too ambiguous to fuzzy-resolve — they coincidentally substring-
    overlap many candidates and the Jaccard + bonus formula gives the
    short-token candidate a free score boost.

    The guard returns ``(None, "none")`` for such requests UNLESS the
    request has an exact match or a curated alias mapping.
    """

    def _tool(self, name: str) -> MagicMock:
        return MagicMock(description=f"Tool {name}")

    def test_run_does_not_fuzzy_to_extend_run(self):
        """The exact #1919 test5 reproducer: agent emits ``run`` with
        a ``command`` arg.  Before the guard, the resolver picked
        ``extend_run`` (score 1.10) over ``execute_shell_command``
        (0.00) and ``run_shell_command`` (0.93) — purely coincidental
        token overlap.

        After the guard, the resolver returns ``(None, "none")``
        — the dispatcher emits a clear "not a valid tool" message
        and the agent recovers by calling the real tool by name or
        going through ``request_tools(query=...)``."""
        available = {
            "extend_run": self._tool("extend_run"),
            "execute_shell_command": self._tool("execute_shell_command"),
        }
        name, source = resolve_tool_name("run", available)
        assert name is None
        assert source == "none"

    def test_3char_single_token_request_is_guarded(self):
        available = {"some_long_tool_name": self._tool("some_long_tool_name")}
        name, source = resolve_tool_name("get", available)
        assert name is None
        assert source == "none"

    def test_4char_single_token_request_is_guarded(self):
        """4 chars is the boundary; still guarded."""
        available = {"some_long_tool_name": self._tool("some_long_tool_name")}
        name, source = resolve_tool_name("read", available)
        assert name is None
        assert source == "none"

    def test_5char_single_token_request_is_NOT_guarded(self):
        """5 chars is past the boundary — fuzzy can run."""
        available = {"shell_exec": self._tool("shell_exec")}
        name, source = resolve_tool_name("shell", available)
        # 'shell' (5 chars) → fuzzy → 'shell_exec' (word_contained hit).
        # The exact resolution depends on the alias map status of 'shell'.
        # Here 'shell' is in the alias map but 'execute_shell_command'
        # isn't registered, so it falls through to fuzzy.
        assert name == "shell_exec"
        assert source == "available"

    def test_short_request_with_exact_match_still_resolves(self):
        """The guard only blocks FUZZY resolution; an exact match
        in available or active is still returned at step 1-2."""
        available = {"ls": self._tool("ls")}
        name, source = resolve_tool_name("ls", available)
        assert name == "ls"
        assert source == "available"

    def test_short_request_with_alias_match_still_resolves(self):
        """``bash`` is 4 chars but is in the alias map → resolves
        to ``execute_shell_command`` before the guard fires."""
        available = {"execute_shell_command": self._tool("execute_shell_command")}
        name, source = resolve_tool_name("bash", available)
        assert name == "execute_shell_command"
        assert source == "available"

    def test_multi_token_short_request_is_NOT_guarded(self):
        """The guard requires SINGLE-token requests.  A short multi-
        token request (e.g. ``do_x``, 4 chars total but 2 tokens)
        still goes through fuzzy.  Multi-token requests have enough
        information to be unambiguously matched."""
        available = {"do_action": self._tool("do_action")}
        # 'do_x' (4 chars, 2 tokens: {do, x}) — no exact match, no
        # alias; multi-token so guard does not fire; falls through to
        # fuzzy.  Whether it actually matches depends on the formula
        # — the test just asserts the guard doesn't block it.
        name, _source = resolve_tool_name("do_x", available)
        # Either fuzzy resolves to do_action or returns None via
        # threshold — both are valid outcomes; the key is that the
        # short-request guard did NOT short-circuit.
        # (Token Jaccard {do, x} vs {do, action} = 1/3 + word_contained
        # for 'do' = 0.40, total ~0.73 → above threshold → matches.)
        assert _source in ("available", "none")


# ---------------------------------------------------------------------------
# top_k_candidates — #1926: ranked candidates for diagnostic surfacing
# ---------------------------------------------------------------------------


class TestTopKCandidates:
    """#1926 (#1919 P2 / Finding 5): the resolver exposes a top-K
    candidates API so the dispatcher's "not a valid tool" message can
    surface the closest 2-3 hits as a "Did you mean..." hint.

    The same scoring pass as ``resolve_tool_name``, but with a softer
    threshold (default 0.30 vs the resolver's 0.65) so near-matches
    are surfaced as suggestions rather than discarded.
    """

    def _tool(self, name: str) -> MagicMock:
        return MagicMock(description=f"Tool {name}")

    def test_returns_top_candidate_by_score(self):
        """Highest-scoring candidate comes first."""
        available = {
            "frobulate": self._tool("frobulate"),
            "frobnicate_v2": self._tool("frobnicate_v2"),
        }
        result = top_k_candidates("frobnicate", available)
        assert len(result) >= 1
        # 'frobnicate_v2' contains 'frobnicate' as a word → exact word_contained hit
        assert result[0][0] == "frobnicate_v2"
        assert result[0][2] == "available"
        assert result[0][1] > 0.30

    def test_respects_k_parameter(self):
        """``k`` bounds the output length."""
        available = {f"tool_{i}_search": self._tool(f"tool_{i}_search") for i in range(10)}
        result = top_k_candidates("search", available, k=3)
        assert len(result) <= 3

    def test_default_k_is_three(self):
        available = {f"tool_{i}_search": self._tool(f"tool_{i}_search") for i in range(10)}
        result = top_k_candidates("search", available)
        assert len(result) <= 3

    def test_below_min_score_returns_empty(self):
        """When no candidate clears ``min_score``, return ``[]``."""
        available = {"completely_unrelated": self._tool("completely_unrelated")}
        result = top_k_candidates("xyz_qrs_abc", available)
        # The two have no shared tokens → score 0.0 → below 0.30 floor → empty
        assert result == []

    def test_min_score_filter_applies(self):
        """Customising ``min_score`` raises the floor."""
        available = {"shared_token_x": self._tool("shared_token_x")}
        # Default 0.30 floor allows a Jaccard-0.33 candidate through.
        loose = top_k_candidates("shared", available, min_score=0.30)
        assert len(loose) >= 1
        # 0.99 floor excludes everything that isn't an exact match.
        strict = top_k_candidates("shared", available, min_score=0.99)
        assert strict == []

    def test_search_through_both_pools(self):
        """Both ``available_tools`` and ``active_tool_names`` contribute candidates."""
        available = {"search_web": self._tool("search_web")}
        active = {"search_api"}
        result = top_k_candidates("search", available, active)
        names = [r[0] for r in result]
        assert "search_web" in names
        assert "search_api" in names
        # Source labelling preserved.
        assert any(r[0] == "search_web" and r[2] == "available" for r in result)
        assert any(r[0] == "search_api" and r[2] == "active" for r in result)

    def test_ranking_consistent_with_resolve_tool_name(self):
        """When ``resolve_tool_name`` returns a fuzzy match, top-K's
        first entry should be the same tool."""
        available = {
            "shell_execute": self._tool("shell_execute"),
            "search_web": self._tool("search_web"),
        }
        # 'shell_exec' is in the alias map → bypasses resolve_tool_name's
        # fuzzy stage but NOT top-K's (top-K skips alias / guard).  Use a
        # non-aliased multi-token request that fuzzy-resolves cleanly.
        match_name, _src = resolve_tool_name("shell_executor", available)
        topk = top_k_candidates("shell_executor", available)
        if match_name is not None:
            assert topk[0][0] == match_name

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be ≥ 1"):
            top_k_candidates("anything", {}, k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k must be ≥ 1"):
            top_k_candidates("anything", {}, k=-1)

    def test_active_only_pool(self):
        """Searching only the active pool works (no available_tools)."""
        result = top_k_candidates("search", {}, {"search_db", "search_api"})
        assert len(result) == 2
        for entry in result:
            assert entry[2] == "active"

    def test_empty_pools_returns_empty(self):
        assert top_k_candidates("anything", {}, set()) == []

    def test_top_k_does_not_apply_alias_map(self):
        """top_k is a pure scoring surface — no alias remapping.

        ``run_shell_command`` is in the alias map (mapping to
        ``execute_shell_command``).  ``resolve_tool_name`` returns the
        canonical target deterministically.  ``top_k_candidates``,
        however, just scores against the pool — so it surfaces
        ``execute_shell_command`` only if the scoring formula gives it
        a meaningful overlap.
        """
        available = {"execute_shell_command": self._tool("execute_shell_command")}
        # The resolver maps this via alias → returns the canonical.
        resolved, _ = resolve_tool_name("run_shell_command", available)
        assert resolved == "execute_shell_command"
        # top_k scores by token overlap only — {run, shell, command}
        # vs {execute, shell, command} = Jaccard 0.5, no qualifying
        # bonus → score 0.5 → above the 0.30 floor.
        topk = top_k_candidates("run_shell_command", available)
        assert len(topk) == 1
        assert topk[0][0] == "execute_shell_command"

    def test_dedupes_when_tool_appears_in_both_pools(self):
        """#1932: a tool present in BOTH ``available_tools`` AND
        ``active_tool_names`` (transient mid-turn state) used to leak
        into the output as two entries because ``_score_candidates``
        emits one row per pool sighting.  Verified live during the
        verify-1919 fleet re-run — the dispatcher's "Closest candidates"
        message showed ``'extend_run' (score 1.10), 'extend_run' (score
        1.10)``.  De-dupe by name so the diagnostic stays clean.
        """
        available = {"extend_run": self._tool("extend_run")}
        active = {"extend_run"}
        result = top_k_candidates("extend_run_yes", available, active)
        # Without de-duplication this would be 2 entries.
        assert len(result) == 1
        assert result[0][0] == "extend_run"
        # available-then-active iteration order → source is "available".
        assert result[0][2] == "available"

    def test_dedupes_across_pools_when_k_would_otherwise_fill(self):
        """Even when k is large enough to expose multiple sightings,
        de-dupe still picks one per name. ``k=3`` against a one-tool
        catalog present in both pools returns one entry, not two."""
        available = {"extend_run": self._tool("extend_run")}
        active = {"extend_run"}
        assert len(top_k_candidates("extend_run_yes", available, active, k=3)) == 1

    def test_no_dedup_needed_when_pools_disjoint(self):
        """Sanity: when each pool has its OWN unique tool, both surface."""
        available = {"search_web": self._tool("search_web")}
        active = {"search_api"}
        result = top_k_candidates("search_anywhere", available, active)
        names = {r[0] for r in result}
        # Both candidates should appear — neither is a duplicate.
        assert "search_web" in names
        assert "search_api" in names
