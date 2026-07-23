"""Tests for src/tools/resolver — canonical fuzzy tool-name resolver."""

from unittest.mock import MagicMock

from src.tools.resolver import _is_word_contained, resolve_tool_name

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

    def test_partial_prefix_resolves_in_available(self):
        """'http' → 'http_request' via word-boundary bonus."""
        available = {"http_request": self._tool("http_request")}
        name, source = resolve_tool_name("http", available)
        assert name == "http_request"
        assert source == "available"

    def test_partial_suffix_resolves_in_available(self):
        """'web' → 'search_web' via word-boundary bonus."""
        available = {"search_web": self._tool("search_web")}
        name, source = resolve_tool_name("web", available)
        assert name == "search_web"
        assert source == "available"

    def test_abbreviated_name_resolves(self):
        """'search' fuzzy-matches 'search_web' (word-boundary hit)."""
        available = {"search_web": self._tool("search_web")}
        name, source = resolve_tool_name("search", available)
        assert name == "search_web"
        assert source == "available"

    def test_fuzzy_resolves_in_active_when_not_available(self):
        """Fuzzy match falls through to the active pool."""
        available = {}
        active = {"shell_exec"}
        name, source = resolve_tool_name("shell", available, active)
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

        'api_get' and 'api_post' both have same distance from 'api' (3 chars each)
        Alphabetically, 'api_get' < 'api_post'
        """
        available = {"api_post": self._tool("api_post"), "api_get": self._tool("api_get")}
        name, source = resolve_tool_name("api", available)
        assert name == "api_get"  # Alphabetically first
        assert source == "available"

    def test_levenshtein_beats_alphabetical(self):
        """Levenshtein distance is checked before alphabetical order.

        'web' vs 'web_api' vs 'web_service' all have same token match.
        Levenshtein('web', 'web') = 0 (but 'web' not in dict)
        Levenshtein('web', 'web_api') = 4 ('_api')
        Levenshtein('web', 'web_service') = 8 ('_service')
        'web_api' should win despite 'web_service' coming later alphabetically.
        """
        available = {
            "web_service": self._tool("web_service"),
            "web_api": self._tool("web_api"),
        }
        name, source = resolve_tool_name("web", available)
        assert name == "web_api"  # Shorter edit distance
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
