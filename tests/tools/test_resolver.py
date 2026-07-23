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
        """When both pools have a fuzzy hit, available wins (first pool)."""
        available = {"search_web": self._tool("search_web")}
        active = {"search_db"}
        name, source = resolve_tool_name("search", available, active)
        assert name == "search_web"
        assert source == "available"

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
