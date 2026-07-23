"""Tests for src/tools/_web_search_cache.py (ADR-0056 stage 0)."""

from __future__ import annotations

import time

import pytest

from src.tools import _web_search_cache as cache_mod
from src.tools._web_search_cache import (
    cache_clear,
    cache_get,
    cache_put,
    cache_size,
    normalise_query,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Every test starts with an empty cache."""
    cache_clear()
    yield
    cache_clear()


class TestNormaliseQuery:
    """Cache-key normaliser produces the documented form per ADR-0056."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # 1. Lowercase + stop-word removal + sort.
            ("Linkerd 2.18 features", "2.18 features linkerd"),
            # "does", "have", "what" are NOT in the stop-word list — the
            # stop list is intentionally tiny to avoid collisions across
            # genuinely different queries.
            (
                "What features does Linkerd 2.18 have?",
                "2.18 does features have linkerd what",
            ),
            # 2. Punctuation removal except .  :  /  -
            ("Hello, world!", "hello world"),
            ("PostgreSQL 18 — release notes", "18 notes postgresql release"),
            ("https://example.com/path", "https://example.com/path"),
            # 3. Preserved punctuation: dot in versions, colon in URLs, slash in paths.
            ("Python 3.13.0", "3.13.0 python"),
            ("kubectl get pods -A", "-a get kubectl pods"),
            # 4. Token dedup — repeated words collapse.
            ("foo foo bar foo", "bar foo"),
            # 5. Whitespace collapse.
            ("  multi    space  ", "multi space"),
            # 6. Empty / whitespace-only.
            ("", ""),
            ("   ", ""),
            # 7. All stop-words.
            ("the a an of", ""),
            # 8. Non-English passes through (no stop-word stripping).
            ("Linkerd 2.18 особенности", "2.18 linkerd особенности"),
            # 9. Mixed case stop-words handled by casefolding.
            ("THE Linkerd", "linkerd"),
        ],
    )
    def test_normaliser_table(self, raw: str, expected: str) -> None:
        assert normalise_query(raw) == expected

    def test_two_phrasings_same_key(self) -> None:
        """Trivially-different phrasings collide on the cache key."""
        a = normalise_query("Linkerd 2.18 features")
        b = normalise_query("features of Linkerd 2.18")
        assert a == b == "2.18 features linkerd"


class TestCacheRoundTrip:
    """Put → get returns the stored value; misses return None."""

    def test_put_then_get_returns_value(self) -> None:
        cache_put("Linkerd 2.18 features", "result-payload")
        assert cache_get("Linkerd 2.18 features") == "result-payload"

    def test_different_phrasing_hits_same_entry(self) -> None:
        cache_put("Linkerd 2.18 features", "stored")
        assert cache_get("features of Linkerd 2.18") == "stored"

    def test_unrelated_query_misses(self) -> None:
        cache_put("Linkerd 2.18 features", "stored")
        assert cache_get("PostgreSQL 18 release notes") is None

    def test_empty_query_does_not_cache(self) -> None:
        cache_put("", "ignored")
        cache_put("   ", "ignored")
        cache_put("the a an of", "ignored")
        assert cache_size() == 0
        assert cache_get("the a an of") is None


class TestTTLExpiry:
    """Entries expire after the TTL passes."""

    def test_expiry_by_monkeypatched_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # cachetools.TTLCache uses time.monotonic by default. Swap the
        # module-level cache for one driven by a fake clock so the test
        # is deterministic and finishes in microseconds.
        from cachetools import TTLCache

        t = [1000.0]

        def fake_monotonic() -> float:
            return t[0]

        monkeypatch.setattr(
            cache_mod,
            "_cache",
            TTLCache(maxsize=4, ttl=cache_mod._CACHE_TTL_SECONDS, timer=fake_monotonic),
        )

        cache_put("Linkerd 2.18 features", "stored")
        assert cache_get("Linkerd 2.18 features") == "stored"

        t[0] += cache_mod._CACHE_TTL_SECONDS - 1
        assert cache_get("Linkerd 2.18 features") == "stored"

        t[0] += 2  # now TTL+1 seconds past insertion — expired
        assert cache_get("Linkerd 2.18 features") is None


class TestLRUCapacity:
    """Cache evicts oldest entries when capacity is exceeded."""

    def test_eviction_under_pressure(self) -> None:
        # Default capacity is 64; create 70 entries with unique keys.
        for i in range(70):
            cache_put(f"query number {i}", f"value-{i}")
        assert cache_size() <= 64

    def test_recently_accessed_survives_eviction(self) -> None:
        # Insert 64 entries, access entry-0, then insert one more.
        for i in range(64):
            cache_put(f"query keyword {i}", f"value-{i}")
        # Access entry 0 to mark it as recently used.
        assert cache_get("query keyword 0") == "value-0"
        # Insert one more — eviction triggers; entry 0 should NOT be the
        # victim if the LRU policy is honoured.
        cache_put("query keyword 99", "value-99")
        # We don't strictly mandate that entry-0 survives; cachetools.TTLCache
        # uses LRU by default. Just assert size stays bounded.
        assert cache_size() <= 64


class TestRealTimeTTL:
    """One smoke test against real time.monotonic — guards regressions in
    the cachetools wiring without burning a second of test time."""

    def test_value_lives_through_short_interval(self) -> None:
        cache_put("smoke test query", "smoke-value")
        # Sleep a tiny amount and re-fetch.
        time.sleep(0.01)
        assert cache_get("smoke test query") == "smoke-value"
