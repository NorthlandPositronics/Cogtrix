"""Per-process query cache for the web_search tool (ADR-0056 stage 0).

Cache entries are keyed by a *normalised* form of the user query so that
trivially-different but semantically-identical queries hit the same entry
("What features does Linkerd 2.18 have?" and "Linkerd 2.18 features" both
key to "2.18 features linkerd").

The normaliser is intentionally minimal — false-positive hits across
genuinely different queries are worse than false-negative misses.
Semantic equivalence (synonyms, embeddings) is deferred per ADR-0056
follow-up #3.

Scope is per-process. Cogtrix is single-tenant by deployment assumption;
this cache is not safe for hypothetical multi-tenant hosting. ADR-0056
"Cache poisoning" risk row covers this.
"""

from __future__ import annotations

import re
import threading
import unicodedata

from cachetools import TTLCache

_CACHE_MAXSIZE = 64
_CACHE_TTL_SECONDS = 300  # 5 minutes; balances freshness vs. follow-up hit rate

_KEEP_PUNCTUATION = frozenset({".", ":", "/", "-"})

# Tiny stop-word list. Bigger lists collide more queries (false-positive
# hits); keep small. English only — non-English queries pass through
# unchanged because removing English stops from them is a no-op.
_STOPWORDS = frozenset({"the", "a", "an", "of", "for", "to", "in", "on", "with"})

_cache: TTLCache = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL_SECONDS)
_cache_lock = threading.Lock()


def normalise_query(query: str) -> str:
    """Return the cache-key form of *query*.

    Steps (per ADR-0056 "Cache key" section):
      1. Lowercase.
      2. Strip Unicode punctuation except .  :  /  -  (kept because
         version numbers and URL fragments in queries carry meaning).
      3. Tokenise on whitespace.
      4. Drop English stop-words.
      5. Sort + dedup the token set.
      6. Join with single space.

    Empty result (e.g. query was only stop-words / punctuation) is
    returned as-is — the caller should treat it as "do not cache."
    """
    s = query.casefold()
    # Replace Unicode punctuation we don't keep with a space, so the
    # surrounding tokens split cleanly.
    out_chars: list[str] = []
    for ch in s:
        if ch.isspace():
            out_chars.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") and ch not in _KEEP_PUNCTUATION:
            out_chars.append(" ")
        else:
            out_chars.append(ch)
    stripped = "".join(out_chars)
    tokens = [t for t in re.split(r"\s+", stripped) if t]
    tokens = [t for t in tokens if t not in _STOPWORDS]
    deduped = sorted(set(tokens))
    return " ".join(deduped)


def cache_get(query: str) -> str | None:
    """Return cached result for *query* or None on miss / unkeyable input."""
    key = normalise_query(query)
    if not key:
        return None
    with _cache_lock:
        return _cache.get(key)


def cache_put(query: str, value: str) -> None:
    """Store *value* under the normalised key for *query*.

    Silently no-ops if the query normalises to empty — there is no useful
    key for "the   ?  !".
    """
    key = normalise_query(query)
    if not key:
        return
    with _cache_lock:
        _cache[key] = value


def cache_clear() -> None:
    """Drop every entry. Test helper; production code does not call this."""
    with _cache_lock:
        _cache.clear()


def cache_size() -> int:
    """Number of live entries. Test helper."""
    with _cache_lock:
        return len(_cache)


__all__ = [
    "cache_clear",
    "cache_get",
    "cache_put",
    "cache_size",
    "normalise_query",
]
