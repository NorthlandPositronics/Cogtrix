"""Domain-class classifier for the web_search tool (ADR-0056).

Maps a URL's registered domain to one of ten classes used by:
  - Stage 2 ranking (authority bonus / de-prioritisation per class).
  - The Sources block in the output schema ("[wiki-encyclopedia · 2026-03]").

The class taxonomy is intentionally biased toward classes a human
researcher would weight differently:
  * Encyclopedic wikis (wikipedia.org) → authority bonus.
  * Community wikis (fandom.com) → labelled but no bonus (uneven moderation).
  * Official-vendor docs, academic → authority bonus.
  * Social media, forum → modest de-prioritisation.

Rule sets live as plain data (mappings + suffix lists) so adding a new
vendor docs domain or news outlet is a one-line edit.

The keyword sets here are intentionally curated rather than crawled —
ADR-0056 explicitly chose this over LLM-based classification for v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlparse

import tldextract

# Pin to the bundled public-suffix list so we don't make a network call
# on first use (ADR-0056 dep note). suffix_list_urls=() disables network
# fetch; cache_dir=None disables on-disk cache; include_psl_private_domains
# kept False to match registered-domain semantics.
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=False,
    cache_dir=None,
)


class DomainClass(StrEnum):
    """The ten domain classes per ADR-0056."""

    WIKI_ENCYCLOPEDIA = "wiki-encyclopedia"
    WIKI_COMMUNITY = "wiki-community"
    OFFICIAL_DOCS = "official-docs"
    NEWS = "news"
    FORUM = "forum"
    BLOG = "blog"
    ACADEMIC = "academic"
    SOCIAL_MEDIA = "social-media"
    E_COMMERCE = "e-commerce"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class _Rule:
    """Match rule for a single class.

    A rule matches if any of:
      * the *full host* (e.g. ``learn.microsoft.com``) is in ``exact_domains``;
      * the *registered domain* (e.g. ``wikipedia.org``) is in ``exact_domains``;
      * the registered domain ends with an entry in ``suffix_domains`` (e.g.
        ``.edu`` matches ``stanford.edu``);
      * the *subdomain* portion equals any entry in ``subdomain_prefixes``
        (e.g., "docs" matches ``docs.acme.com`` regardless of TLD).

    ``exact_domains`` mixes full-host entries (``news.ycombinator.com``) with
    registered-domain entries (``wikipedia.org``) deliberately — the
    classifier checks both forms so rule authors don't have to pre-canonicalise.
    """

    exact_domains: frozenset[str] = frozenset()
    suffix_domains: frozenset[str] = frozenset()
    subdomain_prefixes: frozenset[str] = frozenset()


# Order matters only for documentation. The classifier checks every rule
# and returns the first class that matches; classes earlier in the list
# below win ties. Wikipedia-encyclopedia first so it doesn't fall through
# to the broader community-wiki rule.
_RULES: list[tuple[DomainClass, _Rule]] = [
    (
        DomainClass.WIKI_ENCYCLOPEDIA,
        _Rule(
            exact_domains=frozenset(
                {
                    "wikipedia.org",
                    "mediawiki.org",
                    "wikidata.org",
                    "wiktionary.org",
                    "wikibooks.org",
                    "wikiquote.org",
                    "wikisource.org",
                    "wikinews.org",
                    "wikiversity.org",
                    "wikivoyage.org",
                }
            ),
        ),
    ),
    (
        DomainClass.WIKI_COMMUNITY,
        _Rule(
            exact_domains=frozenset({"fandom.com", "wikia.com"}),
            subdomain_prefixes=frozenset({"wiki"}),
        ),
    ),
    (
        DomainClass.OFFICIAL_DOCS,
        _Rule(
            exact_domains=frozenset(
                {
                    "readthedocs.io",
                    "readthedocs.org",
                    "developer.mozilla.org",
                    "learn.microsoft.com",
                    "docs.microsoft.com",
                    "kubernetes.io",
                    "rust-lang.org",
                    "golang.org",
                    "python.org",
                    "docs.rs",
                    "nodejs.org",
                    "postgresql.org",
                    "redis.io",
                    "nginx.org",
                    "apache.org",
                    "gnu.org",
                }
            ),
            subdomain_prefixes=frozenset({"docs", "developer", "developers", "dev"}),
        ),
    ),
    (
        DomainClass.NEWS,
        _Rule(
            exact_domains=frozenset(
                {
                    "nytimes.com",
                    "bbc.com",
                    "bbc.co.uk",
                    "reuters.com",
                    "theguardian.com",
                    "ft.com",
                    "bloomberg.com",
                    "cnbc.com",
                    "axios.com",
                    "apnews.com",
                    "economist.com",
                    "wsj.com",
                    "washingtonpost.com",
                    "npr.org",
                    "aljazeera.com",
                    "politico.com",
                    "theverge.com",
                    "arstechnica.com",
                    "wired.com",
                    "techcrunch.com",
                }
            ),
        ),
    ),
    (
        DomainClass.FORUM,
        _Rule(
            exact_domains=frozenset(
                {
                    "reddit.com",
                    "stackoverflow.com",
                    "news.ycombinator.com",
                    "lobste.rs",
                    "discourse.org",
                    "discord.com",
                }
            ),
            suffix_domains=frozenset({"stackexchange.com"}),
            subdomain_prefixes=frozenset({"forum", "forums", "discourse", "discuss", "community"}),
        ),
    ),
    (
        DomainClass.BLOG,
        _Rule(
            exact_domains=frozenset(
                {
                    "medium.com",
                    "substack.com",
                    "dev.to",
                    "hashnode.dev",
                    "blogspot.com",
                    "wordpress.com",
                    "ghost.io",
                    "bearblog.dev",
                    "tumblr.com",
                }
            ),
            subdomain_prefixes=frozenset({"blog"}),
        ),
    ),
    (
        DomainClass.ACADEMIC,
        _Rule(
            exact_domains=frozenset(
                {
                    "arxiv.org",
                    "scholar.google.com",
                    "acm.org",
                    "ieee.org",
                    "springer.com",
                    "sciencedirect.com",
                    "jstor.org",
                    "mendeley.com",
                    "pubmed.ncbi.nlm.nih.gov",
                    "nature.com",
                    "science.org",
                    "researchgate.net",
                    "semanticscholar.org",
                }
            ),
            suffix_domains=frozenset({".edu", ".ac.uk", ".ac.jp"}),
        ),
    ),
    (
        DomainClass.SOCIAL_MEDIA,
        _Rule(
            exact_domains=frozenset(
                {
                    "twitter.com",
                    "x.com",
                    "linkedin.com",
                    "facebook.com",
                    "instagram.com",
                    "tiktok.com",
                    "youtube.com",
                    "mastodon.social",
                    "bsky.app",
                    "threads.net",
                    "pinterest.com",
                }
            ),
        ),
    ),
    (
        DomainClass.E_COMMERCE,
        _Rule(
            exact_domains=frozenset(
                {
                    "amazon.com",
                    "amazon.co.uk",
                    "amazon.de",
                    "amazon.fr",
                    "amazon.co.jp",
                    "ebay.com",
                    "alibaba.com",
                    "aliexpress.com",
                    "etsy.com",
                    "walmart.com",
                    "target.com",
                    "bestbuy.com",
                    "costco.com",
                    "ikea.com",
                    "homedepot.com",
                    "lowes.com",
                }
            ),
            suffix_domains=frozenset({"shopify.com"}),
        ),
    ),
]


def _host_parts(url: str) -> tuple[str, str, str] | None:
    """Return (full_host, registered_domain, subdomain) for *url*.

    Returns ``None`` for malformed URLs, IP literals, or hosts with no
    registered domain (bare hostnames). Hostnames are casefolded.
    """
    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return None
    host = (parsed.hostname or "").strip().casefold()
    if not host:
        return None
    extracted = _EXTRACT(host)
    if not extracted.domain or not extracted.suffix:
        return None
    registered = f"{extracted.domain}.{extracted.suffix}"
    return host, registered, extracted.subdomain or ""


def _subdomain_first_token(subdomain: str) -> str:
    """Return the leftmost dotted segment of *subdomain* (e.g. "docs" from "docs.api")."""
    return subdomain.split(".", 1)[0] if subdomain else ""


def _matches_subdomain_prefix(subdomain: str, rule: _Rule) -> bool:
    if not rule.subdomain_prefixes or not subdomain:
        return False
    return _subdomain_first_token(subdomain) in rule.subdomain_prefixes


def _matches_exact_or_suffix(host: str, registered: str, rule: _Rule) -> bool:
    if host in rule.exact_domains:
        return True
    if registered in rule.exact_domains:
        return True
    for suffix in rule.suffix_domains:
        if registered == suffix.lstrip("."):
            return True
        if registered.endswith(suffix):
            return True
    return False


def classify_domain(url: str) -> DomainClass:
    """Return the DomainClass for *url*.

    Classification is two-pass:
      1. Subdomain-prefix matches win first. The subdomain is the most
         specific signal — "discuss.python.org" is a forum even though
         "python.org" is in the official-docs list.
      2. If no subdomain match, fall back to exact-host / registered-
         domain / suffix-domain matches.

    Returns ``DomainClass.UNKNOWN`` for malformed URLs, IP literals, or
    any domain not matching a curated rule.
    """
    parts = _host_parts(url)
    if parts is None:
        return DomainClass.UNKNOWN
    host, registered, subdomain = parts
    for cls, rule in _RULES:
        if _matches_subdomain_prefix(subdomain, rule):
            return cls
    for cls, rule in _RULES:
        if _matches_exact_or_suffix(host, registered, rule):
            return cls
    return DomainClass.UNKNOWN


# Ranking bonus per class. Positive = boost in stage 2 ranking; negative
# = de-prioritisation; 0 = neutral. ADR-0056 calibrates these later in
# tuning; v1 ships with mild values so an unmatched URL is never
# completely de-ranked relative to a matched one.
_AUTHORITY_BONUS: dict[DomainClass, float] = {
    DomainClass.WIKI_ENCYCLOPEDIA: 0.4,
    DomainClass.WIKI_COMMUNITY: 0.0,
    DomainClass.OFFICIAL_DOCS: 0.5,
    DomainClass.NEWS: 0.2,
    DomainClass.ACADEMIC: 0.4,
    DomainClass.BLOG: 0.0,
    DomainClass.FORUM: -0.1,
    DomainClass.SOCIAL_MEDIA: -0.2,
    DomainClass.E_COMMERCE: 0.0,
    DomainClass.UNKNOWN: 0.0,
}


def authority_bonus(cls: DomainClass) -> float:
    """Return the ranking adjustment for *cls* (see ADR-0056 table)."""
    return _AUTHORITY_BONUS.get(cls, 0.0)


__all__ = [
    "DomainClass",
    "authority_bonus",
    "classify_domain",
]
