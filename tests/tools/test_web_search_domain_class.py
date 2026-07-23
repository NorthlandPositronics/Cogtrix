"""Tests for src/tools/_web_search_domain_class.py (ADR-0056)."""

from __future__ import annotations

import pytest

from src.tools._web_search_domain_class import (
    DomainClass,
    authority_bonus,
    classify_domain,
)


class TestWikiEncyclopedia:
    @pytest.mark.parametrize(
        "url",
        [
            "https://en.wikipedia.org/wiki/PostgreSQL",
            "https://wikipedia.org/wiki/Main_Page",
            "https://ru.wikipedia.org/wiki/Россия",
            "https://wikidata.org/wiki/Q42",
            "https://en.wiktionary.org/wiki/synthesis",
        ],
    )
    def test_wikipedia_family(self, url: str) -> None:
        assert classify_domain(url) == DomainClass.WIKI_ENCYCLOPEDIA


class TestWikiCommunity:
    @pytest.mark.parametrize(
        "url",
        [
            "https://harrypotter.fandom.com/wiki/Hogwarts",
            "https://starwars.wikia.com/wiki/Yoda",
            "https://wiki.archlinux.org/title/Installation",
        ],
    )
    def test_community_wikis(self, url: str) -> None:
        assert classify_domain(url) == DomainClass.WIKI_COMMUNITY

    def test_wikipedia_does_not_fall_through_to_community(self) -> None:
        """wikipedia.org would also match the subdomain_prefixes={wiki} rule
        of wiki-community, but encyclopedia rule is checked first."""
        assert classify_domain("https://wikipedia.org/") == DomainClass.WIKI_ENCYCLOPEDIA


class TestOfficialDocs:
    @pytest.mark.parametrize(
        "url",
        [
            "https://docs.python.org/3/library/asyncio.html",
            "https://developer.mozilla.org/en-US/docs/Web/HTTP",
            "https://learn.microsoft.com/en-us/dotnet/",
            "https://kubernetes.io/docs/concepts/",
            "https://docs.rs/tokio/latest/tokio/",
            "https://cogtrix.readthedocs.io/en/latest/",
            "https://developers.google.com/identity",
        ],
    )
    def test_official_docs(self, url: str) -> None:
        assert classify_domain(url) == DomainClass.OFFICIAL_DOCS


class TestNews:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.nytimes.com/2026/04/15/world/article.html",
            "https://www.bbc.com/news/world-europe-12345678",
            "https://www.reuters.com/business/",
            "https://arstechnica.com/gadgets/2026/05/",
        ],
    )
    def test_news_outlets(self, url: str) -> None:
        assert classify_domain(url) == DomainClass.NEWS


class TestForum:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.reddit.com/r/programming/",
            "https://stackoverflow.com/questions/12345",
            "https://news.ycombinator.com/item?id=12345",
            "https://serverfault.stackexchange.com/questions/12345",
            "https://discuss.python.org/t/topic-name/12345",
            "https://forum.djangoproject.com/t/something/123",
        ],
    )
    def test_forums(self, url: str) -> None:
        assert classify_domain(url) == DomainClass.FORUM


class TestBlog:
    @pytest.mark.parametrize(
        "url",
        [
            "https://medium.com/@author/post-title-abcdef",
            "https://substack.com/feed",
            "https://dev.to/author/post-title",
            "https://author.hashnode.dev/post",
            "https://author.blogspot.com/2026/05/post.html",
        ],
    )
    def test_blogs(self, url: str) -> None:
        assert classify_domain(url) == DomainClass.BLOG


class TestAcademic:
    @pytest.mark.parametrize(
        "url",
        [
            "https://arxiv.org/abs/2503.12345",
            "https://www.nature.com/articles/abc123",
            "https://dl.acm.org/doi/10.1145/123456",
            "https://stanford.edu/~prof/paper.pdf",
            "https://www.cam.ac.uk/research/news/something",
        ],
    )
    def test_academic(self, url: str) -> None:
        assert classify_domain(url) == DomainClass.ACADEMIC


class TestSocialMedia:
    @pytest.mark.parametrize(
        "url",
        [
            "https://twitter.com/user/status/12345",
            "https://x.com/user/status/12345",
            "https://www.linkedin.com/in/someone/",
            "https://www.youtube.com/watch?v=abc123",
            "https://bsky.app/profile/handle.bsky.social",
        ],
    )
    def test_social_media(self, url: str) -> None:
        assert classify_domain(url) == DomainClass.SOCIAL_MEDIA


class TestECommerce:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.amazon.com/dp/B0123456",
            "https://www.amazon.co.uk/dp/B0123456",
            "https://www.ebay.com/itm/123456",
            "https://www.etsy.com/listing/12345",
            "https://myshop.shopify.com/products/foo",
        ],
    )
    def test_e_commerce(self, url: str) -> None:
        assert classify_domain(url) == DomainClass.E_COMMERCE


class TestUnknown:
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/article",
            "https://random-startup-456.dev/about",
            "",  # malformed
            "not a url",
            "https://192.168.1.1/admin",  # IP literal
            "https://localhost/foo",
        ],
    )
    def test_unknown_falls_through(self, url: str) -> None:
        # IP literals, bare hosts, and unrecognised curated domains.
        assert classify_domain(url) == DomainClass.UNKNOWN

    def test_blog_subdomain_classifies_as_blog(self) -> None:
        """``blog.<anything>`` is a strong enough signal to win the
        subdomain pass even when the registered domain is uncurated."""
        assert classify_domain("https://blog.example.io/post") == DomainClass.BLOG

    def test_docs_subdomain_classifies_as_official_docs(self) -> None:
        """Same principle for ``docs.<anything>``."""
        assert classify_domain("https://docs.example.io/v2/api") == DomainClass.OFFICIAL_DOCS

    def test_forum_subdomain_wins_over_registered_domain(self) -> None:
        """``discuss.python.org`` should be FORUM even though python.org is
        registered as official-docs — the subdomain signal is more specific."""
        assert classify_domain("https://discuss.python.org/t/topic/123") == DomainClass.FORUM


class TestEdgeCases:
    def test_idn_domain(self) -> None:
        """Internationalised domain names are accepted (passed through to tldextract)."""
        # Just assert it doesn't crash; the specific class isn't important here.
        result = classify_domain("https://例え.jp/path")
        assert isinstance(result, DomainClass)

    def test_malformed_url_returns_unknown(self) -> None:
        assert classify_domain("not://valid") == DomainClass.UNKNOWN

    def test_mailto_returns_unknown(self) -> None:
        assert classify_domain("mailto:user@example.com") == DomainClass.UNKNOWN

    def test_javascript_returns_unknown(self) -> None:
        assert classify_domain("javascript:alert(1)") == DomainClass.UNKNOWN

    def test_uppercase_host_normalises(self) -> None:
        """Host comparison is case-insensitive."""
        assert classify_domain("https://EN.WIKIPEDIA.ORG/wiki/X") == DomainClass.WIKI_ENCYCLOPEDIA

    def test_no_scheme_returns_unknown(self) -> None:
        """urlparse without scheme leaves netloc empty; we return unknown."""
        result = classify_domain("wikipedia.org/wiki/X")
        assert result == DomainClass.UNKNOWN


class TestAuthorityBonus:
    def test_authority_table_covers_all_classes(self) -> None:
        """Every DomainClass has a bonus value (no implicit KeyError)."""
        for cls in DomainClass:
            bonus = authority_bonus(cls)
            assert isinstance(bonus, float)

    def test_wiki_encyclopedia_positive(self) -> None:
        assert authority_bonus(DomainClass.WIKI_ENCYCLOPEDIA) > 0

    def test_wiki_community_neutral(self) -> None:
        assert authority_bonus(DomainClass.WIKI_COMMUNITY) == 0.0

    def test_official_docs_positive(self) -> None:
        assert authority_bonus(DomainClass.OFFICIAL_DOCS) > 0

    def test_social_media_negative(self) -> None:
        assert authority_bonus(DomainClass.SOCIAL_MEDIA) < 0

    def test_forum_negative(self) -> None:
        assert authority_bonus(DomainClass.FORUM) < 0

    def test_unknown_neutral(self) -> None:
        assert authority_bonus(DomainClass.UNKNOWN) == 0.0
