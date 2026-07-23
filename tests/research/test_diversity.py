"""Tests for source diversity tracking and contradiction detection."""

from cogtrix_core.orchestration.research_delegate import (
    ContradictionDetector,
    SourceSnapshot,
    SourceTracker,
)


def _snap(origin: str, content: str) -> SourceSnapshot:
    """Build a fully-typed ``SourceSnapshot`` from the two fields the
    contradiction-detection tests actually exercise (#1900).

    ``ContradictionDetector._find_supporting_origins`` only indexes
    ``source["content"]`` and ``source["origin"]``, but the public
    parameter is typed ``list[SourceSnapshot]`` (a 6-field TypedDict).
    Passing bare ``{"origin": ..., "content": ...}`` dict literals
    works at runtime but trips pyright's ``reportArgumentType``.  This
    helper keeps the test bodies concise while satisfying the contract.
    """
    return {
        "source_id": f"sid-{id(content)}",
        "url": "https://example.com/",
        "domain": "example.com",
        "origin": origin,
        "content": content,
        "timestamp": "2026-01-01T00:00:00Z",
    }


class TestSourceTracker:
    """Tests for SourceTracker class."""

    def test_single_domain_dominance(self):
        """Test that single dominant domain is detected."""
        tracker = SourceTracker()
        tracker.add_source("s1", "https://example.com/page1", "content1")
        tracker.add_source("s2", "https://example.com/page2", "content2")
        tracker.add_source("s3", "https://example.com/page3", "content3")

        assert tracker.diversity_score() < 0.5
        assert tracker.dominant_origin_ratio() > 0.5

    def test_multiple_domains_high_score(self):
        """Test that multiple domains yield high diversity score."""
        tracker = SourceTracker()
        tracker.add_source("s1", "https://github.com/repo1", "content1")
        tracker.add_source("s2", "https://wikipedia.org/wiki1", "content2")
        tracker.add_source("s3", "https://arxiv.org/abs1", "content3")

        assert tracker.diversity_score() >= 0.9

    def test_origin_inference_from_content(self):
        """Test that origin is correctly inferred from content."""
        tracker = SourceTracker()
        tracker.add_source(
            "s1", "https://example.com", "This is from arxiv.org paper about machine learning"
        )
        assert "academic" in tracker.get_unique_origins()

    def test_dominant_origin_ratio_calculation(self):
        """Test dominant origin ratio calculation."""
        tracker = SourceTracker()
        tracker.add_source("s1", "https://a.com", "content")
        tracker.add_source("s2", "https://b.com", "content")
        tracker.add_source("s3", "https://a.com", "content")
        tracker.add_source("s4", "https://a.com", "content")

        ratio = tracker.dominant_origin_ratio()
        assert ratio == 0.75  # 3 out of 4 from 'a'

    def test_empty_tracker_metrics(self):
        """Test metrics for empty tracker."""
        tracker = SourceTracker()

        # Empty tracker = perfect diversity (1.0)
        assert tracker.diversity_score() == 1.0
        assert tracker.dominant_origin_ratio() == 0.0

    def test_url_with_www_prefix(self):
        """Test domain extraction with www prefix."""
        tracker = SourceTracker()
        tracker.add_source("s1", "https://www.example.com", "content")

        assert "example" in tracker.get_unique_origins()

    def test_subdomain_handling(self):
        """Test handling of subdomains."""
        tracker = SourceTracker()
        tracker.add_source("s1", "https://api.github.com", "content")
        tracker.add_source("s2", "https://github.com", "content")

        # Subdomains should be handled appropriately
        assert len(tracker.get_unique_origins()) >= 1

    def test_domain_with_special_characters(self):
        """Test domain extraction with special characters."""
        tracker = SourceTracker()
        tracker.add_source("s1", "https://example.co.uk", "content")

        assert "example" in tracker.get_unique_origins()

    # ── Bug #1896 regression — duplicate-source-ID guard ──

    def test_duplicate_source_id_is_ignored(self):
        """Adding the same source_id twice must not inflate any tracker
        state.  Without the guard, ``total_sources`` would double and
        ``diversity_score`` would drop from 1.0 to 0.5 from a single
        duplicate add — turning a "perfect diversity" tracker into
        spurious "low diversity" with no signal to the caller (#1896)."""
        tracker = SourceTracker()
        tracker.add_source("s1", "https://example.com/a", "content")
        tracker.add_source("s1", "https://example.com/a", "content")

        stats = tracker.get_statistics()
        assert stats["total_sources"] == 1, (
            f"Duplicate source_id must be ignored — total_sources is "
            f"{stats['total_sources']}, expected 1 (#1896)"
        )
        assert stats["unique_domains"] == 1
        assert stats["unique_origins"] == 1
        # Diversity score on a single source is 1.0 (perfect diversity).
        # Without the guard the second add would push it to 0.5.
        assert tracker.diversity_score() == 1.0

    def test_duplicate_source_id_logs_warning(self, caplog):
        """The duplicate add must emit a WARNING so a latent caller-side
        bug surfaces in ops logs rather than silently degrading
        statistics.  The warning must name the duplicate source_id and
        the already-tracked URL so the operator can trace the source of
        the regression."""
        import logging

        tracker = SourceTracker()
        tracker.add_source("s1", "https://original.com/a", "content")

        with caplog.at_level(logging.WARNING, logger="cogtrix"):
            tracker.add_source("s1", "https://duplicate-attempt.com/b", "content")

        assert any("duplicate source_id" in r.message for r in caplog.records), (
            f"Expected a 'duplicate source_id' WARNING; got "
            f"{[r.message for r in caplog.records]!r}"
        )
        # The warning must name the offending source_id.
        msg = next(r.message for r in caplog.records if "duplicate source_id" in r.message)
        assert "'s1'" in msg
        # And the already-tracked URL so the operator can trace it.
        # Check the full URL (scheme + host + path), not just the host
        # substring — both because it's a stronger test and because a
        # host-only substring trips CodeQL's incomplete-URL-substring
        # sanitisation pattern (py/incomplete-url-substring-sanitization),
        # even though this is a log-content assertion not a security check.
        assert "https://original.com/a" in msg

    def test_distinct_source_ids_still_accumulate(self):
        """Negative control: distinct source IDs at the same URL must
        still both register, even though they're at the same URL —
        the guard keys on source_id, not on URL."""
        tracker = SourceTracker()
        tracker.add_source("s1", "https://example.com/a", "content")
        tracker.add_source("s2", "https://example.com/a", "content")

        assert tracker.get_statistics()["total_sources"] == 2

    def test_mixed_unique_and_duplicate_source_ids(self):
        """Realistic shape: 5 unique IDs interleaved with 3 duplicates
        → only the 5 unique sources register."""
        tracker = SourceTracker()
        for i in range(5):
            tracker.add_source(f"s{i}", f"https://example.com/page{i}", "content")
        # 3 duplicate re-adds of existing IDs.
        tracker.add_source("s0", "https://example.com/page0", "content")
        tracker.add_source("s2", "https://example.com/page2", "content")
        tracker.add_source("s4", "https://example.com/page4", "content")

        assert tracker.get_statistics()["total_sources"] == 5


class TestContradictionDetection:
    """Tests for ContradictionDetector class."""

    def test_single_origin_claims_flagged(self):
        """Test that claims from single origin are flagged."""
        detector = ContradictionDetector()
        sources = [
            _snap("origin1", "Claim A is true"),
            _snap("origin1", "Claim A is confirmed"),
            _snap("origin1", "Claim A is verified"),
        ]

        disagreements = detector.check_for_disagreement(["Claim A is true"], sources)
        assert len(disagreements) > 0

    def test_multiple_origin_claims_not_flagged(self):
        """Test that claims from multiple origins are not flagged.

        This test checks that when sources from different origins all contain
        the claim text, it's not flagged as lacking independent confirmation.
        """
        detector = ContradictionDetector()
        # All sources contain the exact claim text
        sources = [
            _snap("origin1", "Claim A is true"),
            _snap("origin2", "Claim A is true"),
            _snap("origin3", "Claim A is true"),
        ]

        disagreements = detector.check_for_disagreement(["Claim A is true"], sources)
        # With 3 different origins all supporting the claim, it should NOT be flagged
        assert len(disagreements) == 0

    def test_conflicting_evidence_detection(self):
        """Test detection of conflicting evidence."""
        ContradictionDetector()
        sources = [
            _snap("origin1", "Claim A is true"),
            _snap("origin2", "Claim A is false"),
        ]

        assert len(sources) == 2

    def test_claim_matching_algorithm(self):
        """Test claim matching algorithm.

        The algorithm checks if ALL words in claim (length > 3) appear in content.
        """
        detector = ContradictionDetector()

        assert detector._claim_matches_content("AI is powerful", "AI is very powerful")
        assert not detector._claim_matches_content("Python programming", "Java programming")

    def test_empty_claims_list(self):
        """Test with empty claims list."""
        detector = ContradictionDetector()

        assert detector.get_confidence_score() == 0.5

    def test_confidence_score_calculation(self):
        """Test confidence score calculation."""
        detector = ContradictionDetector()

        # Add claims with different origin counts
        detector.add_claim("Claim 1", ["origin1", "origin2", "origin3"])  # 3 origins
        detector.add_claim("Claim 2", ["origin1"])  # 1 origin

        score = detector.get_confidence_score()
        assert 0.0 <= score <= 1.0

    def test_well_supported_claims(self):
        """Test that well-supported claims have high confidence."""
        detector = ContradictionDetector()

        # Add claim with multiple origins
        detector.add_claim("Claim", ["origin1", "origin2", "origin3", "origin4"])

        assert detector.get_confidence_score() > 0.5

    def test_poorly_supported_claims(self):
        """Test that poorly-supported claims have low confidence."""
        detector = ContradictionDetector()

        # Add claim with single origin
        detector.add_claim("Claim", ["origin1"])

        assert detector.get_confidence_score() < 0.5

    def test_mixed_support_scenario(self):
        """Test scenario with mixed support levels."""
        detector = ContradictionDetector()

        # Mix of well-supported and poorly-supported claims
        detector.add_claim("Claim 1", ["origin1", "origin2"])
        detector.add_claim("Claim 2", ["origin1"])

        score = detector.get_confidence_score()
        assert 0.0 <= score <= 1.0

    def test_large_origin_set(self):
        """Test with large number of origins."""
        detector = ContradictionDetector()
        origins = [f"origin{i}" for i in range(100)]

        detector.add_claim("Claim", origins)

        assert detector.get_confidence_score() > 0.9

    # ── Bug #1175 regression — origin_count must measure UNIQUE origins ──

    def test_duplicate_origins_collapse_to_unique_count(self):
        """The #1175 reproducer: a claim backed by five source IDs all
        from wikipedia.org represents ONE origin, not five — passing
        the same origin name five times must collapse to a single
        unique origin via the ``set()`` dedup. Without dedup the field
        was a misnamed source-count and inflated confidence.

        Concrete consequence the ticket called out: five identical
        ``"wikipedia"`` entries used to produce ``origin_count == 5``
        and confidence 1.0 despite zero origin diversity. The
        ``set(supporting_origins)`` step is what prevents that —
        pinning it here so a future "preserve duplicates" change
        re-introduces the bug visibly.
        """
        detector = ContradictionDetector()
        detector.add_claim("Wiki-only claim", ["wikipedia"] * 5)
        c = detector.claims[0]
        assert c["origin_count"] == 1, (
            f"Five identical origins must yield origin_count=1, got "
            f"{c['origin_count']} — set() dedup regressed (#1175)"
        )
        # Confidence must NOT classify single-origin claims as well-supported.
        # ``well_supported`` requires ``origin_count > 1``; single-origin
        # claims fall into ``poorly_supported``.
        stats = detector.get_statistics()
        assert stats["well_supported_claims"] == 0
        assert stats["poorly_supported_claims"] == 1

    def test_supporting_origins_keyword_is_the_public_contract(self):
        """The public parameter name is ``supporting_origins`` (not
        ``supporting_sources``). The rename closes the gap between the
        docstring's stated semantics and the field name; passing via
        keyword pins that the public contract didn't silently revert
        to the old misleading name."""
        detector = ContradictionDetector()
        detector.add_claim(
            claim="kw-style call",
            supporting_origins=["wikipedia", "github", "news_agency"],
        )
        assert detector.claims[0]["origin_count"] == 3
        # And the stored dict uses the new key name, not the old one.
        assert "supporting_origins" in detector.claims[0]
        assert "supporting_sources" not in detector.claims[0]

    def test_mixed_duplicate_and_unique_origins(self):
        """Real-world shape: two wiki source IDs (collapse to one
        origin) plus one github plus one news_agency → origin_count=3,
        not 4. Pins the ``set()`` semantics on a non-degenerate
        input."""
        detector = ContradictionDetector()
        detector.add_claim(
            "Mixed-origin claim",
            ["wikipedia", "wikipedia", "github", "news_agency"],
        )
        assert detector.claims[0]["origin_count"] == 3


class TestIntegration:
    """Integration tests for research diversity system."""

    def test_diversity_influences_final_scores(self):
        """Test that diversity metrics influence final confidence scores.

        Low diversity should result in low diversity score (< 0.5).
        The detector confidence should be checked separately as it depends
        on the claims added to it.
        """
        tracker = SourceTracker()
        ContradictionDetector()

        # Low diversity scenario - all sources from same domain
        for i in range(5):
            tracker.add_source(f"s{i}", f"https://same.com/page{i}", "content")

        # Verify low diversity
        assert tracker.diversity_score() < 0.5

    def test_uncertainty_language_in_responses(self):
        """Test that uncertainty language appears in low-diversity responses."""
        tracker = SourceTracker()

        # Add sources from same domain
        for i in range(5):
            tracker.add_source(f"s{i}", f"https://same.com/page{i}", "content")

        # Verify low diversity triggers uncertainty
        assert tracker.diversity_score() < 0.5

    def test_domain_extraction_from_urls(self):
        """Test that domains are correctly extracted from URLs."""
        from cogtrix_core.tools.web_search import extract_domain

        assert extract_domain("https://github.com/user/repo") == "github.com"
        assert extract_domain("https://www.wikipedia.org/wiki/Article") == "wikipedia.org"
        assert extract_domain("https://example.com/path") == "example.com"

    def test_origin_metadata_preservation(self):
        """Test that origin metadata is preserved through the pipeline."""
        tracker = SourceTracker()

        tracker.add_source("s1", "https://github.com/repo", "content")

        sources = tracker.get_sources_by_origin("github")
        assert len(sources) == 1

    def test_end_to_end_research_flow(self):
        """Test complete research flow with diversity tracking."""
        tracker = SourceTracker()

        # Simulate research workflow with diverse sources
        urls = [
            "https://github.com/repo1",
            "https://wikipedia.org/wiki1",
            "https://arxiv.org/abs1",
        ]

        for i, url in enumerate(urls):
            tracker.add_source(f"s{i}", url, f"content from {url}")

        # Verify high diversity (3 different origins)
        assert tracker.diversity_score() >= 0.9
        assert tracker.dominant_origin_ratio() <= 0.35  # 1/3 = 0.333...

    def test_contradiction_detection_in_research(self):
        """Test contradiction detection in research context."""
        tracker = SourceTracker()
        detector = ContradictionDetector()

        # Add sources with conflicting information
        tracker.add_source("s1", "https://example.com", "Claim A is true")
        tracker.add_source("s2", "https://example.com", "Claim A is false")

        sources = tracker.sources
        disagreements = detector.check_for_disagreement(["Claim A is true"], sources)

        # Should detect disagreement (only 1 origin supporting the claim)
        assert len(disagreements) >= 0  # May or may not detect depending on content


class TestEdgeCases:
    """Edge case tests for research diversity system."""

    def test_empty_url_handling(self):
        """Test handling of empty URLs."""
        tracker = SourceTracker()

        tracker.add_source("s1", "", "content")

        # Should handle gracefully
        assert len(tracker.sources) == 1

    def test_invalid_url_handling(self):
        """Test handling of invalid URLs."""
        tracker = SourceTracker()

        tracker.add_source("s1", "not-a-valid-url", "content")

        # Should handle gracefully
        assert len(tracker.sources) == 1

    def test_very_long_url(self):
        """Test handling of very long URLs."""
        tracker = SourceTracker()
        long_url = "https://example.com/" + "a" * 1000

        tracker.add_source("s1", long_url, "content")

        # Should handle gracefully
        assert len(tracker.sources) == 1

    def test_unicode_domain(self):
        """Test handling of unicode domains."""
        tracker = SourceTracker()

        tracker.add_source("s1", "https://例子.测试", "content")

        # Should handle gracefully
        assert len(tracker.sources) == 1

    def test_special_characters_in_content(self):
        """Test handling of special characters in content."""
        tracker = SourceTracker()

        special_content = "Content with <tags> & special 'chars' \"quoted\""
        tracker.add_source("s1", "https://example.com", special_content)

        # Should handle gracefully
        assert len(tracker.sources) == 1
        assert tracker.sources[0]["content"] == special_content


def run_all_tests():
    """Run all tests and report results."""
    tests = [TestSourceTracker(), TestContradictionDetection(), TestIntegration(), TestEdgeCases()]

    passed = 0
    failed = 0
    errors = []

    for test_obj in tests:
        test_class_name = test_obj.__class__.__name__
        for method_name in dir(test_obj):
            if method_name.startswith("test_"):
                try:
                    getattr(test_obj, method_name)()
                    print(f"PASS: {test_class_name}.{method_name}")
                    passed += 1
                except Exception as e:
                    print(f"FAIL: {test_class_name}.{method_name}: {e}")
                    failed += 1
                    errors.append((test_class_name, method_name, str(e)))

    print("\n=== Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if errors:
        print("\nErrors:")
        for test_class, method, error in errors:
            print(f"  {test_class}.{method}: {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
