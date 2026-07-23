"""Source diversity tracking and contradiction detection for research workflows.

This module implements source diversity tracking and contradiction detection
for research workflows. It helps identify when multiple sources trace back
to the same origin, preventing the "everything agrees, so it must be true"
trap.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any, TypedDict

from cogtrix_core.tools.web_search import extract_domain

log = logging.getLogger("cogtrix")


class SourceSnapshot(TypedDict):
    """Snapshot of a source with domain and origin information."""

    source_id: str
    url: str
    domain: str
    origin: str
    content: str
    timestamp: str


class SourceTracker:
    """Track source diversity and origin relationships.

    This class helps identify when multiple sources trace back to the same
    origin, preventing overconfidence in claims supported only by sources
    from a single origin.
    """

    def __init__(self) -> None:
        """Initialize the source tracker."""
        self.sources: list[SourceSnapshot] = []
        self.domains: dict[str, list[str]] = {}  # domain -> [source_ids]
        self.origins: dict[str, list[str]] = {}  # origin -> [source_ids]

    def add_source(self, source_id: str, url: str, content: str) -> None:
        """Add a source and extract domain/origin information.

        Duplicate ``source_id`` calls are ignored — the second call
        logs a warning and returns without mutating any tracker state
        (#1896).  Without this guard a caller that re-ingested the
        same source ID would silently inflate ``total_sources`` and
        the diversity / dominant-origin denominators, turning a
        "perfect diversity" tracker into something noisier with no
        signal to the caller.

        Args:
            source_id: Unique identifier for this source
            url: The URL of the source
            content: The content extracted from the source
        """
        for existing in self.sources:
            if existing["source_id"] == source_id:
                log.warning(
                    "SourceTracker.add_source: duplicate source_id %r ignored "
                    "(already tracked from %r); statistics unchanged",
                    source_id,
                    existing["url"],
                )
                return

        domain = extract_domain(url)
        origin = self._infer_origin(content, url)

        self.sources.append(
            {
                "source_id": source_id,
                "url": url,
                "domain": domain,
                "origin": origin,
                "content": content,
                "timestamp": self._current_timestamp(),
            }
        )

        if domain not in self.domains:
            self.domains[domain] = []
        self.domains[domain].append(source_id)

        if origin not in self.origins:
            self.origins[origin] = []
        self.origins[origin].append(source_id)

    def diversity_score(self) -> float:
        """Calculate source diversity score (0.0 to 1.0).

        Higher score = more diverse sources (better).
        A score of 1.0 means all sources come from different domains.
        A score of 0.0 means all sources come from the same domain.

        For empty tracker (no sources), returns 1.0 (perfect diversity).

        Returns:
            Diversity score between 0.0 and 1.0
        """
        total_sources = len(self.sources)

        # Empty tracker = perfect diversity (1.0)
        if total_sources == 0:
            return 1.0

        return len(self.domains) / total_sources

    def dominant_origin_ratio(self) -> float:
        """Calculate ratio of dominant origin to total sources.

        Lower ratio = more diverse (better).
        A ratio of 1.0 means all sources come from the same origin.
        A ratio of 0.0 means all sources come from different origins.

        For empty tracker (no sources), returns 0.0.

        Returns:
            Dominant origin ratio between 0.0 and 1.0
        """
        if not self.origins:
            return 0.0
        dominant_count = max(len(s) for s in self.origins.values())
        total = sum(len(s) for s in self.origins.values())
        return dominant_count / total if total > 0 else 0.0

    def get_unique_origins(self) -> list[str]:
        """Get list of unique origins.

        Returns:
            List of unique origin names
        """
        return list(self.origins.keys())

    def get_sources_by_origin(self, origin: str) -> list[SourceSnapshot]:
        """Get all sources from a specific origin.

        Args:
            origin: The origin name to filter by

        Returns:
            List of sources from the specified origin
        """
        source_ids = self.origins.get(origin, [])
        return [s for s in self.sources if s["source_id"] in source_ids]

    def get_sources_by_domain(self, domain: str) -> list[SourceSnapshot]:
        """Get all sources from a specific domain.

        Args:
            domain: The domain name to filter by

        Returns:
            List of sources from the specified domain
        """
        source_ids = self.domains.get(domain, [])
        return [s for s in self.sources if s["source_id"] in source_ids]

    def _infer_origin(self, content: str, url: str) -> str:
        """Infer the origin/publication type from content and URL.

        Args:
            content: The content to analyze
            url: The URL for additional context

        Returns:
            Inferred origin (e.g., "github", "wikipedia", "academic", "news_agency")
        """
        # Try to extract publication name from URL
        domain = extract_domain(url)

        # Common patterns for origin identification
        if "wikipedia.org" in domain:
            return "wikipedia"
        elif "github.com" in domain:
            return "github"
        elif "stackoverflow.com" in domain:
            return "stackoverflow"

        # Check content for publication markers
        if re.search(r"(?i)(cnn|fox|bbc|reuters|ap news)", content):
            return "news_agency"
        if re.search(r"(?i)(arxiv|scholar\.google)", content):
            return "academic"

        # Default to domain-based origin
        return domain.split(".")[0] if domain else "unknown"

    def _current_timestamp(self) -> str:
        """Get current timestamp in ISO format.

        Returns:
            ISO formatted timestamp string
        """
        return datetime.now(UTC).isoformat()

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about tracked sources.

        Returns:
            Dictionary with diversity metrics and statistics
        """
        return {
            "total_sources": len(self.sources),
            "unique_domains": len(self.domains),
            "unique_origins": len(self.origins),
            "diversity_score": self.diversity_score(),
            "dominant_origin_ratio": self.dominant_origin_ratio(),
            "domains": list(self.domains.keys()),
            "origins": list(self.origins.keys()),
        }


class ContradictionDetector:
    """Detect contradictions and lack of independent confirmation.

    This class helps identify claims that lack sufficient independent
    confirmation by analysing the diversity of **origins** backing each
    claim.

    ``add_claim`` takes ``supporting_origins`` — origin names like
    ``"wikipedia"`` / ``"github"`` / ``"news_agency"`` — NOT raw source
    IDs. Confidence is then a function of unique-origin count: five
    source IDs all pointing at wikipedia.org should yield ``origin_count
    == 1``, not 5, otherwise a single-origin claim looks "well supported"
    while in fact having zero origin diversity (#1175).

    If a caller has source IDs and needs to derive origins, the sibling
    :class:`SourceTracker` exposes that mapping via ``_infer_origin`` and
    its ``origins`` dict — feed the resulting origin names into
    ``add_claim``.
    """

    def __init__(self) -> None:
        """Initialize the contradiction detector."""
        self.claims: list[dict[str, Any]] = []

    def add_claim(self, claim: str, supporting_origins: list[str]) -> None:
        """Add a claim with its supporting origins.

        Args:
            claim: The claim statement.
            supporting_origins: List of **origin names** supporting this
                claim (e.g. ``["wikipedia", "github", "news_agency"]``).
                Pass origin names — NOT raw source IDs.  Confidence is
                derived from unique-origin diversity; passing source IDs
                instead would silently inflate the count (#1175). Use
                :class:`SourceTracker` to map source IDs to origin names
                when the caller has only the former.
        """
        self.claims.append(
            {
                "claim": claim,
                "supporting_origins": supporting_origins,
                "origin_count": len(set(supporting_origins)),
            }
        )

    def check_for_disagreement(self, claims: list[str], sources: list[SourceSnapshot]) -> list[str]:
        """Check for claims lacking independent confirmation.

        Args:
            claims: List of claims to check
            sources: List of source snapshots to check against

        Returns:
            List of disagreement messages for claims lacking independent support
        """
        disagreements = []
        for claim in claims:
            supporting_origins = self._find_supporting_origins(claim, sources)
            if len(supporting_origins) <= 1:
                disagreements.append(f"Claim lacks independent confirmation: {claim}")
        return disagreements

    def _find_supporting_origins(self, claim: str, sources: list[SourceSnapshot]) -> set[str]:
        """Find unique origins supporting a claim.

        Args:
            claim: The claim to search for
            sources: List of source snapshots to search

        Returns:
            Set of origin names that support the claim
        """
        origins = set()
        for source in sources:
            if self._claim_matches_content(claim, source["content"]):
                origins.add(source["origin"])
        return origins

    def _claim_matches_content(self, claim: str, content: str) -> bool:
        """Check if a claim is supported by content.

        Uses simple keyword matching - can be enhanced with NLP.

        Args:
            claim: The claim to match
            content: The content to search in

        Returns:
            True if claim is supported by content
        """
        claim_lower = claim.lower()
        content_lower = content.lower()
        # Check if ALL words in claim (length > 3) appear in content
        claim_words = [word for word in claim_lower.split() if len(word) > 3]
        return all(word in content_lower for word in claim_words)

    def get_confidence_score(self) -> float:
        """Calculate overall confidence based on source diversity.

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not self.claims:
            return 0.5

        total_claims = len(self.claims)
        well_supported = sum(1 for c in self.claims if c["origin_count"] > 1)
        return well_supported / total_claims if total_claims > 0 else 0.5

    def get_claims_by_confidence(self) -> list[dict[str, Any]]:
        """Get claims sorted by confidence level.

        Returns:
            List of claims with their confidence scores, sorted by confidence
        """
        claims_with_confidence = [
            {
                "claim": c["claim"],
                "origin_count": c["origin_count"],
                "confidence": 1.0 if c["origin_count"] > 1 else 0.0,
            }
            for c in self.claims
        ]
        return sorted(claims_with_confidence, key=lambda x: x["confidence"], reverse=True)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about tracked claims.

        Returns:
            Dictionary with claim statistics and overall confidence
        """
        return {
            "total_claims": len(self.claims),
            "well_supported_claims": sum(1 for c in self.claims if c["origin_count"] > 1),
            "poorly_supported_claims": sum(1 for c in self.claims if c["origin_count"] <= 1),
            "overall_confidence": self.get_confidence_score(),
        }
