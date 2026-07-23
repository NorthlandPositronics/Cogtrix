"""Tests for SAMLNonceCache and assertion replay protection."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from cogtrix_core.api.saml.nonce_cache import SAMLNonceCache


class TestSAMLNonceCache:
    """Test the SAMLNonceCache class."""

    def test_add_new_assertion(self) -> None:
        """Adding a new assertion ID should succeed."""
        cache = SAMLNonceCache(default_ttl_seconds=600)
        result = cache.add("assertion-123")
        assert result is True
        assert cache.contains("assertion-123")

    def test_add_duplicate_assertion(self) -> None:
        """Adding the same assertion ID twice should fail."""
        cache = SAMLNonceCache(default_ttl_seconds=600)
        cache.add("assertion-123")
        result = cache.add("assertion-123")
        assert result is False
        assert cache.contains("assertion-123")

    def test_contains_expired_assertion(self) -> None:
        """An expired assertion should return False."""
        cache = SAMLNonceCache(default_ttl_seconds=1)
        cache.add("assertion-123")
        assert cache.contains("assertion-123")

        # Force expiry by patching time.time
        with patch("cogtrix_core.api.saml.nonce_cache.time.time", return_value=time.time() + 2):
            assert not cache.contains("assertion-123")

    def test_remove_assertion(self) -> None:
        """Removing an assertion should make it unavailable."""
        cache = SAMLNonceCache(default_ttl_seconds=600)
        cache.add("assertion-123")
        result = cache.remove("assertion-123")
        assert result is True
        assert not cache.contains("assertion-123")

    def test_remove_nonexistent_assertion(self) -> None:
        """Removing a non-existent assertion should return False."""
        cache = SAMLNonceCache(default_ttl_seconds=600)
        result = cache.remove("assertion-999")
        assert result is False

    def test_cleanup_expired(self) -> None:
        """Cleanup should remove expired entries and return count."""
        cache = SAMLNonceCache(default_ttl_seconds=1)
        cache.add("assertion-1")
        cache.add("assertion-2")
        cache.add("assertion-3")

        with patch("cogtrix_core.api.saml.nonce_cache.time.time", return_value=time.time() + 2):
            removed = cache.cleanup_expired()
            assert removed == 3
            assert cache.size() == 0

    def test_clear(self) -> None:
        """Clear should remove all entries."""
        cache = SAMLNonceCache(default_ttl_seconds=600)
        cache.add("assertion-1")
        cache.add("assertion-2")
        cache.clear()
        assert cache.size() == 0

    def test_size(self) -> None:
        """Size should return count of non-expired entries."""
        cache = SAMLNonceCache(default_ttl_seconds=600)
        assert cache.size() == 0
        cache.add("assertion-1")
        assert cache.size() == 1
        cache.add("assertion-2")
        assert cache.size() == 2

    def test_custom_ttl(self) -> None:
        """Custom TTL should be respected."""
        cache = SAMLNonceCache(default_ttl_seconds=600)
        cache.add("assertion-1", ttl_seconds=1)
        assert cache.contains("assertion-1")

        with patch("cogtrix_core.api.saml.nonce_cache.time.time", return_value=time.time() + 2):
            assert not cache.contains("assertion-1")


class TestSAMLReplayProtectionIntegration:
    """Test replay protection integration in saml.py routes."""

    @pytest.mark.asyncio
    async def test_acs_rejects_replayed_assertion(self) -> None:
        """ACS should reject a SAMLResponse with a previously used assertion ID."""
        from cogtrix_core.api.routes import saml

        # Reset cache for test
        saml._nonce_cache.clear()

        # First assertion should succeed
        result1 = saml._nonce_cache.add("assertion-replay-test-1")
        assert result1 is True

        # Second attempt with same ID should fail
        result2 = saml._nonce_cache.add("assertion-replay-test-1")
        assert result2 is False
