"""Regression tests for API key debounce race condition and memory leak (issue #1085).

Coverage:
  - Concurrent validate_api_key calls are serialised by asyncio.Lock
  - Debounce window prevents duplicate DB writes for the same key
  - Stale entries are evicted from _API_KEY_LAST_USED after 2x debounce
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api import auth


@pytest.fixture(autouse=True)
def _reset_auth_globals():
    """Reset module-level state before every test."""
    auth._API_KEY_LAST_USED.clear()
    yield
    auth._API_KEY_LAST_USED.clear()


def _make_key_record(key_id: str):
    key = MagicMock()
    key.id = key_id
    key.revoked = False
    key.expires_at = None
    return key


def _make_user():
    user = MagicMock()
    user.id = "user-1"
    user.role = "user"
    user.is_active = True
    return user


def _make_repo_mock(key_id: str = "key-1"):
    """Return an ApiKeyRepository mock that returns a valid key record."""
    mock_repo = MagicMock()
    mock_repo.get_by_hash = AsyncMock(return_value=_make_key_record(key_id))
    mock_repo.update_last_used = AsyncMock()
    return mock_repo


class _MonoClock:
    """Callable that returns an increasing monotonic timestamp.

    Patching ``time.monotonic`` with a constant breaks ``asyncio.sleep``
    because the event loop's ``time()`` method delegates to it.  Using an
    advancing mock keeps the debounce check deterministic while letting
    the event loop schedule timers correctly.
    """

    def __init__(self, start: float = 1000.0, step: float = 1.0) -> None:
        self._t = start
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


class TestApiKeyDebounceLock:
    """Validate that concurrent calls do not race on _API_KEY_LAST_USED."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_only_write_once(self):
        """Two coroutines validating the same key inside the debounce window
        must result in exactly one DB write."""
        mock_repo = _make_repo_mock("key-1")
        mock_db = MagicMock()
        mock_db.commit = AsyncMock()

        call_count = 0

        async def _slow_update(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)  # Force interleaving

        mock_repo.update_last_used.side_effect = _slow_update

        async def _call():
            with patch.object(auth, "_hash_api_key", return_value="hash"):
                with patch(
                    "src.api.db.repositories.api_keys.ApiKeyRepository",
                    return_value=mock_repo,
                ):
                    with patch(
                        "src.api.db.repositories.users.UserRepository",
                        return_value=MagicMock(get_by_id=AsyncMock(return_value=_make_user())),
                    ):
                        return await auth.validate_api_key("ak-test", mock_db)

        # Patch time.monotonic so the debounce check sees a large delta even
        # in CI containers where monotonic clock starts near zero.
        with patch("time.monotonic", _MonoClock()):
            # Fire both calls concurrently
            await asyncio.gather(_call(), _call())

        # Only one DB write should have occurred
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_debounce_prevents_second_write(self):
        """A second call within the debounce window must skip the DB write."""
        mock_repo = _make_repo_mock("key-2")
        mock_db = MagicMock()
        mock_db.commit = AsyncMock()

        async def _call():
            with patch.object(auth, "_hash_api_key", return_value="hash"):
                with patch(
                    "src.api.db.repositories.api_keys.ApiKeyRepository",
                    return_value=mock_repo,
                ):
                    with patch(
                        "src.api.db.repositories.users.UserRepository",
                        return_value=MagicMock(get_by_id=AsyncMock(return_value=_make_user())),
                    ):
                        return await auth.validate_api_key("ak-test", mock_db)

        # Use a large monotonic value so the first call is not debounced.
        with patch("time.monotonic", _MonoClock()):
            await _call()
        assert mock_repo.update_last_used.await_count == 1

        # Immediate second call — inside debounce window
        with patch("time.monotonic", _MonoClock()):
            await _call()
        assert mock_repo.update_last_used.await_count == 1

        # Immediate third call — inside debounce window
        with patch("time.monotonic", _MonoClock()):
            await _call()
        assert mock_repo.update_last_used.await_count == 1


class TestApiKeyLastUsedCleanup:
    """Validate eviction of stale entries from _API_KEY_LAST_USED."""

    @pytest.mark.asyncio
    async def test_stale_entries_removed_after_cleanup(self):
        """Entries older than 2x debounce are evicted on the next write."""
        # Seed the dict with one fresh and one stale entry
        now = time.monotonic()
        auth._API_KEY_LAST_USED["fresh-key"] = now
        auth._API_KEY_LAST_USED["stale-key"] = now - (auth._API_KEY_DEBOUNCE_SECONDS * 3)

        auth._cleanup_stale_api_key_entries()

        assert "fresh-key" in auth._API_KEY_LAST_USED
        assert "stale-key" not in auth._API_KEY_LAST_USED

    @pytest.mark.asyncio
    async def test_cleanup_triggered_on_write(self):
        """Cleanup runs automatically when a write succeeds."""
        # Use a fixed large monotonic value so the debounce check passes
        # even in CI containers where the clock starts near zero.
        with patch("time.monotonic", _MonoClock(start=1000.0)):
            auth._API_KEY_LAST_USED["old-key"] = 1000.0 - (auth._API_KEY_DEBOUNCE_SECONDS * 3)

            mock_repo = _make_repo_mock("new-key")
            mock_db = MagicMock()
            mock_db.commit = AsyncMock()

            with patch.object(auth, "_hash_api_key", return_value="hash"):
                with patch(
                    "src.api.db.repositories.api_keys.ApiKeyRepository",
                    return_value=mock_repo,
                ):
                    with patch(
                        "src.api.db.repositories.users.UserRepository",
                        return_value=MagicMock(get_by_id=AsyncMock(return_value=_make_user())),
                    ):
                        await auth.validate_api_key("ak-test", mock_db)

            assert "new-key" in auth._API_KEY_LAST_USED
            assert "old-key" not in auth._API_KEY_LAST_USED
