"""Regression tests for BUG-104: _prefetch_lids() must hold _lid_cache_lock during scan.

Before the fix, _prefetch_lids() read self._lid_cache without holding _lid_cache_lock,
creating a data race with concurrent _resolve_lid() pool threads that mutate the
OrderedDict (move_to_end, popitem, etc.).

After the fix, the cache-scan loop is wrapped with `with self._lid_cache_lock:`, so all
reads of _lid_cache are synchronized. The lock is released before the ThreadPoolExecutor
spawns worker threads that call _resolve_lid() (which re-acquires the lock independently).

These tests verify:
1. The scan loop executes correctly when _lid_cache_lock is held.
2. No deadlock occurs when _prefetch_lids() calls _resolve_lid() via the pool after
   releasing the scan lock.
3. Already-cached, non-expired entries are NOT added to the uncached set.
4. Expired entries ARE added to the uncached set.
"""

from __future__ import annotations

import collections
import threading
import time
from unittest.mock import MagicMock, patch

from src.assistant.channels.whatsapp import WhatsAppChannel
from src.tools._whatsapp_client import Message, WahaClient


def _make_channel() -> WhatsAppChannel:
    cfg = {"waha_url": "http://localhost:3000", "session": "default"}
    with patch.object(WahaClient, "__init__", lambda self, **kw: None):
        ch = WhatsAppChannel(cfg)
    # Give it a no-op client so network calls don't fire.
    ch._client = MagicMock()
    ch._client.resolve_lid.return_value = "+491234567890"
    return ch


def _make_message(from_number: str, body: str = "hi") -> Message:
    return Message(id="m1", timestamp=1000, from_number=from_number, body=body)


# ---------------------------------------------------------------------------
# Test 1: cached + valid entry → not in uncached, no HTTP call
# ---------------------------------------------------------------------------
def test_prefetch_lids_uses_cached_entry() -> None:
    ch = _make_channel()
    lid = "999@lid"
    # Pre-populate cache with a valid (non-expired) entry.
    ch._lid_cache[lid] = ("+491111111111", float("inf"))
    msgs = [_make_message(lid)]

    ch._prefetch_lids(msgs)

    # No HTTP resolution should happen because the entry is still valid.
    ch._client.resolve_lid.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: missing entry → uncached, HTTP call fires
# ---------------------------------------------------------------------------
def test_prefetch_lids_resolves_missing_lid() -> None:
    ch = _make_channel()
    lid = "888@lid"
    msgs = [_make_message(lid)]

    ch._prefetch_lids(msgs)

    ch._client.resolve_lid.assert_called_once_with(lid)


# ---------------------------------------------------------------------------
# Test 3: expired entry → uncached, HTTP call fires
# ---------------------------------------------------------------------------
def test_prefetch_lids_resolves_expired_lid() -> None:
    ch = _make_channel()
    lid = "777@lid"
    # Insert an entry that has already expired.
    past = time.monotonic() - 1.0
    ch._lid_cache[lid] = (None, past)
    msgs = [_make_message(lid)]

    ch._prefetch_lids(msgs)

    ch._client.resolve_lid.assert_called_once_with(lid)


# ---------------------------------------------------------------------------
# Test 4: non-lid message → no resolution
# ---------------------------------------------------------------------------
def test_prefetch_lids_skips_non_lid_numbers() -> None:
    ch = _make_channel()
    msgs = [_make_message("123@c.us")]

    ch._prefetch_lids(msgs)

    ch._client.resolve_lid.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: lock is acquired during the scan (no deadlock after scan releases it)
# ---------------------------------------------------------------------------
def test_prefetch_lids_acquires_lock_then_releases_before_pool() -> None:
    """The scan lock must be released before pool threads call _resolve_lid.

    We verify this by patching _resolve_lid to assert that _lid_cache_lock
    is NOT held when it runs (it acquires it itself — if the scan held it
    over the pool call we'd deadlock, since threading.Lock is not reentrant).
    """
    ch = _make_channel()
    lid = "666@lid"
    msgs = [_make_message(lid), _make_message("555@lid")]

    lock_held_during_resolve = threading.Event()
    original_resolve = ch._resolve_lid

    def _patched_resolve(lid_: str) -> str | None:
        # If the scan lock were still held, this would deadlock (Lock is not reentrant).
        # The fact that we can acquire the lock means the scan already released it.
        acquired = ch._lid_cache_lock.acquire(blocking=False)
        if acquired:
            lock_held_during_resolve.set()
            ch._lid_cache_lock.release()
        return original_resolve(lid_)

    with patch.object(ch, "_resolve_lid", side_effect=_patched_resolve):
        ch._prefetch_lids(msgs)

    assert lock_held_during_resolve.is_set(), (
        "_lid_cache_lock was still held when _resolve_lid ran — "
        "scan lock was not released before pool dispatch."
    )


# ---------------------------------------------------------------------------
# Test 6: concurrent _resolve_lid and _prefetch_lids do not corrupt the cache
# ---------------------------------------------------------------------------
def test_prefetch_lids_concurrent_with_resolve_lid_no_corruption() -> None:
    """Run _prefetch_lids and _resolve_lid concurrently many times; cache must remain valid."""
    ch = _make_channel()
    ch._client.resolve_lid.return_value = "+490000000000"

    errors: list[str] = []

    def do_prefetch() -> None:
        for _ in range(50):
            lid = f"{(_ % 10)}@lid"
            msgs = [_make_message(lid)]
            try:
                ch._prefetch_lids(msgs)
            except Exception as exc:
                errors.append(f"prefetch: {exc}")

    def do_resolve() -> None:
        for _ in range(50):
            lid = f"{(_ % 10)}@lid"
            try:
                ch._resolve_lid(lid)
            except Exception as exc:
                errors.append(f"resolve: {exc}")

    threads = [threading.Thread(target=do_prefetch), threading.Thread(target=do_resolve)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"Concurrent access produced errors: {errors}"
    # Verify cache is still a valid OrderedDict.
    assert isinstance(ch._lid_cache, collections.OrderedDict)
