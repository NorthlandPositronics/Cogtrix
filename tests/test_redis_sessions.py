"""Tests for src/api/redis_sessions.py — Redis-backed session presence store.

All tests use a mock redis.asyncio client so no real Redis server is needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import src.api.redis_sessions as redis_sessions_mod
from src.api.redis_sessions import SessionPresenceStore, configure_redis, get_store

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(
    redis_url: str = "redis://localhost:6379/0", ttl: int = 7200
) -> SessionPresenceStore:
    """Return a fresh SessionPresenceStore with a mock redis client pre-installed."""
    store = SessionPresenceStore(redis_url=redis_url, ttl_seconds=ttl)
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    store._client = mock_client
    return store


def _detach_client(store: SessionPresenceStore) -> None:
    """Set the client to None to simulate a disconnected state."""
    store._client = None


# ===========================================================================
# SessionPresenceStore — basic properties
# ===========================================================================


def test_is_connected_false_when_no_client():
    store = SessionPresenceStore(redis_url="redis://localhost")
    assert store.is_connected is False


def test_is_connected_true_after_mock_client_injected():
    store = _make_store()
    assert store.is_connected is True


def test_ttl_stored():
    store = SessionPresenceStore(redis_url="redis://localhost", ttl_seconds=3600)
    assert store.ttl_seconds == 3600


def test_default_ttl_is_two_hours():
    store = SessionPresenceStore(redis_url="redis://localhost")
    assert store.ttl_seconds == 7200


# ===========================================================================
# connect() / disconnect()
# ===========================================================================


@pytest.mark.asyncio
async def test_connect_no_op_when_no_redis_package():
    """connect() is a no-op when the redis package is absent."""
    store = SessionPresenceStore(redis_url="redis://localhost")
    with patch.object(redis_sessions_mod, "_HAS_REDIS", False):
        await store.connect()
    assert store._client is None


@pytest.mark.asyncio
async def test_connect_no_op_when_url_is_none():
    store = SessionPresenceStore(redis_url=None)  # type: ignore[arg-type]
    with patch.object(redis_sessions_mod, "_HAS_REDIS", True):
        await store.connect()
    assert store._client is None


@pytest.mark.asyncio
async def test_connect_sets_client_on_success():
    store = SessionPresenceStore(redis_url="redis://localhost:6379")
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)

    with (
        patch.object(redis_sessions_mod, "_HAS_REDIS", True),
        patch.object(redis_sessions_mod, "aioredis") as mock_aioredis,
    ):
        mock_aioredis.from_url.return_value = mock_client
        await store.connect()

    assert store._client is mock_client


@pytest.mark.asyncio
async def test_connect_sets_client_none_on_ping_failure():
    store = SessionPresenceStore(redis_url="redis://localhost:6379")
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(side_effect=ConnectionError("refused"))

    with (
        patch.object(redis_sessions_mod, "_HAS_REDIS", True),
        patch.object(redis_sessions_mod, "aioredis") as mock_aioredis,
    ):
        mock_aioredis.from_url.return_value = mock_client
        await store.connect()

    assert store._client is None


@pytest.mark.asyncio
async def test_disconnect_closes_client():
    store = _make_store()
    client = store._client
    await store.disconnect()
    client.aclose.assert_awaited_once()  # type: ignore[union-attr]
    assert store._client is None


@pytest.mark.asyncio
async def test_disconnect_no_op_when_not_connected():
    store = SessionPresenceStore(redis_url=None)  # type: ignore[arg-type]
    # Should not raise
    await store.disconnect()


# ===========================================================================
# touch()
# ===========================================================================


@pytest.mark.asyncio
async def test_touch_calls_set_with_correct_args():
    store = _make_store(ttl=600)
    await store.touch("sess-abc", 1234567890.5)
    store._client.set.assert_awaited_once_with(  # type: ignore[union-attr]
        "cogtrix:session:sess-abc", "1234567890.5", ex=600
    )


@pytest.mark.asyncio
async def test_touch_no_op_when_disconnected():
    store = _make_store()
    _detach_client(store)
    # Should not raise even without a client
    await store.touch("sess-xyz", 12345.0)


@pytest.mark.asyncio
async def test_touch_swallows_redis_error():
    store = _make_store()
    store._client.set = AsyncMock(side_effect=ConnectionError("gone"))  # type: ignore[union-attr]
    # Should not raise
    await store.touch("sess-err", 999.0)


# ===========================================================================
# get_last_activity()
# ===========================================================================


@pytest.mark.asyncio
async def test_get_last_activity_returns_float():
    store = _make_store()
    store._client.get = AsyncMock(return_value="1700000000.123")  # type: ignore[union-attr]
    result = await store.get_last_activity("sess-x")
    assert result == pytest.approx(1700000000.123)


@pytest.mark.asyncio
async def test_get_last_activity_returns_none_for_missing_key():
    store = _make_store()
    store._client.get = AsyncMock(return_value=None)  # type: ignore[union-attr]
    result = await store.get_last_activity("sess-missing")
    assert result is None


@pytest.mark.asyncio
async def test_get_last_activity_returns_none_when_disconnected():
    store = _make_store()
    _detach_client(store)
    result = await store.get_last_activity("sess-dc")
    assert result is None


@pytest.mark.asyncio
async def test_get_last_activity_swallows_error():
    store = _make_store()
    store._client.get = AsyncMock(side_effect=ConnectionError("gone"))  # type: ignore[union-attr]
    result = await store.get_last_activity("sess-err")
    assert result is None


# ===========================================================================
# remove()
# ===========================================================================


@pytest.mark.asyncio
async def test_remove_calls_delete():
    store = _make_store()
    await store.remove("sess-del")
    store._client.delete.assert_awaited_once_with("cogtrix:session:sess-del")  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_remove_no_op_when_disconnected():
    store = _make_store()
    _detach_client(store)
    await store.remove("sess-dc")  # should not raise


@pytest.mark.asyncio
async def test_remove_swallows_error():
    store = _make_store()
    store._client.delete = AsyncMock(side_effect=ConnectionError("gone"))  # type: ignore[union-attr]
    await store.remove("sess-err")  # should not raise


# ===========================================================================
# list_active()
# ===========================================================================


@pytest.mark.asyncio
async def test_list_active_strips_prefix():
    store = _make_store()

    async def _fake_scan(match=None):  # noqa: ARG001
        for key in ["cogtrix:session:aaa", "cogtrix:session:bbb"]:
            yield key

    store._client.scan_iter = _fake_scan  # type: ignore[union-attr]
    result = await store.list_active()
    assert sorted(result) == ["aaa", "bbb"]


@pytest.mark.asyncio
async def test_list_active_returns_empty_when_disconnected():
    store = _make_store()
    _detach_client(store)
    result = await store.list_active()
    assert result == []


@pytest.mark.asyncio
async def test_list_active_swallows_error():
    store = _make_store()

    async def _bad_scan(match=None):  # noqa: ARG001
        # Yield nothing then raise to exercise the error-handling path.
        # Raising before any yield also works because scan_iter is consumed
        # inside an async for loop that catches exceptions.
        if False:
            yield  # pragma: no cover  — makes this an async generator
        raise ConnectionError("gone")

    store._client.scan_iter = _bad_scan  # type: ignore[union-attr]
    result = await store.list_active()
    assert result == []


# ===========================================================================
# Module-level configure_redis / get_store
# ===========================================================================


def test_configure_redis_creates_store(monkeypatch):
    monkeypatch.setattr(redis_sessions_mod, "_store", None)
    configure_redis("redis://localhost:6379/0", ttl_seconds=3600)
    store = get_store()
    assert store is not None
    assert store.ttl_seconds == 3600
    # Cleanup
    monkeypatch.setattr(redis_sessions_mod, "_store", None)


def test_configure_redis_idempotent(monkeypatch):
    """Second call does NOT replace the existing store."""
    monkeypatch.setattr(redis_sessions_mod, "_store", None)
    configure_redis("redis://localhost/1", ttl_seconds=100)
    first_store = get_store()
    configure_redis("redis://localhost/2", ttl_seconds=200)
    assert get_store() is first_store
    # Cleanup
    monkeypatch.setattr(redis_sessions_mod, "_store", None)


def test_get_store_returns_none_when_not_configured(monkeypatch):
    monkeypatch.setattr(redis_sessions_mod, "_store", None)
    assert get_store() is None


# ===========================================================================
# Key prefix correctness
# ===========================================================================


@pytest.mark.asyncio
async def test_touch_uses_correct_key_prefix():
    store = _make_store()
    await store.touch("my-session", 1.0)
    call_args = store._client.set.call_args  # type: ignore[union-attr]
    key = call_args[0][0]
    assert key == "cogtrix:session:my-session"


@pytest.mark.asyncio
async def test_get_last_activity_uses_correct_key_prefix():
    store = _make_store()
    store._client.get = AsyncMock(return_value=None)  # type: ignore[union-attr]
    await store.get_last_activity("my-session")
    call_args = store._client.get.call_args  # type: ignore[union-attr]
    key = call_args[0][0]
    assert key == "cogtrix:session:my-session"
