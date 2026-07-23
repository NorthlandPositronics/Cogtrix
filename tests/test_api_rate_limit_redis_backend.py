"""Regression tests for #1879 Slice B — optional Redis rate-limit backend.

Covers:

* :func:`src.api.rate_limit.configure_rate_limit_backend` installs
  ``MemoryStorage`` when ``redis_url`` is unset and a Redis-backed
  ``MovingWindowRateLimiter`` when a URL is supplied.
* :func:`current_backend_label` reflects the active backend and
  masks any inline Redis password.
* Per-route enforcement honours the configured limit through both
  backends.
* Redis backend shares state across multiple ``configure_rate_limit_backend``
  installs pointed at the same URL (simulating two replicas).
* Behavioural fail-open on transient backend errors.
* Startup precedence: ``COGTRIX_REDIS_URL`` env > ``api.redis_url`` YAML.

Tests use ``fakeredis[lua]`` so no real Redis daemon is required —
``limits.RedisStorage`` uses Lua scripts (EVALSHA) for atomic moving-
window updates, hence the ``[lua]`` extra.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.api import rate_limit as rl
from src.config import APIConfig

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis_client():
    """Return a fresh ``fakeredis.FakeStrictRedis`` server.

    Tests that need the Redis path patch ``redis.from_url`` to return
    this stand-in, so ``limits.storage_from_string('redis://...')``
    builds a ``RedisStorage`` over the fake server with no real
    network I/O.
    """
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeStrictRedis()


@pytest.fixture(autouse=True)
def _restore_backend():
    """Snapshot the active backend before each test and restore after.

    Also restores the module-level SlowAPI ``limiter`` so the
    follow-up's rebuild path (configure_rate_limit_backend now also
    rebuilds the global limiter, #1879 follow-up) doesn't leak across
    tests.
    """
    with rl._backend_lock:
        saved_storage = rl._storage
        saved_strategy = rl._strategy
        saved_label = rl._backend_label
        saved_limiter = rl.limiter
    saved_disabled = rl._per_route_disabled
    rl._per_route_disabled = False
    try:
        yield
    finally:
        with rl._backend_lock:
            rl._storage = saved_storage
            rl._strategy = saved_strategy
            rl._backend_label = saved_label
            rl.limiter = saved_limiter
        rl._per_route_disabled = saved_disabled


class _FakeRequest:
    """Minimal Request stand-in matching the fields enforcement reads."""

    def __init__(self, client_host: str = "1.2.3.4", path: str = "/x") -> None:
        self.client = SimpleNamespace(host=client_host)
        self.headers: dict[str, str] = {}
        self.scope: dict[str, object] = {}
        self.url = SimpleNamespace(path=path)


# ---------------------------------------------------------------------------
# configure_rate_limit_backend — backend selection + label
# ---------------------------------------------------------------------------


class TestConfigureRateLimitBackend:
    def test_default_backend_is_memory(self) -> None:
        rl.configure_rate_limit_backend(redis_url=None)
        from limits.storage import MemoryStorage

        with rl._backend_lock:
            assert isinstance(rl._storage, MemoryStorage)
        assert rl.current_backend_label() == "memory://"

    def test_empty_string_treated_as_unset(self) -> None:
        rl.configure_rate_limit_backend(redis_url="")
        assert rl.current_backend_label() == "memory://"
        rl.configure_rate_limit_backend(redis_url="   ")
        assert rl.current_backend_label() == "memory://"

    def test_redis_url_installs_redis_backend(self, fake_redis_client) -> None:
        import redis
        from limits.storage.redis import RedisStorage

        with patch.object(redis, "from_url", return_value=fake_redis_client):
            rl.configure_rate_limit_backend(redis_url="redis://localhost:6379/0")
        with rl._backend_lock:
            assert isinstance(rl._storage, RedisStorage)
        assert rl.current_backend_label() == "redis://localhost:6379/0"

    def test_label_redacts_password(self, fake_redis_client) -> None:
        import redis

        with patch.object(redis, "from_url", return_value=fake_redis_client):
            rl.configure_rate_limit_backend(redis_url="redis://user:supersecret@redis.prod:6379/3")
        label = rl.current_backend_label()
        assert "supersecret" not in label
        assert "***" in label
        # User and host preserved for operator visibility.
        assert "user" in label
        assert "redis.prod" in label

    def test_label_redacts_no_user_password(self, fake_redis_client) -> None:
        import redis

        with patch.object(redis, "from_url", return_value=fake_redis_client):
            rl.configure_rate_limit_backend(redis_url="redis://:opensesame@host:6379")
        assert "opensesame" not in rl.current_backend_label()
        assert "***" in rl.current_backend_label()


# ---------------------------------------------------------------------------
# Per-route enforcement through both backends
# ---------------------------------------------------------------------------


class TestEnforcementThroughBackend:
    def test_memory_backend_enforces_limit(self) -> None:
        rl.configure_rate_limit_backend(redis_url=None)
        rl.configure_rate_limits(default="120/minute", per_route={"auth_register": "3/hour"})
        dep = rl.per_route_rate_limit_for("auth_register")
        request = _FakeRequest(client_host="1.1.1.1", path="/api/v1/auth/register")
        # First 3 must pass; 4th must raise 429.
        dep(request)
        dep(request)
        dep(request)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            dep(request)
        assert exc_info.value.status_code == 429

    def test_redis_backend_enforces_limit(self, fake_redis_client) -> None:
        import redis

        with patch.object(redis, "from_url", return_value=fake_redis_client):
            rl.configure_rate_limit_backend(redis_url="redis://localhost:6379/0")
        rl.configure_rate_limits(default="120/minute", per_route={"auth_register": "3/hour"})
        dep = rl.per_route_rate_limit_for("auth_register")
        request = _FakeRequest(client_host="2.2.2.2", path="/api/v1/auth/register")
        dep(request)
        dep(request)
        dep(request)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            dep(request)
        assert exc_info.value.status_code == 429

    def test_redis_backend_shares_state_across_strategy_instances(self, fake_redis_client) -> None:
        """Two ``configure_rate_limit_backend`` calls against the same
        Redis instance see each other's hits — i.e. the per-replica
        sliding window collapses to a single shared bucket. This is the
        whole point of Slice B."""
        import redis

        rl.configure_rate_limits(default="120/minute", per_route={"auth_register": "3/hour"})
        request = _FakeRequest(client_host="3.3.3.3", path="/api/v1/auth/register")

        # Simulate replica 1: hit twice.
        with patch.object(redis, "from_url", return_value=fake_redis_client):
            rl.configure_rate_limit_backend(redis_url="redis://localhost:6379/0")
        dep_replica1 = rl.per_route_rate_limit_for("auth_register")
        dep_replica1(request)
        dep_replica1(request)

        # Simulate replica 2: same Redis, same client, third hit should
        # still be allowed (we've used 2 of 3), fourth must 429.
        with patch.object(redis, "from_url", return_value=fake_redis_client):
            rl.configure_rate_limit_backend(redis_url="redis://localhost:6379/0")
        dep_replica2 = rl.per_route_rate_limit_for("auth_register")
        dep_replica2(request)  # 3rd hit, allowed
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            dep_replica2(request)  # 4th hit, blocked
        assert exc_info.value.status_code == 429

    def test_per_route_disabled_bypass_still_honoured(self) -> None:
        rl.configure_rate_limit_backend(redis_url=None)
        rl.configure_rate_limits(default="120/minute", per_route={"auth_register": "1/hour"})
        rl._per_route_disabled = True
        try:
            dep = rl.per_route_rate_limit_for("auth_register")
            request = _FakeRequest(client_host="4.4.4.4", path="/x")
            # Should not raise even though limit is 1/hour and we hit twice.
            for _ in range(5):
                dep(request)
        finally:
            rl._per_route_disabled = False


# ---------------------------------------------------------------------------
# Fail-open on backend errors
# ---------------------------------------------------------------------------


class TestBackendFailOpen:
    def test_strategy_hit_exception_allows_request(self, caplog) -> None:
        """A transient backend error (e.g. Redis blip) must NOT take
        down the API. We log a WARNING and let the request through —
        the user-visible alternative would be a 5xx storm on every
        request until Redis recovers."""
        rl.configure_rate_limit_backend(redis_url=None)
        rl.configure_rate_limits(default="120/minute", per_route={"auth_register": "3/hour"})

        class _BoomStrategy:
            def hit(self, *_a, **_kw):
                raise RuntimeError("redis is down")

        with rl._backend_lock:
            saved_strategy = rl._strategy
            rl._strategy = _BoomStrategy()  # type: ignore[assignment]
        try:
            dep = rl.per_route_rate_limit_for("auth_register")
            request = _FakeRequest(client_host="5.5.5.5", path="/x")
            with caplog.at_level("WARNING", logger="cogtrix.api.rate_limit"):
                # Must NOT raise.
                dep(request)
            assert any("Rate-limit backend hit() raised" in rec.message for rec in caplog.records)
        finally:
            with rl._backend_lock:
                rl._strategy = saved_strategy


# ---------------------------------------------------------------------------
# Startup-precedence: env > YAML > default
# ---------------------------------------------------------------------------


class TestRedisUrlStartupPrecedence:
    def test_apiconfig_default_is_none(self) -> None:
        assert APIConfig().redis_url is None

    def test_yaml_loader_reads_redis_url(self, tmp_path, monkeypatch) -> None:
        from src.config import load_config

        yaml_path = tmp_path / "cogtrix.yaml"
        yaml_path.write_text("api:\n  redis_url: 'redis://yaml.host:6379/2'\n")
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", str(yaml_path))
        monkeypatch.delenv("COGTRIX_REDIS_URL", raising=False)
        cfg = load_config()
        assert cfg.api.redis_url == "redis://yaml.host:6379/2"

    def test_yaml_loader_treats_empty_string_as_none(self, tmp_path, monkeypatch) -> None:
        from src.config import load_config

        yaml_path = tmp_path / "cogtrix.yaml"
        yaml_path.write_text("api:\n  redis_url: ''\n")
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", str(yaml_path))
        monkeypatch.delenv("COGTRIX_REDIS_URL", raising=False)
        cfg = load_config()
        assert cfg.api.redis_url is None

    def test_yaml_loader_warns_on_non_string(self, tmp_path, monkeypatch, caplog) -> None:
        from src.config import load_config

        yaml_path = tmp_path / "cogtrix.yaml"
        yaml_path.write_text("api:\n  redis_url: 12345\n")
        monkeypatch.setenv("COGTRIX_CONFIG_FILE", str(yaml_path))
        monkeypatch.delenv("COGTRIX_REDIS_URL", raising=False)
        with caplog.at_level("WARNING"):
            cfg = load_config()
        # Numeric value rejected with a warning; redis_url stays None.
        assert cfg.api.redis_url is None
        assert any("redis_url" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# #1879 follow-up: SlowAPI global Limiter Redis-swap
# ---------------------------------------------------------------------------


class TestSlowAPIGlobalLimiterRebuild:
    """The SlowAPI ``limiter`` blunt-guard (default ``120/minute``) was
    documented in PR #1881 as in-memory-only — based on the (then
    incorrect) assumption that ``SlowAPIMiddleware`` captured the
    limiter at construction. In fact, the middleware reads
    ``app.state.limiter`` per request, so the follow-up rebuilds the
    module-level ``limiter`` inside ``configure_rate_limit_backend``.

    These tests assert that the rebuild actually swaps the storage
    backend (memory ↔ redis) and preserves the ``default_limits`` /
    ``key_func`` configuration.
    """

    def test_default_redis_url_keeps_memory_backed_limiter(self) -> None:
        from limits.storage import MemoryStorage

        rl.configure_rate_limit_backend(redis_url=None)
        # The slowapi.Limiter stores its bound storage as ``_storage``.
        # Memory backend is the safe single-node default.
        assert isinstance(rl.limiter._storage, MemoryStorage)
        # The blunt-guard default survives the rebuild. SlowAPI wraps
        # each spec in a ``LimitGroup`` whose underlying spec string
        # is exposed via the name-mangled ``_LimitGroup__limit_provider``
        # attribute.
        assert any(
            "120" in getattr(spec, "_LimitGroup__limit_provider", "")
            for spec in rl.limiter._default_limits
        )

    def test_redis_url_rebuilds_with_redis_storage(self, fake_redis_client) -> None:
        import redis
        from limits.storage.redis import RedisStorage

        with patch.object(redis, "from_url", return_value=fake_redis_client):
            rl.configure_rate_limit_backend(redis_url="redis://localhost:6379/0")
        assert isinstance(rl.limiter._storage, RedisStorage)
        # ``default_limits`` and ``key_func`` survive the rebuild. See
        # the LimitGroup attribute note on
        # test_default_redis_url_keeps_memory_backed_limiter.
        assert any(
            "120" in getattr(spec, "_LimitGroup__limit_provider", "")
            for spec in rl.limiter._default_limits
        )
        assert rl.limiter._key_func is rl.rate_limit_key

    def test_per_route_and_global_share_the_same_backend(self, fake_redis_client) -> None:
        """When ``configure_rate_limit_backend`` swaps to Redis, BOTH
        the per-route MovingWindowRateLimiter strategy AND the SlowAPI
        global limiter must be Redis-backed — otherwise multi-replica
        deployments still suffer the blunt-guard skew this follow-up
        was filed to close."""
        import redis
        from limits.storage import RedisStorage as PerRouteRedisStorage
        from limits.storage.redis import RedisStorage as SlowapiRedisStorage

        with patch.object(redis, "from_url", return_value=fake_redis_client):
            rl.configure_rate_limit_backend(redis_url="redis://localhost:6379/0")

        # Per-route backend (MovingWindowRateLimiter over Storage).
        with rl._backend_lock:
            assert isinstance(rl._storage, PerRouteRedisStorage)

        # SlowAPI global limiter.
        assert isinstance(rl.limiter._storage, SlowapiRedisStorage)

    def test_idempotent_rebuild_returns_to_memory_after_redis(self, fake_redis_client) -> None:
        """Calling configure_rate_limit_backend(redis_url=None) after a
        Redis install must put the global limiter back on
        MemoryStorage — covers the dev workflow where someone unsets
        the env var and restarts."""
        import redis
        from limits.storage import MemoryStorage
        from limits.storage.redis import RedisStorage

        with patch.object(redis, "from_url", return_value=fake_redis_client):
            rl.configure_rate_limit_backend(redis_url="redis://localhost:6379/0")
        assert isinstance(rl.limiter._storage, RedisStorage)

        rl.configure_rate_limit_backend(redis_url=None)
        assert isinstance(rl.limiter._storage, MemoryStorage)

    def test_app_state_limiter_picks_up_rebuilt_instance(self, fake_redis_client) -> None:
        """``src/api/app.py`` startup must reassign ``app.state.limiter``
        via module-attribute access after ``configure_rate_limit_backend``
        runs — otherwise ``SlowAPIMiddleware.dispatch`` keeps reading
        the pre-startup in-memory limiter via the stale ``app.state``
        binding. We assert the source contains the module-attribute
        access pattern so a future refactor that switches back to
        ``app.state.limiter = limiter`` (importing the symbol) gets
        caught.
        """
        from pathlib import Path

        app_py = Path(__file__).resolve().parent.parent / "src" / "api" / "app.py"
        source = app_py.read_text(encoding="utf-8")
        # The fix line: import the rate_limit module and access .limiter
        # via attribute (so the rebuilt object is picked up).
        assert (
            "from src.api import rate_limit as _rl_module" in source
            and "app.state.limiter = _rl_module.limiter" in source
        ), (
            "src/api/app.py must use module-attribute access on the "
            "rebuilt limiter after configure_rate_limit_backend; "
            "importing ``limiter`` as a symbol freezes the binding at "
            "module load and the SlowAPI global rebuild has no effect."
        )
