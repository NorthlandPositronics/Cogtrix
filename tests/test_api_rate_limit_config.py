"""Regression tests for #1879 Slice A — config-driven rate limits.

Covers:

* :func:`src.api.rate_limit.parse_rate_limit_spec` accepts the SlowAPI
  ``"<N>/<window>"`` shape and rejects malformed input.
* :func:`src.api.rate_limit.configure_rate_limits` installs the per-route
  table and the default fallback atomically.
* :func:`src.api.rate_limit.per_route_rate_limit_for` resolves the
  configured limit at request time, falls back to the configured default
  for unknown names, and applies the new limit live across reloads.
* :class:`src.config.APIConfig` validates rate-limit specs and CIDR
  entries at construction and on ``__post_init__`` re-run, and merges
  YAML overrides into the defaults without dropping unspecified keys.
"""

from __future__ import annotations

import pytest

from src.api import rate_limit as rl
from src.config import APIConfig, ConfigError

# ---------------------------------------------------------------------------
# parse_rate_limit_spec
# ---------------------------------------------------------------------------


class TestParseRateLimitSpec:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("3/hour", (3, 3600)),
            ("100/minute", (100, 60)),
            ("1/second", (1, 1)),
            ("500/day", (500, 86400)),
            ("5/m", (5, 60)),
            ("20/h", (20, 3600)),
            ("10/d", (10, 86400)),
            ("  5 / m  ", (5, 60)),  # surrounding whitespace
            ("10/MINUTES", (10, 60)),  # case-insensitive + trailing s
            ("0/hour", (0, 3600)),  # zero is parseable (always-block sentinel)
        ],
    )
    def test_valid_specs(self, spec: str, expected: tuple[int, int]) -> None:
        assert rl.parse_rate_limit_spec(spec) == expected

    @pytest.mark.parametrize(
        "spec",
        [
            "bogus",
            "3 per hour",
            "/hour",
            "3/",
            "3/fortnight",
            "3/years",  # not in the unit table
            "-3/hour",  # negative not allowed
            "",
            "   ",
            "3/hour/extra",
        ],
    )
    def test_invalid_specs_raise(self, spec: str) -> None:
        with pytest.raises(ValueError, match="invalid rate-limit spec"):
            rl.parse_rate_limit_spec(spec)


# ---------------------------------------------------------------------------
# configure_rate_limits + per_route_rate_limit_for
# ---------------------------------------------------------------------------


class _FakeRequest:
    """Minimal stand-in matching the attributes touched by the dependency."""

    def __init__(self, client_host: str = "1.2.3.4", path: str = "/api/v1/x") -> None:
        from types import SimpleNamespace

        # ``slowapi.util.get_remote_address`` reads ``request.client.host``.
        self.client = SimpleNamespace(host=client_host)
        self.headers: dict[str, str] = {}
        # The dependency reads ``request.scope.get("route")`` and
        # ``request.url.path``; default to no route + literal path so it
        # falls back to the path as the bucket key.
        self.scope: dict[str, object] = {}
        self.url = SimpleNamespace(path=path)


@pytest.fixture(autouse=True)
def _restore_rate_limit_state():
    """Snapshot and restore the module-level rate-limit table for isolation."""
    with rl._route_limits_lock:
        saved_limits = dict(rl._route_limits)
        saved_default = rl._default_limit_spec
    saved_counters = dict(rl._hit_counters)
    saved_disabled = rl._per_route_disabled
    rl._per_route_disabled = False
    try:
        yield
    finally:
        with rl._route_limits_lock:
            rl._route_limits = saved_limits
            rl._default_limit_spec = saved_default
        with rl._counters_lock:
            rl._hit_counters.clear()
            rl._hit_counters.update(saved_counters)
        rl._per_route_disabled = saved_disabled


class TestConfigureRateLimits:
    def test_installs_per_route_table(self) -> None:
        rl.configure_rate_limits(
            default="1000/minute",
            per_route={"auth_register": "50/hour", "auth_login": "10/minute"},
        )
        # Behaviourally observable: per_route_rate_limit_for("auth_register")
        # honours the configured value.
        dep = rl.per_route_rate_limit_for("auth_register")
        request = _FakeRequest(client_host="1.2.3.4", path="/api/v1/auth/register")
        # First 50 must pass; 51st must raise 429.
        for _ in range(50):
            dep(request)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            dep(request)
        assert exc_info.value.status_code == 429

    def test_unknown_name_falls_back_to_default(self) -> None:
        rl.configure_rate_limits(
            default="2/minute",
            per_route={"auth_register": "100/minute"},
        )
        dep = rl.per_route_rate_limit_for("not_in_the_table")
        request = _FakeRequest(client_host="5.6.7.8", path="/api/v1/whatever")
        # The default 2/minute applies — third call must raise.
        dep(request)
        dep(request)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            dep(request)
        assert exc_info.value.status_code == 429

    def test_live_reload_replaces_existing_limit(self) -> None:
        """Re-calling configure_rate_limits replaces the previous table
        and the change takes effect on the next request — no stale
        captured closure.

        Slice B semantics note: the underlying ``limits`` library keys
        its sliding-window buckets by ``(spec.amount, spec.multiples,
        *identifiers)``. Changing the limit value (e.g. 1/minute →
        10/minute) therefore starts a FRESH window for the new spec —
        the old bucket's hits are no longer relevant because the bucket
        no longer exists under the new key. Operationally this is the
        right behaviour: a config bump shouldn't immediately block
        legit traffic that fits the new limit. The pre-Slice-B
        in-memory dict shared one bucket across spec changes; we
        explicitly test the new semantics here.
        """
        rl.configure_rate_limits(default="120/minute", per_route={"auth_register": "1/minute"})
        dep = rl.per_route_rate_limit_for("auth_register")
        request = _FakeRequest(client_host="9.9.9.9", path="/api/v1/auth/register")
        dep(request)  # first one OK with the 1/minute limit

        # Bump the limit. Under Slice B semantics the new limit gets a
        # fresh sliding window — 10 hits allowed; the 11th must 429.
        rl.configure_rate_limits(default="120/minute", per_route={"auth_register": "10/minute"})
        for _ in range(10):
            dep(request)
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            dep(request)
        assert exc_info.value.status_code == 429

    def test_invalid_spec_raises_without_corrupting_state(self) -> None:
        rl.configure_rate_limits(default="100/minute", per_route={"auth_register": "5/minute"})
        # Snapshot the existing table.
        with rl._route_limits_lock:
            snapshot = dict(rl._route_limits)
        with pytest.raises(ValueError):
            rl.configure_rate_limits(
                default="100/minute",
                per_route={"auth_register": "5/minute", "bad_one": "totally garbage"},
            )
        # State must be unchanged.
        with rl._route_limits_lock:
            assert rl._route_limits == snapshot


# ---------------------------------------------------------------------------
# APIConfig validation
# ---------------------------------------------------------------------------


class TestAPIConfigValidation:
    def test_default_construction_succeeds(self) -> None:
        cfg = APIConfig()
        assert "default" in cfg.rate_limits
        assert cfg.rate_limits["auth_register"] == "3/hour"
        assert cfg.trusted_proxy_cidrs == []

    def test_invalid_spec_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match="rate_limits.*must be"):
            APIConfig(rate_limits={"default": "totally garbage"})

    def test_missing_default_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match="must define a 'default' key"):
            APIConfig(rate_limits={"auth_register": "3/hour"})

    def test_invalid_cidr_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match="invalid CIDR"):
            APIConfig(trusted_proxy_cidrs=["not.an.address/24"])

    def test_valid_cidrs_accepted(self) -> None:
        cfg = APIConfig(
            trusted_proxy_cidrs=["10.0.0.0/8", "172.16.0.0/12", "::1/128"],
        )
        assert len(cfg.trusted_proxy_cidrs) == 3


class TestLoadConfigAPIBlock:
    """Verify the YAML loader merges ``api:`` overrides into the defaults
    rather than replacing the whole table — unspecified keys must retain
    their built-in values."""

    @staticmethod
    def _load_with_yaml(monkeypatch, yaml_path):
        """Point load_config at the temp YAML via COGTRIX_CONFIG_FILE and
        clear stray env overrides that would otherwise mutate the test
        config.
        """
        import os as _os

        from src.config import load_config

        monkeypatch.setenv("COGTRIX_CONFIG_FILE", str(yaml_path))
        for var in list(_os.environ):
            if var.startswith("COGTRIX_RATE_LIMIT_") or var == "COGTRIX_TRUSTED_PROXY_CIDRS":
                monkeypatch.delenv(var, raising=False)
        return load_config()

    def test_yaml_merge_preserves_unspecified_defaults(self, tmp_path, monkeypatch) -> None:
        yaml_path = tmp_path / "cogtrix.yaml"
        yaml_path.write_text(
            "api:\n"
            "  rate_limits:\n"
            "    auth_register: '100/hour'\n"
            "  trusted_proxy_cidrs:\n"
            "    - '10.0.0.0/8'\n"
        )
        cfg = self._load_with_yaml(monkeypatch, yaml_path)
        # Overridden key reflects the YAML value.
        assert cfg.api.rate_limits["auth_register"] == "100/hour"
        # Unspecified keys retain their defaults.
        assert cfg.api.rate_limits["default"] == "120/minute"
        assert cfg.api.rate_limits["auth_login"] == "5/minute"
        # CIDR list parsed.
        assert cfg.api.trusted_proxy_cidrs == ["10.0.0.0/8"]

    def test_yaml_invalid_spec_raises_at_load(self, tmp_path, monkeypatch) -> None:
        yaml_path = tmp_path / "cogtrix.yaml"
        yaml_path.write_text("api:\n  rate_limits:\n    auth_register: 'every-tuesday'\n")
        with pytest.raises(ConfigError, match="rate_limits"):
            self._load_with_yaml(monkeypatch, yaml_path)

    def test_yaml_cidr_list_or_string(self, tmp_path, monkeypatch) -> None:
        yaml_path = tmp_path / "cogtrix.yaml"
        yaml_path.write_text("api:\n  trusted_proxy_cidrs: '10.0.0.0/8, 172.16.0.0/12'\n")
        cfg = self._load_with_yaml(monkeypatch, yaml_path)
        assert cfg.api.trusted_proxy_cidrs == ["10.0.0.0/8", "172.16.0.0/12"]
