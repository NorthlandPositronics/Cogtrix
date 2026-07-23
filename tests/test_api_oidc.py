"""Tests for the OIDC/SSO token validation module (src/api/oidc.py).

Covers:
- OIDCConfig dataclass construction
- OIDCValidator.validate() — success, expired, wrong audience/issuer, bad signature
- JWKS resolution: kid matching, kid-not-found refresh, no-kid fallback
- JWKS caching: cache hit within TTL, re-fetch after TTL expiry
- Discovery caching: cache hit within TTL, re-fetch after TTL expiry
- discover_jwks_uri: success, network failure, missing field
- map_role: list with admin, list without admin, single string, absent claim
- configure_oidc / get_validator module-level API
- Config OIDC section parsing (oidc_enabled, oidc_issuer, etc.)
- get_current_user OIDC fallback (local fail → OIDC success)
- get_current_user expired local JWT not retried via OIDC
- get_current_user both validators fail → 401
- Async/event-loop safety: validate() must be offloaded via asyncio.to_thread
- Thread safety: concurrent cache access does not corrupt state
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("jwt")
pytest.importorskip("cryptography")

import jwt  # noqa: E402
from cryptography.hazmat.backends import default_backend  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from jwt.algorithms import RSAAlgorithm  # noqa: E402

from src.api.oidc import OIDCConfig, OIDCValidator, configure_oidc, get_validator  # noqa: E402

# ---------------------------------------------------------------------------
# RSA key helpers (module-level, shared across tests)
# ---------------------------------------------------------------------------

_RSA_KEY = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend(),
)
_RSA_KEY_2 = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend(),
)

_ISSUER = "https://auth.example.com"
_AUDIENCE = "cogtrix-app"
_KID = "test-key-id"
_KID_2 = "another-key-id"


def _make_pub_jwk(private_key: Any, kid: str) -> dict:
    pub = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    pub.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return pub


def _jwks_body(private_key: Any, kid: str) -> bytes:
    return json.dumps({"keys": [_make_pub_jwk(private_key, kid)]}).encode()


def _make_token(
    private_key: Any,
    *,
    kid: str = _KID,
    sub: str = "user-123",
    iss: str = _ISSUER,
    aud: str = _AUDIENCE,
    exp_delta: timedelta = timedelta(hours=1),
    extra_claims: dict | None = None,
) -> str:
    now = datetime.now(UTC)
    claims: dict = {
        "sub": sub,
        "iss": iss,
        "aud": aud,
        "iat": now,
        "exp": now + exp_delta,
    }
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


_PATCH_URLOPEN = "src.api.oidc.urlopen"


def _urlopen_returning(body: bytes):
    """Return a mock urlopen that serves *body* from any URL."""
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read = MagicMock(return_value=body)
    return MagicMock(return_value=mock_resp)


def _urlopen_for_jwks(private_key: Any, kid: str):
    return _urlopen_returning(_jwks_body(private_key, kid))


def _make_validator(
    *,
    jwks_uri: str = "https://auth.example.com/jwks",
    allow_insecure_oidc: bool = False,
    production_mode: bool = True,
) -> OIDCValidator:
    return OIDCValidator(
        OIDCConfig(
            issuer=_ISSUER,
            audience=_AUDIENCE,
            jwks_uri=jwks_uri,
            allow_insecure_oidc=allow_insecure_oidc,
            production_mode=production_mode,
        )
    )


# ===========================================================================
# configure_oidc / get_validator
# ===========================================================================


class TestModuleLevelAPI:
    def test_configure_oidc_sets_validator(self) -> None:
        configure_oidc(OIDCConfig(issuer=_ISSUER, audience=_AUDIENCE, jwks_uri="https://x/k"))
        assert get_validator() is not None

    def test_configure_oidc_replaces_previous_validator(self) -> None:
        configure_oidc(OIDCConfig(issuer="https://a.test", audience="a", jwks_uri="https://a/k"))
        v1 = get_validator()
        configure_oidc(OIDCConfig(issuer="https://b.test", audience="b", jwks_uri="https://b/k"))
        v2 = get_validator()
        assert v1 is not v2
        assert v2 is not None
        assert v2._config.issuer == "https://b.test"


# ===========================================================================
# validate()
# ===========================================================================


class TestValidate:
    def test_valid_rs256_token_returns_claims(self) -> None:
        token = _make_token(_RSA_KEY, sub="alice")
        validator = _make_validator()
        with patch(_PATCH_URLOPEN, _urlopen_for_jwks(_RSA_KEY, _KID)):
            claims = validator.validate(token)
        assert claims["sub"] == "alice"
        assert claims["iss"] == _ISSUER

    def test_expired_token_raises_expired_signature_error(self) -> None:
        token = _make_token(_RSA_KEY, exp_delta=timedelta(seconds=-10))
        validator = _make_validator()
        with patch(_PATCH_URLOPEN, _urlopen_for_jwks(_RSA_KEY, _KID)):
            with pytest.raises(jwt.ExpiredSignatureError):
                validator.validate(token)

    def test_wrong_audience_raises_invalid_token_error(self) -> None:
        token = _make_token(_RSA_KEY, aud="wrong-audience")
        validator = _make_validator()
        with patch(_PATCH_URLOPEN, _urlopen_for_jwks(_RSA_KEY, _KID)):
            with pytest.raises(jwt.InvalidTokenError):
                validator.validate(token)

    def test_wrong_issuer_raises_invalid_token_error(self) -> None:
        token = _make_token(_RSA_KEY, iss="https://evil.example.com")
        validator = _make_validator()
        with patch(_PATCH_URLOPEN, _urlopen_for_jwks(_RSA_KEY, _KID)):
            with pytest.raises(jwt.InvalidTokenError):
                validator.validate(token)

    def test_bad_signature_raises_invalid_token_error(self) -> None:
        # Token signed with _RSA_KEY but JWKS advertises _RSA_KEY_2 — mismatch
        token = _make_token(_RSA_KEY)
        validator = _make_validator()
        with patch(_PATCH_URLOPEN, _urlopen_for_jwks(_RSA_KEY_2, _KID)):
            with pytest.raises(jwt.InvalidTokenError):
                validator.validate(token)

    def test_http_jwks_uri_rejected_by_default(self) -> None:
        token = _make_token(_RSA_KEY)
        validator = _make_validator(jwks_uri="http://auth.example.com/jwks")
        with pytest.raises(ValueError, match="JWKS URI uses disallowed scheme 'http'"):
            validator.validate(token)

    def test_http_jwks_uri_allowed_only_in_non_production_when_explicitly_enabled(self) -> None:
        token = _make_token(_RSA_KEY)
        validator = _make_validator(
            jwks_uri="http://auth.example.com/jwks",
            allow_insecure_oidc=True,
            production_mode=False,
        )
        with patch(_PATCH_URLOPEN, _urlopen_for_jwks(_RSA_KEY, _KID)):
            claims = validator.validate(token)
        assert claims["sub"] == "user-123"


# ===========================================================================
# JWKS key resolution
# ===========================================================================


class TestSigningKeyResolution:
    def test_no_kid_in_header_is_rejected(self) -> None:
        """Token without a kid header must be rejected explicitly."""
        now = datetime.now(UTC)
        claims = {
            "sub": "no-kid-user",
            "iss": _ISSUER,
            "aud": _AUDIENCE,
            "iat": now,
            "exp": now + timedelta(hours=1),
        }
        token = jwt.encode(claims, _RSA_KEY, algorithm="RS256")

        validator = _make_validator()
        with patch(_PATCH_URLOPEN, _urlopen_for_jwks(_RSA_KEY, _KID)):
            with pytest.raises(jwt.InvalidTokenError, match="kid header required"):
                validator.validate(token)

    def test_kid_not_found_forces_cache_refresh(self) -> None:
        """When kid is absent from cached JWKS, validator fetches once more."""
        token_kid2 = _make_token(_RSA_KEY_2, kid=_KID_2)
        validator = _make_validator()

        first_body = _jwks_body(_RSA_KEY, _KID)
        second_body = _jwks_body(_RSA_KEY_2, _KID_2)
        call_count = [0]

        def _urlopen(url: str, timeout: int = 5):
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            call_count[0] += 1
            mock_resp.read = MagicMock(
                return_value=first_body if call_count[0] == 1 else second_body
            )
            return mock_resp

        with patch(_PATCH_URLOPEN, _urlopen):
            result = validator.validate(token_kid2)
        assert result["sub"] == "user-123"
        assert call_count[0] == 2  # initial fetch + forced refresh

    def test_kid_not_found_after_refresh_raises(self) -> None:
        """If kid is still absent after refresh, InvalidTokenError is raised."""
        token = _make_token(_RSA_KEY, kid="non-existent-kid")
        validator = _make_validator()
        # Both fetches return JWKS with _KID, not "non-existent-kid"
        with patch(_PATCH_URLOPEN, _urlopen_for_jwks(_RSA_KEY, _KID)):
            with pytest.raises(jwt.InvalidTokenError, match="non-existent-kid"):
                validator.validate(token)


# ===========================================================================
# JWKS caching
# ===========================================================================


class TestJWKSCache:
    def test_cache_hit_within_ttl_avoids_refetch(self) -> None:
        """Two validate() calls within TTL should trigger only one urlopen call."""
        token1 = _make_token(_RSA_KEY)
        token2 = _make_token(_RSA_KEY, sub="bob")
        validator = _make_validator()
        mock_urlopen = _urlopen_for_jwks(_RSA_KEY, _KID)

        with patch(_PATCH_URLOPEN, mock_urlopen):
            validator.validate(token1)
            validator.validate(token2)

        assert mock_urlopen.call_count == 1

    def test_cache_expired_triggers_refetch(self) -> None:
        """After TTL, the next validate() call re-fetches the JWKS."""
        from src.api.oidc import _JWKS_TTL

        token = _make_token(_RSA_KEY)
        validator = _make_validator()
        mock_urlopen = _urlopen_for_jwks(_RSA_KEY, _KID)

        with patch(_PATCH_URLOPEN, mock_urlopen):
            validator.validate(token)
            # Manually expire the cache
            assert validator._jwks_cache is not None
            validator._jwks_cache = (validator._jwks_cache[0], time.monotonic() - _JWKS_TTL - 1)
            validator.validate(token)

        assert mock_urlopen.call_count == 2


# ===========================================================================
# Discovery
# ===========================================================================


class TestDiscovery:
    def test_discover_jwks_uri_success(self) -> None:
        doc = json.dumps({"jwks_uri": "https://auth.example.com/jwks"}).encode()
        validator = OIDCValidator(OIDCConfig(issuer=_ISSUER, audience=_AUDIENCE))
        with patch(_PATCH_URLOPEN, _urlopen_returning(doc)):
            uri = validator.discover_jwks_uri(_ISSUER)
        assert uri == "https://auth.example.com/jwks"

    def test_discover_jwks_uri_network_failure_raises(self) -> None:
        validator = OIDCValidator(OIDCConfig(issuer=_ISSUER, audience=_AUDIENCE))
        with patch(_PATCH_URLOPEN, side_effect=OSError("connection refused")):
            with pytest.raises(RuntimeError, match="OIDC discovery failed"):
                validator.discover_jwks_uri(_ISSUER)

    def test_discover_jwks_uri_missing_field_raises(self) -> None:
        doc = json.dumps({"issuer": _ISSUER}).encode()  # no jwks_uri key
        validator = OIDCValidator(OIDCConfig(issuer=_ISSUER, audience=_AUDIENCE))
        with patch(_PATCH_URLOPEN, _urlopen_returning(doc)):
            with pytest.raises(RuntimeError, match="missing 'jwks_uri'"):
                validator.discover_jwks_uri(_ISSUER)

    def test_http_issuer_rejected_by_default(self) -> None:
        validator = OIDCValidator(OIDCConfig(issuer="http://auth.example.com", audience=_AUDIENCE))
        with pytest.raises(ValueError, match="OIDC issuer uses disallowed scheme 'http'"):
            validator.discover_jwks_uri("http://auth.example.com")

    def test_http_issuer_allowed_only_in_non_production_when_explicitly_enabled(self) -> None:
        doc = json.dumps({"jwks_uri": "https://auth.example.com/jwks"}).encode()
        validator = OIDCValidator(
            OIDCConfig(
                issuer="http://auth.example.com",
                audience=_AUDIENCE,
                allow_insecure_oidc=True,
                production_mode=False,
            )
        )
        with patch(_PATCH_URLOPEN, _urlopen_returning(doc)):
            uri = validator.discover_jwks_uri("http://auth.example.com")
        assert uri == "https://auth.example.com/jwks"

    def test_discovery_cache_reused_within_ttl(self) -> None:
        """_resolve_jwks_uri calls discover_jwks_uri at most once per TTL."""
        discovery_doc = json.dumps({"jwks_uri": "https://auth.example.com/jwks"}).encode()
        call_log: list[str] = []

        def _urlopen(url: str, timeout: int = 5):
            call_log.append(url)
            body = discovery_doc if "openid-configuration" in url else _jwks_body(_RSA_KEY, _KID)
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read = MagicMock(return_value=body)
            return mock_resp

        from src.api.oidc import _JWKS_TTL

        token = _make_token(_RSA_KEY)
        validator = OIDCValidator(OIDCConfig(issuer=_ISSUER, audience=_AUDIENCE))
        with patch(_PATCH_URLOPEN, _urlopen):
            validator.validate(token)
            # Expire only the JWKS cache — discovery cache stays fresh
            assert validator._jwks_cache is not None
            validator._jwks_cache = (validator._jwks_cache[0], time.monotonic() - _JWKS_TTL - 1)
            validator.validate(token)

        discovery_calls = [u for u in call_log if "openid-configuration" in u]
        assert len(discovery_calls) == 1  # discovery fetched only once

    def test_discovery_cache_expired_triggers_refetch(self) -> None:
        discovery_doc = json.dumps({"jwks_uri": "https://auth.example.com/jwks"}).encode()
        call_log: list[str] = []

        def _urlopen(url: str, timeout: int = 5):
            call_log.append(url)
            body = discovery_doc if "openid-configuration" in url else _jwks_body(_RSA_KEY, _KID)
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read = MagicMock(return_value=body)
            return mock_resp

        from src.api.oidc import _DISCOVERY_TTL, _JWKS_TTL

        token = _make_token(_RSA_KEY)
        validator = OIDCValidator(OIDCConfig(issuer=_ISSUER, audience=_AUDIENCE))
        with patch(_PATCH_URLOPEN, _urlopen):
            validator.validate(token)
            # Expire both caches
            assert validator._jwks_cache is not None and validator._discovery_cache is not None
            validator._jwks_cache = (validator._jwks_cache[0], time.monotonic() - _JWKS_TTL - 1)
            validator._discovery_cache = (
                validator._discovery_cache[0],
                time.monotonic() - _DISCOVERY_TTL - 1,
            )
            validator.validate(token)

        discovery_calls = [u for u in call_log if "openid-configuration" in u]
        assert len(discovery_calls) == 2  # re-fetched after TTL


# ===========================================================================
# map_role
# ===========================================================================


class TestMapRole:
    def setup_method(self) -> None:
        self.validator = _make_validator()

    def test_admin_in_role_list_returns_admin(self) -> None:
        claims = {"roles": ["user", "admin", "editor"]}
        assert self.validator.map_role(claims) == "admin"

    def test_no_admin_in_list_returns_first_role(self) -> None:
        claims = {"roles": ["editor", "viewer"]}
        assert self.validator.map_role(claims) == "editor"

    def test_absent_claim_returns_default_role(self) -> None:
        assert self.validator.map_role({}) == "user"

    def test_single_string_admin(self) -> None:
        claims = {"roles": "admin"}
        assert self.validator.map_role(claims) == "admin"

    def test_single_string_non_admin(self) -> None:
        claims = {"roles": "viewer"}
        assert self.validator.map_role(claims) == "viewer"

    def test_custom_role_claim(self) -> None:
        v = OIDCValidator(
            OIDCConfig(issuer=_ISSUER, audience=_AUDIENCE, jwks_uri="x", role_claim="grp")
        )
        claims = {"grp": ["admin"]}
        assert v.map_role(claims) == "admin"

    def test_custom_default_role(self) -> None:
        v = OIDCValidator(
            OIDCConfig(issuer=_ISSUER, audience=_AUDIENCE, jwks_uri="x", default_role="admin")
        )
        assert v.map_role({}) == "admin"


# ===========================================================================
# Config OIDC section parsing
# ===========================================================================


class TestConfigOIDCParsing:
    @staticmethod
    def _load_from_yaml(tmp_path: Path, yaml_text: str):
        from src.config import Config, _apply_config_file  # type: ignore[attr-defined]

        cfg_file = tmp_path / ".cogtrix.yaml"
        cfg_file.write_text(yaml_text)
        cfg = Config()
        _apply_config_file(cfg, cfg_file)
        return cfg

    def test_oidc_fields_parsed_from_config(self, tmp_path: Path) -> None:
        cfg = self._load_from_yaml(
            tmp_path,
            "oidc:\n"
            "  enabled: true\n"
            "  issuer: https://auth.example.com\n"
            "  audience: my-app\n"
            "  jwks_uri: https://auth.example.com/jwks\n"
            "  role_claim: grp\n"
            "  default_role: admin\n",
        )
        assert cfg.oidc_enabled is True
        assert cfg.oidc_issuer == "https://auth.example.com"
        assert cfg.oidc_audience == "my-app"
        assert cfg.oidc_jwks_uri == "https://auth.example.com/jwks"
        assert cfg.oidc_allow_insecure_oidc is False
        assert cfg.oidc_role_claim == "grp"
        assert cfg.oidc_default_role == "admin"

    def test_oidc_defaults_when_section_absent(self, tmp_path: Path) -> None:
        cfg = self._load_from_yaml(tmp_path, "session: default\n")
        assert cfg.oidc_enabled is False
        assert cfg.oidc_issuer is None
        assert cfg.oidc_audience is None
        assert cfg.oidc_jwks_uri is None
        assert cfg.oidc_allow_insecure_oidc is False
        assert cfg.oidc_role_claim == "roles"
        assert cfg.oidc_default_role == "user"

    def test_invalid_default_role_ignored(self, tmp_path: Path) -> None:
        cfg = self._load_from_yaml(tmp_path, "oidc:\n  enabled: true\n  default_role: superuser\n")
        assert cfg.oidc_default_role == "user"  # invalid value — stays at default

    def test_empty_issuer_and_audience_ignored(self, tmp_path: Path) -> None:
        cfg = self._load_from_yaml(
            tmp_path, "oidc:\n  issuer: '   '\n  audience: ''\n  jwks_uri: '  '\n"
        )
        assert cfg.oidc_issuer is None
        assert cfg.oidc_audience is None
        assert cfg.oidc_jwks_uri is None

    def test_allow_insecure_oidc_parsed_from_config(self, tmp_path: Path) -> None:
        cfg = self._load_from_yaml(
            tmp_path,
            "oidc:\n"
            "  enabled: true\n"
            "  issuer: http://auth.example.com\n"
            "  audience: my-app\n"
            "  allow_insecure_oidc: true\n",
        )
        assert cfg.oidc_enabled is True
        assert cfg.oidc_allow_insecure_oidc is True


# ===========================================================================
# Event-loop blocking
# ===========================================================================


class TestEventLoopBlocking:
    """Verify that OIDCValidator.validate() blocks the event loop when called
    directly from async code, and that asyncio.to_thread() prevents blocking.

    Regression for PR #1024 — src/api/routes/messages.py called
    validator.validate(raw_token) directly inside an async handler, freezing
    the event loop for up to _HTTP_TIMEOUT seconds on every OIDC fallback.
    """

    @pytest.mark.asyncio
    async def test_direct_validate_call_blocks_event_loop(self) -> None:
        """Calling validate() directly from async code blocks the event loop."""
        token = _make_token(_RSA_KEY)
        validator = _make_validator()
        progress = asyncio.Event()

        def _slow_urlopen(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.2)
            return _urlopen_returning(_jwks_body(_RSA_KEY, _KID))()

        async def _background() -> None:
            await asyncio.sleep(0.01)
            progress.set()

        with patch(_PATCH_URLOPEN, _slow_urlopen):
            bg_task = asyncio.create_task(_background())
            validator.validate(token)
            # validate() blocked the loop, so background hasn't run yet
            assert not progress.is_set()
            await bg_task

        assert progress.is_set()

    @pytest.mark.asyncio
    async def test_validate_via_to_thread_keeps_loop_responsive(self) -> None:
        """Offloading validate() via asyncio.to_thread lets other tasks run."""
        token = _make_token(_RSA_KEY)
        validator = _make_validator()
        progress = asyncio.Event()

        def _slow_urlopen(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.2)
            return _urlopen_returning(_jwks_body(_RSA_KEY, _KID))()

        async def _background() -> None:
            await asyncio.sleep(0.01)
            progress.set()

        with patch(_PATCH_URLOPEN, _slow_urlopen):
            bg_task = asyncio.create_task(_background())
            await asyncio.to_thread(validator.validate, token)
            # background ran while validate() executed on a worker thread
            assert progress.is_set()
            await bg_task


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    """Verify OIDCValidator internal caches tolerate concurrent access."""

    def test_concurrent_validate_calls_with_cache_miss(self) -> None:
        """Multiple threads with JWKS cache miss should not raise or corrupt state."""
        token = _make_token(_RSA_KEY)
        validator = _make_validator()
        call_count = [0]
        lock = threading.Lock()

        def _urlopen(url: str, timeout: int = 5) -> Any:
            with lock:
                call_count[0] += 1
            time.sleep(0.01)
            return _urlopen_returning(_jwks_body(_RSA_KEY, _KID))()

        results: list[Any] = []
        errors: list[Exception] = []

        def _worker() -> None:
            try:
                with patch(_PATCH_URLOPEN, _urlopen):
                    results.append(validator.validate(token))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 10
        assert all(r["sub"] == "user-123" for r in results)
        # Duplicate fetches are acceptable; verify no unbounded growth
        assert 1 <= call_count[0] <= 10

    def test_concurrent_discovery_calls_with_cache_miss(self) -> None:
        """Multiple threads with discovery cache miss should not raise or corrupt state."""
        validator = _make_validator(jwks_uri=None)
        call_count = [0]
        lock = threading.Lock()

        def _urlopen(url: str, timeout: int = 5) -> Any:
            with lock:
                call_count[0] += 1
            time.sleep(0.01)
            if "openid-configuration" in url:
                doc = json.dumps({"jwks_uri": "https://auth.example.com/jwks"}).encode()
                return _urlopen_returning(doc)()
            return _urlopen_returning(_jwks_body(_RSA_KEY, _KID))()

        results: list[Any] = []
        errors: list[Exception] = []

        def _worker() -> None:
            try:
                with patch(_PATCH_URLOPEN, _urlopen):
                    token = _make_token(_RSA_KEY)
                    results.append(validator.validate(token))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 10
        assert all(r["sub"] == "user-123" for r in results)


# ===========================================================================
# get_current_user OIDC fallback (integration via TestClient)
# ===========================================================================

pytest.importorskip("fastapi")

import asyncio as _asyncio  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from src.api.db.engine import Base, get_db  # noqa: E402

os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")


def _make_expired_local_jwt(user_id: str = "uid", role: str = "user") -> str:
    import jwt as _jwt

    now = datetime.now(UTC)
    return _jwt.encode(
        {"sub": user_id, "role": role, "iat": now, "exp": now - timedelta(hours=1)},
        os.environ["COGTRIX_JWT_SECRET"],
        algorithm="HS256",
    )


@pytest.fixture()
def _app():
    """Minimal FastAPI app with in-memory SQLite; OIDC validator reset per test."""
    from src.api import oidc as _oidc_mod
    from src.api.app import create_app

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _create() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    _asyncio.run(_create())

    old_validator = _oidc_mod._validator
    _oidc_mod._validator = None

    the_app = create_app()

    async def _override():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    the_app.dependency_overrides[get_db] = _override
    yield the_app

    _oidc_mod._validator = old_validator
    _asyncio.run(engine.dispose())


@pytest.fixture()
def _client(_app):
    with TestClient(_app, raise_server_exceptions=False) as c:
        yield c


def _auth(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


def _register_and_login(client: Any) -> str:
    import uuid

    uname = f"u_{uuid.uuid4().hex[:8]}"
    client.post(
        "/api/v1/auth/register",
        json={"username": uname, "email": f"{uname}@t.com", "password": "TestPass1!"},
    )
    resp = client.post("/api/v1/auth/login", json={"username": uname, "password": "TestPass1!"})
    return resp.json()["data"]["access_token"]


class TestGetCurrentUserOIDCFallback:
    def test_valid_local_jwt_accepted_without_oidc(self, _client: Any) -> None:
        tok = _register_and_login(_client)
        resp = _client.get("/api/v1/auth/me", headers=_auth(tok))
        assert resp.status_code == 200

    def test_no_token_returns_401(self, _client: Any) -> None:
        resp = _client.get("/api/v1/auth/me")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"

    def test_expired_local_jwt_not_retried_via_oidc(self, _client: Any) -> None:
        """An expired HS256 JWT must return TOKEN_EXPIRED — OIDC must not be called."""
        from src.api import oidc as _oidc_mod

        mock_validator = MagicMock()
        mock_validator.validate = MagicMock(return_value={"sub": "x"})
        _oidc_mod._validator = mock_validator

        try:
            expired_tok = _make_expired_local_jwt()
            resp = _client.get("/api/v1/auth/me", headers=_auth(expired_tok))
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "TOKEN_EXPIRED"
            mock_validator.validate.assert_not_called()
        finally:
            _oidc_mod._validator = None

    def test_invalid_local_jwt_no_oidc_returns_401(self, _client: Any) -> None:
        resp = _client.get("/api/v1/auth/me", headers=_auth("not.a.jwt.at.all"))
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"

    def test_invalid_local_jwt_oidc_validator_is_called(self, _client: Any) -> None:
        """Bad local token: OIDC validator is invoked on the UNAUTHORIZED fallback path."""
        from src.api import oidc as _oidc_mod

        oidc_claims = {
            "sub": "oidc-user-99",
            "iss": _ISSUER,
            "aud": _AUDIENCE,
            "roles": ["user"],
        }
        mock_validator = MagicMock()
        mock_validator.validate = MagicMock(return_value=oidc_claims)
        mock_validator.map_role = MagicMock(return_value="user")
        _oidc_mod._validator = mock_validator

        try:
            token = _make_token(_RSA_KEY, sub="oidc-user-99")
            # The /me endpoint raises 401 when the OIDC user doesn't exist in the local DB
            # (OIDC validates the token but no matching local account exists — that's expected).
            # The key assertion is that the OIDC validator WAS invoked on the fallback path.
            _client.get("/api/v1/auth/me", headers=_auth(token))
            mock_validator.validate.assert_called_once_with(token)
            mock_validator.map_role.assert_called_once_with(oidc_claims)
        finally:
            _oidc_mod._validator = None

    def test_invalid_local_jwt_oidc_also_fails_returns_401(self, _client: Any) -> None:
        """Both local JWT and OIDC fail → 401 UNAUTHORIZED."""
        from src.api import oidc as _oidc_mod

        mock_validator = MagicMock()
        mock_validator.validate = MagicMock(side_effect=jwt.InvalidTokenError("bad token"))
        _oidc_mod._validator = mock_validator

        try:
            resp = _client.get("/api/v1/auth/me", headers=_auth("bad.token.here"))
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "UNAUTHORIZED"
        finally:
            _oidc_mod._validator = None
