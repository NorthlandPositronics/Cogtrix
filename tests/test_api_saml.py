"""Tests for SAML 2.0 SP — config, provider, and routes (Enterprise Phase 1 — task 1.2.1).

python3-saml is an optional dependency.  Tests that require it are skipped
automatically when it is not installed (pytest.importorskip).

Tests that do NOT require python3-saml (config, 503 responses, settings serialisation)
run unconditionally.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")

os.environ.setdefault("COGTRIX_JWT_SECRET", "testsecret_mustbe32chars_minimum00")
os.environ.setdefault("COGTRIX_DB_URL", "sqlite+aiosqlite:///:memory:")

from cogtrix_core.api.saml.config import (  # noqa: E402
    SAMLConfig,
    SAMLIdPConfig,
    configure_saml,
    get_saml_config,
    is_saml_configured,
)

# ---------------------------------------------------------------------------
# Fixture: minimal SAML config
# ---------------------------------------------------------------------------

_FAKE_CERT = (
    "MIIDEzCCAfugAwIBAgIJAKoK/heBjcOuMA0GCSqGSIb3DQEBBQUAMCAxHjAcBgNV"
    "BAMTFXNhbWwuZXhhbXBsZS5jb20wHhcNMTcwMTEwMDAwMDAwWhcNMjcwMTA4MDAwMDAw"
    "WjAgMR4wHAYDVQQDExVzYW1sLmV4YW1wbGUuY29tMIIBIjANBgkqhkiG9w0BAQEFAAOC"
    "AQ8AMIIBCgKCAQEA7Q2TCJJCtT/CMhc7DRiHTU3mVp09UiR0HNYH7EGT6U4sG"
    "fake_cert_data_for_test"
)


@pytest.fixture(autouse=True)
def reset_saml_config():
    """Reset global SAML config before and after each test."""
    import cogtrix_core.api.saml.config as _cfg

    _cfg._saml_config = None
    yield
    _cfg._saml_config = None


def _make_config(**overrides) -> SAMLConfig:
    idp = overrides.pop(
        "idp",
        SAMLIdPConfig(
            entity_id="https://idp.example.com",
            sso_url="https://idp.example.com/sso",
            certificate=_FAKE_CERT,
        ),
    )
    return SAMLConfig(
        sp_entity_id=overrides.pop("sp_entity_id", "https://sp.example.com/saml/metadata"),
        sp_acs_url=overrides.pop("sp_acs_url", "https://sp.example.com/api/v1/saml/acs"),
        idp=idp,
        **overrides,
    )


# ---------------------------------------------------------------------------
# SAMLConfig
# ---------------------------------------------------------------------------


class TestSAMLConfig:
    def test_configure_and_get(self):
        cfg = _make_config()
        assert not is_saml_configured()
        configure_saml(cfg)
        assert is_saml_configured()
        assert get_saml_config() is cfg

    def test_configure_replaces_prior(self):
        cfg1 = _make_config(sp_entity_id="https://sp1.example.com/metadata")
        cfg2 = _make_config(sp_entity_id="https://sp2.example.com/metadata")
        configure_saml(cfg1)
        configure_saml(cfg2)
        assert get_saml_config() is cfg2

    def test_not_configured_by_default(self):
        assert get_saml_config() is None
        assert not is_saml_configured()

    def test_to_python3_saml_settings_structure(self):
        cfg = _make_config()
        settings = cfg.to_python3_saml_settings()
        assert "sp" in settings
        assert "idp" in settings
        assert settings["sp"]["entityId"] == cfg.sp_entity_id
        assert settings["sp"]["assertionConsumerService"]["url"] == cfg.sp_acs_url
        assert settings["idp"]["entityId"] == cfg.idp.entity_id
        assert settings["idp"]["singleSignOnService"]["url"] == cfg.idp.sso_url

    def test_security_block_enforces_signed_assertions(self):
        """python3-saml defaults wantAssertionsSigned/wantMessagesSigned to False.
        strict:True alone does NOT enforce signatures — the security block is required.
        Regression for #295 (unsigned SAMLResponse authenticated successfully)."""
        cfg = _make_config()
        settings = cfg.to_python3_saml_settings()
        sec = settings.get("security", {})
        assert (
            sec.get("wantAssertionsSigned") is True
        ), "wantAssertionsSigned must be True — unsigned assertions must be rejected"
        assert (
            sec.get("wantMessagesSigned") is True
        ), "wantMessagesSigned must be True — unsigned messages must be rejected"
        assert sec.get("rejectDeprecatedAlgorithm") is True
        assert "sha256" in sec.get("signatureAlgorithm", "").lower()
        assert "sha256" in sec.get("digestAlgorithm", "").lower()

    def test_sp_cert_included_when_set(self):
        cfg = _make_config(sp_certificate="mycert", sp_private_key="mykey")
        settings = cfg.to_python3_saml_settings()
        assert settings["sp"]["x509cert"] == "mycert"
        assert settings["sp"]["privateKey"] == "mykey"

    def test_sp_cert_omitted_when_empty(self):
        cfg = _make_config()
        settings = cfg.to_python3_saml_settings()
        assert "x509cert" not in settings["sp"]
        assert "privateKey" not in settings["sp"]

    def test_slo_url_included_when_set(self):
        idp = SAMLIdPConfig(
            entity_id="https://idp.example.com",
            sso_url="https://idp.example.com/sso",
            certificate=_FAKE_CERT,
            slo_url="https://idp.example.com/slo",
        )
        cfg = _make_config(idp=idp)
        settings = cfg.to_python3_saml_settings()
        assert "singleLogoutService" in settings["idp"]
        assert settings["idp"]["singleLogoutService"]["url"] == "https://idp.example.com/slo"

    def test_default_attribute_map(self):
        cfg = _make_config()
        assert "email" in cfg.attribute_map
        assert "username" in cfg.attribute_map

    def test_default_role_is_user(self):
        cfg = _make_config()
        assert cfg.default_role == "user"


# ---------------------------------------------------------------------------
# SAMLIdPConfig
# ---------------------------------------------------------------------------


class TestSAMLIdPConfig:
    def test_required_fields(self):
        idp = SAMLIdPConfig(
            entity_id="https://idp.example.com",
            sso_url="https://idp.example.com/sso",
            certificate="cert",
        )
        assert idp.entity_id == "https://idp.example.com"
        assert idp.slo_url is None

    def test_optional_slo_url(self):
        idp = SAMLIdPConfig(
            entity_id="eid",
            sso_url="sso",
            certificate="cert",
            slo_url="slo",
        )
        assert idp.slo_url == "slo"


# ---------------------------------------------------------------------------
# Routes — 503 when not configured or not installed
# ---------------------------------------------------------------------------


@pytest.fixture()
def saml_client():
    import asyncio

    from fastapi.testclient import TestClient
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    from cogtrix_core.api.app import create_app
    from cogtrix_core.api.db.engine import Base, get_db

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_setup())
    factory = async_sessionmaker(engine, expire_on_commit=False)

    from unittest.mock import patch

    with patch.dict(os.environ, {"COGTRIX_JWT_SECRET": "testsecret_mustbe32chars_minimum00"}):
        app = create_app()

        async def _override():
            async with factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

        app.dependency_overrides[get_db] = _override
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    asyncio.run(engine.dispose())


class TestSAMLRoutes:
    def test_metadata_503_when_not_configured(self, saml_client):
        r = saml_client.get("/api/v1/saml/metadata")
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "SAML_NOT_CONFIGURED"

    def test_sso_503_when_not_configured(self, saml_client):
        r = saml_client.get("/api/v1/saml/sso")
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "SAML_NOT_CONFIGURED"

    def test_acs_503_when_not_configured(self, saml_client):
        r = saml_client.post("/api/v1/saml/acs", data={"SAMLResponse": "fake"})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "SAML_NOT_CONFIGURED"


# ---------------------------------------------------------------------------
# Provider — import-error path (python3-saml not installed)
# ---------------------------------------------------------------------------


class TestSAMLProvider:
    def test_get_metadata_xml_raises_import_error_when_no_saml(self):
        """When python3-saml is absent, get_metadata_xml raises ImportError."""
        import sys
        from unittest.mock import patch

        # Force import failure of onelogin.saml2
        with patch.dict(
            sys.modules,
            {
                "onelogin": None,
                "onelogin.saml2": None,
                "onelogin.saml2.metadata": None,
                "onelogin.saml2.settings": None,
            },
        ):
            from importlib import reload

            import cogtrix_core.api.saml.provider as _prov

            cfg = _make_config()
            with pytest.raises(ImportError, match="saml"):
                reload(_prov)
                _prov.get_metadata_xml(cfg)

    def test_provider_first_attr_helper(self):
        """_first_attr returns first value or empty string."""
        from cogtrix_core.api.saml.provider import _first_attr

        assert _first_attr({"k": ["a", "b"]}, "k") == "a"
        assert _first_attr({"k": []}, "k") == ""
        assert _first_attr({}, "missing") == ""


# ---------------------------------------------------------------------------
# Routes — HTTP-level integration tests (#229)
# Covers the three route handlers with SAML configured and provider mocked.
# ---------------------------------------------------------------------------


class TestSAMLRoutesConfigured:
    """HTTP integration tests for SAML routes with a configured SP."""

    @pytest.fixture()
    def configured_client(self, saml_client):
        """saml_client with SAML configured."""
        configure_saml(_make_config())
        return saml_client

    # ── metadata ──────────────────────────────────────────────────────────

    def test_metadata_returns_xml_when_configured(self, configured_client):
        """GET /metadata with a working provider returns 200 + XML."""
        from unittest.mock import patch

        with patch(
            "cogtrix_core.api.saml.provider.get_metadata_xml", return_value="<md:EntityDescriptor/>"
        ):
            r = configured_client.get("/api/v1/saml/metadata")
        assert r.status_code == 200
        assert "EntityDescriptor" in r.text
        assert r.headers["content-type"].startswith("application/xml")

    def test_metadata_503_when_saml_not_installed(self, configured_client):
        """GET /metadata raises 503 SAML_NOT_INSTALLED when import fails."""
        from unittest.mock import patch

        with patch(
            "cogtrix_core.api.saml.provider.get_metadata_xml", side_effect=ImportError("no saml")
        ):
            r = configured_client.get("/api/v1/saml/metadata")
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "SAML_NOT_INSTALLED"

    def test_metadata_500_on_value_error(self, configured_client):
        """GET /metadata returns 500 when metadata generation raises ValueError."""
        from unittest.mock import patch

        with patch(
            "cogtrix_core.api.saml.provider.get_metadata_xml",
            side_effect=ValueError("bad config"),
        ):
            r = configured_client.get("/api/v1/saml/metadata")
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "SAML_METADATA_ERROR"

    # ── sso ───────────────────────────────────────────────────────────────

    def test_sso_redirects_to_idp(self, configured_client):
        """GET /sso with working provider returns 302 redirect to IdP."""
        from unittest.mock import patch

        with patch(
            "cogtrix_core.api.saml.provider.get_sso_redirect_url",
            return_value="https://idp.example.com/sso?SAMLRequest=abc123",
        ):
            r = configured_client.get("/api/v1/saml/sso", follow_redirects=False)
        assert r.status_code == 302
        assert "idp.example.com" in r.headers["location"]

    def test_sso_503_when_saml_not_installed(self, configured_client):
        """GET /sso raises 503 SAML_NOT_INSTALLED when import fails."""
        from unittest.mock import patch

        with patch(
            "cogtrix_core.api.saml.provider.get_sso_redirect_url",
            side_effect=ImportError("no saml"),
        ):
            r = configured_client.get("/api/v1/saml/sso")
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "SAML_NOT_INSTALLED"

    # ── acs ───────────────────────────────────────────────────────────────

    def test_acs_422_missing_saml_response(self, configured_client):
        """POST /acs without SAMLResponse body field returns 422."""
        r = configured_client.post("/api/v1/saml/acs", data={})
        assert r.status_code == 422

    def test_acs_401_on_invalid_saml_response(self, configured_client):
        """POST /acs with a bad SAMLResponse returns 401 SAML_INVALID_RESPONSE."""
        from unittest.mock import patch

        with patch(
            "cogtrix_core.api.saml.provider.process_saml_response",
            side_effect=ValueError("signature invalid"),
        ):
            r = configured_client.post("/api/v1/saml/acs", data={"SAMLResponse": "bm90dmFsaWQ="})
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "SAML_INVALID_RESPONSE"

    def test_acs_503_when_saml_not_installed(self, configured_client):
        """POST /acs raises 503 SAML_NOT_INSTALLED when python3-saml absent."""
        import sys
        from unittest.mock import patch

        # The ACS handler does `from cogtrix_core.api.saml.provider import process_saml_response`
        # inside the function body; we must hide the provider module so that import
        # raises ImportError (as it would when python3-saml is not installed).
        with patch.dict(sys.modules, {"cogtrix_core.api.saml.provider": None}):
            r = configured_client.post("/api/v1/saml/acs", data={"SAMLResponse": "bm90dmFsaWQ="})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "SAML_NOT_INSTALLED"

    def test_acs_is_rate_limited_after_five_requests(self, configured_client):
        """POST /acs should reject brute-force traffic after the 5/minute route limit."""
        import uuid
        from types import SimpleNamespace
        from unittest.mock import patch

        # Each call must yield a fresh assertion_id; the nonce cache (ADR-0054,
        # PR #1531) would otherwise reject the 2nd+ requests as replays before
        # the rate-limit gate is hit.
        def _fresh_assertion(*_args, **_kwargs):
            return SimpleNamespace(assertion_id=f"ass-{uuid.uuid4()}")

        with patch(
            "cogtrix_core.api.routes.saml._provision_user", return_value=(object(), "token")
        ):
            with patch(
                "cogtrix_core.api.saml.provider.process_saml_response",
                side_effect=_fresh_assertion,
            ):
                for _ in range(5):
                    r = configured_client.post(
                        "/api/v1/saml/acs",
                        data={"SAMLResponse": "bm90dmFsaWQ="},
                    )
                    assert r.status_code == 200, r.text

                r = configured_client.post(
                    "/api/v1/saml/acs",
                    data={"SAMLResponse": "bm90dmFsaWQ="},
                )
                assert r.status_code == 429, r.text


# ---------------------------------------------------------------------------
# Security regression — Host header injection (#285)
# ---------------------------------------------------------------------------


class TestBuildRequestDataHostPinning:
    """_build_request_data must derive http_host from sp_acs_url, not from
    the incoming request Host header (CVE class: Host header injection)."""

    def test_http_host_comes_from_config_not_request_header(self):
        """http_host is pinned to sp_acs_url even when a spoofed Host header is present."""
        from unittest.mock import MagicMock

        from cogtrix_core.api.routes.saml import _build_request_data

        config = _make_config(sp_acs_url="https://sp.example.com/api/v1/saml/acs")

        request = MagicMock()
        request.headers = {"host": "attacker.evil.com"}
        request.url.path = "/api/v1/saml/acs"
        request.url.port = None
        request.url.scheme = "https"
        request.query_params = {}

        data = _build_request_data(request, config)

        assert data["http_host"] == "sp.example.com"
        assert data["https"] == "on"
        assert data["server_port"] == "443"

    def test_http_host_includes_nonstandard_port(self):
        """Non-standard port in sp_acs_url is included in http_host."""
        from unittest.mock import MagicMock

        from cogtrix_core.api.routes.saml import _build_request_data

        config = _make_config(sp_acs_url="https://sp.example.com:8443/api/v1/saml/acs")

        request = MagicMock()
        request.headers = {"host": "attacker.evil.com:9999"}
        request.url.path = "/api/v1/saml/acs"
        request.url.port = 9999
        request.url.scheme = "http"
        request.query_params = {}

        data = _build_request_data(request, config)

        assert data["http_host"] == "sp.example.com:8443"
        assert data["https"] == "on"
        assert data["server_port"] == "8443"

    def test_http_scheme_from_config_not_request(self):
        """https flag is derived from sp_acs_url scheme, not the request scheme."""
        from unittest.mock import MagicMock

        from cogtrix_core.api.routes.saml import _build_request_data

        config = _make_config(sp_acs_url="https://sp.example.com/api/v1/saml/acs")

        request = MagicMock()
        request.headers = {"host": "sp.example.com"}
        request.url.path = "/api/v1/saml/acs"
        request.url.port = None
        request.url.scheme = "http"  # attacker-downgraded scheme
        request.query_params = {}

        data = _build_request_data(request, config)

        # Must use config's https, not the request's http
        assert data["https"] == "on"
