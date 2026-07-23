"""FastAPI application factory for the Cogtrix API.

Creates and configures the FastAPI app instance:
- Mounts all routers under /api/v1
- Mounts WebSocket routers under /ws/v1
- Configures CORS for the React dev server and production origin
- Attaches lifespan context manager (startup/shutdown hooks)
- Registers global exception handlers for consistent error envelopes

Usage:
    uvicorn src.api.app:app --reload --port 8000

Environment variables:
    COGTRIX_JWT_SECRET       — required; JWT signing secret (min 32 chars)
    COGTRIX_TRUSTED_PROXY_CIDRS — optional comma-separated reverse-proxy CIDR allowlist
    COGTRIX_CORS_ORIGINS     — comma-separated allowed origins (overrides defaults)
    COGTRIX_API_HOST         — bind host (default 0.0.0.0)
    COGTRIX_API_PORT         — bind port (default 8000)
    OTEL_SERVICE_NAME        — optional; OpenTelemetry service name (default cogtrix)
    OTEL_EXPORTER_OTLP_ENDPOINT — optional; OTLP gRPC collector endpoint

All environment variables are read via src.config to participate in the
hierarchical configuration system.  Never read os.environ directly in this file.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.rate_limit import (
    configure_rate_limit_backend,
    configure_rate_limits,
    configure_trusted_proxy_cidrs,
    current_backend_label,
    limiter,
    reset_rate_limits,
)
from src.api.routes import (
    admin,
    agents,
    assistant,
    auth,
    billing,
    config,
    cross_workspace,
    enforcement,
    health,
    jit,
    ldap,
    mcp,
    memory,
    messages,
    metrics,
    organizations,
    plans,
    rag,
    saml,
    scim,
    sessions,
    system,
    tasks,
    teams,
    tools,
    usage,
    users,
    workflows,
    workspaces,
)
from src.api.schemas.common import APIError, APIResponse
from src.api.telemetry import setup_telemetry

log = logging.getLogger("cogtrix.api")

# ---------------------------------------------------------------------------
# Shutdown state tracking
# ---------------------------------------------------------------------------
_shutdown_initiated: bool = False


# Synchronous signal handler for SIGTERM (must be sync, not async)
def _handle_sigterm_for_api_sync(signum: int, frame: Any) -> None:
    """Synchronous handler for SIGTERM signal during graceful shutdown.

    This function is called when SIGTERM is received. It sets the shutdown
    flag and logs the shutdown start. The actual shutdown happens in the
    lifespan shutdown section. This must be a synchronous function because
    signal handlers in Python must be sync; async functions cannot be used
    directly as signal handlers.
    """
    global _shutdown_initiated
    if _shutdown_initiated:
        log.warning("Received second SIGTERM, force exiting...")
        # Force exit after a brief delay
        import asyncio

        # Schedule the async cleanup in the event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_async_sigterm_cleanup())
        except Exception:
            pass  # If we can't schedule, just exit
        os._exit(1)

    _shutdown_initiated = True
    log.info("SIGTERM received, initiating graceful shutdown...")


async def _async_sigterm_cleanup() -> None:
    """Async cleanup for SIGTERM shutdown sequence."""
    global _shutdown_initiated
    # This will be called from _handle_sigterm_for_api_sync
    # The actual shutdown sequence is handled by lifespan
    # This is a placeholder for any async cleanup that might be needed


# Register SIGTERM handler at module level (called from lifespan)
def _register_sigterm_handler() -> None:
    """Register SIGTERM signal handler for graceful shutdown.

    Called from lifespan startup to ensure the handler is registered
    in the main process (not in worker processes where signal handling
    may behave differently).

    Skipped in test environments (PYTEST_CURRENT_TEST is set): the handler
    intercepts SIGTERM without terminating the process, which causes test
    runners to hang when they send SIGTERM for teardown or timeouts, and
    corrupts in-flight tests by leaving _shutdown_initiated=True.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        # Only register in the main process if not already registered
        # signal.SIG_DFL is the default handler (does nothing, exits with code 0)
        current_handler = signal.getsignal(signal.SIGTERM)
        if current_handler in (signal.SIG_DFL, signal.SIG_IGN):
            signal.signal(signal.SIGTERM, _handle_sigterm_for_api_sync)
            log.debug("SIGTERM handler registered for API")
    except (OSError, ValueError) as exc:
        # SIGTERM not available on some platforms (e.g., Windows)
        log.debug(f"Could not register SIGTERM handler: {exc}")


# ---------------------------------------------------------------------------
# Allowed CORS origins
# ---------------------------------------------------------------------------

# Localhost-only fallback used when Config cannot be loaded. A production
# origin is NOT shipped here on purpose (#2059): a misconfigured prod must
# fail loudly (browser blocks) rather than silently half-allow a placeholder.
# Real origins are set via api.cors_origins / COGTRIX_CORS_ORIGINS / Helm.
_DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:5173",  # Vite React dev server
    "http://localhost:3000",  # Create-React-App dev server (fallback)
]


def _get_cors_origins() -> list[str]:
    """Return the list of allowed CORS origins via the Config hierarchy (#2059).

    Origins resolve through ``Config`` (CLI → env ``COGTRIX_CORS_ORIGINS`` →
    ``api.cors_origins`` config file → default), honouring the documented
    precedence instead of reading ``os.environ`` directly. Falls back to the
    localhost-only default if Config cannot be loaded.
    """
    try:
        from src.config import get_cached_config

        # #2101: reuse the process-wide resolved config (env read once).
        origins = get_cached_config().api.cors_origins
        if origins:
            return list(origins)
    except Exception:  # noqa: BLE001 — never let config failure break CORS setup
        log.warning("Failed to resolve CORS origins from Config; using localhost default")
    return _DEFAULT_CORS_ORIGINS


# ---------------------------------------------------------------------------
# Lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: startup and shutdown hooks.

    Startup:
        - Validate COGTRIX_JWT_SECRET is set and meets length requirements.
        - Create DB tables via SQLAlchemy metadata (dev convenience; production uses Alembic).
        - Load the Cogtrix Config and attach to app.state.
        - Initialize the tool registry.
        - Log API version and startup summary.

    Shutdown:
        - Dispose the async DB engine connection pool.
        - Flush any pending log records.
    """
    # ---- startup ----
    # Reset shutdown flag — required for tests that spin up multiple TestClient
    # instances in the same process (module-level flag survives between instances).
    global _shutdown_initiated
    _shutdown_initiated = False

    # Set up Cogtrix logging from env vars (set by __main__.py or docker env).
    # This is a no-op when __main__.py already called setup_logging() — the
    # logger will already have handlers and this just ensures coverage for
    # bare ``uvicorn src.api.app:app`` invocations.
    try:
        from src.logging_config import setup_logging

        log_file = os.environ.get("COGTRIX_API_LOG_FILE")
        debug = bool(os.environ.get("COGTRIX_DEBUG"))
        stream_output = bool(os.environ.get("COGTRIX_LOG_STREAM"))
        if stream_output and log_file is None:
            # --debug without --log-file: route DEBUG/INFO→stdout, WARNING+→stderr
            setup_logging(log_file=None, debug=debug, verbose=debug, stream_output=True)
        elif log_file is not None or debug:
            if log_file is None:
                log_file = "cogtrix-api.log"
            setup_logging(log_file=log_file, debug=debug, console_output=True, verbose=debug)
        else:
            # Ensure the cogtrix logger is at INFO level even without --log/--debug
            # so that log records propagate to the WebSocket log stream handler.
            cogtrix_logger = logging.getLogger("cogtrix")
            if cogtrix_logger.level == logging.WARNING or cogtrix_logger.level == 0:
                cogtrix_logger.setLevel(logging.INFO)
    except Exception:
        pass  # logging setup is best-effort

    log.info("Cogtrix API starting up")

    # Validate and snapshot JWT secret for auth helpers. Resolve via the #2103
    # _FILE convention so COGTRIX_JWT_SECRET_FILE=/run/secrets/jwt_secret works.
    from src.config import secret_from_env_or_file

    jwt_secret = secret_from_env_or_file("COGTRIX_JWT_SECRET") or ""
    from src.api.auth import configure_dummy_password_hash, configure_jwt_secret

    configure_jwt_secret(jwt_secret)
    log.info("JWT secret validated")

    # Pre-compute the constant-time dummy bcrypt hash so the first
    # no-such-user login after a fresh boot doesn't leak ~150 ms of timing
    # signal (forge audit B5, second-order to H3).
    configure_dummy_password_hash()

    # ── Rate-limit + trusted-proxy config (#1879 Slice A) ──────────────
    # Precedence: env var > .cogtrix.yaml > built-in default. We tolerate
    # a missing/unreadable config file (matches the pre-#1879 behaviour
    # where the limit was hardcoded and the only knob was the env var) —
    # but a malformed ``api:`` block surfaces cleanly as a startup error
    # via ``ConfigError`` / ``ValueError`` rather than a 500 on the
    # first request.
    try:
        from src.config import APIConfig as _APIConfig
        from src.config import get_cached_config as _get_cached_config

        try:
            # #2101: reuse the process-wide resolved config (env read once).
            _app_cfg_api = _get_cached_config().api
        except Exception as _cfg_exc:  # noqa: BLE001
            log.info(
                "API rate-limit / trusted-proxy config falling back to "
                "built-in defaults (could not load .cogtrix.yaml: %s)",
                _cfg_exc,
            )
            _app_cfg_api = _APIConfig()

        _rate_limits_cfg = dict(_app_cfg_api.rate_limits)
        for _name in list(_rate_limits_cfg):
            _env_val = os.environ.get(f"COGTRIX_RATE_LIMIT_{_name.upper()}")
            if _env_val:
                _rate_limits_cfg[_name] = _env_val
        _default_spec = _rate_limits_cfg.pop(
            "default",
            os.environ.get("COGTRIX_RATE_LIMIT_DEFAULT", "120/minute"),
        )
        configure_rate_limits(default=_default_spec, per_route=_rate_limits_cfg)

        # Trusted-proxy CIDRs: env-var wins; YAML next; empty otherwise.
        _trusted_env = os.environ.get("COGTRIX_TRUSTED_PROXY_CIDRS")
        if _trusted_env is not None:
            configure_trusted_proxy_cidrs(_trusted_env)
        elif _app_cfg_api.trusted_proxy_cidrs:
            configure_trusted_proxy_cidrs(_app_cfg_api.trusted_proxy_cidrs)
        else:
            configure_trusted_proxy_cidrs(None)

        # ── Rate-limit storage backend (#1879 Slice B) ─────────────
        # COGTRIX_REDIS_URL > api.redis_url (YAML) > None (in-memory).
        # When unset, the per-process MemoryStorage default is used —
        # correct for single-node deployments but jitters under
        # horizontal scaling; we log the chosen backend so operators
        # can see what they got at startup.
        _redis_env = os.environ.get("COGTRIX_REDIS_URL")
        if _redis_env is not None and _redis_env.strip():
            _redis_url = _redis_env.strip()
        else:
            _redis_url = (_app_cfg_api.redis_url or "").strip() or None
        try:
            configure_rate_limit_backend(redis_url=_redis_url)
        except ImportError as exc:
            # ``limits.storage_from_string('redis://...')`` lazy-imports
            # the ``redis`` package. Operators who set a Redis URL
            # without installing ``cogtrix[redis]`` get a clear error
            # at startup rather than a 500 on the first request.
            raise RuntimeError(
                f"COGTRIX_REDIS_URL / api.redis_url is set but the 'redis' "
                f"package is not installed. Install with: "
                f"pip install cogtrix[api,redis]  (original error: {exc})"
            ) from exc
        if _redis_url is None:
            log.info(
                "Rate-limit backend: in-memory (per-process). "
                "Set COGTRIX_REDIS_URL or api.redis_url for multi-replica "
                "deployments."
            )
        else:
            log.info(
                "Rate-limit backend: shared counter at %s",
                current_backend_label(),
            )

        # #1879 follow-up: ``configure_rate_limit_backend`` also rebuilt
        # the module-level SlowAPI ``limiter`` with the same backend so
        # the global 120/min blunt guard shares state across replicas.
        # ``SlowAPIMiddleware.dispatch`` reads ``app.state.limiter`` on
        # every request, so reassigning here lets the rebuilt limiter
        # take effect on the next inbound request — no middleware
        # recreation required. Access via module attribute so we pick
        # up the rebuilt instance, not the one imported at module load.
        from src.api import rate_limit as _rl_module

        app.state.limiter = _rl_module.limiter
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    # Safety check: STRIPE_ALLOW_UNSIGNED must not be enabled in production.
    _env = os.environ.get("COGTRIX_ENV", "development").lower()
    if os.environ.get("STRIPE_ALLOW_UNSIGNED") == "1" and _env == "production":
        raise RuntimeError(
            "STRIPE_ALLOW_UNSIGNED=1 is not permitted in production. "
            "Remove the variable or set COGTRIX_ENV to a non-production value."
        )

    # Create database tables (idempotent; no-op when tables exist).
    #
    # CRITICAL: resolve the engine module via ``sys.modules`` rather
    # than ``import src.api.db.engine``.  The plain import statement
    # reads ``src.api.db.engine`` as an attribute on the parent
    # ``src.api.db`` package — and that attribute is mutated by
    # ``tests/test_api_db_url_resolution.py::_reimport_engine`` when it
    # calls ``importlib.import_module('src.api.db.engine')`` (Python's
    # import machinery sets the new submodule object on the parent
    # package).  The polluter's teardown restores
    # ``sys.modules['src.api.db.engine']`` but does NOT restore the
    # parent-package attribute, so post-teardown the plain ``import``
    # returns the orphaned re-imported module (empty ``Base.metadata``,
    # fresh ``_engine``) while the package-level ``get_db`` / models
    # still bind to the original module — table creation lands on one
    # engine and route queries hit another, producing
    # ``OperationalError: no such table: users``.
    #
    # Reading directly from ``sys.modules`` bypasses the parent
    # attribute and always returns the module that holds the
    # registered models and the get_db / session factory closures.
    import src.api.db.models  # noqa: F401 — registers all ORM model classes

    _engine_mod = sys.modules["src.api.db.engine"]
    Base = _engine_mod.Base
    validate_connection = _engine_mod.validate_connection
    engine = _engine_mod.engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables ready")
    await validate_connection()
    log.info("Database connection validated")

    # Stash the captured module reference on app.state so the shutdown
    # block below resets the exact same cache that startup populated.
    app.state._db_engine_module = _engine_mod

    # Load Cogtrix config
    cfg = None
    try:
        from src.config import get_cached_config

        # #2101: resolve once and reuse process-wide. This seeds the cache at
        # startup so every later runtime path (CORS, RAG ingest, DB-URL resolver,
        # tools) reuses the same instance instead of re-reading os.environ.
        cfg = get_cached_config()
        app.state.config = cfg
        log.info(
            "Config loaded (provider=%s, file=%s)",
            getattr(cfg, "provider", "unknown"),
            cfg.config_file_path or "defaults",
        )
    except Exception as exc:
        log.warning("Could not load Cogtrix config: %s", exc)
        app.state.config = None

    # Propagate data_dir to env so API route helpers (_get_uploads_dir)
    # use the same root as configure_rag_tool() when COGTRIX_DATA_DIR
    # is not explicitly set.
    if cfg is not None and not os.environ.get("COGTRIX_DATA_DIR"):
        os.environ["COGTRIX_DATA_DIR"] = cfg.data_dir

    # Initialize OIDC validator early so startup fails fast on insecure or
    # malformed SSO configuration.
    try:
        if cfg is not None and cfg.oidc_enabled:
            if not cfg.oidc_issuer or not cfg.oidc_audience:
                raise RuntimeError("OIDC is enabled but issuer/audience are missing")
            from src.api.oidc import OIDCConfig, configure_oidc

            configure_oidc(
                OIDCConfig(
                    issuer=cfg.oidc_issuer,
                    audience=cfg.oidc_audience,
                    jwks_uri=cfg.oidc_jwks_uri,
                    allow_insecure_oidc=cfg.oidc_allow_insecure_oidc,
                    production_mode=not cfg.debug,
                    role_claim=cfg.oidc_role_claim,
                    default_role=cfg.oidc_default_role,
                )
            )
            log.info("OIDC validator initialized (issuer=%s)", cfg.oidc_issuer)
    except Exception as exc:
        log.warning("Could not initialize OIDC validator: %s", exc)
        raise

    # Initialize agent registry from config + AGENTS.md
    try:
        from src.agent import registry as _agent_registry
        from src.agent.agents_md import load_default_agents as _load_agents_md

        _agents_md = _load_agents_md()
        if cfg is not None:
            _agent_registry.load_from_config(cfg)
        _agent_registry.merge_from_agents_md(_agents_md)
        log.debug("Agent registry initialized (%d agent(s))", len(_agent_registry.list_agents()))
    except Exception as exc:
        log.warning("Could not initialize agent registry: %s", exc)

    # Initialize tool registry
    try:
        from src.registry import ToolRegistry

        tool_registry = ToolRegistry()
        tool_registry.load_all_tools(config=cfg)
        app.state.tool_registry = tool_registry
        log.info("Tool registry initialized (%d tools discovered)", len(tool_registry.tools))
    except Exception as exc:
        log.warning("Could not initialize tool registry: %s", exc)
        app.state.tool_registry = None

    # Connect MCP servers and register their tools
    app.state.mcp_manager = None
    app.state.pinned_mcp_tool_names = set()  # type: ignore[var-annotated]
    if cfg is not None and getattr(cfg, "mcp_servers", None):
        try:
            from src.mcp_client import MCP_AVAILABLE, MCPManager, MCPServerConfig

            if MCP_AVAILABLE:
                mcp_manager = MCPManager()
                # Shared with cogtrix.py — see ``src/mcp_client.py`` for the
                # KNOWN vs DOC_ONLY split rationale.
                from src.mcp_client import KNOWN_MCP_FIELDS

                mcp_configs = []
                for _name, _srv_cfg in cfg.mcp_servers.items():
                    _filtered = {k: v for k, v in _srv_cfg.items() if k in KNOWN_MCP_FIELDS}
                    mcp_configs.append(MCPServerConfig(name=_name, **_filtered))

                _mcp_pin_map = {c.name: c.pin for c in mcp_configs}
                tool_registry = app.state.tool_registry
                mcp_tools = mcp_manager.connect_all(
                    mcp_configs,
                    builtin_tool_names=set((tool_registry.tools if tool_registry else {}).keys()),
                )
                # Mirror the discovered tools into the live registry via the
                # shared helper so startup and the runtime /mcp routes register
                # tools identically (#2151/#2153).
                from src.api.mcp_runtime import register_mcp_tools

                register_mcp_tools(
                    tool_registry,
                    app.state.pinned_mcp_tool_names,
                    mcp_tools,
                    _mcp_pin_map,
                )

                app.state.mcp_manager = mcp_manager
                log.info(
                    "MCP: connected %d server(s), %d tool(s) registered (%d pinned)",
                    len(mcp_configs),
                    len(mcp_tools),
                    len(app.state.pinned_mcp_tool_names),
                )
                if len(app.state.pinned_mcp_tool_names) > 50:
                    log.warning(
                        "MCP: %d tools are pinned (pin=True). "
                        "This adds ~%d tokens of overhead per API turn. "
                        "Consider setting pin: false for large MCP servers.",
                        len(app.state.pinned_mcp_tool_names),
                        len(app.state.pinned_mcp_tool_names) * 300,
                    )
        except Exception as exc:
            log.warning("Could not initialize MCP servers: %s", exc)

    # Configure tool modules that rely on provider/model settings from config.
    if cfg is not None:
        try:
            from src.tools.configure import configure_rag_tool

            configure_rag_tool(cfg)
            log.debug("RAG tool configured with embedding settings from config")
        except Exception as exc:
            log.warning("Could not configure RAG tool: %s", exc)

        try:
            from src.tools.configure import configure_delegate_tool

            configure_delegate_tool(cfg)
            log.debug("Delegate tool configured from config")
        except Exception as exc:
            log.warning("Could not configure delegate tool: %s", exc)

        try:
            from src.tools.configure import configure_deep_think_tool

            configure_deep_think_tool(cfg)
            log.debug("Deep-think tool configured from config")
        except Exception as exc:
            log.warning("Could not configure deep-think tool: %s", exc)

        try:
            from src.tools.configure import configure_python_exec_tool

            configure_python_exec_tool(cfg)
            log.debug(
                "Python exec tool configured from config (enable_datascience_modules=%s)",
                cfg.enable_datascience_modules,
            )
        except Exception as exc:
            log.warning("Could not configure python_exec tool: %s", exc)

    # Initialize session registry (Phase 2)
    try:
        from src.api.session_bridge import ApiSessionRegistry

        session_registry = ApiSessionRegistry(app.state)
        app.state.session_registry = session_registry
        session_registry.start_eviction_loop()
        log.info("Session registry initialized")
    except Exception as exc:
        log.warning("Could not initialize session registry: %s", exc)
        app.state.session_registry = None

    # Initialize workflow registry
    try:
        from src.assistant.workflows import WorkflowRegistry

        _data_dir = os.environ.get("COGTRIX_DATA_DIR", "data")
        app.state.workflow_registry = WorkflowRegistry(data_dir=_data_dir)
        log.info("Workflow registry initialized")
    except Exception as exc:
        log.warning("Could not initialize workflow registry: %s", exc)
        app.state.workflow_registry = None

    # Placeholders for Phase 2+ state
    app.state.assistant_service = None
    app.state.message_scheduler = None
    app.state.deferral_manager = None
    app.state.guardrail_pipeline = None
    app.state.knowledge_store = None

    # API content guardrails (#2056). The assistant/messaging path runs a
    # GuardrailPipeline; the API chat path historically had none. Build it here
    # so the turn runner can screen input and sanitize output. Default OFF: the
    # pipeline is only constructed when ``api.guardrails.enabled`` is truthy, so
    # existing deployments are unchanged. If the operator DID enable guardrails
    # but construction fails, we re-raise (fail closed) rather than silently
    # serve unprotected — a security control must not degrade silently.
    api_guardrails: dict[str, Any] = (
        dict(getattr(getattr(cfg, "api", None), "guardrails", {}) or {}) if cfg is not None else {}
    )
    if api_guardrails.get("enabled"):
        from pathlib import Path as _Path

        from src.assistant.guardrails import GuardrailPipeline

        if "violations_persist_path" not in api_guardrails:
            _data_dir = getattr(cfg, "data_dir", "data")
            api_guardrails["violations_persist_path"] = str(
                _Path(_data_dir) / "api" / "violations.json"
            )
        # Optional LLM judge — only when enabled AND a model is named (the API has
        # no guaranteed default chat LLM at startup to fall back on).
        judge_llm = None
        judge_cfg = api_guardrails.get("llm_judge", {}) or {}
        if judge_cfg.get("enabled", False):
            judge_model = judge_cfg.get("model")
            if judge_model:
                from src.assistant.knowledge import create_extraction_llm

                judge_llm = create_extraction_llm(judge_model, cfg)
            else:
                log.warning(
                    "api.guardrails.llm_judge.enabled is true but no model is set; "
                    "the LLM judge will be disabled on the API path."
                )
        app.state.guardrail_pipeline = GuardrailPipeline(
            config={"guardrails": api_guardrails}, llm=judge_llm
        )
        log.info("API content guardrails enabled")

    # Auto-start assistant if configured
    if cfg is not None and app.state.tool_registry is not None:
        assistant_cfg = getattr(cfg, "assistant_config", None) or {}
        if assistant_cfg.get("auto_start", False):
            try:
                from src.api.assistant_lifecycle import create_and_start_assistant

                svc = await create_and_start_assistant(cfg, app.state.tool_registry)
                app.state.assistant_service = svc
                log.info("Assistant service auto-started")
            except Exception as exc:
                log.warning("Assistant auto-start failed: %s", exc)

    # Register SIGTERM handler for graceful shutdown
    _register_sigterm_handler()

    log.info("Cogtrix API startup complete")
    yield  # application runs here

    # ---- shutdown ----
    log.info("Cogtrix API shutting down")

    # Mark shutdown as initiated to stop accepting new connections
    _shutdown_initiated = True
    log.info("Shutdown initiated, stopping new connection acceptance")

    # Attempt to drain active WebSocket connections (30s timeout)
    # Note: HTTP connection draining requires uvicorn Server instance access,
    # which isn't available through the FastAPI lifespan. The middleware
    # (_request_context_middleware) now checks _shutdown_initiated to reject
    # new HTTP requests, effectively draining connections over time.
    log.info("Draining WebSocket sessions (30s timeout)...")

    # Give active WebSocket sessions time to complete
    try:
        import asyncio as _asyncio

        await _asyncio.sleep(0.5)  # Allow brief time for in-flight messages
    except Exception as exc:
        log.warning("Error during connection drain wait: %s", exc)

    # Stop assistant service if running
    try:
        svc = getattr(app.state, "assistant_service", None)
        if svc is not None:
            import asyncio as _asyncio

            from src.api.assistant_lifecycle import shutdown_assistant_sync

            await _asyncio.to_thread(shutdown_assistant_sync, svc)
            app.state.assistant_service = None
            log.info("Assistant service stopped")
    except Exception as exc:
        log.warning("Error stopping assistant service: %s", exc)

    # Drain WebSocket sessions with 30-second timeout
    # In test environments, skip stop_eviction_loop() — it calls save_all() which
    # creates aiosqlite connections that strand in the TestClient portal event loop,
    # blocking thread.join() and triggering pytest-timeout on every API test.
    import os as _os_shutdown

    _in_test = bool(_os_shutdown.environ.get("PYTEST_CURRENT_TEST"))
    try:
        registry = getattr(app.state, "session_registry", None)
        if registry is not None:
            if _in_test:
                # Cancel eviction task only — no DB writes during test teardown
                eviction_task = getattr(registry, "_eviction_task", None)
                if eviction_task and not eviction_task.done():
                    eviction_task.cancel()
            else:
                drain_timeout = 30.0
                log.info(f"Draining WebSocket sessions (timeout: {drain_timeout}s)...")
                await registry.stop_eviction_loop()
                log.info("WebSocket sessions drained and sessions saved")
    except Exception as exc:
        log.warning("Error draining WebSocket sessions: %s", exc)

    # Wait for in-flight APP background tasks to complete.
    # Only wait for tasks the app created (named tasks); skip anyio/starlette
    # infrastructure tasks which run for the lifetime of the portal and must
    # not be cancelled here.
    import asyncio as _asyncio
    import os as _os

    _in_test = bool(_os.environ.get("PYTEST_CURRENT_TEST"))
    # In tests, skip entirely — anyio portal tasks would be misidentified as
    # app work and cancelling them breaks the portal teardown.
    if not _in_test:
        _drain_timeout = 60.0
        log.info(
            "Waiting for in-flight background tasks to complete (%ss timeout)...", _drain_timeout
        )
        try:
            await _asyncio.sleep(0.1)
            # Only consider tasks with names that the app explicitly set.
            _app_task_names = {"session-eviction", "compression", "background"}
            pending_tasks = [
                t
                for t in _asyncio.all_tasks()
                if t is not _asyncio.current_task()
                and any(n in (t.get_name() or "") for n in _app_task_names)
            ]
            if pending_tasks:
                log.debug("Found %d pending app task(s)", len(pending_tasks))
                done, pending = await _asyncio.wait(
                    pending_tasks,
                    timeout=_drain_timeout,
                    return_when=_asyncio.ALL_COMPLETED,
                )
                if pending:
                    log.warning(
                        "%d task(s) did not complete within %ss, cancelling...",
                        len(pending),
                        _drain_timeout,
                    )
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except _asyncio.CancelledError:
                            pass
        except Exception as exc:
            log.warning("Error waiting for background tasks: %s", exc)

    _mcp_mgr = getattr(app.state, "mcp_manager", None)
    if _mcp_mgr is not None:
        try:
            _mcp_mgr.close_all()
        except Exception as exc:
            log.warning("Error stopping MCP manager: %s", exc)

    # Dispose the async engine connection pool (only if it was built),
    # then reset the module-level cache so the next lifespan starts
    # with a fresh engine bound to its own event loop.  Without the
    # reset, consecutive ``with TestClient(...)`` blocks reuse the
    # disposed engine whose aiosqlite worker threads are still tied to
    # the previous (now-closed) loop — any in-flight callback then
    # fires on the closed loop and surfaces as
    # ``PytestUnhandledThreadExceptionWarning: RuntimeError: Event loop
    # is closed``.
    #
    # CRITICAL: uses the engine module reference captured at startup
    # (``app.state._db_engine_module``), not a fresh ``import``.  If a
    # test earlier in the run re-imported ``src.api.db.engine`` (e.g.
    # ``test_api_db_url_resolution._reimport_engine``), a fresh
    # ``import`` here would resolve a different module whose ``_engine``
    # is ``None`` — the dispose-and-reset branch would be silently
    # skipped and the ORIGINAL module's engine would survive across
    # TestClient cycles, breaking per-test data isolation in tests that
    # share ``:memory:`` SQLite via the global cache.
    import asyncio as _asyncio

    _db_engine_mod = getattr(app.state, "_db_engine_module", None)
    if _db_engine_mod is not None and _db_engine_mod._engine is not None:
        try:
            await _db_engine_mod._engine.dispose()
            # One event-loop tick after dispose gives aiosqlite worker
            # callbacks a chance to land before the loop closes.
            await _asyncio.sleep(0)
            log.info("Database engine disposed")
        except Exception as exc:
            log.warning("Error disposing DB engine: %s", exc)
        finally:
            _db_engine_mod._engine = None
            _db_engine_mod._session_factory = None

    log.info("Cogtrix API shutdown complete")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


_STATUS_CODE_MAP: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "TOO_MANY_REQUESTS",
    501: "NOT_IMPLEMENTED",
    503: "SERVICE_UNAVAILABLE",
}


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert FastAPI/Starlette HTTPException to the standard APIResponse envelope."""
    http_exc: HTTPException = exc  # type: ignore[assignment]
    detail = http_exc.detail
    if isinstance(detail, dict):
        code = detail.get("code", "INTERNAL_ERROR")
        message = detail.get("message", str(detail))
    else:
        # Infer a sensible code from the HTTP status when no structured detail
        # was provided — avoids returning code="INTERNAL_ERROR" for 4xx responses.
        code = _STATUS_CODE_MAP.get(http_exc.status_code, "INTERNAL_ERROR")
        message = str(detail) if detail else "An unexpected server error occurred."

    envelope = APIResponse(
        data=None,
        error=APIError(code=code, message=message),
    )
    return JSONResponse(
        status_code=http_exc.status_code,
        content=envelope.model_dump(mode="json"),
    )


async def _validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert Pydantic / FastAPI validation errors to the standard error envelope."""
    from src.api.validation import translate_validation_errors

    validation_exc = exc  # type: ignore[assignment]
    details: dict = {}
    if hasattr(validation_exc, "errors"):
        try:
            raw_errors = validation_exc.errors()  # type: ignore[union-attr]
            details = translate_validation_errors(raw_errors)
        except Exception as exc_inner:
            log.debug("Validation error translation failed: %s", exc_inner)

    envelope = APIResponse(
        data=None,
        error=APIError(
            code="VALIDATION_ERROR",
            message="Request body or query parameter validation failed.",
            details=details or None,
        ),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=envelope.model_dump(mode="json"),
    )


async def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert unhandled exceptions to the standard INTERNAL_ERROR envelope.

    Logs the traceback at ERROR level.  Never leaks stack traces to clients.
    """
    log.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    envelope = APIResponse(
        data=None,
        error=APIError(
            code="INTERNAL_ERROR",
            message="An unexpected server error occurred.",
        ),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=envelope.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns a fully wired FastAPI app ready for uvicorn.  Called once at
    module load time to produce the ``app`` singleton used by uvicorn.
    """
    app = FastAPI(
        title="Cogtrix API",
        description=(
            "REST + WebSocket API for the Cogtrix AI assistant platform. "
            "Powers the React web frontend with full access to sessions, "
            "messages, tools, memory, MCP servers, assistant mode, RAG, and system controls."
        ),
        version="1.1.0",
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        lifespan=lifespan,
    )

    # Rate limiting — reset counters on startup; keep SlowAPI for global 120/min guard
    reset_rate_limits()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # CORS — must be added before routers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def _request_context_middleware(request: Request, call_next: Any) -> Response:
        """Attach a request-scoped logging context to each HTTP request."""
        from src.logging_config import clear_request_id, new_request_id

        # Refuse new connections during shutdown
        if _shutdown_initiated:
            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=503,
                content={"detail": "Service shutting down. Please retry later."},
                headers={"Retry-After": "30"},
            )

        request_id = new_request_id()
        try:
            response = await call_next(request)
            response.headers.setdefault("X-Request-ID", request_id)
            return response
        finally:
            clear_request_id()

    # Exception handlers
    app.add_exception_handler(HTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _generic_exception_handler)  # type: ignore[arg-type]

    # OpenTelemetry is opt-in via the OTLP endpoint env var to keep dev startup no-op.
    otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name = os.environ.get("OTEL_SERVICE_NAME", "cogtrix")
    if setup_telemetry(otel_service_name, otel_endpoint):
        try:
            FastAPIInstrumentor().instrument_app(app)
            log.info(
                "OpenTelemetry tracing enabled (service=%s, endpoint=%s)",
                otel_service_name,
                otel_endpoint,
            )
        except Exception as exc:
            log.warning("Could not instrument FastAPI for OpenTelemetry: %s", exc)

    # REST routers — all prefixed /api/v1
    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(metrics.router, prefix=api_prefix)
    app.include_router(auth.router, prefix=api_prefix)
    app.include_router(agents.router, prefix=api_prefix)
    app.include_router(sessions.router, prefix=api_prefix)
    app.include_router(messages.router, prefix=api_prefix)
    app.include_router(memory.router, prefix=api_prefix)
    app.include_router(tasks.router, prefix=api_prefix)
    app.include_router(tools.router, prefix=api_prefix)
    app.include_router(config.router, prefix=api_prefix)
    app.include_router(mcp.router, prefix=api_prefix)
    app.include_router(assistant.router, prefix=api_prefix)
    app.include_router(rag.router, prefix=api_prefix)
    app.include_router(system.router, prefix=api_prefix)
    app.include_router(users.router, prefix=api_prefix)
    app.include_router(organizations.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)
    app.include_router(workflows.router, prefix=api_prefix)
    app.include_router(saml.router, prefix=api_prefix)
    app.include_router(ldap.router, prefix=api_prefix)
    app.include_router(teams.router, prefix=api_prefix)
    app.include_router(jit.router, prefix=api_prefix)
    app.include_router(workspaces.router, prefix=api_prefix)
    app.include_router(cross_workspace.router, prefix=api_prefix)
    app.include_router(plans.router, prefix=api_prefix)
    app.include_router(plans.org_plan_router, prefix=api_prefix)
    app.include_router(usage.router, prefix=api_prefix)
    app.include_router(enforcement.router, prefix=api_prefix)
    app.include_router(billing.router, prefix=api_prefix)
    # SCIM is mounted at /scim/v2/ (no api_prefix — standard SCIM path)
    app.include_router(scim.router)

    # WebSocket routers — prefixed /ws/v1 (embedded in route modules)
    app.include_router(messages.ws_router)
    app.include_router(system.ws_router)

    return app


# Module-level singleton — uvicorn targets `src.api.app:app`
app = create_app()
