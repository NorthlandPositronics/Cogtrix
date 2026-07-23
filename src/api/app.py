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
    COGTRIX_CORS_ORIGINS     — comma-separated allowed origins (overrides defaults)
    COGTRIX_API_HOST         — bind host (default 0.0.0.0)
    COGTRIX_API_PORT         — bind port (default 8000)

All environment variables are read via src.config to participate in the
hierarchical configuration system.  Never read os.environ directly in this file.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import (
    assistant,
    auth,
    config,
    health,
    mcp,
    memory,
    messages,
    rag,
    sessions,
    system,
    tools,
    users,
    workflows,
)
from src.api.schemas.common import APIError, APIResponse

log = logging.getLogger("cogtrix.api")

# ---------------------------------------------------------------------------
# Allowed CORS origins
# ---------------------------------------------------------------------------

_DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:5173",  # Vite React dev server
    "http://localhost:3000",  # Create-React-App dev server (fallback)
    "https://app.cogtrix.ai",  # Production origin placeholder — update before deploy
]


def _get_cors_origins() -> list[str]:
    """Return the list of allowed CORS origins.

    Reads COGTRIX_CORS_ORIGINS from the environment if set (comma-separated);
    falls back to _DEFAULT_CORS_ORIGINS.
    """
    raw = os.environ.get("COGTRIX_CORS_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
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

    # Validate JWT secret
    jwt_secret = os.environ.get("COGTRIX_JWT_SECRET", "")
    if len(jwt_secret) < 32:
        raise RuntimeError(
            "COGTRIX_JWT_SECRET must be set to at least 32 characters. "
            'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
        )
    log.info("JWT secret validated")

    # Create database tables (idempotent; no-op when tables exist)
    import src.api.db.models  # noqa: F401 — registers all ORM model classes
    from src.api.db.engine import Base, engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables ready")

    # Load Cogtrix config
    cfg = None
    try:
        from src.config import load_config

        cfg = load_config()
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

    # Initialize tool registry
    try:
        from src.registry import ToolRegistry

        tool_registry = ToolRegistry()
        tool_registry.load_all_tools()
        app.state.tool_registry = tool_registry
        log.info("Tool registry initialized (%d tools discovered)", len(tool_registry.tools))
    except Exception as exc:
        log.warning("Could not initialize tool registry: %s", exc)
        app.state.tool_registry = None

    # Configure RAG tool with embedding settings from config
    if cfg is not None:
        try:
            from src.tools.configure import configure_rag_tool

            configure_rag_tool(cfg)
            log.debug("RAG tool configured with embedding settings from config")
        except Exception as exc:
            log.warning("Could not configure RAG tool: %s", exc)

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

    log.info("Cogtrix API startup complete")
    yield  # application runs here

    # ---- shutdown ----
    log.info("Cogtrix API shutting down")

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

    # Save all in-memory sessions before shutting down
    try:
        registry = getattr(app.state, "session_registry", None)
        if registry is not None:
            await registry.stop_eviction_loop()
            log.info("Session registry stopped and sessions saved")
    except Exception as exc:
        log.warning("Error stopping session registry: %s", exc)

    # Dispose the async engine connection pool
    try:
        from src.api.db.engine import engine as _engine

        await _engine.dispose()
        log.info("Database engine disposed")
    except Exception as exc:
        log.warning("Error disposing DB engine: %s", exc)

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

    # CORS — must be added before routers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # Exception handlers
    app.add_exception_handler(HTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _generic_exception_handler)  # type: ignore[arg-type]

    # REST routers — all prefixed /api/v1
    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(auth.router, prefix=api_prefix)
    app.include_router(sessions.router, prefix=api_prefix)
    app.include_router(messages.router, prefix=api_prefix)
    app.include_router(memory.router, prefix=api_prefix)
    app.include_router(tools.router, prefix=api_prefix)
    app.include_router(config.router, prefix=api_prefix)
    app.include_router(mcp.router, prefix=api_prefix)
    app.include_router(assistant.router, prefix=api_prefix)
    app.include_router(rag.router, prefix=api_prefix)
    app.include_router(system.router, prefix=api_prefix)
    app.include_router(users.router, prefix=api_prefix)
    app.include_router(workflows.router, prefix=api_prefix)

    # WebSocket routers — prefixed /ws/v1 (embedded in route modules)
    app.include_router(messages.ws_router)
    app.include_router(system.ws_router)

    return app


# Module-level singleton — uvicorn targets `src.api.app:app`
app = create_app()
