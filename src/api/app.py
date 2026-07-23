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
    try:
        from src.config import Config

        cfg = Config()
        app.state.config = cfg
        log.info("Config loaded (provider=%s)", getattr(cfg, "provider", "unknown"))
    except Exception as exc:
        log.warning("Could not load Cogtrix config: %s", exc)
        app.state.config = None

    # Initialize tool registry
    try:
        from src.registry import ToolRegistry

        tool_registry = ToolRegistry()
        tool_registry.scan_tools()
        app.state.tool_registry = tool_registry
        log.info("Tool registry initialized (%d tools discovered)", len(tool_registry.tools))
    except Exception as exc:
        log.warning("Could not initialize tool registry: %s", exc)
        app.state.tool_registry = None

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

    # Placeholders for Phase 2+ state
    app.state.assistant_service = None
    app.state.message_scheduler = None
    app.state.deferral_manager = None
    app.state.guardrail_pipeline = None
    app.state.knowledge_store = None

    log.info("Cogtrix API startup complete")
    yield  # application runs here

    # ---- shutdown ----
    log.info("Cogtrix API shutting down")

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
    validation_exc = exc  # type: ignore[assignment]
    details: dict = {}
    if hasattr(validation_exc, "errors"):
        try:
            details = {"errors": validation_exc.errors()}  # type: ignore[union-attr]
        except Exception:
            pass

    envelope = APIResponse(
        data=None,
        error=APIError(
            code="VALIDATION_ERROR",
            message="Request body or query parameter validation failed.",
            details=details or None,
        ),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
        version="1.0.0",
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

    # WebSocket routers — prefixed /ws/v1 (embedded in route modules)
    app.include_router(messages.ws_router)
    app.include_router(system.ws_router)

    return app


# Module-level singleton — uvicorn targets `src.api.app:app`
app = create_app()
