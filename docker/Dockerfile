ARG TARGETPLATFORM
ARG BUILDPLATFORM

# ── Stage 1: Build ────────────────────────────────────────────
FROM python:3.13-slim@sha256:a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d AS builder

# Pin to a specific patch version for reproducibility.
# Update this when intentionally upgrading uv.
# uv 0.10.12 publishes multi-arch manifests for linux/amd64 and linux/arm64.
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1

# Build-time system libraries required by python3-saml (xmlsec C extension).
# These headers are only needed during wheel compilation; the runtime stage
# installs only the shared libraries (libxmlsec1, libxml2) instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxmlsec1-dev libxml2-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Dependency manifests first — changes here bust only the install layer.
COPY pyproject.toml uv.lock ./

# Install all production extras; strip test trees and dist-info RECORD files
# to reduce the layer size before the venv is copied to the runtime stage.
# __pycache__ cleanup is included for safety (no-op without --compile-bytecode).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
    --extra search \
    --extra anthropic \
    --extra google \
    --extra mcp \
    --extra science \
    --extra api \
    --extra saml \
    --extra ldap \
    --extra rag && \
    find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name tests     -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name test      -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -name '*.dist-info/RECORD' -delete 2>/dev/null || true && \
    find /app/.venv -name 'direct_url.json' -delete 2>/dev/null || true && \
    rm -rf /root/.cache/pip

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.13-slim@sha256:a0779d7c12fc20be6ec6b4ddc901a4fd7657b8a6bc9def9d3fde89ed5efe0a3d AS runtime

ARG TARGETPLATFORM
ARG BUILDPLATFORM
LABEL org.opencontainers.image.title="Cogtrix" \
      org.opencontainers.image.description="Modular AI assistant" \
      org.opencontainers.image.source="https://github.com/NorthlandPositronics/Cogtrix" \
      org.opencontainers.image.licenses="LicenseRef-Cogtrix-Source-Available-1.0"

# Runtime shared libraries for python3-saml's xmlsec C extension.
# libxmlsec1-dev is builder-only; libxmlsec1 + libxml2 are needed at runtime.
# python:3.13-slim already ships libxml2; libxmlsec1 must be added explicitly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxmlsec1 \
    && rm -rf /var/lib/apt/lists/*

# --no-log-init prevents sparse utmp/wtmp files for high-numbered UIDs
RUN groupadd --gid 1000 cogtrix && \
    useradd --uid 1000 --gid cogtrix --no-log-init --create-home cogtrix

WORKDIR /app

# PYTHONPATH ensures "python -m src.api" and "import src.*" resolve from /app
# COGTRIX_DATA_DIR=/data moves all runtime writes outside /app, keeping /app read-only.
# COGTRIX_ALLOWED_WRITE_PATHS uses comma separators to match file_ops.py runtime parser.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    COGTRIX_DATA_DIR="/data" \
    COGTRIX_ALLOWED_WRITE_PATHS="/tmp,/data/output,/data/log"

# Virtual environment from builder — --link enables independent layer caching.
# Root-owned so /app/.venv is read-only for the cogtrix runtime user.
COPY --from=builder --link /app/.venv /app/.venv

# Create the data directory tree at /data (not /app/data) so that /app
# can be mounted or enforced as read-only at runtime.
# This runs as root so we can chown; USER is set to cogtrix below.
RUN mkdir -p \
        /data/history \
        /data/knowledge \
        /data/vectordb \
        /data/api/uploads \
        /data/assistant \
        /data/workflows \
        /data/output \
        /data/log && \
    chown -R cogtrix:cogtrix /data && \
    # Pre-create the API-mode sentinel directory in /run so the cogtrix user
    # can write /run/cogtrix/ at runtime without needing root.
    mkdir -p /run/cogtrix && \
    chown cogtrix:cogtrix /run/cogtrix

# Application source — root-owned so /app is read-only for the cogtrix runtime user.
# Ordering: least-to-most-frequently-changed for cache efficiency.
# alembic.ini is included so "python -m alembic" works inside the container.
# docs/ is intentionally excluded — no runtime code imports from it, and
# including it would expose internal architecture documents in production images.
COPY docker/docker-entrypoint.sh alembic.ini ./
RUN chmod +x /app/docker-entrypoint.sh
COPY alembic/ ./alembic/
COPY cogtrix.py ./
COPY src/ ./src/

VOLUME /data

# API server port
EXPOSE 8000

# Uvicorn handles SIGTERM for graceful shutdown
STOPSIGNAL SIGTERM

# Healthcheck for container orchestrators.
# In CLI mode the HTTP server is not running, so the probe exits 0 immediately
# (healthy by convention — the process is simply not a server). In API mode the
# entrypoint creates /run/cogtrix/api-mode before exec, which signals that the
# HTTP endpoint must be reachable.
# Uses Python's built-in urllib — no curl/wget required in the slim image.
# The 4-second socket timeout keeps the probe within Docker's 5-second deadline.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", \
         "import os, sys; sys.exit(0) if not os.path.exists('/run/cogtrix/api-mode') else None; import urllib.request; r = urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=4); sys.exit(0 if r.status == 200 else 1)"]

USER cogtrix

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
