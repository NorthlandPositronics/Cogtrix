# ── Stage 1: Build ────────────────────────────────────────────
FROM python:3.13-slim AS builder

# Pin to a specific patch version for reproducibility.
# Update this when intentionally upgrading uv.
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency manifests first — changes here bust only the install layer.
COPY pyproject.toml uv.lock ./

# Install all production extras; strip bytecode caches and test trees to
# reduce the layer size before the venv is copied to the runtime stage.
RUN uv sync --frozen --no-dev --no-install-project \
    --extra search \
    --extra anthropic \
    --extra google \
    --extra mcp \
    --extra science \
    --extra api && \
    find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name tests     -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name test      -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -name '*.dist-info/RECORD' -delete 2>/dev/null || true && \
    rm -rf /root/.cache/uv /root/.cache/pip

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="Cogtrix" \
      org.opencontainers.image.description="Modular AI assistant with 51 built-in tools and REST/WebSocket API" \
      org.opencontainers.image.source="https://github.com/NorthlandPositronics/Cogtrix" \
      org.opencontainers.image.licenses="LicenseRef-Cogtrix-Source-Available-1.0"

# --no-log-init prevents sparse utmp/wtmp files for high-numbered UIDs
RUN groupadd --gid 1000 cogtrix && \
    useradd --uid 1000 --gid cogtrix --no-log-init --create-home cogtrix

WORKDIR /app

# PYTHONPATH ensures "python -m src.api" and "import src.*" resolve from /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Virtual environment from builder — --link enables independent layer caching
COPY --from=builder --link --chown=cogtrix:cogtrix /app/.venv /app/.venv

# Application source ordered least-to-most-frequently-changed for cache efficiency.
# alembic.ini is included so "python -m alembic" works inside the container.
COPY --chown=cogtrix:cogtrix docker-entrypoint.sh alembic.ini ./
COPY --chown=cogtrix:cogtrix alembic/ ./alembic/
COPY --chown=cogtrix:cogtrix cogtrix.py ./
COPY --chown=cogtrix:cogtrix src/ ./src/
COPY --chown=cogtrix:cogtrix docs/ ./docs/

# Create the full data-directory tree that the application expects.
# This runs as root so we can chown; USER is set to cogtrix below.
# The VOLUME declaration comes after so Docker initialises the mount point
# with these pre-created directories and ownership.
RUN mkdir -p \
        /app/data/history \
        /app/data/knowledge \
        /app/data/vectordb \
        /app/data/api/uploads \
        /app/data/assistant \
        /app/data/workflows && \
    chown -R cogtrix:cogtrix /app/data && \
    chmod +x /app/docker-entrypoint.sh

VOLUME /app/data

# API server port
EXPOSE 8000

# Uvicorn handles SIGTERM for graceful shutdown
STOPSIGNAL SIGTERM

# Healthcheck for container orchestrators (API mode only).
# Uses Python's built-in urllib — no curl/wget required in the slim image.
# The 4-second socket timeout keeps the probe within Docker's 5-second deadline.
# Exits 0 on HTTP 200, non-zero on any error.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", \
         "import urllib.request, sys; r = urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=4); sys.exit(0 if r.status == 200 else 1)"]

USER cogtrix

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
