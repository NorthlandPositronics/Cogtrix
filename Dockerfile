# ── Stage 1: Build ────────────────────────────────────────────
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency manifests (layer caching)
COPY pyproject.toml uv.lock ./

# Install production deps — no dev, no project itself
RUN uv sync --frozen --no-dev --no-install-project \
    --extra search \
    --extra anthropic \
    --extra google \
    --extra mcp \
    --extra science \
    --extra api && \
    find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -type d -name tests -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -type d -name test -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -name '*.dist-info/RECORD' -delete 2>/dev/null; \
    rm -rf /root/.cache/uv

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="Cogtrix" \
      org.opencontainers.image.description="Modular AI assistant with 51 built-in tools and REST/WebSocket API" \
      org.opencontainers.image.source="https://github.com/NorthlandPositronics/Cogtrix"

# Non-root user for security
RUN groupadd --gid 1000 cogtrix && \
    useradd --uid 1000 --gid cogtrix --create-home cogtrix

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Virtual environment from builder (--link enables independent layer caching)
COPY --from=builder --link --chown=cogtrix:cogtrix /app/.venv /app/.venv

# Application source — ordered by change frequency (least → most) for layer caching
COPY --chown=cogtrix:cogtrix docker-entrypoint.sh alembic.ini ./
COPY --chown=cogtrix:cogtrix alembic/ ./alembic/
COPY --chown=cogtrix:cogtrix cogtrix.py ./
COPY --chown=cogtrix:cogtrix src/ ./src/
COPY --chown=cogtrix:cogtrix docs/ ./docs/

# Data directories + entrypoint permissions
RUN mkdir -p /app/data/history /app/data/knowledge /app/data/vectordb \
             /app/data/api /app/data/api/uploads /app/data/assistant \
             /app/data/workflows && \
    chown -R cogtrix:cogtrix /app/data && \
    chmod +x /app/docker-entrypoint.sh

VOLUME /app/data

# API server port
EXPOSE 8000

# Graceful shutdown — uvicorn handles SIGTERM
STOPSIGNAL SIGTERM

# Health check for container orchestrators (API mode only).
# Uses the lightweight /health endpoint; runs as a no-import one-liner
# to avoid loading the full application on each probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"]

USER cogtrix

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
