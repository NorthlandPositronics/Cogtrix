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
    --extra science && \
    find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
    rm -rf /root/.cache/uv

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="Cogtrix" \
      org.opencontainers.image.description="Modular AI assistant with 51 built-in tools" \
      org.opencontainers.image.source="https://github.com/NorthlandPositronics/Cogtrix"

# Non-root user for security
RUN groupadd --gid 1000 cogtrix && \
    useradd --uid 1000 --gid cogtrix --create-home cogtrix

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Virtual environment from builder
COPY --from=builder --chown=cogtrix:cogtrix /app/.venv /app/.venv

# Application source — only what's needed at runtime
COPY --chown=cogtrix:cogtrix cogtrix.py ./
COPY --chown=cogtrix:cogtrix src/ ./src/
COPY --chown=cogtrix:cogtrix docs/ ./docs/
COPY --chown=cogtrix:cogtrix docker-entrypoint.sh ./

# Data directories + entrypoint permissions
RUN mkdir -p /app/data/history /app/data/knowledge /app/data/vectordb && \
    chown -R cogtrix:cogtrix /app/data && \
    chmod +x /app/docker-entrypoint.sh

VOLUME /app/data

USER cogtrix

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
