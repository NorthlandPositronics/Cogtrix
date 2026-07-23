# ── Stage 1: Build ────────────────────────────────────────────
FROM python:3.13-slim AS builder

# Install uv for fast, reproducible dependency installation
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first (layer caching)
COPY pyproject.toml uv.lock ./

# Install production dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project --extra search

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1000 cogtrix && \
    useradd --uid 1000 --gid cogtrix --create-home cogtrix

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy virtual environment from builder (owned by cogtrix)
COPY --from=builder --chown=cogtrix:cogtrix /app/.venv /app/.venv

# Copy application source (owned by cogtrix)
COPY --chown=cogtrix:cogtrix . .

# Ensure runtime data directories exist (owned by cogtrix)
RUN mkdir -p /app/data/history /app/data/knowledge /app/data/vectordb && \
    chown -R cogtrix:cogtrix /app/data

VOLUME /app/data

RUN chmod +x /app/docker-entrypoint.sh

USER cogtrix

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
