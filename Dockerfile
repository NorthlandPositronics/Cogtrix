# ── Stage 1: Build ────────────────────────────────────────────
FROM python:3.13-slim AS builder

# Install uv for fast, reproducible dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first (layer caching)
COPY pyproject.toml uv.lock ./

# Install production dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project --extra search

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Minimal runtime packages (none required for now, but keeps
# the door open for future native dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends tini && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1000 cogtrix && \
    useradd --uid 1000 --gid cogtrix --create-home cogtrix

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY . .

# Ensure the virtualenv is on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Writable directories for runtime data
RUN mkdir -p /app/data/history /app/data/vectordb && \
    chown -R cogtrix:cogtrix /app/data

USER cogtrix

# Health check — import succeeds and --help exits cleanly
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "cogtrix.py", "--check-config"] || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["python", "cogtrix.py"]
