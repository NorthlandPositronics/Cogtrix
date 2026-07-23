#!/bin/sh
set -e

# Resolve data directory — allows overriding the default /app/data mount point
# via COGTRIX_DATA_DIR, e.g. when the volume is mounted elsewhere in compose.
DATA_DIR="${COGTRIX_DATA_DIR:-/app/data}"

# Ensure the full directory tree exists and is writable regardless of how the
# volume was mounted (freshly created volume, host bind-mount, etc.).
mkdir -p \
    "$DATA_DIR/history" \
    "$DATA_DIR/knowledge" \
    "$DATA_DIR/vectordb" \
    "$DATA_DIR/api/uploads" \
    "$DATA_DIR/assistant" \
    "$DATA_DIR/workflows"

# ── API server mode ──────────────────────────────────────────
# Invoked as:
#   docker run cogtrix api
#   docker run cogtrix --api
#   docker run cogtrix api --debug
if [ "${1}" = "api" ] || [ "${1}" = "--api" ]; then
    shift

    # Run Alembic migrations before starting the server.
    # A non-zero exit here is treated as a warning so an already-migrated DB
    # (which returns exit 0) doesn't block startup, and partial failures are
    # surfaced on stderr for visibility.
    if ! alembic upgrade head 2>&1; then
        echo "Warning: Alembic migration step returned non-zero (tables may already exist)" >&2
    fi

    # Mark API mode so the HEALTHCHECK can skip the HTTP probe in CLI mode
    # (BUG-236). The sentinel is created before exec so it persists across the
    # process replacement and is visible to subsequent `docker exec` probes.
    touch /tmp/.cogtrix-api-mode

    exec python -m src.api "$@"
fi

# ── Interactive CLI mode ─────────────────────────────────────
# Auto-start the setup wizard when all of the following are true:
#   1. No config file is present at /app/.cogtrix.yaml, /app/.cogtrix.yml,
#      or /app/.cogtrix.json
#   2. No provider API key env var is set
#   3. stdin is a TTY (i.e. the container is running interactively)
#   4. No arguments were passed — explicit args mean the user knows what they
#      want; pass them straight through to cogtrix.py instead of the wizard
#
# The COGTRIX_CONFIG_FILE env var is intentionally NOT checked here — if the
# user has set that, python cogtrix.py will find it and the wizard is skipped.
#
# Tip: to reach an Ollama instance running on the Docker host, add
# --network host so the wizard can auto-detect it at 127.0.0.1:11434.
if [ $# -eq 0 ] && [ ! -f /app/.cogtrix.yaml ] && [ ! -f /app/.cogtrix.yml ] && [ ! -f /app/.cogtrix.json ] && [ -t 0 ]; then
    if [ -z "$OPENAI_API_KEY" ] && \
       [ -z "$ANTHROPIC_API_KEY" ] && \
       [ -z "$GEMINI_API_KEY" ] && \
       [ -z "$XAI_API_KEY" ] && \
       [ -z "$DEEPSEEK_API_KEY" ] && \
       [ -z "$COGTRIX_OLLAMA" ] && \
       [ -z "$OLLAMA_BASE_URL" ]; then
        exec python cogtrix.py --setup
    fi
fi

exec python cogtrix.py "$@"
