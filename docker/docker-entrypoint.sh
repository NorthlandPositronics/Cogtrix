#!/bin/bash
set -euo pipefail

# Resolve data directory — allows overriding the default /app/data mount point
# via COGTRIX_DATA_DIR, e.g. when the volume is mounted elsewhere in compose.
DATA_DIR="${COGTRIX_DATA_DIR:-/data}"

# Ensure the full directory tree exists and is writable regardless of how the
# volume was mounted (freshly created volume, host bind-mount, etc.).
mkdir -p \
    "$DATA_DIR/history" \
    "$DATA_DIR/knowledge" \
    "$DATA_DIR/vectordb" \
    "$DATA_DIR/api/uploads" \
    "$DATA_DIR/assistant" \
    "$DATA_DIR/workflows" \
    "$DATA_DIR/output" \
    "$DATA_DIR/log"

# ── API server mode ──────────────────────────────────────────
# Invoked as:
#   docker run cogtrix api
#   docker run cogtrix --api
#   docker run cogtrix api --debug
if [ "${1}" = "api" ] || [ "${1}" = "--api" ]; then
    shift

    # Run Alembic migrations before starting the server.
    # alembic upgrade head returns 0 for both "applied" and "already up to date",
    # so a non-zero exit always indicates a genuine failure (locked table, schema
    # conflict, missing revision). Let it propagate and kill the container rather
    # than starting uvicorn against a potentially broken schema.
    echo "Running database migrations..."
    alembic upgrade head
    echo "Migrations complete."

    # Mark API mode so the HEALTHCHECK can skip the HTTP probe in CLI mode
    # (BUG-236). The sentinel is created before exec so it persists across the
    # process replacement and is visible to subsequent `docker exec` probes.
    mkdir -p /run/cogtrix
    touch /run/cogtrix/api-mode

    exec python -m src.api "$@"
fi

# ── Interactive CLI mode ─────────────────────────────────────
# Auto-start the setup wizard when all of the following are true:
#   1. No config file is found by the Python config resolver (find_config_file)
#   2. No provider API key env var is set
#   3. stdin is a TTY (i.e. the container is running interactively)
#   4. No arguments were passed — explicit args mean the user knows what they
#      want; pass them straight through to cogtrix.py instead of the wizard
#
# Tip: to reach an Ollama instance running on the Docker host, add
# --network host so the wizard can auto-detect it at 127.0.0.1:11434.

# Check all paths the Python config resolver searches (src/config.py:find_config_file).
# Missing any of these would wrongly trigger the setup wizard even when a config exists.
_cogtrix_has_config() {
    [ -f "/app/.cogtrix.yaml" ] || [ -f "/app/.cogtrix.yml" ] || [ -f "/app/.cogtrix.json" ] || \
    [ -f "${HOME}/.cogtrix.yaml" ] || [ -f "${HOME}/.cogtrix.yml" ] || [ -f "${HOME}/.cogtrix.json" ] || \
    [ -f "${XDG_CONFIG_HOME:-${HOME}/.config}/cogtrix/.cogtrix.yaml" ] || \
    [ -f "${XDG_CONFIG_HOME:-${HOME}/.config}/cogtrix/.cogtrix.yml" ] || \
    [ -n "${COGTRIX_CONFIG_FILE:-}" ]
}
if [ $# -eq 0 ] && ! _cogtrix_has_config && [ -t 0 ]; then
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
