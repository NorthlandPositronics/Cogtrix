#!/bin/sh
set -e

# Ensure data directory exists and is writable
DATA_DIR="${COGTRIX_DATA_DIR:-/app/data}"
mkdir -p "$DATA_DIR/history" "$DATA_DIR/knowledge" "$DATA_DIR/vectordb" \
         "$DATA_DIR/api" "$DATA_DIR/api/uploads" "$DATA_DIR/assistant" \
         "$DATA_DIR/workflows"

# ── API server mode ──────────────────────────────────────────
# Start the FastAPI server when invoked with "api" or --api flag.
#   docker run -p 8000:8000 -e COGTRIX_JWT_SECRET=... cogtrix api
#   docker run -p 8000:8000 -e COGTRIX_JWT_SECRET=... cogtrix api --debug
#   docker run -p 8000:8000 -e COGTRIX_JWT_SECRET=... cogtrix api --log
if [ "${1}" = "api" ] || [ "${1}" = "--api" ]; then
    shift

    # Run Alembic migrations before starting the server
    if ! python -m alembic upgrade head 2>&1; then
        echo "Warning: Alembic migration failed (non-fatal — tables may already exist)" >&2
    fi

    exec python -m src.api "$@"
fi

# ── Interactive CLI mode ─────────────────────────────────────
# If no config exists and no provider key/Ollama is available, nothing will work
# regardless of which flags were passed.  Always start the setup wizard so the
# user can configure a provider before continuing.
if [ ! -f /app/.cogtrix.yaml ] && [ ! -f /app/.cogtrix.json ] && [ -t 0 ]; then
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$GEMINI_API_KEY" ] && [ -z "$XAI_API_KEY" ] && [ -z "$COGTRIX_OLLAMA" ] && [ -z "$OLLAMA_BASE_URL" ]; then
        exec python cogtrix.py --setup
    fi
fi

exec python cogtrix.py "$@"
