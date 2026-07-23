#!/bin/sh
set -e

# Ensure data directory exists and is writable
DATA_DIR="${COGTRIX_DATA_DIR:-/app/data}"
mkdir -p "$DATA_DIR/history" "$DATA_DIR/knowledge" "$DATA_DIR/vectordb" \
         "$DATA_DIR/api" "$DATA_DIR/api/uploads" "$DATA_DIR/assistant"

# ── API server mode ──────────────────────────────────────────
# Start the FastAPI server when invoked with "api" or --api flag.
#   docker run -p 8000:8000 -e COGTRIX_JWT_SECRET=... cogtrix api
#   docker run -p 8000:8000 -e COGTRIX_JWT_SECRET=... cogtrix --api
if [ "${1}" = "api" ] || [ "${1}" = "--api" ]; then
    shift
    HOST="${COGTRIX_API_HOST:-0.0.0.0}"
    PORT="${COGTRIX_API_PORT:-8000}"
    WORKERS="${COGTRIX_API_WORKERS:-1}"

    # Run Alembic migrations before starting the server
    python -m alembic upgrade head 2>/dev/null || true

    exec uvicorn src.api.app:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info \
        "$@"
fi

# ── Interactive CLI mode ─────────────────────────────────────
# If no config file exists and running interactively, suggest --setup
if [ ! -f /app/.cogtrix.yaml ] && [ ! -f /app/.cogtrix.json ] && [ -t 0 ]; then
    # Check if any provider key is set
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$GEMINI_API_KEY" ] && [ -z "$XAI_API_KEY" ] && [ -z "$COGTRIX_OLLAMA" ] && [ -z "$OLLAMA_BASE_URL" ]; then
        echo ""
        echo "  No LLM provider configured."
        echo "  Run with --setup to create a configuration file,"
        echo "  or set a provider API key via environment variable."
        echo ""
        echo "  Examples:"
        echo "    docker run -it -e OPENAI_API_KEY=sk-... cogtrix"
        echo "    docker run -it cogtrix --setup"
        echo "    docker run -p 8000:8000 -e COGTRIX_JWT_SECRET=... cogtrix api"
        echo ""
    fi
fi

exec python cogtrix.py "$@"
