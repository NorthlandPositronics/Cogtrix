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
    "$DATA_DIR/log" \
    "$DATA_DIR/documents"

# Resolve a WRITABLE runtime working directory and cd into it before exec
# (#2068). The image WORKDIR is /app, which is read-only at runtime by design,
# so the agent's default cwd was non-writable: any file tool writing relative
# to the cwd (e.g. write_file("foo.txt")) failed unless the operator passed
# --workdir. Default to "$DATA_DIR/work" (override via COGTRIX_WORKDIR); it
# lives under the writable data volume. PYTHONPATH=/app still resolves imports.
WORK_DIR="${COGTRIX_WORKDIR:-$DATA_DIR/work}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
# NOTE: after this cd the cwd is no longer /app, so every subsequent exec must
# use absolute paths / module form (``python -m cogtrix_core.api`` via PYTHONPATH=/app,
# ``/app/cogtrix.py``, ``alembic -c /app/alembic.ini``) — a relative
# ``python cogtrix.py`` would resolve against $WORK_DIR and fail (#2068).

# Config discovery searches the CWD for .cogtrix.{yaml,yml,json}; the conventional
# mount point is /app/.cogtrix.yaml (docker-compose, Helm ConfigMap), which the
# cd above moved out of the search path. Point COGTRIX_CONFIG_FILE at it when a
# config is actually present so it's still found from $WORK_DIR (#2068). This is
# conditional on purpose: an unconditional value would make load_config raise
# "Config file not found" when no config is mounted (e.g. env-only API runs).
# An explicit COGTRIX_CONFIG_FILE override is always honored.
if [ -z "${COGTRIX_CONFIG_FILE:-}" ]; then
    for _cfg in /app/.cogtrix.yaml /app/.cogtrix.yml /app/.cogtrix.json; do
        if [ -f "$_cfg" ]; then
            export COGTRIX_CONFIG_FILE="$_cfg"
            break
        fi
    done
fi

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
    # Explicit -c so migrations work from any cwd: the entrypoint cd's into a
    # writable runtime workdir (#2068), but alembic.ini lives at /app. Its
    # script_location uses %(here)s, so it resolves correctly regardless of cwd.
    echo "Running database migrations..."
    alembic -c /app/alembic.ini upgrade head
    echo "Migrations complete."

    # Mark API mode so the HEALTHCHECK can skip the HTTP probe in CLI mode
    # (BUG-236). The sentinel is created before exec so it persists across the
    # process replacement and is visible to subsequent `docker exec` probes.
    mkdir -p /run/cogtrix
    touch /run/cogtrix/api-mode

    exec python -m cogtrix_core.api "$@"
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

# Check all paths the Python config resolver searches (cogtrix_core/config.py:find_config_file).
# Missing any of these would wrongly trigger the setup wizard even when a config exists.
# XDG config files match what ``cogtrix_core/config.py:find_config_file`` searches
# for: ``cogtrix.json``/``cogtrix.yml``/``cogtrix.yaml`` with no leading
# dot.  An earlier version of this script used ``.cogtrix.yaml`` here,
# which silently disagreed with the Python resolver — anyone mounting an
# XDG-style config got dropped into the setup wizard instead.
_cogtrix_has_config() {
    [ -f "/app/.cogtrix.yaml" ] || [ -f "/app/.cogtrix.yml" ] || [ -f "/app/.cogtrix.json" ] || \
    [ -f "${HOME}/.cogtrix.yaml" ] || [ -f "${HOME}/.cogtrix.yml" ] || [ -f "${HOME}/.cogtrix.json" ] || \
    [ -f "${XDG_CONFIG_HOME:-${HOME}/.config}/cogtrix/cogtrix.yaml" ] || \
    [ -f "${XDG_CONFIG_HOME:-${HOME}/.config}/cogtrix/cogtrix.yml" ] || \
    [ -f "${XDG_CONFIG_HOME:-${HOME}/.config}/cogtrix/cogtrix.json" ] || \
    [ -n "${COGTRIX_CONFIG_FILE:-}" ]
}
# ``${VAR:-}`` is required everywhere below because the script runs under
# ``set -euo pipefail``; a bare ``[ -z "$OPENAI_API_KEY" ]`` against an
# unset variable would abort the script with "unbound variable" and the
# user would never reach the wizard.
if [ $# -eq 0 ] && ! _cogtrix_has_config && [ -t 0 ]; then
    if [ -z "${OPENAI_API_KEY:-}" ] && \
       [ -z "${ANTHROPIC_API_KEY:-}" ] && \
       [ -z "${GEMINI_API_KEY:-}" ] && \
       [ -z "${XAI_API_KEY:-}" ] && \
       [ -z "${DEEPSEEK_API_KEY:-}" ] && \
       [ -z "${COGTRIX_OLLAMA:-}" ] && \
       [ -z "${OLLAMA_BASE_URL:-}" ]; then
        exec python /app/cogtrix.py --setup
    fi
fi

exec python /app/cogtrix.py "$@"
