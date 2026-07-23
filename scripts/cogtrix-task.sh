#!/usr/bin/env bash
# cogtrix-task.sh — run Cogtrix inside Docker against any project directory.
#
# The target project is mounted at /workspace inside the container and Cogtrix
# operates there as its working directory, so all file reads/writes and git
# commands affect the real project files on the host.
#
# Usage:
#   ./scripts/cogtrix-task.sh <project-dir> "Fix all bugs in src/api/"
#   ./scripts/cogtrix-task.sh <project-dir> "Add pagination to /users endpoint"
#   ./scripts/cogtrix-task.sh <project-dir> -f task.txt
#   ./scripts/cogtrix-task.sh <project-dir> "Audit for bugs" --think
#   ./scripts/cogtrix-task.sh <project-dir> "Refactor auth module" --delegate
#
# Config (checked in order — first match wins):
#   1. <project-dir>/.cogtrix.yaml  — project-local config
#   2. ~/.cogtrix.yaml              — your personal config
#   3. ~/.config/cogtrix/config.yaml
#   OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / XAI_API_KEY /
#   DEEPSEEK_API_KEY are always forwarded from the host environment.
#
# Options:
#   -f <file>        Read the task prompt from a file instead of the command line
#   --think          Force deep reasoning pass after the agent run
#   --delegate       Force parallel sub-agent delegation
#   --tools <list>   Comma-separated tool names to activate (overrides default set)
#   --image <name>   Docker image to use (default: cogtrix; or set COGTRIX_IMAGE)
#
# Note: git_add and git_commit require confirmation by default. The agent will
# pause and ask before committing. To auto-approve, add those tools to the
# prompt: "... then git add and commit without asking me".
#
# Requires: Docker; image built with `docker build -t cogtrix .`

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────

IMAGE="${COGTRIX_IMAGE:-cogtrix}"

# Tools pre-loaded into the session. The agent can still load additional
# on-demand tools via request_tools during its run.
DEFAULT_TOOLS="read_file,list_directory,file_info,write_file,patch_file,execute_shell_command,git_status,git_diff,git_log,git_add,git_commit"

# ── Argument parsing ──────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 <project-dir> \"<prompt>\" [--think|--delegate] [--tools <list>] [--image <name>]"
    echo "       $0 <project-dir> -f <task-file> [--think|--delegate]"
    exit 1
}

[[ $# -lt 2 ]] && usage

TARGET_DIR="$(realpath "$1")"; shift

TOOLS="$DEFAULT_TOOLS"
EXTRA_ARGS=()
PROMPT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--file)
            [[ -f "$2" ]] || { echo "Error: file not found: $2"; exit 1; }
            PROMPT_ARGS=(--prompt "$(< "$2")")
            shift 2 ;;
        --think)
            EXTRA_ARGS+=(--think); shift ;;
        --delegate)
            EXTRA_ARGS+=(--delegate); shift ;;
        --tools)
            TOOLS="$2"; shift 2 ;;
        --image)
            IMAGE="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1"; usage ;;
        *)
            PROMPT_ARGS=(--prompt "$1"); shift ;;
    esac
done

[[ ${#PROMPT_ARGS[@]} -eq 0 ]] && { echo "Error: no prompt supplied"; usage; }

[[ -d "$TARGET_DIR" ]] || { echo "Error: project directory not found: $TARGET_DIR"; exit 1; }

# ── Config file discovery ─────────────────────────────────────────────────────
# Mount host config at /workspace/.cogtrix.yaml so the container picks up the
# LLM provider settings. Project-local config takes precedence.

CONFIG_MOUNT=()
if [[ -f "$TARGET_DIR/.cogtrix.yaml" ]]; then
    CONFIG_MOUNT=(-v "$TARGET_DIR/.cogtrix.yaml:/workspace/.cogtrix.yaml:ro")
elif [[ -f "$HOME/.cogtrix.yaml" ]]; then
    CONFIG_MOUNT=(-v "$HOME/.cogtrix.yaml:/workspace/.cogtrix.yaml:ro")
elif [[ -f "$HOME/.config/cogtrix/config.yaml" ]]; then
    CONFIG_MOUNT=(-v "$HOME/.config/cogtrix/config.yaml:/workspace/.cogtrix.yaml:ro")
fi

# ── API key forwarding ────────────────────────────────────────────────────────

ENV_ARGS=()
for var in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY XAI_API_KEY \
           DEEPSEEK_API_KEY COGTRIX_OLLAMA OLLAMA_BASE_URL TAVILY_API_KEY EXA_API_KEY; do
    [[ -n "${!var:-}" ]] && ENV_ARGS+=(-e "$var=${!var}")
done

# ── Run ───────────────────────────────────────────────────────────────────────
# Override the entrypoint so we can set -w /workspace (the project root) while
# still invoking /app/cogtrix.py from its installed location inside the image.

TTY_FLAGS=(-i)
[[ -t 0 && -t 1 ]] && TTY_FLAGS=(-it)

echo "Project : $TARGET_DIR"
echo "Image   : $IMAGE"
echo "Tools   : $TOOLS"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "Flags   : ${EXTRA_ARGS[*]}"
echo "─────────────────────────────────────────────────────────────────────"

exec docker run --rm \
    "${TTY_FLAGS[@]}" \
    --entrypoint python \
    -v "$TARGET_DIR:/workspace" \
    -w /workspace \
    "${CONFIG_MOUNT[@]}" \
    "${ENV_ARGS[@]}" \
    "$IMAGE" \
    /app/cogtrix.py \
    --activate-tools "$TOOLS" \
    "${EXTRA_ARGS[@]}" \
    "${PROMPT_ARGS[@]}"
