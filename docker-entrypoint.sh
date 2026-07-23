#!/bin/sh
set -e

# Ensure data directory exists and is writable
mkdir -p /app/data/history /app/data/knowledge /app/data/vectordb

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
        echo ""
    fi
fi

exec python cogtrix.py "$@"
