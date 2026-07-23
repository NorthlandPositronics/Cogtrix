# Cogtrix Agent

A modular LangChain-based AI agent with extensible tools, multi-provider support, and intelligent memory management.

## Features

- **Multi-Provider Support** — OpenAI, Ollama, and any OpenAI-compatible API
- **43 Built-in Tools** — File operations, web search (DuckDuckGo, Tavily, Exa, Brave, Google, SerpAPI), code execution, deep reasoning, and more
- **Interactive CLI** — Slash commands, line editing with history (arrow keys, Home/End)
- **Live Configuration Switching** — Change model, provider, memory mode, and session at runtime via slash commands
- **Memory Modes** — Optimized for conversation, coding, or strategic reasoning
- **Deep Reasoning** — Tree-of-Thought with Chain-of-Thought Reflection via `/think`
- **Task Delegation** — Distribute subtasks across multiple LLM models
- **Non-interactive Mode** — Single prompt with file I/O for scripting and automation
- **Safety Layer** — Human confirmation for sensitive operations (toggleable with `/noconfirm`)
- **Debug & Logging** — Comprehensive logging with verbose LLM observability

## Quick Start

### Installation

```bash
# Clone and navigate to project
cd cogtrix

# Install dependencies and run (using uv — recommended)
uv sync
uv run python cogtrix.py

# Or using pip (traditional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python cogtrix.py
```

### Basic Usage

```bash
# Default (uses config file or built-in defaults)
python cogtrix.py

# Specify provider and model
python cogtrix.py -p ollama -m llama3:70b

# Use code development memory mode
python cogtrix.py -M code

# Non-interactive mode (single prompt, then exit)
python cogtrix.py --prompt "What is the capital of France?"

# Save response to file
python cogtrix.py --prompt "Summarize this code" -o summary.md

# Enable debug logging
python cogtrix.py --debug
```

## Configuration

Create `.cogtrix.json` (or `.cogtrix.yml` / `.cogtrix.yaml`) in your current directory, home directory, or `~/.config/cogtrix/`:

```json
{
  "provider": "my-server",

  "providers": {
    "my-server": {
      "type": "ollama",
      "base_url": "http://192.168.1.100:11434",
      "model": "llama3:70b"
    },
    "openai": {
      "type": "openai",
      "model": "gpt-4o"
    },
    "groq": {
      "type": "openai",
      "base_url": "https://api.groq.com/openai/v1",
      "api_key": "gsk-...",
      "model": "llama-3.3-70b-versatile"
    }
  },

  "memory": {
    "mode": "conversation"
  },

  "delegate": {
    "enabled": true,
    "model_aliases": {
      "fast": "my-server/llama3:8b",
      "smart": "openai/gpt-4o"
    }
  }
}
```

### Configuration Priority

1. **Command line arguments** — highest priority
2. **Environment variables** — `COGTRIX_PROVIDER`, `COGTRIX_MODEL`, `OPENAI_API_KEY`
3. **Config file** — `.cogtrix.json` / `.cogtrix.yml` / `.cogtrix.yaml`
4. **Built-in defaults** — fallback values

## Command Line Options

| Option | Description |
|--------|-------------|
| `-p, --provider` | LLM provider name |
| `-m, --model` | Model name (or model alias from delegate config) |
| `-s, --session` | Session ID for memory persistence |
| `-M, --memory-mode` | Memory mode: `conversation`, `code`, `reasoning` |
| `--debug` | Enable debug mode (auto-enables `--log` and `--verbose`) |
| `-v, --verbose` | Log full LLM interactions: tokens, thinking, tool calls |
| `--log [FILE]` | Log to file (default: `cogtrix.log`) |
| `--tools LIST` | Comma-separated tools to load (`none`, `minimal`, or names) |
| `--check-config` | Validate configuration and exit |
| `--prompt TEXT` | Send a single prompt and exit (non-interactive) |
| `--prompt-file FILE` | Read prompt from file and exit (non-interactive) |
| `-o, --output FILE` | Write response to file (use with `--prompt`) |
| `--no-stream` | Disable streaming output (useful for scripting) |
| `--ingest` | Build vector database from documents and exit |
| `--docs-dir PATH` | Documents directory for ingestion |
| `--embedding-provider` | Embedding provider: `openai` or `ollama` |

## Interactive Commands

During an interactive session, use slash commands:

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help [cmd]` | `/h`, `/?` | List commands or show detailed help |
| `/info` | `/i` | Show session information (provider, model, mode, etc.) |
| `/tools [search]` | `/t` | List loaded tools (with optional filter) |
| `/think <task>` | — | Run deep Tree-of-Thought reasoning |
| `/mode [name]` | `/M` | Show / switch memory mode |
| `/model [name]` | `/m` | Show / switch LLM model |
| `/provider [name]` | `/p` | Show / switch LLM provider |
| `/session [id]` | `/s` | Show / switch session |
| `/debug` | — | Toggle debug mode |
| `/verbose` | `/v` | Toggle verbose logging |
| `/noconfirm` | `/y` | Toggle tool auto-approval |
| `/paste` | — | Enter multi-line paste mode |
| `/clear` | — | Clear conversation history |
| `/quit` | `/exit`, `/q` | Exit the session |

The CLI also supports full line editing: arrow keys, Home/End, and input history (via `readline`).

**Note:** Short aliases match the corresponding CLI flags (e.g., `-M` for mode maps to `/M`, `-m` for model maps to `/m`).

## Memory Modes

| Mode | Best For | Working Memory |
|------|----------|----------------|
| `conversation` | General chat, Q&A, research | 20 messages |
| `code` | Programming, debugging | 8 messages + file tracking |
| `reasoning` | Planning, decisions | 6 messages + goal tracking |

```bash
python cogtrix.py -M code        # Code development
python cogtrix.py -M reasoning   # Strategic planning
```

## Task Delegation

Delegate subtasks to specialized models:

```json
{
  "delegate": {
    "model_aliases": {
      "code": "ollama/codellama:34b",
      "fast": "groq/llama-3.3-70b-versatile",
      "smart": "openai/gpt-4o",
      "deep": {
        "provider": "ollama",
        "model": "llama3:70b",
        "num_ctx": 32768,
        "temperature": 0.3
      }
    }
  }
}
```

The agent can then delegate: *"Analyze this code using the 'code' model"*

Aliases also work with `-m`: `python cogtrix.py -m fast`

## Built-in Tools

### Search (10 tools)
- `search_web`, `search_news` — DuckDuckGo search (no API key needed)
- `tavily_search`, `tavily_extract` — AI-optimised search with content extraction (Tavily API)
- `exa_search`, `exa_find_similar`, `exa_get_contents` — Semantic search with neural embeddings (Exa API)
- `brave_search` — Privacy-focused search with independent index (Brave API)
- `google_search` — Official Google Custom Search API
- `serpapi_search` — Structured Google/Bing results with answer boxes and knowledge graph (SerpAPI)

### System & Files
- `execute_shell_command` — Run shell commands *(requires confirmation)*
- `execute_python` — Execute Python with persistent state, history, NumPy/Pandas support *(requires confirmation)*
- `read_file`, `write_file`, `append_file` — File operations
- `list_directory`, `file_info` — Directory operations

### Text & Data
- `calculate` — Math expressions
- `parse_json`, `format_json`, `query_json`, `extract_json`, `json_to_text` — JSON processing
- `word_count`, `find_replace`, `extract_urls`, `extract_emails`, `text_compare`, `split_text`, `trim_text` — Text utilities
- `analyze_sentiment`, `summarize_text`, `extract_keywords` — NLP

### Web & HTTP
- `http_get`, `http_post` — HTTP requests

### Date & Time
- `get_current_datetime`, `convert_timezone`, `parse_date`
- `get_weather` — Weather information (OpenWeather API)

### Knowledge & Delegation
- `query_knowledge_base` — RAG queries
- `delegate_task`, `delegate_parallel` — Multi-model delegation

### Deep Reasoning
- `deep_think` — Tree-of-Thought with iterative Chain-of-Thought reflection (also available as `/think` command)

**Note:** Search tools that require API keys (Tavily, Exa, Brave, Google, SerpAPI) are automatically disabled when the key is not configured. DuckDuckGo search is always available.

## Debugging & Logging

Enable logging to troubleshoot tool calls and agent behavior:

```bash
# Basic logging
python cogtrix.py --log

# Log full LLM interactions (tokens, thinking)
python cogtrix.py --log -v

# Full debug mode (auto-enables --log and --verbose)
python cogtrix.py --debug
```

Debug mode logs: user messages, tool calls (with inputs/outputs), agent responses, memory context. See [Configuration](docs/CONFIGURATION.md#debugging--logging) for details.

## Adding Custom Tools

Create a file in `src/tools/`:

```python
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    query: str = Field(description="The input query")

def my_tool(query: str) -> str:
    """Description shown to the LLM."""
    return f"Processed: {query}"

TOOL_CONFIG = {
    "name": "my_tool",
    "description": "What this tool does",
    "input_schema": MyToolInput,
    "requires_confirmation": False,
}
```

The tool is automatically discovered on next startup.

## RAG (Knowledge Base)

Build a searchable knowledge base from your documents:

```bash
# Add documents to docs/ directory
mkdir -p docs
cp your-documents.pdf docs/

# Build vector database
python cogtrix.py --ingest

# Use Ollama for embeddings (local, free)
python cogtrix.py --ingest --embedding-provider ollama

# Custom directories
python cogtrix.py --ingest --docs-dir ./my-docs --vectordb-dir ./my-vectordb
```

**Supported file types:** PDF, Markdown, CSV, TXT

**Configuration:**

```json
{
  "rag": {
    "docs_dir": "docs",
    "vectordb_dir": "data/vectordb",
    "embedding_provider": "my-server",
    "embedding_model": "nomic-embed-text"
  }
}
```

**Note:** `embedding_provider` can be `openai`, `ollama`, or any named provider from your `providers` config.

After ingestion, query in conversation:

```
You: What does the policy say about remote work?
```

## Project Structure

```
cogtrix/
├── cogtrix.py            # CLI entry point
├── pyproject.toml        # Project metadata & dependencies
├── uv.lock               # Locked dependency versions
├── requirements.txt      # Pip-compatible deps (auto-generated)
├── src/
│   ├── config.py         # Configuration management
│   ├── registry.py       # Tool discovery & loading
│   ├── logging_config.py # Logging setup
│   ├── agent/
│   │   ├── core.py       # LangChain agent setup
│   │   └── safety.py     # Confirmation layer
│   ├── memory/
│   │   ├── factory.py    # Memory mode factory
│   │   └── modes/        # conversation, code, reasoning
│   ├── rag/
│   │   └── ingest.py     # Document ingestion
│   └── tools/            # 43 built-in tools
├── tests/                # Test suite
├── docs/                 # Documentation
└── data/                 # Session history & vector DB
```

## Documentation

| Document | Description |
|----------|-------------|
| **[Configuration](docs/CONFIGURATION.md)** | Complete configuration reference |
| **[Architecture](docs/ARCHITECTURE.md)** | System design and internals |
| **[Memory Modes](docs/MEMORY_MODES.md)** | Conversation, code, and reasoning modes |
| **[Tools Reference](docs/TOOLS_REFERENCE.md)** | All 43 tools with parameters |
| **[Providers](docs/PROVIDERS.md)** | OpenAI, Ollama, Groq setup guides |
| **[Deep Think](docs/DEEPTHINK.md)** | Tree-of-Thought reasoning engine |
| **[RAG Guide](docs/RAG_GUIDE.md)** | Knowledge base setup |
| **[Development](docs/DEVELOPMENT.md)** | Adding tools, extending the system |

## Testing

```bash
# Run all tests (using uv)
uv run pytest tests/ -v

# Or with pip/venv
python -m pytest tests/ -v

# Run specific test file
uv run pytest tests/test_provider_config.py -v
```

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- One of: OpenAI API key, Ollama server, or OpenAI-compatible API

## License

Copyright © 2025–2026 Northland Positronics (FZE). All rights reserved.

This software is released under the **Cogtrix Source-Available License 1.0**. You may use and run the Software, but may not modify, create derivative works, rebrand, or redistribute it without prior written consent. See [LICENSE](LICENSE) for full terms.
