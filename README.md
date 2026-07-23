# Cogtrix Agent

A modular LangChain-based AI agent with extensible tools, multi-provider support, and intelligent memory management.

---

## What Is Cogtrix?

Cogtrix is an **interactive command-line AI assistant** that connects to large language models (LLMs) and extends them with tools — web search, file operations, code execution, deep reasoning, and more. You type a question or task; the agent reasons about it, calls tools as needed, and returns the result.

It works with **OpenAI**, **Ollama** (local models), and any **OpenAI-compatible API** (Groq, Together, vLLM, LocalAI, etc.).

### Key capabilities

- **43 built-in tools** — web search, file I/O, shell commands, Python execution, HTTP requests, JSON processing, NLP, and more
- **6 search providers** — DuckDuckGo (free, no key), Tavily, Exa, Brave, Google, SerpAPI
- **Memory modes** — optimized for conversation, coding, or strategic reasoning
- **Deep reasoning** — Tree-of-Thought with Chain-of-Thought Reflection via `/think`
- **Task delegation** — distribute subtasks across multiple LLM models
- **Non-interactive mode** — single prompt with file I/O for scripting and automation
- **Safety layer** — human confirmation for sensitive operations (shell, code execution)
- **Live configuration** — change model, provider, memory mode at runtime via slash commands

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.13+** | Check with `python3 --version` |
| **[uv](https://docs.astral.sh/uv/)** (recommended) or pip | `uv` handles dependencies and virtual environments |
| **An LLM backend** | One of: OpenAI API key, running Ollama server, or any OpenAI-compatible API |

---

## Quick Start

Follow these steps to go from zero to a working agent.

### 1. Clone and install

```bash
git clone <repository-url> cogtrix
cd cogtrix

# Using uv (recommended)
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure an LLM provider

You need at least one LLM backend. Pick one:

**Option A — OpenAI (cloud)**

```bash
export OPENAI_API_KEY="sk-..."
```

**Option B — Ollama (local, free)**

Install [Ollama](https://ollama.com/), then pull a model:

```bash
ollama pull llama3:8b
```

**Option C — Any OpenAI-compatible API** (Groq, Together, vLLM, etc.)

Create a config file — see [Configuration](#configuration) below.

### 3. Run

```bash
# With uv
uv run python cogtrix.py

# With pip/venv
python cogtrix.py
```

You should see an interactive prompt. Type a question and press Enter.

### 4. Try it out

```
You: What is the capital of New Zealand?
You: /think search for top 10 news affecting the stock market
You: /help
You: /quit
```

### Common launch examples

```bash
# Specify provider and model
python cogtrix.py -p ollama -m llama3:70b

# Code development memory mode
python cogtrix.py -M code

# Non-interactive mode (single prompt, then exit)
python cogtrix.py --prompt "What is the capital of France?"

# Save response to file
python cogtrix.py --prompt "Summarize this code" -o summary.md

# Enable debug logging
python cogtrix.py --debug
```

---

## Configuration

Create `.cogtrix.json` (or `.cogtrix.yml` / `.cogtrix.yaml`) in your current directory, home directory, or `~/.config/cogtrix/`.

### Minimal example (Ollama)

```json
{
  "provider": "ollama"
}
```

That's it — Cogtrix will connect to `http://localhost:11434` with the default model.

### Full example

```json
{
  "provider": "my-server",

  "inference": {
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

  "services": {
    "tavily":      { "api_key": "tvly-..." },
    "exa":         { "api_key": "exa-..." },
    "brave":       { "api_key": "BSA..." },
    "serpapi":     { "api_key": "..." },
    "google":      { "api_key": "AIza...", "cse_id": "..." },
    "openweather": { "api_key": "..." }
  },

  "model_aliases": {
    "fast": "my-server/llama3:8b",
    "smart": "openai/gpt-4o"
  },

  "memory": {
    "mode": "conversation"
  }
}
```

> **Note:** The key `"inference"` is preferred. The legacy key `"providers"` still works for backward compatibility.

### Configuration priority

1. **Command line arguments** — highest priority
2. **Environment variables** — `COGTRIX_PROVIDER`, `COGTRIX_MODEL`, `OPENAI_API_KEY`, etc.
3. **Config file** — `.cogtrix.json` / `.cogtrix.yml` / `.cogtrix.yaml`
4. **Built-in defaults** — fallback values

---

## Search Providers

Cogtrix ships with **six search providers**. DuckDuckGo works immediately with no setup. The other five are premium providers that require an API key and, in some cases, an additional Python package.

### Overview

| Provider | Package required | API key | Free tier | Best for |
|----------|-----------------|---------|-----------|----------|
| **DuckDuckGo** | Included (`ddgs`) | None | Unlimited | Quick, no-setup search |
| **Tavily** | `tavily-python` | `TAVILY_API_KEY` | 1 000/month | AI-optimized search with full-page extraction |
| **Exa** | `exa-py` | `EXA_API_KEY` | 1 000/month | Semantic/neural search |
| **Brave** | Included (`requests`) | `BRAVE_API_KEY` | 2 000/month | Privacy-focused, independent index |
| **Google** | Included (`requests`) | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | 100/day | Official Google results |
| **SerpAPI** | `google-search-results` | `SERPAPI_API_KEY` | 100/month | Structured Google/Bing with knowledge graph |

**DuckDuckGo is always available.** Premium providers are automatically hidden from the agent when their API key is missing — no errors, they simply don't appear.

### Step 1: Install optional search packages

Tavily, Exa, and SerpAPI require extra Python packages that are **not** installed by default. Install them all at once:

```bash
# With uv (recommended)
uv sync --extra search

# With pip
pip install tavily-python exa-py google-search-results
```

Brave and Google use only `requests`, which is already included — no extra install needed.

### Step 2: Configure API keys

You can set API keys via **environment variables** or the **config file**. Both methods work; use whichever you prefer.

**Environment variables:**

```bash
export TAVILY_API_KEY="tvly-..."
export EXA_API_KEY="exa-..."
export BRAVE_API_KEY="BSA..."
export SERPAPI_API_KEY="..."
export GOOGLE_API_KEY="AIza..."
export GOOGLE_CSE_ID="abc123..."
```

**Config file** (`.cogtrix.json`):

```json
{
  "services": {
    "tavily":  { "api_key": "tvly-..." },
    "exa":     { "api_key": "exa-..." },
    "brave":   { "api_key": "BSA..." },
    "serpapi":  { "api_key": "..." },
    "google":  { "api_key": "AIza...", "cse_id": "abc123..." }
  }
}
```

### Step 3: Verify

Start Cogtrix and run `/tools search` to see which search tools are loaded:

```
You: /tools search
```

You should see entries like `search_web`, `search_news`, `tavily_search`, `exa_search`, etc., depending on which packages and keys you configured.

### Where to get API keys

| Provider | Sign-up URL |
|----------|-------------|
| Tavily | <https://tavily.com/> |
| Exa | <https://exa.ai/> |
| Brave | <https://brave.com/search/api/> |
| Google | <https://console.cloud.google.com/> (enable Custom Search API, then create a search engine at <https://programmablesearchengine.google.com/>) |
| SerpAPI | <https://serpapi.com/> |

---

## Command Line Options

| Option | Description |
|--------|-------------|
| `-p, --provider` | LLM provider name |
| `-m, --model` | Model name (or model alias from config) |
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

---

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

---

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

---

## Task Delegation

Delegate subtasks to specialized models:

```json
{
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
```

The agent can then delegate: *"Analyze this code using the 'code' model"*

Aliases also work with `-m`: `python cogtrix.py -m fast`

---

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

---

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

---

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

---

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

**Note:** `embedding_provider` can be `openai`, `ollama`, or any named provider from your `inference` config.

After ingestion, query in conversation:

```
You: What does the policy say about remote work?
```

---

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

---

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

---

## Testing

```bash
# Run all tests (using uv)
uv run pytest tests/ -v

# Or with pip/venv
python -m pytest tests/ -v

# Run specific test file
uv run pytest tests/test_provider_config.py -v
```

---

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- One of: OpenAI API key, Ollama server, or OpenAI-compatible API

---

## License

Copyright © 2025–2026 Northland Positronics (FZE). All rights reserved.

This software is released under the **Cogtrix Source-Available License 1.0**. You may use and run the Software, but may not modify, create derivative works, rebrand, or redistribute it without prior written consent. See [LICENSE](LICENSE) for full terms.
