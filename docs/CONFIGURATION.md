# Cogtrix Configuration Reference

Complete reference for all configuration options in Cogtrix.

## Table of Contents

- [Configuration Priority](#configuration-priority)
- [Configuration File](#configuration-file)
- [Environment Variables](#environment-variables)
- [Command Line Arguments](#command-line-arguments)
- [Complete Configuration Example](#complete-configuration-example)
- [Interactive Commands](#interactive-commands)
- [Debugging & Logging](#debugging--logging)

---

## Configuration Priority

Configuration is loaded from multiple sources with the following priority (highest to lowest):

1. **Command line arguments** — Override everything
2. **Environment variables** — Override config file
3. **Configuration file** (`.cogtrix.json` / `.cogtrix.yml` / `.cogtrix.yaml`) — Base settings
4. **Built-in defaults** — Fallback values

---

## Configuration File

Both JSON and YAML formats are supported. Create a config file in one of these locations (first found wins):

1. `./.cogtrix.json`
2. `./.cogtrix.yml` or `./.cogtrix.yaml`
3. `~/.cogtrix.json`
4. `~/.cogtrix.yml` or `~/.cogtrix.yaml`
5. `~/.config/cogtrix/cogtrix.json`
6. `~/.config/cogtrix/cogtrix.yml` or `~/.config/cogtrix/cogtrix.yaml`

Within each directory, JSON is checked first, then `.yml`, then `.yaml`.

### General Settings

```json
{
  "provider": "openai",
  "model": "gpt-4o",
  "session": "default"
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | string | `"openai"` | Active provider name |
| `model` | string | Provider-specific | Model to use (overrides provider default) |
| `session` | string | `"default"` | Session ID for memory persistence |

### Providers Section

Define named LLM providers with custom configurations:

```json
{
  "providers": {
    "my-ollama": {
      "type": "ollama",
      "base_url": "http://192.168.1.100:11434",
      "model": "llama3:70b"
    },
    "openai": {
      "type": "openai",
      "model": "gpt-4o",
      "api_key": "sk-..."
    },
    "groq": {
      "type": "openai",
      "base_url": "https://api.groq.com/openai/v1",
      "api_key": "gsk-...",
      "model": "llama-3.3-70b-versatile"
    }
  }
}
```

#### Provider Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `type` | string | Yes | Provider type: `"openai"` or `"ollama"` |
| `base_url` | string | No | API endpoint URL |
| `model` | string | No | Default model for this provider |
| `api_key` | string | No | API key (OpenAI-compatible providers) |
| `temperature` | float | No | Response temperature (0.0-2.0) |
| `num_ctx` | int | No | Context window size (Ollama only) |

#### Provider Types

| Type | Use For | Default Base URL |
|------|---------|------------------|
| `openai` | OpenAI, Groq, Together, vLLM, LocalAI | `https://api.openai.com/v1` |
| `ollama` | Ollama servers | `http://localhost:11434` |

### Legacy Provider Format

For backward compatibility, you can use the legacy format:

```json
{
  "provider": "ollama",
  "openai": {
    "api_key": "sk-...",
    "model": "gpt-4o-mini"
  },
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "qwen3:32b"
  }
}
```

### Memory Section

Configure memory management:

```json
{
  "memory": {
    "mode": "conversation",
    "modes": {
      "conversation": {
        "working_memory_size": 20
      },
      "code": {
        "working_memory_size": 8,
        "max_files": 20
      },
      "reasoning": {
        "working_memory_size": 6,
        "max_decisions": 20
      }
    }
  }
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | string | `"conversation"` | Active memory mode |
| `modes` | object | `{}` | Mode-specific configurations |

See [MEMORY_MODES.md](MEMORY_MODES.md) for detailed mode options.

### RAG Section

Configure document ingestion for knowledge base:

```json
{
  "rag": {
    "docs_dir": "docs",
    "vectordb_dir": "data/vectordb",
    "chunk_size": 1200,
    "chunk_overlap": 200,
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small"
  }
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `docs_dir` | string | `"docs"` | Source documents directory |
| `vectordb_dir` | string | `"data/vectordb"` | Vector database output directory |
| `chunk_size` | int | `1200` | Text chunk size in characters |
| `chunk_overlap` | int | `200` | Overlap between chunks |
| `embedding_provider` | string | `"openai"` | Embedding provider: `"openai"`, `"ollama"`, or a named provider |
| `embedding_model` | string | Auto | Embedding model name |

**Note:** You can use a named provider (e.g., `"gpu-server"`) as `embedding_provider`. It will resolve to the provider's type and base_url.

See [RAG_GUIDE.md](RAG_GUIDE.md) for detailed setup instructions.

### Delegate Section

Configure task delegation to other models:

```json
{
  "delegate": {
    "enabled": true,
    "default_timeout": 60,
    "max_depth": 3,
    "allowed_providers": ["openai", "ollama"],
    "model_aliases": {
      "fast": "ollama/llama3:8b",
      "smart": "openai/gpt-4o",
      "code": "ollama/codellama:34b",
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

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable/disable delegation |
| `default_timeout` | int | `60` | Default timeout in seconds |
| `max_depth` | int | `3` | Maximum delegation depth |
| `allowed_providers` | array | All providers | Providers allowed for delegation |
| `model_aliases` | object | `{}` | Shortcuts for provider/model combinations |

#### Model Alias Formats

Aliases can be defined in two formats:

**String format:** `"provider/model"` or just `"model"`
```json
"fast": "ollama/llama3:8b"
```

**Object format:** With additional overrides (`num_ctx`, `temperature`, `timeout`)
```json
"deep": {
  "provider": "ollama",
  "model": "llama3:70b",
  "num_ctx": 32768,
  "temperature": 0.3,
  "timeout": 300
}
```

Model aliases can also be used with the `-m` CLI flag to quickly switch configurations:
```bash
python cogtrix.py -m fast     # Resolves alias to ollama/llama3:8b
python cogtrix.py -m deep     # Resolves to ollama/llama3:70b with num_ctx=32768
```

### Services Section

Configure API keys for external services (search providers, weather, etc.) in a single place:

```json
{
  "services": {
    "tavily":      { "api_key": "tvly-..." },
    "exa":         { "api_key": "exa-..." },
    "brave":       { "api_key": "BSA..." },
    "serpapi":     { "api_key": "..." },
    "google":      { "api_key": "AIza...", "cse_id": "abc123..." },
    "openweather": { "api_key": "..." }
  }
}
```

Tools that require an API key are **automatically hidden** from the agent when the key is not configured — no errors, they simply don't appear in the tool list.

#### Search Providers

Cogtrix includes six search providers. DuckDuckGo is always available with no setup. The other five require an API key and some require an additional Python package.

| Provider | Tools | Package | API Key | Free Tier |
|----------|-------|---------|---------|-----------|
| DuckDuckGo | `search_web`, `search_news` | Included (`ddgs`) | None | Unlimited |
| Tavily | `tavily_search`, `tavily_extract` | `tavily-python` | `TAVILY_API_KEY` | 1 000/month |
| Exa | `exa_search`, `exa_find_similar`, `exa_get_contents` | `exa-py` | `EXA_API_KEY` | 1 000/month |
| Brave | `brave_search` | Included (`requests`) | `BRAVE_API_KEY` | 2 000/month |
| Google | `google_search` | Included (`requests`) | `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | 100/day |
| SerpAPI | `serpapi_search` | `google-search-results` | `SERPAPI_API_KEY` | 100/month |

**Installing optional search packages:**

Tavily, Exa, and SerpAPI need extra Python packages not included by default:

```bash
# All at once (recommended)
uv sync --extra search

# Or individually with pip
pip install tavily-python exa-py google-search-results
```

Brave and Google use only `requests`, which is already a core dependency.

#### Legacy service format

For backward compatibility, top-level service keys still work:

```json
{
  "openweather": { "api_key": "..." },
  "tavily":      { "api_key": "..." }
}
```

The `"services"` section takes priority when both are present.

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `COGTRIX_PROVIDER` | LLM provider | `ollama` |
| `COGTRIX_MODEL` | Model name | `llama3:70b` |
| `COGTRIX_SESSION` | Session ID | `my-project` |
| `COGTRIX_MEMORY_MODE` | Memory mode | `code` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://192.168.1.100:11434` |
| `OPENWEATHER_API_KEY` | OpenWeather API key | `abc123` |
| `COGTRIX_EMBEDDING_PROVIDER` | RAG embedding provider | `ollama` |
| `OLLAMA_EMBEDDING_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `TAVILY_API_KEY` | Tavily search API key | `tvly-...` |
| `EXA_API_KEY` | Exa search API key | `exa-...` |
| `BRAVE_API_KEY` | Brave search API key | `BSA...` |
| `GOOGLE_API_KEY` | Google Custom Search API key | `AIza...` |
| `GOOGLE_CSE_ID` | Google Programmable Search Engine ID | `abc123...` |
| `SERPAPI_API_KEY` | SerpAPI search API key | `...` |

---

## Command Line Arguments

### General Options

```bash
python cogtrix.py [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--provider NAME` | `-p` | LLM provider name |
| `--model NAME` | `-m` | Model name or model alias from delegate config |
| `--session ID` | `-s` | Session ID for memory persistence |
| `--memory-mode MODE` | `-M` | Memory mode: `conversation`, `code`, `reasoning` |
| `--debug` | | Enable debug mode (auto-enables `--log` and `--verbose`) |
| `--verbose` | `-v` | Log full LLM interactions: tokens, thinking, tool calls |
| `--log [FILE]` | | Enable logging to file (default: `cogtrix.log`) |
| `--tools LIST` | | Comma-separated tools to load (default: all) |
| `--check-config` | | Validate configuration and exit |

### Non-interactive Mode

Process a single prompt and exit (useful for scripting and automation):

```bash
python cogtrix.py --prompt "What is 2+2?"
python cogtrix.py --prompt-file task.txt
python cogtrix.py --prompt "Summarize this" -o summary.md
python cogtrix.py --prompt "Generate JSON" --no-stream -o data.json
```

| Option | Short | Description |
|--------|-------|-------------|
| `--prompt TEXT` | | Send a single prompt and exit |
| `--prompt-file FILE` | | Read prompt from file and exit |
| `--output FILE` | `-o` | Write response to file |
| `--no-stream` | | Disable streaming output |

### Tool Filtering

Control which tools are loaded at startup:

```bash
python cogtrix.py --tools none                    # No tools (pure LLM chat)
python cogtrix.py --tools minimal                 # Basic set (file ops + calculate)
python cogtrix.py --tools "search_web,calculate"  # Specific tools only
```

### RAG Ingestion Options

```bash
python cogtrix.py --ingest [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--ingest` | Build vector database and exit |
| `--docs-dir PATH` | Documents directory |
| `--vectordb-dir PATH` | Vector database output directory |
| `--embedding-provider NAME` | Embedding provider: `openai` or `ollama` |
| `--embedding-model NAME` | Embedding model name |

---

## Complete Configuration Example

```json
{
  "provider": "my-server",
  "session": "default",

  "inference": {
    "my-server": {
      "type": "ollama",
      "base_url": "http://192.168.1.100:11434",
      "model": "llama3:70b",
      "temperature": 0.7,
      "num_ctx": 32768
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
    },
    "local-gpu": {
      "type": "ollama",
      "base_url": "http://192.168.1.101:11434",
      "model": "codellama:34b"
    }
  },

  "services": {
    "tavily":      { "api_key": "tvly-..." },
    "exa":         { "api_key": "exa-..." },
    "brave":       { "api_key": "BSA..." },
    "openweather": { "api_key": "..." }
  },

  "memory": {
    "mode": "conversation",
    "modes": {
      "conversation": { "working_memory_size": 20 },
      "code": { "working_memory_size": 8, "max_files": 20 },
      "reasoning": { "working_memory_size": 6, "max_decisions": 20 }
    }
  },

  "rag": {
    "docs_dir": "docs",
    "vectordb_dir": "data/vectordb",
    "embedding_provider": "local-gpu",
    "embedding_model": "nomic-embed-text"
  },

  "delegate": {
    "enabled": true,
    "default_timeout": 60,
    "model_aliases": {
      "fast": "my-server/llama3:8b",
      "smart": "openai/gpt-4o",
      "code": "local-gpu/codellama:34b"
    }
  }
}
```

> **Note:** This example uses `"inference"` (preferred). The legacy key `"providers"` still works.

---

## Interactive Commands

During an interactive session, slash commands provide quick access to session management and tools.

### Available Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help [cmd]` | `/h`, `/?` | List all commands, or detailed help for a specific command |
| `/quit` | `/exit`, `/q` | End the session (history is preserved) |
| `/info` | `/i` | Show session information (provider, model, mode, etc.) |
| `/tools [search]` | `/t` | List loaded tools, optionally filtered by name |
| `/think <task>` | — | Run deep Tree-of-Thought reasoning directly |
| `/mode [name]` | `/M` | Show / switch memory mode |
| `/model [name]` | `/m` | Show / switch LLM model |
| `/provider [name]` | `/p` | Show / switch LLM provider |
| `/session [id]` | `/s` | Show / switch session |
| `/debug` | — | Toggle debug mode |
| `/verbose` | `/v` | Toggle verbose logging |
| `/noconfirm` | `/y` | Toggle tool auto-approval |
| `/paste` | — | Enter multi-line paste mode |
| `/clear` | — | Clear conversation history (cannot be undone) |

Legacy bare commands `exit`, `quit`, and `q` (without `/`) still work for backward compatibility.

**Note:** Commands with `[name]` or `[id]` arguments work in two modes: without arguments they display current state, with an argument they switch to the specified value at runtime.

### Line Editing

The interactive prompt supports full line editing via Python's `readline` module:

- **Left/Right arrows** — Move cursor within the line
- **Home/End** — Jump to beginning/end of line
- **Up/Down arrows** — Navigate input history
- **Ctrl+A / Ctrl+E** — Beginning/end of line (Emacs-style)
- **Ctrl+W** — Delete previous word

This works out of the box on Linux and macOS. On Windows, install `pyreadline3` for equivalent functionality.

---

## Debugging & Logging

Enable logging to troubleshoot issues:

```bash
# Enable logging to default file (cogtrix.log)
python cogtrix.py --log

# Enable logging to specific file
python cogtrix.py --log ~/my-logs/session.log

# Log full LLM interactions (tokens, thinking, tool calls)
python cogtrix.py --log -v

# Enable debug mode (auto-enables --log and --verbose)
python cogtrix.py --debug
python cogtrix.py --debug --log ~/debug.log
```

### Log Levels

| Mode | Level | What's Logged |
|------|-------|---------------|
| `--log` | INFO | User messages, agent responses, tool calls, errors |
| `--log -v` | INFO | Above plus: full LLM interactions, tokens, thinking content |
| `--debug` | DEBUG | All of the above plus: message details, context info, tool inputs/outputs |

### What Gets Logged

| Event | Level | Example |
|-------|-------|---------|
| User message | INFO | `User: What's the weather?` |
| Agent response | INFO | `Agent response` |
| Tool execution | INFO | `Tool: get_weather` |
| Tool input | DEBUG | `Tool input: {'location': 'Auckland'}` |
| Tool output | DEBUG | `Tool output: Current weather in...` |
| Memory context | DEBUG | `Context: mode=conversation, 10 messages` |
| Errors | ERROR | `Tool failed: get_weather - Connection error` |

### Example Log Output

```
2025-01-15 10:30:15.123 [INFO] [a1b2c3d4] User: What's the weather in Auckland?
2025-01-15 10:30:15.124 [DEBUG] [a1b2c3d4] Context: mode=conversation, 5 messages, ~1200 tokens
2025-01-15 10:30:16.500 [INFO] [a1b2c3d4] Tool: get_weather
2025-01-15 10:30:16.500 [DEBUG] [a1b2c3d4] Tool input: {'location': 'Auckland, New Zealand', 'units': 'metric'}
2025-01-15 10:30:17.200 [DEBUG] [a1b2c3d4] Tool output: Current weather in Auckland: 18°C, partly cloudy...
2025-01-15 10:30:18.500 [INFO] [a1b2c3d4] Agent response
```

The `[a1b2c3d4]` is a request ID that groups all log entries for a single user query.

### Debugging Tips

1. **Tool not being called?** Check if the agent outputs JSON text instead of calling the tool. This may indicate conversation history issues — try a fresh session with `-s new_session`.

2. **Timeout errors?** The model may be slow. Check the provider's status and consider using a faster model.

3. **Connection errors?** Verify the provider URL and that the service is running.

---

## See Also

- [PROVIDERS.md](PROVIDERS.md) — Provider setup guides
- [MEMORY_MODES.md](MEMORY_MODES.md) — Memory mode details
- [RAG_GUIDE.md](RAG_GUIDE.md) — Knowledge base setup
