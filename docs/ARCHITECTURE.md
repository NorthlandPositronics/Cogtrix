# Cogtrix Architecture

System design, components, and internal workings of Cogtrix.

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Extension Points](#extension-points)
- [Security Model](#security-model)

---

## System Overview

Cogtrix is a modular LangChain-based AI agent built with a layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLI Interface Layer                         │
│                         (cogtrix.py)                               │
│  • Interactive & non-interactive modes  • Slash command system   │
│  • Session management  • Line editing (readline)                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                     Configuration Layer                          │
│                       (src/config.py)                            │
│  • Multi-provider config  • Priority resolution  • Validation    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                      Agent Core Layer                            │
│                    (src/agent/core.py)                           │
│  • LangGraph ReAct agent  • LLM orchestration  • Tool execution  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                       Safety Layer                               │
│                    (src/agent/safety.py)                         │
│  • Tool confirmation  • Approval tracking  • Execution control   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                      Tool Registry                               │
│                     (src/registry.py)                            │
│  • Dynamic discovery  • Tool wrapping  • Metadata management     │
└───────────────┬─────────────────────────────────┬───────────────┘
                │                                 │
┌───────────────┴───────────────┐ ┌───────────────┴───────────────┐
│        Memory System          │ │        Tool Modules           │
│       (src/memory/)           │ │       (src/tools/)            │
│  • Mode managers              │ │  • 43 built-in tools          │
│  • Context preparation        │ │  • Auto-discovery             │
│  • JSON persistence           │ │  • Pydantic schemas           │
└───────────────────────────────┘ └───────────────────────────────┘
```

---

## Component Architecture

### Directory Structure

```
src/
├── config.py              # Configuration management
├── registry.py            # Tool discovery and registration
├── logging_config.py      # Logging infrastructure
├── agent/
│   ├── core.py            # LangGraph agent setup
│   └── safety.py          # Tool confirmation wrapper
├── memory/
│   ├── base.py            # Abstract base classes
│   ├── factory.py         # Memory mode factory
│   ├── manager.py         # Base memory manager
│   ├── context.py         # Context data structures
│   ├── json_store.py      # JSON file persistence
│   └── modes/
│       ├── conversation.py  # General chat mode
│       ├── code.py          # Code development mode
│       └── reasoning.py     # Planning/reasoning mode
├── rag/
│   ├── __init__.py        # RAG module
│   └── ingest.py          # Document ingestion
└── tools/
    ├── brave_search.py    # Brave Search API
    ├── calculator.py      # Math expressions
    ├── datetime_tool.py   # Date/time utilities
    ├── deep_think.py      # Tree-of-Thought reasoning
    ├── delegate.py        # Task delegation
    ├── exa_search.py      # Exa semantic search (3 tools)
    ├── file_ops.py        # File operations
    ├── google_search.py   # Google Custom Search API
    ├── http_request.py    # HTTP requests
    ├── json_tool.py       # JSON processing
    ├── nlp_tools.py       # NLP (sentiment, summarization)
    ├── python_exec.py     # Python execution
    ├── rag.py             # Knowledge base queries
    ├── serpapi_search.py  # SerpAPI (Google/Bing structured)
    ├── shell.py           # Shell commands
    ├── tavily_search.py   # Tavily AI search (2 tools)
    ├── text_tools.py      # Text processing
    ├── weather.py         # Weather information
    └── web_search.py      # DuckDuckGo search
```

---

## Core Components

### 1. Configuration (`src/config.py`)

Manages multi-source configuration with priority resolution.

**Key Classes:**

```python
@dataclass
class ProviderConfig:
    name: str
    type: str              # "openai" or "ollama"
    base_url: Optional[str]
    model: Optional[str]
    api_key: Optional[str]
    temperature: Optional[float]
    num_ctx: Optional[int]  # Context window size (Ollama only)

@dataclass
class Config:
    provider: str
    model: Optional[str]
    session: str
    providers: Dict[str, ProviderConfig]
    memory_mode: str
    rag: RAGConfig
    debug: bool
    verbose: bool
    log_file: Optional[str]
    # ... more fields
```

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `load_config(cli_args)` | Load and merge configuration from all sources |
| `find_config_file()` | Locate `.cogtrix.json` in cwd or home |
| `Config.get_provider_config(name)` | Get provider configuration by name |
| `Config.list_providers()` | List all available provider names |

### 2. CLI Interface (`cogtrix.py`)

The entry point handles both interactive and non-interactive modes.

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `SlashCommandRegistry` | Registers, resolves, and dispatches `/commands` |
| `SlashCommand` | Dataclass: name, handler, help text, aliases |
| `ToolCallLogger` | `BaseCallbackHandler` for LLM/tool observability |
| `readline` import | Enables arrow keys, Home/End, input history |

**Slash Command Dispatch:**

```
User Input
    │
    ├── Starts with "/"  → SlashCommandRegistry.dispatch()
    │                        ├── /help, /info, /tools, /mode, /model, /provider
    │                        ├── /session, /debug, /verbose, /noconfirm, /clear
    │                        ├── /think <task> → deep_think() directly
    │                        ├── /paste → multi-line input mode
    │                        └── /quit → exit
    │
    └── Regular text     → Agent processing pipeline
```

**Built-in Commands:**

| Command | Aliases | Description |
|---------|---------|-------------|
| `/help` | `/h`, `/?` | List commands or detailed help |
| `/quit` | `/exit`, `/q` | End session |
| `/info` | `/i` | Session information (provider, model, mode) |
| `/tools` | `/t` | List loaded tools |
| `/think` | — | Deep Tree-of-Thought reasoning |
| `/mode` | `/M` | Show / switch memory mode |
| `/model` | `/m` | Show / switch LLM model |
| `/provider` | `/p` | Show / switch LLM provider |
| `/session` | `/s` | Show / switch session |
| `/debug` | — | Toggle debug mode |
| `/verbose` | `/v` | Toggle verbose logging |
| `/noconfirm` | `/y` | Toggle tool auto-approval |
| `/paste` | — | Enter multi-line paste mode |
| `/clear` | — | Clear conversation history |

### 3. Agent Core (`src/agent/core.py`)

Builds and configures the LangGraph ReAct agent.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `build_agent_executor(tools, ...)` | Create compiled agent with tools |
| `create_llm_from_provider_config(config)` | LLM factory from ProviderConfig |
| `build_system_prompt(mode_additions)` | Generate system prompt with mode context |
| `prepare_messages_with_context(...)` | Prepare messages for agent invocation |

**Agent Architecture:**

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│         LangGraph ReAct Agent       │
│  ┌─────────────────────────────┐    │
│  │     System Prompt           │    │
│  │  (base + mode additions)    │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │          LLM                │    │
│  │  (OpenAI/Ollama/Custom)     │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │      Tool Bindings          │    │
│  │  (43 tools with schemas)    │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    │
    ▼
Agent Response
```

### 4. Safety Layer (`src/agent/safety.py`)

Wraps sensitive tools with confirmation prompts.

**Mechanism:**

1. Tools marked with `requires_confirmation: True` are wrapped
2. Wrapper intercepts execution and prompts user
3. User can approve once (`y`), deny (`n`), or approve for session (`all`)
4. Approvals tracked in session-scoped set

**Sensitive Tools:**
- `execute_shell_command`
- `execute_python`
- `write_file`
- `append_file`
- `http_post`

### 5. Tool Registry (`src/registry.py`)

Discovers and loads tools dynamically.

**Discovery Process:**

```
1. Scan src/tools/ directory
       │
       ▼
2. Import each .py module
       │
       ▼
3. Look for TOOL_CONFIG or TOOL_CONFIGS
       │
       ▼
4. Extract function, schema, metadata
       │
       ▼
5. Create LangChain StructuredTool
       │
       ▼
6. Register in tools dictionary
```

**Tool Configuration Format:**

```python
TOOL_CONFIG = {
    "name": "function_name",
    "description": "What this tool does",
    "input_schema": PydanticModel,
    "requires_confirmation": False,
}
```

### 6. Memory System (`src/memory/`)

Pluggable memory management with multiple modes.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      MemoryFactory                               │
│                   create(mode, store, ...)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Conversation  │     │     Code      │     │   Reasoning   │
│    Memory     │     │   Memory      │     │    Memory     │
│   Manager     │     │   Manager     │     │   Manager     │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ JsonFileMemory  │
                    │     Store       │
                    └─────────────────┘
```

**Base Interface:**

```python
class BaseMemoryManager(ABC):
    def prepare_context(self, user_input: str) -> MemoryContext
    def update(self, user_input: str, ai_response: str) -> None
    def save(self) -> None
    def load(self) -> None
    def get_system_prompt_additions(self) -> str
```

### 7. RAG System (`src/rag/`)

Document ingestion and knowledge base queries.

**Ingestion Pipeline:**

```
Documents (docs/)
       │
       ▼
┌─────────────────┐
│  Document       │  PDF, Markdown, CSV, TXT
│  Loaders        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text           │  chunk_size: 1200
│  Splitter       │  chunk_overlap: 200
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embeddings     │  OpenAI or Ollama
│  Model          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS          │  data/vectordb/faiss_index
│  Vector Store   │
└─────────────────┘
```

---

## Data Flow

### Request Processing Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Memory: prepare_context()        │
│    - Get working memory             │
│    - Build context prefix           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Agent: invoke()                  │
│    - Process with LLM               │
│    - Select and execute tools       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Safety: check confirmation       │
│    - Prompt if required             │
│    - Execute or deny                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Memory: update()                 │
│    - Add to history                 │
│    - Update mode-specific tracking  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. Memory: save()                   │
│    - Persist to JSON file           │
└─────────────────────────────────────┘
    │
    ▼
Response to User
```

### Tool Execution Flow

```
Agent decides to use tool
    │
    ▼
┌─────────────────────────────────────┐
│ Tool Registry: get tool             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Safety Wrapper: check               │
│                                     │
│ Requires confirmation?              │
│   ├── No  → Execute directly        │
│   └── Yes → Prompt user             │
│             ├── y   → Execute once  │
│             ├── all → Add to set    │
│             └── n   → Return denial │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Tool Function: execute              │
└─────────────────────────────────────┘
    │
    ▼
Result to Agent
```

---

## Extension Points

### Adding a New Tool

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed instructions.

### Adding a New Memory Mode

1. Create `src/memory/modes/my_mode.py`
2. Extend `BaseMemoryManager`
3. Implement required methods
4. Register in `src/memory/factory.py`

### Adding a New Provider Type

Currently supports `openai` and `ollama` types. To add a new type:

1. Update `src/agent/core.py`:
   - Add case in `create_llm_from_provider_config()`
2. Update `src/config.py`:
   - Add validation in `_parse_providers_section()`

### Adding a New Interface

The agent core is interface-agnostic:

```python
from src.config import load_config
from src.registry import ToolRegistry
from src.agent.core import build_agent_executor, create_llm_from_provider_config
from src.memory import MemoryFactory, JsonFileMemoryStore

# Load config
config = load_config()

# Setup tools
registry = ToolRegistry()
registry.load_all_tools()
tools = list(registry.tools.values())

# Create LLM and agent
provider_config = config.get_provider_config()
llm = create_llm_from_provider_config(provider_config)
agent = build_agent_executor(tools, llm=llm)

# Setup memory
memory_store = JsonFileMemoryStore()
memory_manager = MemoryFactory.create(mode="conversation", store=memory_store)
memory_manager.load()

# Use agent
context = memory_manager.prepare_context(user_input)
result = agent.invoke({"messages": context.messages})
memory_manager.update(user_input, str(result))
memory_manager.save()
```

---

## Security Model

### Tool Safety

| Category | Tools | Confirmation |
|----------|-------|--------------|
| Read-only | `read_file`, `list_directory`, `search_web` | No |
| Sensitive | `execute_shell_command`, `write_file`, `execute_python` | Yes |
| External | `http_post` | Yes |

### API Key Management

- Store in environment variables (preferred)
- Or in config file (`.cogtrix.json` / `.cogtrix.yml`)
- Never commit to version control (`.gitignore` excludes config files and `.env*`)
- Keys hidden in logs (`api_key: "***"`)

### Session Isolation

- Each session has separate memory file
- Approvals are session-scoped (not persisted)
- History stored in `data/history/{session_id}.json`

---

## Dependencies

Dependencies are managed via `pyproject.toml` (with `uv`) and exported to `requirements.txt` for pip compatibility.

### Core

| Package | Purpose |
|---------|---------|
| `langchain-core` | Base LangChain functionality |
| `langchain-openai` | OpenAI LLM integration |
| `langchain-ollama` | Ollama LLM integration |
| `langchain-community` | Community tool integrations |
| `langgraph` | ReAct agent implementation |
| `pydantic` | Schema validation |
| `python-dotenv` | Environment variable loading |

### Tools

| Package | Purpose |
|---------|---------|
| `ddgs` | DuckDuckGo search |
| `textblob` | NLP tools |
| `faiss-cpu` | Vector store |
| `pypdf` | PDF loading |
| `python-docx` | DOCX file support |
| `tiktoken` | Token counting |

### CLI

| Package | Purpose |
|---------|---------|
| `rich` | Terminal formatting |

---

## See Also

- [CONFIGURATION.md](CONFIGURATION.md) — Configuration reference
- [DEVELOPMENT.md](DEVELOPMENT.md) — Extension guide
- [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md) — Tool documentation
