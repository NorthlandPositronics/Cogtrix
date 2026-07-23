# Cogtrix Development Guide

Everything you need to extend Cogtrix with new tools, memory modes, or slash commands. If you want to understand the system architecture first, read [Architecture](ARCHITECTURE.md). For contribution guidelines and code style rules, see [CONTRIBUTING](../CONTRIBUTING.md).

## Table of Contents

- [Adding Custom Tools](#adding-custom-tools)
- [Tool Examples](#tool-examples)
- [Adding Memory Modes](#adding-memory-modes)
- [Adding Slash Commands](#adding-slash-commands)
- [Testing](#testing)
- [Code Style](#code-style)
- [Project Structure](#project-structure)

---

## Adding Custom Tools

Tools are automatically discovered from `src/tools/`. To add a new tool:

### Step 1: Create Tool File

Create `src/tools/my_tool.py`:

```python
"""
My custom tool - Brief description.
"""

from pydantic import BaseModel, Field


class MyToolInput(BaseModel):
    """Input schema for my tool."""
    
    query: str = Field(
        description="The input query to process"
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results"
    )


def my_tool(query: str, max_results: int = 10) -> str:
    """
    Process a query and return results.
    
    This description is shown to the LLM to help it understand
    when and how to use this tool.
    
    Args:
        query: The input query to process
        max_results: Maximum number of results
        
    Returns:
        Processed results as a string
    """
    # Your implementation here
    result = f"Processed '{query}' with max {max_results} results"
    return result


# Tool configuration for registry
TOOL_CONFIG = {
    "name": "my_tool",
    "description": "Process queries and return results. Use this when you need to...",
    "input_schema": MyToolInput,
    "requires_confirmation": False,
}

__all__ = ["my_tool", "MyToolInput", "TOOL_CONFIG"]
```

### Step 2: Configuration Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | str | Yes | Function name (must match) |
| `description` | str | Yes | LLM-readable description |
| `input_schema` | BaseModel | Yes | Pydantic input schema |
| `requires_confirmation` | bool | Yes | Prompt user before execution |

### Step 3: Optional — API Key Gating

If your tool requires an external service or API key, add an `is_configured()` function. The registry checks this before loading — if it returns `False`, the tool is silently skipped (no error, just absent from the toolbox).

```python
def is_configured() -> bool:
    """Return True if the tool has the required credentials."""
    return bool(os.getenv("MY_SERVICE_API_KEY"))
```

This pattern is used by all search providers, weather, and WhatsApp tools.

### Step 4: Restart

The tool is automatically discovered:

```bash
python cogtrix.py
# ✓ Loaded 35 tool(s):
#   - my_tool
```

---

## Tool Examples

### Simple Tool (No Confirmation)

```python
from pydantic import BaseModel, Field


class GreetInput(BaseModel):
    name: str = Field(description="Name to greet")


def greet(name: str) -> str:
    """Generate a greeting for a person."""
    return f"Hello, {name}!"


TOOL_CONFIG = {
    "name": "greet",
    "description": "Generate a friendly greeting",
    "input_schema": GreetInput,
    "requires_confirmation": False,
}
```

### Sensitive Tool (Requires Confirmation)

```python
from pydantic import BaseModel, Field
import subprocess


class RunScriptInput(BaseModel):
    script_path: str = Field(description="Path to script")
    args: str = Field(default="", description="Script arguments")


def run_script(script_path: str, args: str = "") -> str:
    """Execute a script file."""
    cmd = f"{script_path} {args}".strip()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout or result.stderr


TOOL_CONFIG = {
    "name": "run_script",
    "description": "Execute a script file with optional arguments",
    "input_schema": RunScriptInput,
    "requires_confirmation": True,  # User must approve
}
```

### Multiple Tools Per File

Use `TOOL_CONFIGS` (list) instead of `TOOL_CONFIG`:

```python
from pydantic import BaseModel, Field


class AddInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


class MultiplyInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    return str(a * b)


TOOL_CONFIGS = [
    {
        "name": "add",
        "description": "Add two numbers together",
        "input_schema": AddInput,
        "requires_confirmation": False,
        "function": add,
    },
    {
        "name": "multiply",
        "description": "Multiply two numbers",
        "input_schema": MultiplyInput,
        "requires_confirmation": False,
        "function": multiply,
    },
]
```

### Tool with External API

```python
import os
import requests
from pydantic import BaseModel, Field


class StockPriceInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol (e.g., AAPL)")


def get_stock_price(symbol: str) -> str:
    """Get current stock price for a ticker symbol."""
    api_key = os.getenv("STOCK_API_KEY")
    if not api_key:
        return "Error: STOCK_API_KEY not configured"
    
    try:
        response = requests.get(
            f"https://api.example.com/stock/{symbol}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return f"{symbol}: ${data['price']:.2f}"
    except Exception as e:
        return f"Error fetching stock price: {e}"


TOOL_CONFIG = {
    "name": "get_stock_price",
    "description": "Get the current stock price for a ticker symbol",
    "input_schema": StockPriceInput,
    "requires_confirmation": False,
}
```

---

## Adding Memory Modes

### Step 1: Create Mode File

Create `src/memory/modes/custom.py`:

```python
"""Custom memory mode for specific use case."""

from typing import Any, Dict, List, Optional
from ..manager import BaseMemoryManager
from ..context import MemoryContext


class CustomMemoryManager(BaseMemoryManager):
    """Memory manager for custom use case."""
    
    def __init__(
        self,
        store,
        session_id: str = "default",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(store, session_id, config)
        self.working_memory_size = self.config.get("working_memory_size", 10)
        # Custom tracking
        self.custom_data: List[str] = []
    
    def prepare_context(self, user_input: str) -> MemoryContext:
        """Prepare context for LLM."""
        # Capture user-input timestamp (used later in update())
        self._pending_user_ts = self._now_ts()

        # Get recent messages
        messages = self._get_recent_messages(self.working_memory_size)

        # Inject timestamps so the LLM has temporal awareness
        messages = self._inject_timestamps(messages)
        
        # Build context prefix
        context_parts = []
        if self.custom_data:
            context_parts.append(f"Custom context: {self.custom_data}")
        
        context_prefix = "\n".join(context_parts) if context_parts else None
        
        return MemoryContext(
            messages=messages,
            context_prefix=context_prefix,
        )
    
    def update(self, user_input: str, ai_response: str) -> None:
        """Update memory after interaction."""
        # Create messages
        from langchain_core.messages import AIMessage, HumanMessage
        human_msg = HumanMessage(content=user_input)
        ai_msg = AIMessage(content=ai_response)

        # Stamp: user msg with ts captured in prepare_context(),
        # AI msg with current time (shows LLM processing duration)
        self._set_msg_ts(human_msg, self._pending_user_ts)
        self._pending_user_ts = None
        self._set_msg_ts(ai_msg)

        self._messages.append(human_msg)
        self._messages.append(ai_msg)
        
        # Custom tracking logic
        if "important" in user_input.lower():
            self.custom_data.append(user_input)
    
    def get_system_prompt_additions(self) -> str:
        """Additional instructions for LLM."""
        return "You are operating in custom mode. Focus on..."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mode statistics."""
        stats = super().get_stats()
        stats["custom_items"] = len(self.custom_data)
        return stats
```

### Step 2: Register in Factory

Edit `src/memory/factory.py`:

```python
from .modes.custom import CustomMemoryManager

class MemoryFactory:
    _modes = {
        "conversation": ConversationMemoryManager,
        "code": CodeDevelopmentMemoryManager,
        "reasoning": ReasoningMemoryManager,
        "custom": CustomMemoryManager,  # Add your mode
    }
```

### Step 3: Use the Mode

```bash
python cogtrix.py -M custom
```

---

## Adding Slash Commands

Slash commands are registered in `cogtrix.py` via `_build_slash_commands()`.

### Step 1: Add Handler Method

Add a static method to `SlashCommandRegistry`:

```python
@staticmethod
def _cmd_mycommand(self, args: str) -> str:
    """Handler for /mycommand."""
    # self = SlashCommandRegistry instance (has .config, .memory_manager, .tools)
    # args = everything after the command name (e.g. "/mycommand foo bar" → "foo bar")
    print(f"My command received: {args}")
    return "continue"  # or "quit" to exit
```

### Step 2: Register the Command

In `_build_slash_commands()`:

```python
reg.register(SlashCommand(
    name="mycommand",
    handler=SlashCommandRegistry._cmd_mycommand,
    short_help="Brief description for /help listing",
    long_help=(
        "  Usage: /mycommand [args]\n\n"
        "  Detailed description shown by /help mycommand.\n"
        "  Include usage examples here."
    ),
    aliases=["mc"],  # optional
))
```

### Handler Context

Handlers receive `self` (the registry instance) which provides access to:

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.config` | `Config` | Application configuration |
| `self.memory_manager` | `BaseMemoryManager` | Current memory manager |
| `self.tools` | `dict` | Loaded tools dictionary |

### Return Values

| Return | Effect |
|--------|--------|
| `"continue"` | Resume the input loop |
| `"quit"` | Exit the application |
| `"switch_mode:<name>"` | Switch memory mode at runtime |
| `"switch_model:<name>"` | Switch LLM model at runtime |
| `"switch_provider:<name>"` | Switch LLM provider at runtime |
| `"switch_session:<id>"` | Switch session at runtime |
| `"rebuild_callbacks"` | Rebuild observability callbacks |

---

## Testing

### Run All Tests

```bash
# Using uv (recommended)
uv run pytest tests/ -v

# Or with pip/venv
python -m pytest tests/ -v
```

### Run Specific Tests

```bash
# Memory tests
uv run pytest tests/memory/ -v

# Tool tests
uv run pytest tests/tools/ -v

# Config tests
uv run pytest tests/test_provider_config.py -v
```

### Writing Tests

Create `tests/tools/test_my_tool.py`:

```python
import pytest
from src.tools.my_tool import my_tool, MyToolInput


class TestMyTool:
    def test_basic_functionality(self):
        result = my_tool("test query")
        assert "test query" in result
    
    def test_with_max_results(self):
        result = my_tool("query", max_results=5)
        assert "5" in result
    
    def test_input_schema(self):
        # Test Pydantic validation
        input_obj = MyToolInput(query="test", max_results=10)
        assert input_obj.query == "test"
        assert input_obj.max_results == 10
    
    def test_invalid_input(self):
        with pytest.raises(ValueError):
            MyToolInput(query="", max_results=-1)
```

---

## Code Style

### Formatting

```bash
# Format with black
uv run black cogtrix.py src/ tests/

# Lint with ruff
uv run ruff check cogtrix.py src/ tests/

# Type check with pyright
uv run pyright cogtrix.py src/ tests/

# Security scan with bandit
uv run bandit -r src/ cogtrix.py -q
```

### Guidelines

1. **Type hints** — Use type hints for function signatures
2. **Docstrings** — Document all public functions and classes
3. **Line length** — Max 100 characters (configured in `pyproject.toml`)
4. **Imports** — Group: stdlib, third-party, local

### Example

```python
"""Module description."""

from pathlib import Path
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel

from src.config import Config


def process_data(
    input_data: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Process input data and return result.
    
    Args:
        input_data: List of strings to process
        config: Optional configuration dictionary
        
    Returns:
        Processed result as string
        
    Raises:
        ValueError: If input_data is empty
    """
    if not input_data:
        raise ValueError("input_data cannot be empty")
    
    # Implementation
    return "result"
```

---

## Project Structure

```
cogtrix/
├── cogtrix.py                # CLI entry point
├── pyproject.toml            # Project metadata & dependencies
├── uv.lock                   # Locked dependency versions
├── requirements.txt          # Pip-compatible deps (auto-generated)
│
├── src/
│   ├── config.py             # Configuration management
│   ├── registry.py           # Tool discovery & registration
│   ├── logging_config.py     # Logging infrastructure
│   ├── setup_wizard.py       # Interactive --setup wizard
│   │
│   ├── providers/
│   │   ├── __init__.py       # Registry: create_chat_model(), create_embeddings()
│   │   ├── defaults.py       # Default models, base URLs, env vars, presets
│   │   ├── openai.py         # OpenAI and compatible APIs
│   │   ├── ollama.py         # Ollama local inference
│   │   ├── anthropic.py      # Anthropic Claude
│   │   └── google.py         # Google Gemini
│   │
│   ├── agent/
│   │   ├── core.py           # LangGraph agent setup
│   │   └── safety.py         # Tool confirmation wrapper
│   │
│   ├── assistant/
│   │   ├── __init__.py      # Package exports
│   │   ├── channel.py       # Channel ABC + IncomingMessage
│   │   ├── channels/
│   │   │   ├── whatsapp.py  # WhatsApp via Waha
│   │   │   └── telegram.py  # Telegram Bot API
│   │   ├── session.py       # Chat session lifecycle
│   │   ├── handler.py       # Message → agent → reply
│   │   ├── poller.py        # Per-channel polling threads
│   │   ├── knowledge.py     # Cross-chat knowledge store
│   │   ├── guardrails.py    # Security guardrails (input/output/rate-limit/LLM judge)
│   │   └── service.py       # Main orchestrator
│   │
│   ├── memory/
│   │   ├── base.py           # Abstract base classes
│   │   ├── factory.py        # Memory mode factory
│   │   ├── manager.py        # Base memory manager + hybrid memory logic
│   │   ├── context.py        # Context data structures
│   │   ├── json_store.py     # JSON file persistence
│   │   ├── summarizer.py     # LLM-based incremental summarization
│   │   ├── recall.py         # Per-session FAISS vector store
│   │   └── modes/
│   │       ├── conversation.py  # General chat mode
│   │       ├── code.py          # Code development mode
│   │       └── reasoning.py     # Planning/reasoning mode
│   │
│   ├── rag/
│   │   └── ingest.py         # Document ingestion
│   │
│   └── tools/                # Built-in tool modules (52 tools)
│       ├── brave_search.py   # Brave Search API
│       ├── calculator.py     # Math expressions
│       ├── datetime_tool.py  # Date/time utilities
│       ├── deep_think.py     # Tree-of-Thought reasoning
│       ├── delegate.py       # Task delegation
│       ├── exa_search.py     # Exa semantic search
│       ├── file_ops.py       # File operations
│       ├── google_search.py  # Google Custom Search
│       ├── http_request.py   # HTTP requests
│       ├── json_tool.py      # JSON processing
│       ├── nlp_tools.py      # NLP (sentiment, summarization)
│       ├── python_exec.py    # Python execution
│       ├── rag.py            # Knowledge base queries
│       ├── serpapi_search.py # SerpAPI (Google/Bing)
│       ├── shell.py          # Shell commands
│       ├── tavily_search.py  # Tavily AI search
│       ├── text_tools.py     # Text processing
│       ├── weather.py        # Weather information
│       ├── web_search.py     # DuckDuckGo search
│       ├── whatsapp.py       # WhatsApp messaging
│       ├── _whatsapp_client.py # Waha HTTP client
│       ├── telegram.py       # Telegram messaging
│       └── _telegram_client.py # Telegram Bot API client
│
├── tests/
│   ├── memory/               # Memory mode tests
│   ├── tools/                # Tool tests
│   ├── test_assistant_*.py   # Assistant mode tests (including guardrails)
│   ├── test_setup_wizard.py  # Setup wizard tests
│   └── test_*.py             # Config & integration tests
│
├── docs/                     # Documentation
│
└── data/                     # Runtime data
    ├── history/              # Session history + hybrid meta files
    ├── knowledge/            # Cross-chat knowledge store (facts.json)
    └── vectordb/             # FAISS vector indexes (RAG + per-session recall + knowledge)
```

---

## Contributing

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run all checks:
   ```bash
   uv run black .
   uv run ruff check .
   uv run pytest tests/ -v
   ```
5. Submit pull request

### Commit Messages

```
type: brief description

- Detail 1
- Detail 2
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) — System internals
- [TOOLS_REFERENCE.md](TOOLS_REFERENCE.md) — Existing tools
- [CONFIGURATION.md](CONFIGURATION.md) — Configuration options
