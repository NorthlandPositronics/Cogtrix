# Authoring Cogtrix Tools

This guide explains how to add new tools to Cogtrix — either by dropping a `.py`
file into a directory (file-drop) or by packaging a tool as an installable Python
package (entry-point).

---

## Quick overview

Cogtrix discovers tools from three sources, in this order:

| Track | How | Use when |
|-------|-----|----------|
| **Built-in** | `src/tools/*.py` in the package | Contributing to the project itself |
| **File-drop** | `*.py` files in a configured directory | Local one-off tools, rapid prototyping |
| **Entry-point** | Installed package declares `cogtrix.tools` EP | Distributable, versioned tool packages |

---

## 1. Tool module format

Every tool module — built-in, file-drop, or entry-point — uses the same format.

### Minimal example

```python
# my_tools.py
from pydantic import BaseModel, Field


class SayHelloInput(BaseModel):
    name: str = Field(..., description="Name to greet.")


def say_hello(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"


TOOL_CONFIGS = [
    {
        "name": "say_hello",
        "description": "Greet a person by name.",
        "input_schema": SayHelloInput,
        "function": say_hello,
        "requires_confirmation": False,
    }
]
```

### TOOL_CONFIG vs TOOL_CONFIGS

| Variable | Use |
|----------|-----|
| `TOOL_CONFIGS` | List of dicts — preferred when a module provides multiple tools |
| `TOOL_CONFIG` | Single dict — shorthand for single-tool modules |

Each config dict must contain:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | `str` | ✅ | Unique tool name (lowercase, underscores) |
| `description` | `str` | ✅ | One-sentence description shown in `/tools` |
| `input_schema` | `BaseModel` subclass | ✅ | Pydantic schema for the tool's arguments |
| `function` | `callable` | ✅ | The Python function that implements the tool |
| `requires_confirmation` | `bool` | ❌ | If `True`, user must approve before execution (default `False`) |

### Optional: TOOL_SETUP

If your module needs access to the Cogtrix `Config` object at load time (e.g. to
read API keys from the config file), define a `TOOL_SETUP` function:

```python
def TOOL_SETUP(config) -> None:
    """Called automatically by ToolRegistry after this module is loaded."""
    global _api_key
    _api_key = config.services.get("my_service", {}).get("api_key", "")
```

`TOOL_SETUP` is called once per process, right after the module is imported and
before tools are registered.

### Optional: is_configured

If your tool requires external credentials or dependencies that may not be
available, define `is_configured()` to gate registration:

```python
def is_configured() -> bool:
    """Return True only if the tool is ready to use."""
    return bool(_api_key)
```

When `is_configured()` returns `False`, the module is silently skipped and its
tools are not added to the catalog.

---

## 2. File-drop tools

The simplest way to add a custom tool is to drop a `.py` file into a directory
and tell Cogtrix to scan it.

### Configure the scan directory

In your Cogtrix config file (`.cogtrix.yaml`):

```yaml
tool_dirs:
  - ~/.cogtrix/tools          # user-specific tools
  - /opt/company/cogtrix/tools  # team-shared tools
```

Or use the environment variable (colon-separated):

```bash
export COGTRIX_TOOL_DIRS="$HOME/.cogtrix/tools:/opt/company/cogtrix/tools"
```

### Rules for file-drop modules

- File name must end in `.py`
- Files whose names start with `_` are skipped (use this for helpers)
- Symlinks that resolve outside the declared directory are rejected
- The module is imported with a synthetic name `cogtrix_plugin_<stem>` so it
  never conflicts with built-in tool modules

### Example directory layout

```
~/.cogtrix/tools/
    my_calendar.py    ← loaded
    jira_tools.py     ← loaded
    _helpers.py       ← skipped (leading underscore)
```

---

## 3. Entry-point (installable) tools

For tools you want to distribute as a Python package:

### Step 1 — Write your plugin class

```python
# my_cogtrix_plugin/tools.py
from pydantic import BaseModel, Field
from cogtrix.plugins import hookimpl   # or: from src.plugins import hookimpl


class EchoInput(BaseModel):
    text: str = Field(..., description="Text to echo back.")


def echo(text: str) -> str:
    """Echo text back to the caller."""
    return text


class MyPlugin:
    @hookimpl
    def cogtrix_tools(self) -> list[dict]:
        return [
            {
                "name": "echo",
                "description": "Echo text back unchanged.",
                "input_schema": EchoInput,
                "function": echo,
            }
        ]
```

> **Note:** `hookimpl` is the [pluggy](https://pluggy.readthedocs.io/) hook
> implementation marker.  It is optional but recommended because it future-proofs
> your plugin against hook signature changes.

### Step 2 — Declare the entry-point

In `pyproject.toml`:

```toml
[project.entry-points."cogtrix.tools"]
my_plugin = "my_cogtrix_plugin.tools:MyPlugin"
```

In `setup.cfg`:

```ini
[options.entry_points]
cogtrix.tools =
    my_plugin = my_cogtrix_plugin.tools:MyPlugin
```

### Step 3 — Install and run

```bash
pip install .          # or: uv add .
cogtrix               # MyPlugin's tools appear in /tools
```

### Alternative: point directly to a module

If your module already has `TOOL_CONFIGS` at the top level, the entry-point can
point directly to the module (no class needed):

```toml
[project.entry-points."cogtrix.tools"]
my_plugin = "my_cogtrix_plugin.tools"
```

Cogtrix detects that the loaded object is a module and reads `TOOL_CONFIGS` or
`TOOL_CONFIG` from it directly.

---

## 4. Testing your tool

### Unit test

```python
# tests/test_my_tool.py
from my_cogtrix_plugin.tools import echo

def test_echo():
    assert echo("hello") == "hello"
```

### Integration test (file-drop)

```python
import tempfile, textwrap, pathlib
from src.plugins.loader import ToolPluginLoader

def test_file_drop():
    src = textwrap.dedent('''
        from pydantic import BaseModel, Field

        class PingInput(BaseModel):
            msg: str = Field(..., description="Message.")

        def ping(msg: str) -> str:
            return msg

        TOOL_CONFIGS = [{
            "name": "ping",
            "description": "Ping.",
            "input_schema": PingInput,
            "function": ping,
        }]
    ''')
    with tempfile.TemporaryDirectory() as tmp:
        pathlib.Path(tmp, "ping.py").write_text(src)
        loader = ToolPluginLoader()
        modules = loader.load_all([tmp])
    assert len(modules) == 1
    assert modules[0].TOOL_CONFIGS[0]["name"] == "ping"
```

---

## 5. Reference: discovery order

1. Built-in tools in `src/tools/` are always loaded first.
2. File-drop directories are scanned in the order they appear in `tool_dirs`.
3. Entry-point plugins are loaded last (order within the group is implementation-defined).

If two tools share the same `name`, the last one registered wins with a WARNING
logged.  Use unique, namespaced names to avoid collisions (e.g. `acme_search`
rather than `search`).

---

## 6. Reference: full TOOL_CONFIG schema

```python
{
    # Required
    "name": str,               # "my_tool"
    "description": str,        # "Does X given Y."
    "input_schema": type,      # class MyToolInput(BaseModel): ...
    "function": callable,      # def my_tool(...) -> str: ...

    # Optional
    "requires_confirmation": bool,  # default False
}
```

The `function` must be synchronous (no `async def`).  For async operations, run
them with `asyncio.run()` or `loop.run_until_complete()` inside the function body.

---

## 7. Error string conventions

### Path-policy errors (file-touching tools)

Tools that touch the filesystem (`read_file`, `write_file`, `list_directory`, or
any custom tool that resolves paths) should emit **canonical** error strings
when a path falls outside the permitted area, so the agent recognises the same
failure class consistently regardless of which tool produced it. The canonical
strings live in `src/tools/_path_policy.py`:

```python
from src.tools._path_policy import (
    format_write_outside_error,   # path outside writable area
    format_read_outside_error,    # path outside readable area
    format_traversal_error,       # ../ escape attempt
    is_path_policy_error,         # downstream classifier
)

def my_tool_that_writes(path: str) -> str:
    if not _is_safe_write_path(path):
        return format_write_outside_error(path)
    # ... proceed with write ...
```

The canonical strings start with `"Error: "` so the existing tool-error
detection in `tests/evaluation/runner.py` continues to classify them. They
each include `{path}` so the agent sees exactly what was rejected. `Permission
denied: <path>` produced by an OS-level `PermissionError` is intentionally
kept as a *separate* class — it signals a real environment issue rather than
the agent picking a bad path.

`is_path_policy_error(message)` is the recommended classifier for downstream
consumers (dispatcher diagnostics, test harness) that want to distinguish
agent-recoverable path choices from environmental failures.

### Dispatcher-synthesised ToolMessage kinds

The `process_tools` dispatcher uses `additional_kwargs["cogtrix.kind"]` to mark
ToolMessages it synthesises when a tool call cannot actually execute (tool
not loaded, name unresolvable, denied by user, ...). The four kinds are
defined in `src/orchestration/tool_message_kinds.py`:

| Kind | Trigger |
|---|---|
| `tool_not_loaded` | Match exists in catalog; agent should issue `request_tools(add=...)` |
| `tool_disabled` | `session_state.is_denied(match)` |
| `tool_name_invalid` | Fuzzy match points to an already-active tool ("Did you mean 'X'?" hint) |
| `tool_resolution_failed` | No match found |

Most tool authors don't need to interact with this set — the dispatcher
attaches the kind automatically when it synthesises a failure response.
**You only need to know about it if your tool produces ToolMessages itself**
(for example, a meta-tool that dispatches to sub-tools and wraps their
results, or a tool that intercepts and rewrites tool-call traffic). In that
case, attach the appropriate kind so the fabricated-success detector in
`response_detectors.py` recognises the synthetic failure:

```python
from langchain_core.messages import ToolMessage
from src.orchestration.tool_message_kinds import (
    COGTRIX_KIND_KEY,
    KIND_TOOL_RESOLUTION_FAILED,
)

ToolMessage(
    content=f"'{tool_name}' is not a valid tool and could not be resolved.",
    tool_call_id=call["id"],
    name=tool_name,
    additional_kwargs={COGTRIX_KIND_KEY: KIND_TOOL_RESOLUTION_FAILED},
)
```

Without the kind marker, a downstream detector falls back to substring
matching on the content text — which drifts over time as messages are
rephrased. The kind is the structural signal that won't drift.

See `src/orchestration/tool_message_kinds.py` for the full set of kinds
and `TOOL_RESOLUTION_FAILURE_KINDS` (the canonical set the detector chain
treats as tool-error events).
