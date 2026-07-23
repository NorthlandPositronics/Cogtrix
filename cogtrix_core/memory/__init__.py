"""
Cogtrix Memory System

Provides modular memory management with multiple modes:
- conversation: General chat and Q&A
- code: Programming assistance
- reasoning: Strategic planning

Usage:
    from cogtrix_core.memory import MemoryFactory, JsonFileMemoryStore

    store = JsonFileMemoryStore()
    manager = MemoryFactory.create("conversation", store, "session-123")
    manager.load()

    context = manager.prepare_context(user_input)
    # ... send context.messages to LLM ...
    manager.update(user_input, ai_response)
    manager.save()
"""

# Import modes to trigger registration
# This is done at the end to avoid circular imports
from cogtrix_core.memory import modes  # noqa: F401
from cogtrix_core.memory.base import BaseMemoryStore
from cogtrix_core.memory.context import MemoryContext
from cogtrix_core.memory.factory import MemoryFactory
from cogtrix_core.memory.json_store import JsonFileMemoryStore
from cogtrix_core.memory.manager import BaseMemoryManager
from cogtrix_core.memory.recall import SessionVectorStore
from cogtrix_core.memory.summarizer import generate_summary

__all__ = [
    "BaseMemoryStore",
    "JsonFileMemoryStore",
    "MemoryContext",
    "BaseMemoryManager",
    "MemoryFactory",
    "SessionVectorStore",
    "generate_summary",
]
