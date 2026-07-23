"""
Cogtrix Memory System

Provides modular memory management with multiple modes:
- conversation: General chat and Q&A
- code: Programming assistance
- reasoning: Strategic planning

Usage:
    from src.memory import MemoryFactory, JsonFileMemoryStore

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
from src.memory import modes  # noqa: F401
from src.memory.base import BaseMemoryStore
from src.memory.context import MemoryContext
from src.memory.factory import MemoryFactory
from src.memory.json_store import JsonFileMemoryStore
from src.memory.manager import BaseMemoryManager

__all__ = [
    "BaseMemoryStore",
    "JsonFileMemoryStore",
    "MemoryContext",
    "BaseMemoryManager",
    "MemoryFactory",
]
