"""
Memory mode implementations.

Each mode provides a specialized memory management strategy:
- conversation: General chat, Q&A, research
- code: Programming assistance, debugging
- reasoning: Strategic planning, decision-making

Modes are auto-registered with MemoryFactory when imported.
"""

from cogtrix_core.memory.factory import MemoryFactory
from cogtrix_core.memory.modes.code import CodeDevelopmentMemoryManager
from cogtrix_core.memory.modes.conversation import ConversationMemoryManager
from cogtrix_core.memory.modes.reasoning import ReasoningMemoryManager

# Register all modes
MemoryFactory.register("conversation", ConversationMemoryManager)
MemoryFactory.register("code", CodeDevelopmentMemoryManager)
MemoryFactory.register("reasoning", ReasoningMemoryManager)

__all__ = [
    "ConversationMemoryManager",
    "CodeDevelopmentMemoryManager",
    "ReasoningMemoryManager",
]
