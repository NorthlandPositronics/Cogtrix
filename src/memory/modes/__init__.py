"""
Memory mode implementations.

Each mode provides a specialized memory management strategy:
- conversation: General chat, Q&A, research
- code: Programming assistance, debugging
- reasoning: Strategic planning, decision-making

Modes are auto-registered with MemoryFactory when imported.
"""

from src.memory.factory import MemoryFactory
from src.memory.modes.code import CodeDevelopmentMemoryManager
from src.memory.modes.conversation import ConversationMemoryManager
from src.memory.modes.reasoning import ReasoningMemoryManager

# Register all modes
MemoryFactory.register("conversation", ConversationMemoryManager)
MemoryFactory.register("code", CodeDevelopmentMemoryManager)
MemoryFactory.register("reasoning", ReasoningMemoryManager)

__all__ = [
    "ConversationMemoryManager",
    "CodeDevelopmentMemoryManager",
    "ReasoningMemoryManager",
]
