"""
Memory context structure returned by memory managers.
Contains prepared messages and metadata for LLM consumption.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryContext:
    """
    Context prepared by a memory manager for LLM consumption.

    This is the standard interface between memory managers and the agent.
    All memory managers must return this structure from prepare_context().

    Attributes:
        messages: List of message objects to send to LLM (LangChain format)
        system_additions: Extra content to append to system prompt
        context_prefix: Context to prepend before conversation messages
        mode: Name of the memory mode that created this context
        total_messages_stored: Total messages in storage (for display)
        context_messages_count: Messages included in this context
        token_estimate: Rough estimate of tokens (optional)
        metadata: Additional mode-specific metadata
    """

    # Core content
    messages: list[Any] = field(default_factory=list)
    system_additions: str | None = None
    context_prefix: str | None = None

    # Metadata
    mode: str = "unknown"
    total_messages_stored: int = 0
    context_messages_count: int = 0
    token_estimate: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.context_messages_count == 0 and self.messages:
            self.context_messages_count = len(self.messages)

    def has_context_prefix(self) -> bool:
        """Check if context prefix is present."""
        return bool(self.context_prefix)

    def has_system_additions(self) -> bool:
        """Check if system additions are present."""
        return bool(self.system_additions)
