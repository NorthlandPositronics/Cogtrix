"""
Conversation memory mode for general chat, Q&A, and research.

This is the default memory mode, providing:
- Working memory: Last N messages in context
- Session summary: Compressed older messages (future)
- Entity store: Extracted facts and preferences (future)
- RAG integration: Long-term retrieval (future)
"""

from typing import Any

from src.logging_config import log_memory_context
from src.memory.base import BaseMemoryStore
from src.memory.context import MemoryContext
from src.memory.manager import BaseMemoryManager

# Optional LangChain message classes
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
except ImportError:
    HumanMessage = None  # type: ignore[misc, assignment]
    AIMessage = None  # type: ignore[misc, assignment]
    BaseMessage = None  # type: ignore[misc, assignment]


class ConversationMemoryManager(BaseMemoryManager):
    """
    Memory manager for general conversation, Q&A, and research.

    Optimized for:
    - Information search and analysis
    - General question answering
    - Research assistance
    - Casual conversation

    Configuration options:
        working_memory_size (int): Messages to keep in context (default: 20)
        summary_threshold (int): When to trigger summarization (default: 30)
        entity_extraction (bool): Enable entity tracking (default: False)
        rag_enabled (bool): Enable RAG retrieval (default: False)
        rag_top_k (int): Number of RAG results (default: 3)
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "working_memory_size": 20,
        "summary_threshold": 30,
        "entity_extraction": False,
        "rag_enabled": False,
        "rag_top_k": 3,
    }

    def __init__(
        self,
        store: BaseMemoryStore,
        session_id: str,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize conversation memory manager.

        Args:
            store: Storage backend
            session_id: Session identifier
            config: Mode-specific configuration overrides
        """
        super().__init__(store, session_id, config)

        # Merge defaults with provided config
        self._mode_config = {**self.DEFAULT_CONFIG, **(config or {})}

        # Working memory - recent messages
        self._messages: list[Any] = []

        # Session summary (future feature)
        self._summary: str | None = None

        # Entity store (future feature)
        self._entities: dict[str, Any] = {}

        # Topics discussed (future feature)
        self._topics: list[str] = []

    @property
    def mode_name(self) -> str:
        """Return mode identifier."""
        return "conversation"

    def load(self) -> None:
        """Load conversation history from storage, sanitizing bad entries."""
        self._messages = self.store.load_history(self.session_id)
        self._messages = self.sanitize_history(self._messages)
        self._loaded = True

    def save(self) -> None:
        """Save conversation history to storage."""
        self.store.save_history(self.session_id, self._messages)

    def prepare_context(self, user_input: str) -> MemoryContext:
        """
        Prepare conversation context for LLM.

        Selects recent messages within the working memory window
        and prepends any session summary.

        Args:
            user_input: Current user input (for future relevance filtering)

        Returns:
            MemoryContext with messages and metadata
        """
        # Get working memory window
        window_size = self._mode_config["working_memory_size"]

        if self._messages:
            context_messages = self._messages[-window_size:]
        else:
            context_messages = []

        # Build context prefix (summary + entities if available)
        prefix_parts = []

        if self._summary:
            prefix_parts.append(f"Previous conversation summary:\n{self._summary}")

        if self._entities:
            entity_str = ", ".join(f"{k}: {v}" for k, v in self._entities.items())
            prefix_parts.append(f"Known facts: {entity_str}")

        context_prefix = "\n\n".join(prefix_parts) if prefix_parts else None

        token_estimate = self._estimate_tokens(context_messages)

        # Log context preparation
        log_memory_context(
            mode=self.mode_name,
            message_count=len(context_messages),
            token_estimate=token_estimate,
        )

        return MemoryContext(
            messages=context_messages,
            system_additions=self.get_system_prompt_additions(),
            context_prefix=context_prefix,
            mode=self.mode_name,
            total_messages_stored=len(self._messages),
            context_messages_count=len(context_messages),
            token_estimate=token_estimate,
            metadata={
                "has_summary": self._summary is not None,
                "entity_count": len(self._entities),
            },
        )

    def update(self, user_input: str, ai_response: str) -> None:
        """
        Add new turn to conversation memory.

        Creates message objects and appends to history.
        Future: trigger summarization if threshold exceeded.

        Args:
            user_input: User's input
            ai_response: AI's response
        """
        # Create message objects
        if HumanMessage is not None and AIMessage is not None:
            human_msg = HumanMessage(content=user_input)
            ai_msg = AIMessage(content=ai_response)
        else:
            human_msg = {"type": "human", "content": user_input}
            ai_msg = {"type": "ai", "content": ai_response}

        self._messages.append(human_msg)
        self._messages.append(ai_msg)

        # Check if summarization needed (future feature)
        # threshold = self._mode_config["summary_threshold"]
        # if len(self._messages) > threshold:
        #     self._trigger_summarization()

    def get_system_prompt_additions(self) -> str | None:
        """Return conversation-mode system prompt additions."""
        # Reinforce task completion and accuracy in conversation mode
        return (
            "In conversation mode: answer questions fully, complete requested tasks, "
            "and use tools proactively to gather information you need. "
            "Don't stop to ask clarifying questions when the task is clear.\n"
            "ACCURACY: Base factual claims strictly on data gathered by tools. "
            "Do NOT invent numbers, dates, parameter counts, URLs, or other "
            "specifics not found in tool results. If the information was not "
            "found, say so explicitly."
        )

    def clear(self) -> None:
        """Clear all conversation memory."""
        self._messages = []
        self._summary = None
        self._entities = {}
        self._topics = []

    def get_message_count(self) -> int:
        """Return total number of messages stored."""
        return len(self._messages)

    def get_stats(self) -> dict[str, Any]:
        """Return conversation memory statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "total_messages": len(self._messages),
            "working_memory_size": self._mode_config["working_memory_size"],
            "has_summary": self._summary is not None,
            "entity_count": len(self._entities),
            "topic_count": len(self._topics),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize conversation state."""
        base = super().to_dict()

        # Serialize messages
        messages_data = []
        for msg in self._messages:
            if BaseMessage is not None and isinstance(msg, BaseMessage):
                if HumanMessage is not None and isinstance(msg, HumanMessage):
                    msg_type = "human"
                else:
                    msg_type = "ai"
                messages_data.append(
                    {
                        "type": msg_type,
                        "content": msg.content or "",
                    }
                )
            elif isinstance(msg, dict):
                messages_data.append(msg)
            else:
                messages_data.append(
                    {
                        "type": "unknown",
                        "content": str(msg),
                    }
                )

        return {
            **base,
            "messages": messages_data,
            "summary": self._summary,
            "entities": self._entities,
            "topics": self._topics,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore conversation state from dictionary."""
        super().from_dict(data)

        # Restore messages
        messages_data = data.get("messages", [])
        self._messages = []

        for msg_data in messages_data:
            msg_type = msg_data.get("type", "human")
            content = msg_data.get("content", "")

            if HumanMessage is not None and AIMessage is not None:
                if msg_type == "ai":
                    self._messages.append(AIMessage(content=content))
                else:
                    self._messages.append(HumanMessage(content=content))
            else:
                self._messages.append(msg_data)

        self._summary = data.get("summary")
        self._entities = data.get("entities", {})
        self._topics = data.get("topics", [])
        self._loaded = True

    def _estimate_tokens(self, messages: list[Any]) -> int:
        """
        Rough token estimation for messages.

        Uses simple heuristic: ~4 characters per token.

        Args:
            messages: List of messages

        Returns:
            Estimated token count
        """
        total_chars = 0
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                total_chars += len(msg.content)
            elif isinstance(msg, dict) and msg.get("content"):
                total_chars += len(msg["content"])

        return total_chars // 4

    # --- Future features ---

    def _trigger_summarization(self) -> None:
        """
        Summarize older messages to compress history.

        Future feature: Use LLM to summarize messages beyond
        the working memory window.
        """

    def _extract_entities(self, text: str) -> None:
        """
        Extract entities and facts from text.

        Future feature: Use NLP to identify and store
        key facts, names, preferences.
        """
