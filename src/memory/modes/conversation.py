"""
Conversation memory mode for general chat, Q&A, and research.

This is the default memory mode, providing:
- Working memory: Last N messages in context (sliding window)
- Session summary: LLM-compressed older messages (incremental)
- Vector recall: Semantic search over evicted messages (optional)
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
        working_memory_size (int): Messages to keep in context (default: 25)
        summary_threshold (int): When to trigger summarization (default: 35)
        entity_extraction (bool): Enable entity tracking (default: False)
        rag_enabled (bool): Enable RAG retrieval (default: False)
        rag_top_k (int): Number of RAG results (default: 3)
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "working_memory_size": 25,
        "summary_threshold": 35,
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
        self._load_hybrid_meta()
        self._clamp_summary_idx()
        self._loaded = True

    def save(self) -> None:
        """Save conversation history to storage."""
        self.store.save_history(self.session_id, self._messages)
        super().save()

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
        # Record the moment the user sent this message
        self._pending_user_ts = self._now_ts()

        # Get working memory window
        window_size = self._mode_config["working_memory_size"]

        if self._messages:
            context_messages = self._messages[-window_size:]
        else:
            context_messages = []

        # Inject timestamps so the LLM has temporal awareness
        context_messages = self._inject_timestamps(context_messages)

        # Build context prefix (hybrid summary + recall + entities)
        prefix_parts = []

        hybrid = self._build_hybrid_prefix(user_input)
        if hybrid:
            prefix_parts.append(hybrid)

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
                "summary_coverage": self._summary_msg_idx,
                "entity_count": len(self._entities),
            },
        )

    def update(
        self,
        user_input: str,
        ai_response: str,
        agent_messages: list[Any] | None = None,
    ) -> None:
        """
        Add new turn to conversation memory.

        If *agent_messages* is provided the full tool-call chain is
        stored (enabling the Ralph Loop — the agent can see its
        previous tool usage on restart).  Otherwise a simple
        Human / AI pair is stored.

        Args:
            user_input: User's input
            ai_response: AI's response
            agent_messages: Optional full chain from the agent run
        """
        # --- Build the human message (always needed) ----------------
        if HumanMessage is not None:
            human_msg: Any = HumanMessage(content=user_input)
        else:
            human_msg = {"type": "human", "content": user_input}
        self._set_msg_ts(human_msg, self._pending_user_ts)
        self._pending_user_ts = None

        self._messages.append(human_msg)

        # --- Append the agent's messages ---------------------------
        if agent_messages:
            # agent_messages already contains the full chain
            # (AI tool_calls, ToolMessages, final AI).
            # Stamp the final AI message with the current time.
            for m in agent_messages:
                self._messages.append(m)
            # Stamp only the *last* AI message (the final answer)
            last = agent_messages[-1]
            if hasattr(last, "content") or isinstance(last, dict):
                self._set_msg_ts(last)
        else:
            # Legacy path: just a plain AI text response
            if AIMessage is not None:
                ai_msg: Any = AIMessage(content=ai_response)
            else:
                ai_msg = {"type": "ai", "content": ai_response}
            self._set_msg_ts(ai_msg)
            self._messages.append(ai_msg)

        # Incrementally summarize messages outside the sliding window
        window_size = self._mode_config["working_memory_size"]
        self._schedule_slow_path(self._messages, window_size)

    def get_system_prompt_additions(self) -> str | None:
        """Return conversation-mode system prompt additions."""
        # Reinforce task completion and accuracy in conversation mode
        return (
            "In conversation mode: answer questions fully, complete requested tasks, "
            "and use tools proactively to gather information you need. "
            "Don't stop to ask clarifying questions when the task is clear."
        )

    def clear(self) -> None:
        """Clear all conversation memory."""
        super().clear()
        self._messages = []
        self._entities = {}
        self._topics = []

    def get_message_count(self) -> int:
        """Return total number of messages stored."""
        return len(self._messages)

    def get_stats(self) -> dict[str, Any]:
        """Return conversation memory statistics."""
        base_stats = super().get_stats()
        vs = self._vector_store
        return {
            **base_stats,
            "total_messages": len(self._messages),
            "working_memory_size": self._mode_config["working_memory_size"],
            "has_summary": self._summary is not None,
            "summary_coverage": self._summary_msg_idx,
            "vector_recall_ready": vs is not None and vs.ready,
            "entity_count": len(self._entities),
            "topic_count": len(self._topics),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize conversation state."""
        from src.memory.json_store import _message_to_dict

        base = super().to_dict()

        messages_data = [_message_to_dict(m) for m in self._messages]

        return {
            **base,
            "messages": messages_data,
            "entities": self._entities,
            "topics": self._topics,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore conversation state from dictionary."""
        from src.memory.json_store import _dict_to_message

        super().from_dict(data)

        self._messages = [_dict_to_message(d) for d in data.get("messages", [])]

        # Legacy "summary" key → migrate to base-class _summary
        if self._summary is None and data.get("summary"):
            self._summary = data["summary"]

        self._entities = data.get("entities", {})
        self._topics = data.get("topics", [])
        self._loaded = True

    # --- Future features ---

    def _extract_entities(self, text: str) -> None:
        """
        Extract entities and facts from text.

        Future feature: Use NLP to identify and store
        key facts, names, preferences.
        """
