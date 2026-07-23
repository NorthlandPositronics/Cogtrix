"""
Conversation memory mode for general chat, Q&A, and research.

This is the default memory mode, providing:
- Working memory: Last N messages in context (sliding window)
- Session summary: LLM-compressed older messages (incremental)
- Vector recall: Semantic search over evicted messages (optional)
- Entity store: Extracted facts and preferences (future)
- RAG integration: Long-term retrieval (future)
"""

import logging
import threading
from typing import Any

from cogtrix_core.logging_config import log_memory_context
from cogtrix_core.memory.base import BaseMemoryStore
from cogtrix_core.memory.context import MemoryContext
from cogtrix_core.memory.manager import BaseMemoryManager

log = logging.getLogger("cogtrix")

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
        "summary_max_age_hours": None,
        "summary_max_uncovered_tokens": 16_000,
        "distill_on_expire": False,
        "facts_ttl_days": 7,
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

        # Lock protecting mode-specific mutable state (_messages, _entities, _topics)
        self._mode_lock = threading.Lock()

    @property
    def mode_name(self) -> str:
        """Return mode identifier."""
        return "conversation"

    def load(self) -> None:
        """Load conversation history from storage, sanitizing bad entries."""
        self._messages = self.store.load_history(self.session_id)
        self._messages = self.sanitize_history(self._messages)
        self._load_hybrid_meta()
        self._check_summary_ttl()
        self._check_summary_token_ttl()
        self._load_tier_cache()
        self._load_mode_meta()
        self._clamp_summary_idx()
        self._initial_mode = self.mode_name
        self._loaded = True

    def _restore_mode_state(self, data: dict) -> None:
        """Restore conversation-specific state from mode_state.json."""
        with self._mode_lock:
            self._entities = data.get("entities", {})
            self._topics = data.get("topics", [])

    def save(self) -> None:
        """Save conversation history to storage."""
        with self._mode_lock:
            self.store.save_history(self.session_id, self._messages)
        super().save()

    def _pending_path(self) -> "Any":
        from pathlib import Path

        from cogtrix_core.memory.manager import _sanitize_session_id

        base = getattr(self.store, "base_path", None) or Path("data/history")
        sanitized = _sanitize_session_id(self.session_id)
        return Path(str(base)) / f"{sanitized}_pending.json"

    def prerecord_user(self, text: str) -> None:
        """Write user message to disk as a pending turn for shutdown durability.

        Does not call update() so existing call-count assertions are unaffected.
        The pending file is cleaned up by the subsequent update() call or by
        discard_prerecord() on deferral/suppress paths.
        """
        import json
        import time

        try:
            path = self._pending_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"user": text, "ts": time.time()}))
        except Exception:
            pass

    def discard_prerecord(self) -> None:
        """Remove the pending pre-record without affecting message history."""
        try:
            path = self._pending_path()
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def prepare_context(self, user_input: str) -> MemoryContext:
        """Prepare conversation-optimized context for LLM."""
        # Record the moment the user sent this message.
        # Protected by _hybrid_lock so concurrent prepare_context() calls
        # do not silently overwrite each other's timestamps before update()
        # can consume them (issue #1344).
        with self._hybrid_lock:
            self._captured_user_ts = self._now_ts()
            self._pending_user_ts = self._captured_user_ts

        # ── Mode-specific prefix (entities + hybrid recall) ──────────────
        # Computed regardless of the message-selection path so both the
        # tier-cache path and the sliding-window path return a consistent prefix.
        prefix_parts = []

        hybrid = self._build_hybrid_prefix(user_input)
        if hybrid:
            prefix_parts.append(hybrid)

        # Acquire mode lock for safe reads of mode-specific state. Snapshot the
        # message list here too (#2131 C3): update() appends to self._messages
        # under _mode_lock, so reading it unlocked below would race a concurrent
        # turn and could hand assemble_from_tiers / the cold path a half-mutated
        # list. Use the snapshot for every read in this method.
        with self._mode_lock:
            messages_snapshot = list(self._messages)
            if self._entities:
                entity_str = ", ".join(f"{k}: {v}" for k, v in self._entities.items())
                prefix_parts.append(f"Known facts: {entity_str}")

        context_prefix = "\n\n".join(prefix_parts) if prefix_parts else None

        # ── Tiered context assembly ──────────────────────────────────────
        with self._hybrid_lock:
            tier_ready = self._tier_cache_ready
            tier_cache = self._tier_cache
            summary = self._summary or ""
            summary_msg_idx = self._summary_msg_idx

        if tier_ready and tier_cache is not None:
            from cogtrix_core.memory.tier_cache import assemble_from_tiers

            assembled, tier_counts = assemble_from_tiers(
                snapshot=tier_cache,
                messages=messages_snapshot,
                summary=summary,
                summary_msg_idx=summary_msg_idx,
            )
            total_tokens = sum(tier_counts.values())

            if log.isEnabledFor(logging.DEBUG):
                log_memory_context(
                    mode=self.mode_name,
                    message_count=len(assembled),
                    token_estimate=total_tokens,
                )

            return MemoryContext(
                messages=assembled,
                system_additions=self.get_system_prompt_additions(),
                context_prefix=context_prefix,
                mode=self.mode_name,
                total_messages_stored=len(messages_snapshot),
                context_messages_count=len(assembled),
                token_estimate=total_tokens,
                tier_token_counts=tier_counts,
                metadata={
                    "has_summary": bool(summary),
                    "summary_coverage": summary_msg_idx,
                    "entity_count": len(self._entities),
                },
            )

        # ── Sliding window fallback (cold cache) ─────────────────────────
        # Return all messages to the LLM on cold-cache paths. The 25-message
        # cap was applied BEFORE summarization, causing messages outside the
        # window to be lost (never seen by the LLM) before they could be
        # compressed into the summary. Now we let the LLM see all messages;
        # background summarization will compress older messages into the tier
        # cache, and subsequent turns will use the compressed tiers instead.
        context_messages = list(messages_snapshot)

        # Inject timestamps so the LLM has temporal awareness
        context_messages = self._inject_timestamps(context_messages)

        token_estimate = self._estimate_tokens(context_messages)

        # Log context preparation (debug-only — skip the call on non-debug paths)
        if log.isEnabledFor(logging.DEBUG):
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
        """Update memory with conversation context."""
        # Capture timestamp under _hybrid_lock (matches lock used in
        # prepare_context to write it — issue #1344).
        with self._hybrid_lock:
            ts_to_apply = self._pending_user_ts
            self._pending_user_ts = None
        with self._mode_lock:
            # --- Build the human message (always needed) ----------------
            if HumanMessage is not None:
                human_msg: Any = HumanMessage(content=user_input)
            else:
                human_msg = {"type": "human", "content": user_input}
            self._set_msg_ts(human_msg, ts_to_apply)

            self._messages.append(human_msg)

            # --- Append the agent's messages ---------------------------
            if agent_messages is not None:
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

            # Snapshot messages under lock so downstream scheduling
            # sees a consistent view even if clear() or another update()
            # mutates self._messages concurrently.
            messages_snapshot = list(self._messages)

        # Layer-1a: accumulate tokens since last summary update (protected by _hybrid_lock in base)
        from cogtrix_core.memory.manager import _msg_tokens

        with self._hybrid_lock:
            self._tokens_since_summary += _msg_tokens(human_msg)
            if agent_messages is not None:
                self._tokens_since_summary += _msg_tokens(agent_messages[-1])
            else:
                self._tokens_since_summary += _msg_tokens(ai_msg)
            self._check_summary_token_ttl_locked()

        # Incrementally summarize messages outside the sliding window
        window_size = self._mode_config["working_memory_size"]
        self._schedule_slow_path(messages_snapshot, window_size)

        # Schedule tier cache roll-forward only when history exceeds the window.
        # No-op for short sessions that still fit in the verbatim sliding window.
        if len(messages_snapshot) > window_size:
            try:
                self.schedule_tier_roll_forward(
                    max_context_tokens=getattr(self, "_max_context_tokens", 0) or 128_000,
                    llm=getattr(self, "_compression_llm", None),
                )
            except Exception as exc:
                log.debug("Tier roll-forward scheduling failed: %s", exc)

        # Clean up any pending pre-record file now that the turn completed.
        self.discard_prerecord()

        # ── Domain-shift detection ─────────────────────────────────────
        # Check if recent conversation patterns indicate a topic-domain shift
        # that warrants resetting the rolling summary. Called outside all locks.
        prompts = self._extract_recent_user_prompts(self._messages, limit=3)
        self._check_domain_shift(prompts)

    def get_system_prompt_additions(self) -> str | None:
        """Return conversation-mode system prompt additions."""
        # Reinforce task completion and accuracy in conversation mode
        return (
            "In conversation mode: answer questions fully, complete requested tasks, "
            "and use tools proactively to gather information you need. "
            "For low-risk or reversible tasks where the intent is clear, proceed "
            "directly without asking for clarification."
        )

    def clear(self) -> None:
        """Clear all conversation memory."""
        with self._mode_lock:
            super().clear()
            self._messages = []
            self._entities = {}
            self._topics = []
        # Remove the pending pre-record file so a cleared session doesn't leak
        # {id}_pending.json (#2131 C5). discard_prerecord is best-effort.
        self.discard_prerecord()

    def get_message_count(self) -> int:
        """Return total number of messages stored."""
        with self._mode_lock:
            return len(self._messages)

    def pop_last_turn(self) -> int:
        """Remove the last user+assistant exchange from memory.

        Scans _messages backwards for the last HumanMessage and removes
        everything from that index to the end. Returns the number of
        messages removed (0 if nothing to remove).
        """
        with self._mode_lock:
            if not self._messages:
                return 0
            # Find the last HumanMessage index
            last_human_idx = None
            for i in range(len(self._messages) - 1, -1, -1):
                msg = self._messages[i]
                is_human = (HumanMessage is not None and isinstance(msg, HumanMessage)) or (
                    isinstance(msg, dict) and msg.get("type") == "human"
                )
                if is_human:
                    last_human_idx = i
                    break
            if last_human_idx is None:
                return 0
            removed = len(self._messages) - last_human_idx
            self._messages = self._messages[:last_human_idx]
        self.save()
        return removed

    def get_stats(self) -> dict[str, Any]:
        """Return conversation memory statistics."""
        with self._mode_lock:
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
        from cogtrix_core.memory.json_store import _message_to_dict

        with self._mode_lock:
            base = super().to_dict()

            messages_data = [_message_to_dict(m) for m in self._messages]

            return {
                **base,
                "messages": messages_data,
                "entities": self._entities,
                "topics": self._topics,
            }

    def _mode_state_dict(self) -> dict[str, Any]:
        """Persist conversation-specific state without messages or hybrid data."""
        return {
            "entities": self._entities,
            "topics": self._topics,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore conversation state from dictionary."""
        from cogtrix_core.memory.json_store import _dict_to_message

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
