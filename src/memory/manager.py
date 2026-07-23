"""
Abstract base class for memory management strategies.
All memory modes must inherit from this class.

Includes hybrid memory support:
- **Sliding window**: subclasses keep last N messages verbatim
- **Incremental summary**: older messages are summarised by the LLM
- **Vector recall** (optional): evicted messages are embedded for
  semantic retrieval, improving long-term awareness
"""

import atexit
import json
import logging
import os

from src.utils.path_safety import _sanitize_session_id

__all__ = [
    "_sanitize_session_id",
]

import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy as _shallow_copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.message_validation import _coerce_content, is_bad_ai_content
from src.logging_config import is_verbose
from src.memory.base import BaseMemoryStore
from src.memory.context import MemoryContext

# Optional LangChain types — imported lazily in helpers
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
except ImportError:  # pragma: no cover
    BaseMessage = None  # type: ignore[misc, assignment]
    HumanMessage = None  # type: ignore[misc, assignment]
    AIMessage = None  # type: ignore[misc, assignment]
    ToolMessage = None  # type: ignore[misc, assignment]

log = logging.getLogger("cogtrix")

TS_DISPLAY_FORMAT = "%Y-%m-%d %H:%M:%S"

# Summarize when at least this many messages have fallen out of the
# sliding window since the last summary was generated.
_SUMMARY_BATCH_SIZE = 10

# Minimum amount of *meaningful* conversation content (human messages
# plus final AI responses — tool-call intermediaries excluded) before
# summarization kicks in.  A single turn with a long tool-call chain
# can overflow the sliding window, but there is nothing to compress yet.
_MIN_MEANINGFUL_MSGS_FOR_SUMMARY = 4  # at least 2 full turns (H+A pairs)
_MIN_MEANINGFUL_CHARS_FOR_SUMMARY = 5000  # ignore tiny exchanges

_SESSION_ID_MAX_LEN = 200
_SLOW_PATH_MAX_FAILURES = 3
_CHARS_PER_TOKEN = 2  # matches src.memory.tier_cache
# Max seconds a background summarization job may run before it is considered
# stuck. The thread continues (Python cannot cancel it) but we stop waiting
# and allow a fresh submission on the next turn.
_BG_JOB_TIMEOUT_SECONDS = 120

_SUMMARIZATION_POOL: ThreadPoolExecutor | None = None
_SUMMARIZATION_POOL_LOCK = threading.Lock()


def _get_summarization_pool() -> ThreadPoolExecutor:
    global _SUMMARIZATION_POOL
    if _SUMMARIZATION_POOL is None:
        with _SUMMARIZATION_POOL_LOCK:
            if _SUMMARIZATION_POOL is None:
                _SUMMARIZATION_POOL = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="summarize"
                )
                atexit.register(_SUMMARIZATION_POOL.shutdown, wait=False, cancel_futures=True)
    return _SUMMARIZATION_POOL


def _msg_tokens(msg: Any) -> int:
    """Estimate token count for a single message (chars // 2)."""
    c = getattr(msg, "content", None)
    if c is None and isinstance(msg, dict):
        c = msg.get("content")
    if isinstance(c, str):
        return max(1, len(c) // _CHARS_PER_TOKEN)
    if isinstance(c, list):
        total = 0
        for item in c:
            if isinstance(item, str):
                total += len(item)
            elif isinstance(item, dict):
                total += len(item.get("text", ""))
        return max(1, total // _CHARS_PER_TOKEN)
    return 1


class BaseMemoryManager(ABC):
    """
    Abstract interface for memory management strategies.

    Each memory mode (conversation, code, reasoning) implements this
    interface to provide mode-specific context preparation and updates.

    Lifecycle:
        1. __init__() - Create manager with store and session
        2. load() - Load existing state from storage
        3. prepare_context() - Get context for LLM (called before each turn)
        4. update() - Update memory after LLM response
        5. save() - Persist state to storage

    Attributes:
        store: Storage backend for persistence
        session_id: Unique identifier for this session
        config: Mode-specific configuration options
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        session_id: str,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the memory manager.

        Args:
            store: Storage backend (e.g., JsonFileMemoryStore)
            session_id: Unique session identifier
            config: Optional mode-specific configuration
        """
        self.store = store
        self.session_id = session_id
        self.config = config or {}
        self._loaded = False
        # Set by prepare_context(); consumed by update() for the human message
        self._pending_user_ts: str | None = None

        # ── Hybrid memory state ──────────────────────────────────────
        self._llm: Any = None  # set via set_llm()
        self._summary: str | None = None
        # How many messages (from the start) are covered by the summary.
        self._summary_msg_idx: int = 0
        # When the rolling summary was last refreshed.
        self._summary_last_updated_at: datetime | None = None
        # Tokens accumulated since the last summary update (Layer-1a TTL).
        self._tokens_since_summary: int = 0

        # Optional vector recall (set via set_embeddings())
        self._vector_store: Any = None  # SessionVectorStore | None
        # True when _summary was updated by a background job but not yet persisted.
        self._summary_dirty: bool = False
        self._facts_store: Any = None

        # Lazy embedding config — populated by set_embedding_config(); the
        # provider is instantiated on first actual use to avoid blocking startup.
        self._lazy_emb_type: str | None = None
        self._lazy_emb_model: str | None = None
        self._lazy_emb_base_url: str | None = None
        self._lazy_emb_api_key: str | None = None
        self._lazy_emb_vector_store_dir: str | None = None
        self._lazy_emb_resolved: bool = False  # True once the provider was created
        self._lazy_emb_warned: bool = False  # One-shot: warning emitted on first failure only

        # ── Background slow-path threading ───────────────────────────
        self._hybrid_lock = threading.Lock()
        self._bg_future: Future | None = None
        self._bg_submitted_at: float = 0.0  # monotonic timestamp of last submit
        self._slow_path_failures: int = 0

        # ── Tiered Context Cache (TCC) ────────────────────────────────
        # Phase 1: data structures and persistence only.
        # Roll-forward (Phase 3) and context assembly (Phase 2) are
        # implemented in later phases.
        from src.memory.tier_cache import TierCacheSnapshot

        self._tier_cache: TierCacheSnapshot | None = None
        self._tier_cache_ready: bool = False

    # ── Hybrid memory wiring ────────────────────────────────────────

    def set_llm(self, llm: Any) -> None:
        """Attach an LLM for summarization.

        Called from ``cogtrix.py`` after the LLM is created (and again
        whenever the user switches model/provider at runtime).
        """
        self._llm = llm

    def configure_compression(
        self, max_context_tokens: int, compression_llm: Any | None = None
    ) -> None:
        """Set TCC (Tiered Context Compression) parameters.

        Called from ``session_bridge.py`` after ``AgentRunConfig`` is built
        so that background roll-forward jobs use the correct context budget
        and compression LLM.

        Args:
            max_context_tokens: Maximum tokens allowed in the context window.
            compression_llm: Optional dedicated LLM for compression tasks.
        """
        self._max_context_tokens = max_context_tokens
        self._compression_llm = compression_llm

    def set_embeddings(
        self, embedding_fn: Any, embedding_model: str, vector_store_dir: str | None = None
    ) -> None:
        """Attach an embedding function for vector recall.

        Creates (or reconfigures) a per-session ``SessionVectorStore``.
        Skipped when embedding is unavailable — vector recall simply
        won't contribute to context.
        """
        from src.memory.recall import SessionVectorStore

        if self._vector_store is None:
            self._vector_store = SessionVectorStore(
                self.session_id,
                storage_dir=vector_store_dir or "data/vectordb/sessions",
            )
        self._vector_store.configure(embedding_fn, embedding_model)
        self._lazy_emb_resolved = True

    def set_embedding_config(
        self,
        emb_type: str,
        emb_model: str,
        emb_base_url: str | None,
        emb_api_key: str | None,
        vector_store_dir: str | None = None,
    ) -> None:
        """Store embedding config for lazy initialisation.

        The embedding provider is NOT created here — it is created on the
        first call to ``_ensure_embeddings_initialized()`` (triggered by
        ``_build_hybrid_prefix()`` or ``_run_slow_path()``).  This avoids
        the ~280 ms provider-SDK init cost at session startup.

        Calling ``set_embeddings()`` directly still works and marks the
        provider as already resolved.
        """
        self._lazy_emb_type = emb_type
        self._lazy_emb_model = emb_model
        self._lazy_emb_base_url = emb_base_url
        self._lazy_emb_api_key = emb_api_key
        self._lazy_emb_vector_store_dir = vector_store_dir
        self._lazy_emb_resolved = False

    def _ensure_embeddings_initialized(self) -> None:
        """Create the embedding provider on first use if only config was stored.

        Thread-safe: protected by ``_hybrid_lock``.  No-op if the provider is
        already resolved or if no lazy config was stored.
        """
        if self._lazy_emb_resolved or self._lazy_emb_type is None:
            return
        with self._hybrid_lock:
            if self._lazy_emb_resolved:
                return
            try:
                from src.providers import create_embeddings_from_config

                fn, tag = create_embeddings_from_config(
                    self._lazy_emb_type,
                    model=self._lazy_emb_model,
                    base_url=self._lazy_emb_base_url,
                    api_key=self._lazy_emb_api_key,
                )
                self.set_embeddings(fn, tag, vector_store_dir=self._lazy_emb_vector_store_dir)
                log.debug("Lazy embedding init: using %s", tag)
            except Exception as exc:
                if not self._lazy_emb_warned:
                    log.warning(
                        "Lazy embedding provider '%s' unavailable: %s",
                        self._lazy_emb_type,
                        exc,
                    )
                    self._lazy_emb_warned = True
                self._lazy_emb_resolved = True

    def _hybrid_meta_path(self) -> Path:
        """Return the path for the hybrid-state meta file.

        Derived from the store's base directory when available,
        otherwise falls back to ``data/history/``.
        """
        safe_id = _sanitize_session_id(self.session_id)
        base: Path = getattr(self.store, "base_path", Path("data/history"))
        result = (base / f"{safe_id}_hybrid.json").resolve()
        try:
            result.relative_to(base.resolve())
        except ValueError:
            raise ValueError(
                f"Path traversal detected in session_id: {self.session_id!r}"
            ) from None
        return result

    def _save_hybrid_meta(self, *, block: bool = True, timeout: float = 0.0) -> None:
        """Persist summary state and vector store metadata to a file."""
        snapshot = self._get_hybrid_snapshot(block=block, timeout=timeout)
        if snapshot is None:
            # Lock unavailable in non-blocking mode; skip to avoid persisting
            # a torn (inconsistent) snapshot. Previous meta file remains valid.
            return
        summary, summary_idx, summary_updated_at = snapshot

        if summary is None and summary_idx == 0:
            meta_path = self._hybrid_meta_path()
            if meta_path.exists():
                meta_path.unlink(missing_ok=True)
            return
        meta = {
            "_summary": summary,
            "_summary_msg_idx": summary_idx,
        }
        if summary_updated_at is not None:
            meta["_summary_last_updated_at"] = summary_updated_at.isoformat()
        try:
            from src.utils.atomic_write import atomic_write_json

            meta_path = self._hybrid_meta_path()
            with atomic_write_json(meta_path) as f:
                json.dump(meta, f)
        except Exception as exc:
            log.warning("Failed to save hybrid meta: %s", exc)

    def _load_hybrid_meta(self) -> None:
        """Restore summary state from the meta file (if present)."""
        meta_path = self._hybrid_meta_path()
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._summary = meta.get("_summary")
            self._summary_msg_idx = meta.get("_summary_msg_idx", 0)
            raw_summary_updated_at = meta.get("_summary_last_updated_at")
            if raw_summary_updated_at:
                self._summary_last_updated_at = datetime.fromisoformat(raw_summary_updated_at)
            else:
                self._summary_last_updated_at = None
        except Exception as exc:
            log.warning("Failed to load hybrid meta: %s", exc)

    def _get_facts_store(self) -> Any:
        """Return the persistent facts store for this session."""
        if self._facts_store is None:
            with self._hybrid_lock:
                if self._facts_store is None:
                    from src.memory.facts import PersistentFactsStore

                    base: Path = getattr(self.store, "base_path", Path("data/history"))
                    storage_dir = base.parent / "memory" / "facts"
                    self._facts_store = PersistentFactsStore(
                        self.session_id,
                        storage_dir=str(storage_dir),
                    )
        return self._facts_store

    def _get_summary_max_age_hours(self) -> float | int | None:
        """Return the configured summary TTL in hours, if any."""
        mode_config = getattr(self, "_mode_config", None)
        if isinstance(mode_config, dict):
            return mode_config.get("summary_max_age_hours")
        return self.config.get("summary_max_age_hours")

    def _bg_in_flight(self) -> bool:
        """Return True when a background summarizer is currently running.

        Used by the TTL checks to defer reset while a BG is producing a fresh
        summary — see ``_check_summary_token_ttl`` for the race this guards.
        """
        fut = self._bg_future
        return fut is not None and not fut.done()

    def _check_summary_ttl(self) -> None:
        """Reset an expired rolling summary when a TTL is configured."""
        max_age_hours = self._get_summary_max_age_hours()
        if max_age_hours is None:
            return

        with self._hybrid_lock:
            summary = self._summary
            summary_updated_at = self._summary_last_updated_at

        if summary is None or summary_updated_at is None:
            return

        age_hours = (datetime.now(UTC) - summary_updated_at).total_seconds() / 3600
        if age_hours > float(max_age_hours):
            # If a background summarizer is running, defer reset — it will write
            # a fresh summary shortly. Resetting now risks wiping that write
            # without scheduling a follow-up (the next _schedule_slow_path skips
            # while the BG is still in flight, so the reset's wipe would persist).
            if self._bg_in_flight():
                log.debug(
                    "Rolling summary age %.1fh > %sh, but BG summarizer is in "
                    "flight — deferring reset for session %s",
                    age_hours,
                    max_age_hours,
                    self.session_id,
                )
                return
            log.info(
                "Rolling summary expired (%.1fh > %sh) — resetting",
                age_hours,
                max_age_hours,
            )
            if not self._reset_summary_state(expected_summary_last_updated_at=summary_updated_at):
                log.debug(
                    "Rolling summary refreshed during TTL check for session %s; skipping reset",
                    self.session_id,
                )

    def _check_summary_token_ttl(self) -> None:
        """Reset the rolling summary when token churn exceeds threshold."""
        mode_config = getattr(self, "_mode_config", None)
        if isinstance(mode_config, dict):
            max_tokens = mode_config.get("summary_max_uncovered_tokens")
        else:
            max_tokens = self.config.get("summary_max_uncovered_tokens")
        if max_tokens is None:
            return

        with self._hybrid_lock:
            tokens_since = self._tokens_since_summary
            # Capture timestamp for TOCTOU guard — same pattern as _check_summary_ttl.
            summary_updated_at = self._summary_last_updated_at

        if tokens_since >= int(max_tokens):
            # If a background summarizer is running, defer reset. The TOCTOU
            # guard on _reset_summary_state is not sufficient here: once a BG
            # has written a fresh summary, the captured ``summary_updated_at``
            # equals the BG's write timestamp, so the guard passes and the
            # reset wipes the BG's just-written summary. With the BG already
            # past its writeback (in _save_hybrid_meta etc.), no follow-up
            # _schedule_slow_path call ever fires — leaving _summary stuck at
            # None. Skipping while in flight lets the BG's write stand.
            if self._bg_in_flight():
                log.debug(
                    "Token churn %d >= %d, but BG summarizer is in flight — "
                    "deferring reset for session %s",
                    tokens_since,
                    max_tokens,
                    self.session_id,
                )
                return
            log.info(
                "Rolling summary stale: %d tokens since update (threshold %d) — resetting",
                tokens_since,
                max_tokens,
            )
            if not self._reset_summary_state(expected_summary_last_updated_at=summary_updated_at):
                log.debug(
                    "Rolling summary refreshed during token-TTL check for session %s; skipping reset",
                    self.session_id,
                )

    def _get_hybrid_snapshot(
        self, *, block: bool = True, timeout: float = 0.0
    ) -> tuple[str | None, int, datetime | None] | None:
        """Return a snapshot of the summary state.

        ``block=False`` is used during shutdown to avoid deadlocking on the
        background summarizer's lock. When the lock cannot be acquired in
        non-blocking mode, ``None`` is returned so callers can skip the
        save rather than persist a torn (inconsistent) snapshot.
        """
        if block:
            with self._hybrid_lock:
                return self._summary, self._summary_msg_idx, self._summary_last_updated_at

        if self._hybrid_lock.acquire(timeout=timeout):
            try:
                return self._summary, self._summary_msg_idx, self._summary_last_updated_at
            finally:
                self._hybrid_lock.release()

        return None

    def _mode_meta_path(self) -> Path:
        """Return the path for mode-specific state."""
        hybrid = self._hybrid_meta_path()
        return hybrid.parent / hybrid.name.replace("_hybrid.json", "_mode_state.json")

    def _mode_state_dict(self) -> dict[str, Any]:
        """Return mode-specific state for persistence.

        Subclasses override this to persist their own fields while keeping the
        messages, summary, and base config out of the mode metadata file.
        """
        return {}

    def _save_mode_meta(self) -> None:
        """Persist mode-specific state to a file."""
        data = self._mode_state_dict()
        if not data:
            meta_path = self._mode_meta_path()
            if meta_path.exists():
                meta_path.unlink(missing_ok=True)
            return
        try:
            from src.utils.atomic_write import atomic_write_json

            meta_path = self._mode_meta_path()
            with atomic_write_json(meta_path) as f:
                json.dump(data, f)
        except Exception as exc:
            log.warning("Failed to save mode meta: %s", exc)

    def _load_mode_meta(self) -> None:
        """Restore mode-specific state from the meta file (if present)."""
        meta_path = self._mode_meta_path()
        if not meta_path.exists():
            return
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            self._restore_mode_state(data)
        except Exception as exc:
            log.warning("Failed to load mode meta: %s", exc)

    def _restore_mode_state(self, data: dict) -> None:  # noqa: B027
        """Override in subclasses to restore mode-specific state from dict.

        Called during load() to restore mode-specific state that was persisted
        by _save_mode_meta(). The dict contains only mode-specific keys
        (messages, hybrid state, and base keys are excluded).
        """
        pass

    # ── Tiered Context Cache persistence ─────────────────────────────

    def _tier_cache_path(self) -> Path:
        """Return the path for the tier cache JSON file.

        Follows the same pattern as ``_hybrid_meta_path()``.
        """
        safe_id = _sanitize_session_id(self.session_id)
        base: Path = getattr(self.store, "base_path", Path("data/history"))
        result = (base / f"{safe_id}_tier_cache.json").resolve()
        try:
            result.relative_to(base.resolve())
        except ValueError:
            raise ValueError(
                f"Path traversal detected in session_id: {self.session_id!r}"
            ) from None
        return result

    def _load_tier_cache(self) -> None:
        """Load tier cache from disk into ``self._tier_cache``.

        Missing or corrupt files silently leave the cache uninitialized
        (``_tier_cache_ready = False``).  Called from ``load()``.
        """
        from src.memory.tier_cache import TierCacheSnapshot

        cache_path = self._tier_cache_path()
        if not cache_path.exists():
            with self._hybrid_lock:
                self._tier_cache = None
                self._tier_cache_ready = False
            return
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            snapshot = TierCacheSnapshot.from_dict(data)
            with self._hybrid_lock:
                self._tier_cache = snapshot
                self._tier_cache_ready = True
        except Exception as exc:
            log.warning("Failed to load tier cache: %s", exc)
            with self._hybrid_lock:
                self._tier_cache = None
                self._tier_cache_ready = False

    def _get_tier_cache_snapshot(
        self, *, block: bool = True, timeout: float = 0.0
    ) -> tuple[Any | None, bool]:
        """Return a snapshot of the tier cache state."""
        if block:
            with self._hybrid_lock:
                return self._tier_cache, self._tier_cache_ready

        if self._hybrid_lock.acquire(timeout=timeout):
            try:
                return self._tier_cache, self._tier_cache_ready
            finally:
                self._hybrid_lock.release()

        return self._tier_cache, self._tier_cache_ready

    def _save_tier_cache(self, *, block: bool = True, timeout: float = 0.0) -> None:
        """Persist the tier cache to disk atomically.

        No-op when ``_tier_cache`` is ``None``.  Called from ``save()``.
        """
        snapshot, _ = self._get_tier_cache_snapshot(block=block, timeout=timeout)

        if snapshot is None:
            return
        try:
            from src.utils.atomic_write import atomic_write_json

            cache_path = self._tier_cache_path()
            with atomic_write_json(cache_path) as f:
                json.dump(snapshot.to_dict(), f)
        except Exception as exc:
            log.warning("Failed to save tier cache: %s", exc)

    def _clamp_summary_idx(self) -> None:
        """Ensure ``_summary_msg_idx`` does not exceed the message count.

        After ``sanitize_history()`` removes entries the stored index
        may point past the end of the (now shorter) message list.
        Clamping prevents messages from being skipped or re-summarized.
        """
        msg_count = self.get_message_count()
        if self._summary_msg_idx > msg_count:
            self._summary_msg_idx = msg_count

    def _schedule_slow_path(self, messages: list[Any], window_size: int) -> None:
        """Schedule background summarization + embedding if needed."""
        if not self._llm:
            return

        if not self.config.get("summarization", True):
            return

        with self._hybrid_lock:
            summary_idx = self._summary_msg_idx
            total = len(messages)
            window_start = max(0, total - window_size)

        unsummarized_start = min(summary_idx, total)
        unsummarized_end = window_start

        if unsummarized_start >= unsummarized_end:
            return

        unsummarized_count = unsummarized_end - unsummarized_start
        if unsummarized_count < _SUMMARY_BATCH_SIZE:
            return

        # Gate: only summarize when there is enough *meaningful*
        # conversation content.  Meaningful = human messages + final AI
        # responses.  Tool-call intermediaries (AIMessages with
        # tool_calls, ToolMessages) are excluded — they inflate the raw
        # message count but carry no conversational substance.
        # Either enough meaningful messages OR enough total chars
        # is sufficient to proceed.
        meaningful_count = 0
        meaningful_chars = 0
        for m in messages[unsummarized_start:unsummarized_end]:
            if self._is_meaningful_message(m):
                meaningful_count += 1
                content = getattr(m, "content", None)
                if content is None and isinstance(m, dict):
                    content = m.get("content")
                if isinstance(content, str):
                    meaningful_chars += len(content)
        if (
            meaningful_count < _MIN_MEANINGFUL_MSGS_FOR_SUMMARY
            and meaningful_chars < _MIN_MEANINGFUL_CHARS_FOR_SUMMARY
        ):
            return

        with self._hybrid_lock:
            slow_path_disabled = self._slow_path_failures >= _SLOW_PATH_MAX_FAILURES
        if slow_path_disabled:
            log.warning(
                "Background memory slow-path disabled after %d consecutive failures — "
                "summarization LLM may be unreachable",
                _SLOW_PATH_MAX_FAILURES,
            )
            return

        if self._bg_future is not None and not self._bg_future.done():
            elapsed = time.monotonic() - self._bg_submitted_at
            if elapsed > _BG_JOB_TIMEOUT_SECONDS:
                log.warning(
                    "Background memory summarization has been running for %.0fs "
                    "(limit %ds) — treating as stuck and allowing a fresh job. "
                    "The original thread will continue until it finishes naturally.",
                    elapsed,
                    _BG_JOB_TIMEOUT_SECONDS,
                )
                # Fall through and submit a fresh job; old thread runs to completion.
            else:
                if is_verbose():
                    log.debug("Background memory job still running — skipping")
                return

        batch = [_shallow_copy(m) for m in messages[unsummarized_start:unsummarized_end]]
        unsummarized_end_snapshot = unsummarized_end

        self._bg_future = _get_summarization_pool().submit(
            self._run_slow_path, batch, unsummarized_end_snapshot
        )
        self._bg_submitted_at = time.monotonic()

    def schedule_tier_roll_forward(
        self,
        max_context_tokens: int,
        llm: Any | None = None,
        compression_cache: dict[str, Any] | None = None,
    ) -> None:
        """Schedule a background roll-forward of the tier cache.

        Runs on the existing summarization pool.  Thread-safe: reads message
        state under ``_hybrid_lock``, performs LLM calls outside the lock,
        writes back under the lock.

        No-op when the pool cannot be acquired (logs a warning).
        """
        with self._hybrid_lock:
            messages_snapshot = list(getattr(self, "_messages", []))
            current_snapshot = self._tier_cache
            summary = self._summary or ""
            summary_msg_idx = self._summary_msg_idx

        if not messages_snapshot:
            return

        def _do_roll_forward() -> None:
            from src.memory.tier_cache import roll_forward

            try:
                new_snapshot = roll_forward(
                    messages=messages_snapshot,
                    current_snapshot=current_snapshot,
                    summary=summary,
                    summary_msg_idx=summary_msg_idx,
                    max_context_tokens=max_context_tokens,
                    llm=llm,
                    compression_cache=compression_cache,
                )
                # Only activate the tier path when the cache provides actual value:
                # either some messages have been pushed into T1/T2, or the Tier 0
                # boundary is non-zero (older messages excluded from verbatim window).
                # An all-T0 snapshot with no compressed content is equivalent to
                # the cold sliding-window fallback and should not override it.
                has_value = (
                    new_snapshot.tier0_boundary_idx > 0
                    or new_snapshot.tier1_messages
                    or new_snapshot.tier2_messages
                )
                if has_value:
                    with self._hybrid_lock:
                        self._tier_cache = new_snapshot
                        self._tier_cache_ready = True
                    log.debug(
                        "Tier roll-forward complete: boundary=%d, t1=%d msgs, t2=%d msgs",
                        new_snapshot.tier0_boundary_idx,
                        len(new_snapshot.tier1_messages),
                        len(new_snapshot.tier2_messages),
                    )
                else:
                    log.debug(
                        "Tier roll-forward produced empty snapshot (all messages fit in T0) "
                        "— keeping cold-cache fallback"
                    )
            except Exception as exc:
                log.warning(
                    "Tier roll-forward background job failed — agent will use "
                    "cold sliding-window fallback this session: %s",
                    exc,
                    exc_info=True,
                )

        try:
            _get_summarization_pool().submit(_do_roll_forward)
        except Exception as exc:
            log.warning("Failed to submit tier roll-forward to pool: %s", exc)

    def _run_slow_path(self, batch: list[Any], unsummarized_end: int) -> None:
        """Background: run summarization LLM + vector embedding + disk save."""
        t0 = time.monotonic()
        try:
            from src.memory.summarizer import generate_summary

            # Materialise the embedding provider on first use — safe to call
            # from the background thread; _ensure_embeddings_initialized is
            # protected by _hybrid_lock internally.
            self._ensure_embeddings_initialized()

            with self._hybrid_lock:
                summary_before = self._summary

            new_summary = generate_summary(self._llm, batch, summary_before)
            if new_summary is not None:
                with self._hybrid_lock:
                    self._summary = new_summary
                    self._summary_msg_idx = unsummarized_end
                    self._summary_last_updated_at = datetime.now(UTC)
                    # Reset the token counter so _check_summary_token_ttl does not
                    # immediately wipe the summary we just wrote (#486).
                    self._tokens_since_summary = 0
                    self._summary_dirty = True
                if is_verbose():
                    log.debug(
                        "Background summary updated in %.2fs, covers messages 0..%d (%d tokens est.)",
                        time.monotonic() - t0,
                        unsummarized_end,
                        len(new_summary) // 4,
                    )

                if self._vector_store is not None:
                    t1 = time.monotonic()
                    self._vector_store.add_messages(batch)
                    if is_verbose():
                        log.debug(
                            "Background vector embedding completed in %.2fs",
                            time.monotonic() - t1,
                        )

            self._save_hybrid_meta()
            with self._hybrid_lock:
                self._summary_dirty = False
            if self._vector_store is not None:
                self._vector_store.save()
            with self._hybrid_lock:
                self._slow_path_failures = 0
        except Exception as exc:
            with self._hybrid_lock:
                self._slow_path_failures += 1
            log.warning("Background memory update failed: %s", exc)

    def join_background(self, timeout: float = 60.0) -> None:
        """Block until any running background memory job completes."""
        fut = self._bg_future
        if fut is not None and not fut.done():
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass

    def _build_hybrid_prefix(self, user_input: str) -> str | None:
        """Build the hybrid-memory portion of the context prefix.

        Returns a string containing the rolling summary and any
        vector-recalled past exchanges, or ``None`` if neither is
        available.
        """
        parts: list[str] = []

        with self._hybrid_lock:
            summary = self._summary
        if summary:
            parts.append(f"Conversation summary (older context):\n{summary}")

        facts_snapshot = self._get_facts_store().load()
        if facts_snapshot and facts_snapshot.facts:
            parts.append("Persistent context from prior sessions:")
            parts.extend(f"• {fact}" for fact in facts_snapshot.facts)

        # Materialise the embedding provider on first use if only config was stored.
        self._ensure_embeddings_initialized()

        # Vector recall — skip for trivial inputs (greetings, single words)
        # to avoid a wasted embedding API call (~200ms+ round trip)
        if (
            self._vector_store is not None
            and self._vector_store.ready
            and len(user_input.split()) >= 3
        ):
            # Support config file and environment variable
            recall_k = self.config.get(
                "vector_recall_k",
                int(os.environ.get("COGTRIX_VECTOR_RECALL_K", "3")),
            )
            recall_threshold = self.config.get("vector_recall_threshold", 0.25)
            recalled = self._vector_store.recall(
                user_input, k=recall_k, score_threshold=recall_threshold
            )
            if recalled:
                joined = "\n---\n".join(recalled)
                parts.append(f"Related past exchanges:\n{joined}")

        return "\n\n".join(parts) if parts else None

    # ── Abstract interface ───────────────────────────────────────────

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """
        Return the mode identifier.

        Returns:
            Mode name string (e.g., 'conversation', 'code', 'reasoning')
        """
        pass

    @abstractmethod
    def prepare_context(self, user_input: str) -> MemoryContext:
        """
        Prepare context for the LLM given new user input.

        This method is called before each LLM invocation. It should:
        1. Select relevant messages from history
        2. Add any mode-specific context (summaries, entities, etc.)
        3. Return a MemoryContext with everything the LLM needs

        Args:
            user_input: The user's current input (for relevance filtering)

        Returns:
            MemoryContext with messages and metadata
        """
        pass

    @abstractmethod
    def update(
        self,
        user_input: str,
        ai_response: str,
        agent_messages: list[Any] | None = None,
    ) -> None:
        """
        Update memory after a conversation turn.

        Called after the LLM response is received. Should:
        1. Add new messages to history
        2. Extract any mode-specific information
        3. Trigger any necessary processing (summarization, etc.)

        Args:
            user_input: The user's input for this turn
            ai_response: The AI's response for this turn
            agent_messages: If provided, the *full* agent message chain
                for this turn (AIMessages with tool_calls, ToolMessages,
                and the final AIMessage).  Saving these enables the
                "Ralph Loop" — the agent can see its previous tool usage
                and continue iterating on complex tasks across restarts.
        """
        pass

    def prerecord_user(self, text: str) -> None:  # noqa: B027
        """Persist the user message for shutdown durability before the LLM call.

        Called outside the session lock so the message survives even if
        save_all() times out waiting for the lock during graceful shutdown.
        Does NOT call update() — call counts for tests are unaffected.
        Subclasses should override to write a lightweight pending-turn file.
        """

    def discard_prerecord(self) -> None:  # noqa: B027
        """Remove the pending pre-record without updating message history.

        Called when a turn is deferred or suppressed so the ephemeral pending
        file does not persist across restarts.  No-op if no pending file exists.
        """

    def get_system_prompt_additions(self) -> str | None:
        """
        Return mode-specific additions to the system prompt.

        Override this to add mode-specific instructions to the LLM.

        Returns:
            String to append to system prompt, or None
        """
        return None

    def load(self) -> None:
        """
        Load state from storage.

        Called once at session start. Override to load mode-specific state.
        Base implementation just marks as loaded.
        """
        self._load_tier_cache()
        self._loaded = True

    @staticmethod
    def _has_tool_calls(msg: Any) -> bool:
        """Return True if *msg* is an AI intermediate step with tool_calls."""
        tc = getattr(msg, "tool_calls", None) or (isinstance(msg, dict) and msg.get("tool_calls"))
        return bool(tc)

    @staticmethod
    def _is_meaningful_message(msg: Any) -> bool:
        """Return True for human messages and final AI responses.

        Intermediate tool-calling steps (AIMessages with ``tool_calls``)
        and ToolMessages are excluded — they inflate the raw message count
        but carry no conversational substance worth summarizing.
        """
        msg_type = getattr(msg, "type", None)
        if msg_type is None and isinstance(msg, dict):
            msg_type = msg.get("type")
        if msg_type is None:
            type_name = type(msg).__name__.lower()
            msg_type = type_name.replace("message", "")

        # Human messages are always meaningful
        if msg_type in ("human", "humanmessage"):
            return True

        # AI messages are meaningful only if they are final responses
        # (have actual content and are NOT intermediate tool-call steps)
        if msg_type in ("ai", "aimessage"):
            if BaseMemoryManager._has_tool_calls(msg):
                return False
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            return bool(content)

        # ToolMessages and anything else: not meaningful
        return False

    @staticmethod
    def sanitize_history(messages: list[Any]) -> list[Any]:
        """
        Remove poisoned entries from conversation history.

        Filters out:
        - AI messages with empty content (but **not** intermediate
          tool-calling steps that legitimately have empty text)
        - AI messages containing error responses
        - Orphaned human messages (where the AI response was removed)

        ToolMessages and AIMessages with ``tool_calls`` (intermediate
        agent steps) are always preserved — they form the agent's
        working chain needed for iterative continuation (Ralph Loop).

        Args:
            messages: List of message dicts or BaseMessage objects

        Returns:
            Cleaned list of messages with invalid pairs removed
        """
        cleaned: list[Any] = []
        removed = 0
        i = 0

        while i < len(messages):
            msg = messages[i]

            # Get content and type from message
            if isinstance(msg, dict):
                content = _coerce_content(msg.get("content", ""))
                msg_type = msg.get("type", "")
            elif hasattr(msg, "content"):
                content = _coerce_content(msg.content)
                msg_type = type(msg).__name__.lower()
            else:
                cleaned.append(msg)
                i += 1
                continue

            # ToolMessages are intermediate agent steps — always keep.
            is_tool = msg_type in ("tool", "toolmessage")
            if is_tool:
                cleaned.append(msg)
                i += 1
                continue

            # AIMessages with tool_calls are intermediate steps — always keep.
            is_ai = msg_type in ("ai", "aimessage")
            if is_ai and BaseMemoryManager._has_tool_calls(msg):
                cleaned.append(msg)
                i += 1
                continue

            # Check if this is a human message followed by a bad AI response,
            # possibly with an intermediate tool chain (BUG-064).
            is_human = msg_type in ("human", "humanmessage")
            if is_human and i + 1 < len(messages):
                # Scan forward past any tool chain (ai-tc + tool messages)
                # to find the terminal AI response.
                j = i + 1
                while j < len(messages):
                    scan_msg = messages[j]
                    if isinstance(scan_msg, dict):
                        scan_type = scan_msg.get("type", "")
                    elif hasattr(scan_msg, "content"):
                        scan_type = type(scan_msg).__name__.lower()
                    else:
                        scan_type = ""
                    scan_is_tool = scan_type in ("tool", "toolmessage")
                    scan_is_ai_tc = scan_type in (
                        "ai",
                        "aimessage",
                    ) and BaseMemoryManager._has_tool_calls(scan_msg)
                    if scan_is_tool or scan_is_ai_tc:
                        j += 1
                    else:
                        break

                # j now points to the first non-tool-chain message after human
                if j < len(messages):
                    terminal_msg = messages[j]
                    if isinstance(terminal_msg, dict):
                        terminal_content = _coerce_content(terminal_msg.get("content", ""))
                        terminal_type = terminal_msg.get("type", "")
                    elif hasattr(terminal_msg, "content"):
                        terminal_content = _coerce_content(terminal_msg.content)
                        terminal_type = type(terminal_msg).__name__.lower()
                    else:
                        terminal_content = ""
                        terminal_type = ""

                    terminal_is_ai = terminal_type in ("ai", "aimessage")
                    # Remove the entire chain (human + tool-chain + bad AI terminal)
                    if (
                        terminal_is_ai
                        and not BaseMemoryManager._has_tool_calls(terminal_msg)
                        and is_bad_ai_content(terminal_content)
                    ):
                        chain_len = j - i + 1  # human + all tool-chain msgs + terminal
                        removed += chain_len
                        i = j + 1
                        continue

            # Check standalone AI message with bad content
            if is_ai and is_bad_ai_content(content):
                removed += 1
                while cleaned:
                    tail = cleaned[-1]
                    if isinstance(tail, dict):
                        tail_type = tail.get("type", "")
                    elif hasattr(tail, "content"):
                        tail_type = type(tail).__name__.lower()
                    else:
                        break
                    is_tool_tail = tail_type in ("tool", "toolmessage")
                    is_ai_tc_tail = tail_type in (
                        "ai",
                        "aimessage",
                    ) and BaseMemoryManager._has_tool_calls(tail)
                    if is_tool_tail or is_ai_tc_tail:
                        cleaned.pop()
                        removed += 1
                    else:
                        break
                # Also remove the triggering HumanMessage at the start of the chain
                # (BUG-033: prevents lone HumanMessage with no following response)
                if cleaned:
                    tail = cleaned[-1]
                    if isinstance(tail, dict):
                        tail_type = tail.get("type", "")
                    elif hasattr(tail, "content"):
                        tail_type = type(tail).__name__.lower()
                    else:
                        tail_type = ""
                    if tail_type in ("human", "humanmessage"):
                        cleaned.pop()
                        removed += 1
                i += 1
                continue

            cleaned.append(msg)
            i += 1

        if removed > 0:
            log.info(
                "History sanitized: removed %d poisoned entries (%d -> %d messages)",
                removed,
                len(messages),
                len(cleaned),
            )

        return cleaned

    # ── Timestamp utilities ─────────────────────────────────────────

    @staticmethod
    def _now_ts() -> str:
        """Return the current UTC time as an ISO 8601 string with 'Z' suffix."""
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _get_msg_ts(msg: Any) -> str | None:
        """Extract the Cogtrix ``_ts`` timestamp from a message, if any."""
        if hasattr(msg, "additional_kwargs"):
            return (msg.additional_kwargs or {}).get("_ts")
        if isinstance(msg, dict):
            return msg.get("timestamp")
        return None

    @staticmethod
    def _set_msg_ts(msg: Any, ts: str | None = None) -> None:
        """Attach a timestamp to *msg* in-place."""
        if ts is None:
            ts = BaseMemoryManager._now_ts()
        if hasattr(msg, "additional_kwargs"):
            if msg.additional_kwargs is None:
                msg.additional_kwargs = {}
            msg.additional_kwargs["_ts"] = ts
        elif isinstance(msg, dict):
            msg["timestamp"] = ts

    @staticmethod
    def _inject_timestamps(messages: list[Any]) -> list[Any]:
        """Return copies of *messages* with ``[YYYY-MM-DD HH:MM:SS UTC]`` prepended.

        Messages that have no stored timestamp are returned as-is.
        ToolMessages and AIMessages that are intermediate tool-calling
        steps (have ``tool_calls`` but no meaningful text) are passed
        through unchanged — timestamps are only displayed on the
        "bookend" Human/AI messages.  The originals are never mutated.
        """

        def _str_content(m: Any) -> str:
            c = m.content if hasattr(m, "content") else m.get("content", "")
            return c if isinstance(c, str) else str(c) if c else ""

        result: list[Any] = []
        for msg in messages:
            # ToolMessages: pass through unchanged (no timestamp prefix)
            if ToolMessage is not None and isinstance(msg, ToolMessage):
                result.append(msg)
                continue
            if isinstance(msg, dict) and msg.get("type", "") in ("tool", "toolmessage"):
                result.append(msg)
                continue

            # Intermediate AI steps (tool_calls, empty/minimal text): keep as-is
            if (AIMessage is not None and isinstance(msg, AIMessage)) or (
                isinstance(msg, dict) and msg.get("type", "") in ("ai", "aimessage")
            ):
                if BaseMemoryManager._has_tool_calls(msg):
                    result.append(msg)
                    continue

            # No timestamp stored → keep as-is
            ts = BaseMemoryManager._get_msg_ts(msg)
            if not ts:
                result.append(msg)
                continue

            # Build compact display string — _now_ts() always produces YYYY-MM-DDTHH:MM:SSZ
            try:
                display = ts[:10] + " " + ts[11:19] + " UTC"
            except (IndexError, TypeError):
                display = ts

            prefix = f"[{display}] "

            if HumanMessage is not None and isinstance(msg, HumanMessage):
                result.append(msg.model_copy(update={"content": prefix + _str_content(msg)}))
            elif AIMessage is not None and isinstance(msg, AIMessage):
                # Don't prefix AI messages — the LLM mimics the timestamp
                # pattern and outputs it in new responses.
                result.append(msg)
                continue
            elif isinstance(msg, dict) and "content" in msg:
                if msg.get("type", "") in ("ai", "aimessage"):
                    result.append(msg)
                    continue
                result.append({**msg, "content": prefix + _str_content(msg)})
            else:
                result.append(msg)
        return result

    # ── Persistence hooks ────────────────────────────────────────

    def save(self) -> None:  # noqa: B027
        """
        Persist current state to storage.

        Called after each turn. Override to save mode-specific state.
        Subclasses should call ``super().save()`` to persist vector store
        and hybrid summary meta.
        """
        fut = self._bg_future
        if fut is None or fut.done():
            self._save_hybrid_meta()
        elif self._summary_dirty:
            self._save_hybrid_meta()
            with self._hybrid_lock:
                self._summary_dirty = False
        self._save_mode_meta()
        self._save_tier_cache()
        if self._vector_store is not None:
            self._vector_store.save()

    def _reset_summary_state(
        self, expected_summary_last_updated_at: datetime | None = None
    ) -> bool:
        """Clear just the rolling summary and its persisted metadata.

        Returns True when the reset is applied. If an expected timestamp is
        provided and the live summary was refreshed in the meantime, the reset
        is skipped and False is returned.
        """
        with self._hybrid_lock:
            if (
                expected_summary_last_updated_at is not None
                and self._summary_last_updated_at != expected_summary_last_updated_at
            ):
                return False
            self._summary = None
            self._summary_msg_idx = 0
            self._summary_last_updated_at = None
            self._tokens_since_summary = 0
        try:
            meta_path = self._hybrid_meta_path()
            if meta_path.exists():
                meta_path.unlink(missing_ok=True)
        except Exception:
            pass
        return True

    def reset_summary_state(self) -> None:
        """Reset rolling summary state without disturbing the message history."""
        self._reset_summary_state()
        log.info("Rolling summary state reset for session %s", self.session_id)

    def save_messages_only(self) -> None:
        """Persist only the message history.

        This path intentionally avoids the hybrid-lock protected summary and
        tier-cache state so shutdown can salvage the latest turn even when the
        background summarizer is mid-flight.
        """
        self.store.save_history(self.session_id, getattr(self, "_messages", []))

    def shutdown(self) -> None:
        """Cancel background work and save final state.

        Called once at process exit.  The caller is expected to use
        ``os._exit()`` afterward to prevent Python's internal
        ``_python_exit()`` from blocking on ThreadPoolExecutor threads.
        """
        fut = self._bg_future
        bg_still_running = fut is not None and fut.running()
        if fut is not None and not fut.done():
            fut.cancel()
        self._bg_future = None
        if bg_still_running:
            # The background thread may be holding _hybrid_lock while doing
            # summarization or embedding work.  A full save() would block on
            # that lock and risk losing the latest turn.  Save the messages
            # immediately, then best-effort snapshot the hybrid, mode, and
            # tier-cache state without waiting for the background job.
            self.save_messages_only()
            self._save_hybrid_meta(block=False)
            self._save_mode_meta()
            self._save_tier_cache(block=False)
            return
        try:
            self.save()
        except (KeyboardInterrupt, SystemExit, OSError, PermissionError):
            # Ctrl-C during shutdown — save was interrupted; suppress the
            # traceback because os._exit() follows immediately anyway.
            # OSError (disk-full, etc.) and PermissionError are also suppressed
            # since there's nothing recoverable at shutdown time.
            pass

    def reset_summary(self) -> None:
        """Clear the rolling summary without destroying message history.

        Use this when the conversation topic shifts significantly (e.g.
        from internal engineering work to an external business query) so
        that older domain data does not contaminate future turns.

        The message history and vector-store embeddings are preserved;
        only the incremental summary and its coverage index are reset.
        """
        self.join_background()
        should_distill = bool(
            self._summary and self._llm is not None and self.config.get("distill_on_expire", True)
        )
        if should_distill:
            try:
                from src.memory.distillation import distill_summary

                facts = distill_summary(self._llm, self._summary or "")
                if facts:
                    self._get_facts_store().save(
                        facts,
                        ttl_days=int(self.config.get("facts_ttl_days", 7)),
                    )
            except Exception as exc:
                log.warning("Fact distillation failed: %s", exc)
        with self._hybrid_lock:
            self._summary = None
            self._summary_msg_idx = 0
            self._summary_last_updated_at = None
        self._save_hybrid_meta()
        log.info("Rolling summary reset for session %s", self.session_id)

    def clear(self) -> None:  # noqa: B027
        """
        Clear all memory for this session.

        Resets to initial state. Override to clear mode-specific data.
        Default is a no-op (subclasses should call super().clear()).
        """
        self.join_background()
        self._reset_summary_state()
        with self._hybrid_lock:
            self._tier_cache = None
            self._tier_cache_ready = False
        if self._vector_store is not None:
            self._vector_store.clear()
        self._get_facts_store().clear()
        # Remove persisted meta file
        try:
            meta_path = self._hybrid_meta_path()
            if meta_path.exists():
                meta_path.unlink(missing_ok=True)
        except Exception:
            pass
        # Remove tier cache file
        try:
            cache_path = self._tier_cache_path()
            if cache_path.exists():
                cache_path.unlink(missing_ok=True)
        except Exception:
            pass

    def get_stats(self) -> dict[str, Any]:
        """
        Return statistics about current memory state.

        Useful for debugging and display.

        Returns:
            Dictionary with mode-specific statistics
        """
        with self._hybrid_lock:
            tier_ready = self._tier_cache_ready
            tier_cache = self._tier_cache

        stats: dict[str, Any] = {
            "mode": self.mode_name,
            "session_id": self.session_id,
            "loaded": self._loaded,
            "tier_cache_ready": tier_ready,
        }
        if tier_cache is not None:
            stats["tier0_boundary_idx"] = tier_cache.tier0_boundary_idx
            stats["tier1_token_count"] = tier_cache.tier1_token_count
            stats["tier2_token_count"] = tier_cache.tier2_token_count
        return stats

    def get_message_count(self) -> int:
        """
        Return the total number of messages stored.

        Override in subclasses that track messages.

        Returns:
            Number of messages in memory
        """
        return 0

    def pop_last_turn(self) -> int:
        """Remove the last user+assistant exchange from memory.

        Default implementation is a no-op. Subclasses that maintain
        a message list should override this.

        Returns:
            Number of messages removed (0 if not supported or nothing to remove).
        """
        return 0

    @staticmethod
    def _estimate_tokens(messages: list[Any]) -> int:
        """Rough token estimation for a list of messages.

        Uses simple heuristic: ~4 characters per token.
        Handles both string and list (multimodal) content.

        Args:
            messages: List of messages

        Returns:
            Estimated token count
        """
        total_chars = 0
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                total_chars += len(_coerce_content(msg.content))
            elif isinstance(msg, dict) and msg.get("content"):
                total_chars += len(_coerce_content(msg["content"]))
        return total_chars // 4

    # --- Serialization interface (for persistence) ---

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize full state to dictionary.

        Override to include mode-specific state.

        Returns:
            Dictionary representation of state
        """
        d: dict[str, Any] = {
            "mode": self.mode_name,
            "version": 1,
            "session_id": self.session_id,
            "config": self.config,
        }
        # Hybrid state — read under lock for consistency with background summarizer
        with self._hybrid_lock:
            summary = self._summary
            summary_idx = self._summary_msg_idx
            summary_updated_at = self._summary_last_updated_at
        if summary is not None:
            d["_summary"] = summary
            d["_summary_msg_idx"] = summary_idx
        if summary_updated_at is not None:
            d["_summary_last_updated_at"] = summary_updated_at.isoformat()
        return d

    def from_dict(self, data: dict[str, Any]) -> None:
        """
        Restore state from dictionary.

        Override to restore mode-specific state.
        Only restores serialized config if no runtime config was provided
        during __init__, so that live/reloaded config is not overwritten.

        Args:
            data: Dictionary from to_dict()

        Raises:
            ValueError: If mode in data doesn't match this manager's mode
        """
        if data.get("mode") != self.mode_name:
            raise ValueError(f"Mode mismatch: expected {self.mode_name}, got {data.get('mode')}")
        # Only apply serialized config when no runtime config was set
        if not self.config:
            self.config = data.get("config", {})

        # Restore hybrid state
        with self._hybrid_lock:
            self._summary = data.get("_summary")
            self._summary_msg_idx = data.get("_summary_msg_idx", 0)
            raw_summary_updated_at = data.get("_summary_last_updated_at")
            if raw_summary_updated_at:
                self._summary_last_updated_at = datetime.fromisoformat(raw_summary_updated_at)
            else:
                self._summary_last_updated_at = None
