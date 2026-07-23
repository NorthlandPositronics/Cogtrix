"""
Code development memory mode for programming assistance.

Optimized for:
- Code writing and editing
- Debugging and error resolution
- Code review and refactoring
- Project-wide understanding

Features:
- Smaller working memory (code context is expensive)
- File tracking (what files are being worked on)
- Task tracking (current objective and progress)
- Error tracking (recent errors for debugging context)
- Code extraction (identify file references in conversation)
"""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.memory.base import BaseMemoryStore
from src.memory.context import MemoryContext
from src.memory.manager import BaseMemoryManager

# Optional LangChain imports
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
except ImportError:
    HumanMessage = None  # type: ignore[misc, assignment]
    AIMessage = None  # type: ignore[misc, assignment]
    BaseMessage = None  # type: ignore[misc, assignment]


@dataclass
class FileContext:
    """Information about a file being worked on."""

    path: str
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    snippet: str | None = None
    line_range: tuple | None = None  # (start, end) lines


@dataclass
class TaskProgress:
    """Tracks progress on current coding task."""

    description: str
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    steps_completed: list[str] = field(default_factory=list)
    current_step: str | None = None
    blockers: list[str] = field(default_factory=list)


class CodeDevelopmentMemoryManager(BaseMemoryManager):
    """
    Memory manager optimized for code development assistance.

    Configuration options:
        working_memory_size (int): Messages in context (default: 30)
        track_files (bool): Track mentioned files (default: True)
        track_errors (bool): Track errors mentioned (default: True)
        max_errors (int): Max errors to keep (default: 5)
        max_files (int): Max files to track (default: 20)
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "working_memory_size": 30,
        "track_files": True,
        "track_errors": True,
        "max_errors": 5,
        "max_files": 20,
    }

    # Patterns for extracting file references
    FILE_PATTERNS = [
        r"`([^`]+\.(?:py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|hpp|"
        r"md|json|yaml|yml|toml|sql|sh|bash|css|html|xml))`",
        r'"([^"]+\.(?:py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|hpp|'
        r'md|json|yaml|yml|toml|sql|sh|bash|css|html|xml))"',
        r"'([^']+\.(?:py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|hpp|"
        r"md|json|yaml|yml|toml|sql|sh|bash|css|html|xml))'",
    ]

    # Patterns for error extraction
    ERROR_PATTERNS = [
        r"((?:Error|Exception|Traceback)[^\n]{0,100})",
        r"((?:TypeError|ValueError|KeyError|AttributeError|"
        r"ImportError|SyntaxError|NameError):\s*[^\n]{0,100})",
        r"(error\[E\d+\]:[^\n]{0,100})",  # Rust errors
        r"(error TS\d+:[^\n]{0,100})",  # TypeScript errors
    ]

    def __init__(
        self,
        store: BaseMemoryStore,
        session_id: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(store, session_id, config)
        self._mode_config = {**self.DEFAULT_CONFIG, **(config or {})}

        # Working memory
        self._messages: list[Any] = []

        # Task tracking
        self._current_task: TaskProgress | None = None

        # File tracking
        self._files: dict[str, FileContext] = {}
        self._current_file: str | None = None

        # Error tracking
        self._recent_errors: list[str] = []

        # Change tracking
        self._changes_made: list[str] = []

    @property
    def mode_name(self) -> str:
        return "code"

    # --- Public API for code context ---

    def set_task(self, description: str) -> None:
        """Set the current coding task."""
        self._current_task = TaskProgress(description=description)

    def add_progress(self, step: str) -> None:
        """Record a completed step."""
        if self._current_task:
            self._current_task.steps_completed.append(step)

    def set_current_step(self, step: str) -> None:
        """Set what we're currently working on."""
        if self._current_task:
            self._current_task.current_step = step

    def add_blocker(self, blocker: str) -> None:
        """Record a blocker."""
        if self._current_task:
            self._current_task.blockers.append(blocker)

    def set_file_context(
        self,
        path: str,
        snippet: str | None = None,
        line_range: tuple | None = None,
    ) -> None:
        """Set context for a specific file."""
        max_files = self._mode_config["max_files"]

        # If file not already tracked and at limit, remove oldest
        if path not in self._files and len(self._files) >= max_files:
            oldest = min(
                self._files,
                key=lambda p: self._files[p].last_accessed,
            )
            del self._files[oldest]

        self._files[path] = FileContext(
            path=path,
            snippet=snippet,
            line_range=line_range,
        )
        self._current_file = path

    def add_error(self, error: str) -> None:
        """Record an error."""
        max_errors = self._mode_config["max_errors"]
        self._recent_errors.append(error)
        self._recent_errors = self._recent_errors[-max_errors:]

    def record_change(self, description: str) -> None:
        """Record a file change."""
        timestamp = datetime.now(UTC).strftime("%H:%M")
        self._changes_made.append(f"{timestamp} - {description}")

    def get_tracked_files(self) -> list[str]:
        """Return list of tracked file paths."""
        return list(self._files.keys())

    def get_recent_errors(self) -> list[str]:
        """Return list of recent errors."""
        return list(self._recent_errors)

    # --- Memory Manager Interface ---

    def load(self) -> None:
        """Load code session from storage, sanitizing bad entries."""
        self._messages = self.store.load_history(self.session_id)
        self._messages = self.sanitize_history(self._messages)
        self._load_hybrid_meta()
        self._clamp_summary_idx()
        self._loaded = True

    def save(self) -> None:
        """Save code session to storage."""
        self.store.save_history(self.session_id, self._messages)
        super().save()

    def prepare_context(self, user_input: str) -> MemoryContext:
        """Prepare code-optimized context for LLM."""
        # Record the moment the user sent this message
        self._pending_user_ts = self._now_ts()

        window_size = self._mode_config["working_memory_size"]
        context_messages = self._messages[-window_size:] if self._messages else []

        # Inject timestamps so the LLM has temporal awareness
        context_messages = self._inject_timestamps(context_messages)

        # Build context prefix with code-specific information
        prefix_parts = []

        # Hybrid memory (summary + recall)
        hybrid = self._build_hybrid_prefix(user_input)
        if hybrid:
            prefix_parts.append(hybrid)

        # Task context
        if self._current_task:
            task_desc = self._current_task.description
            task_lines = [f"**Current Task:** {task_desc}"]

            if self._current_task.steps_completed:
                steps = self._current_task.steps_completed[-5:]
                steps_text = "\n".join(f"  ✓ {s}" for s in steps)
                task_lines.append(f"**Completed:**\n{steps_text}")

            if self._current_task.current_step:
                task_lines.append(f"**Working on:** {self._current_task.current_step}")

            if self._current_task.blockers:
                blockers = "\n".join(f"  ⚠ {b}" for b in self._current_task.blockers)
                task_lines.append(f"**Blockers:**\n{blockers}")

            prefix_parts.append("\n".join(task_lines))

        # File context
        if self._files:
            recent_files = sorted(
                self._files.values(),
                key=lambda f: f.last_accessed,
                reverse=True,
            )[:5]
            files_list = ", ".join(f"`{f.path}`" for f in recent_files)
            prefix_parts.append(f"**Recent files:** {files_list}")

        # Current file with snippet
        if self._current_file and self._current_file in self._files:
            fc = self._files[self._current_file]
            if fc.snippet:
                snippet_preview = fc.snippet[:500] + "..." if len(fc.snippet) > 500 else fc.snippet
                lines_info = ""
                if fc.line_range:
                    start, end = fc.line_range
                    lines_info = f" (lines {start}-{end})"
                prefix_parts.append(
                    f"**Current file:** `{fc.path}`{lines_info}\n```\n{snippet_preview}\n```"
                )

        # Recent errors
        if self._recent_errors:
            errors_text = "\n".join(f"  • {e[:100]}" for e in self._recent_errors)
            prefix_parts.append(f"**Recent errors:**\n{errors_text}")

        # Recent changes
        if self._changes_made:
            changes_text = "\n".join(f"  • {c}" for c in self._changes_made[-5:])
            prefix_parts.append(f"**Recent changes:**\n{changes_text}")

        context_prefix = "\n\n".join(prefix_parts) if prefix_parts else None

        return MemoryContext(
            messages=context_messages,
            system_additions=self.get_system_prompt_additions(),
            context_prefix=context_prefix,
            mode=self.mode_name,
            total_messages_stored=len(self._messages),
            context_messages_count=len(context_messages),
            token_estimate=self._estimate_tokens(context_messages),
            metadata={
                "files_tracked": len(self._files),
                "current_file": self._current_file,
                "has_task": self._current_task is not None,
                "error_count": len(self._recent_errors),
            },
        )

    def update(
        self,
        user_input: str,
        ai_response: str,
        agent_messages: list[Any] | None = None,
    ) -> None:
        """Update memory and extract code-related information."""
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
            for m in agent_messages:
                self._messages.append(m)
            last = agent_messages[-1]
            if hasattr(last, "content") or isinstance(last, dict):
                self._set_msg_ts(last)
        else:
            if AIMessage is not None:
                ai_msg: Any = AIMessage(content=ai_response)
            else:
                ai_msg = {"type": "ai", "content": ai_response}
            self._set_msg_ts(ai_msg)
            self._messages.append(ai_msg)

        # Incrementally summarize messages outside the sliding window
        window_size = self._mode_config["working_memory_size"]
        self._schedule_slow_path(self._messages, window_size)

        # Extract file references
        if self._mode_config["track_files"]:
            self._extract_files(user_input)
            self._extract_files(ai_response)

        # Extract errors
        if self._mode_config["track_errors"]:
            self._extract_errors(user_input)

    def get_system_prompt_additions(self) -> str | None:
        """Return code-mode system prompt additions."""
        return (
            "You are an expert programmer. COMPLETE coding tasks end-to-end. "
            "When asked to analyze/modify code: read files, understand context, "
            "make changes or provide complete solutions. "
            "Don't stop to ask what to do — execute the requested work. "
            "Be concise, show code examples, track file paths and errors."
        )

    def clear(self) -> None:
        """Clear all code development memory."""
        super().clear()
        self._messages = []
        self._current_task = None
        self._files = {}
        self._current_file = None
        self._recent_errors = []
        self._changes_made = []

    def get_message_count(self) -> int:
        """Return total number of messages stored."""
        return len(self._messages)

    def get_stats(self) -> dict[str, Any]:
        """Return code development statistics."""
        return {
            **super().get_stats(),
            "total_messages": len(self._messages),
            "working_memory_size": self._mode_config["working_memory_size"],
            "files_tracked": len(self._files),
            "current_file": self._current_file,
            "has_task": self._current_task is not None,
            "error_count": len(self._recent_errors),
            "changes_count": len(self._changes_made),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize code development state."""
        from src.memory.json_store import _message_to_dict

        base = super().to_dict()

        messages_data = [_message_to_dict(m) for m in self._messages]

        # Serialize task
        task_data = None
        if self._current_task:
            task_data = {
                "description": self._current_task.description,
                "started_at": self._current_task.started_at.isoformat(),
                "steps_completed": self._current_task.steps_completed,
                "current_step": self._current_task.current_step,
                "blockers": self._current_task.blockers,
            }

        # Serialize files
        files_data = {}
        for path, fc in self._files.items():
            files_data[path] = {
                "path": fc.path,
                "last_accessed": fc.last_accessed.isoformat(),
                "snippet": fc.snippet,
                "line_range": list(fc.line_range) if fc.line_range else None,
            }

        return {
            **base,
            "messages": messages_data,
            "task": task_data,
            "files": files_data,
            "current_file": self._current_file,
            "recent_errors": self._recent_errors,
            "changes_made": self._changes_made,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore code development state."""
        from src.memory.json_store import _dict_to_message

        super().from_dict(data)

        self._messages = [_dict_to_message(d) for d in data.get("messages", [])]

        # Restore task
        task_data = data.get("task")
        if task_data:
            self._current_task = TaskProgress(
                description=task_data["description"],
                started_at=datetime.fromisoformat(task_data["started_at"]),
                steps_completed=task_data.get("steps_completed", []),
                current_step=task_data.get("current_step"),
                blockers=task_data.get("blockers", []),
            )
        else:
            self._current_task = None

        # Restore files
        self._files = {}
        for path, fc_data in data.get("files", {}).items():
            line_range = fc_data.get("line_range")
            self._files[path] = FileContext(
                path=fc_data["path"],
                last_accessed=datetime.fromisoformat(fc_data["last_accessed"]),
                snippet=fc_data.get("snippet"),
                line_range=tuple(line_range) if line_range else None,
            )

        self._current_file = data.get("current_file")
        self._recent_errors = data.get("recent_errors", [])
        self._changes_made = data.get("changes_made", [])
        self._loaded = True

    # --- Private methods ---

    def _extract_files(self, text: str) -> None:
        """Extract file paths mentioned in text."""
        max_files = self._mode_config["max_files"]

        for pattern in self.FILE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                path = match if isinstance(match, str) else match[0]
                # Skip very short matches that might be false positives
                if len(path) < 3:
                    continue
                if path not in self._files:
                    if len(self._files) >= max_files:
                        # Remove oldest
                        oldest = min(
                            self._files,
                            key=lambda p: self._files[p].last_accessed,
                        )
                        del self._files[oldest]
                    self._files[path] = FileContext(path=path)
                else:
                    self._files[path].last_accessed = datetime.now(UTC)

    def _extract_errors(self, text: str) -> None:
        """Extract error messages from text."""
        for pattern in self.ERROR_PATTERNS:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                error = match.strip()[:200]  # Limit length
                if error and error not in self._recent_errors:
                    self.add_error(error)

    # _estimate_tokens() is inherited from BaseMemoryManager
