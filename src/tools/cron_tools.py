"""Cron scheduling tools — schedule recurring LLM prompts.

Tools:
    cron_add    — add a recurring prompt on a cron schedule
    cron_list   — list all scheduled cron jobs
    cron_remove — remove a scheduled cron job by ID

When a job fires, the registered LLM factory is called and the prompt is
sent directly to the LLM.  Jobs can also opt into inherited session context
when the host process provides a runner callback.  Output is written to the
cron logger with a [CRON] prefix so it is captured in the log file without
polluting the interactive console.

Configuration:
    Call configure_cron(data_dir, llm_factory, job_runner) once at startup.
    llm_factory is a zero-argument callable that returns a ChatModel;
    it is called fresh each time a job fires so it always reflects the
    current provider / model settings.
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel, Field
else:
    try:
        from pydantic import BaseModel, Field
    except ImportError:  # pragma: no cover
        BaseModel = object  # type: ignore[assignment,misc]
        Field = lambda *a, **kw: None  # type: ignore[assignment]  # noqa: E731

log = logging.getLogger("cogtrix.tools.cron")

# ── Optional croniter import ──────────────────────────────────────────────────

try:
    from croniter import croniter as _croniter

    _HAS_CRONITER = True
except ImportError:  # pragma: no cover
    _HAS_CRONITER = False
    _croniter = None  # type: ignore[assignment]

try:
    from langchain_core.messages import HumanMessage as _HumanMessage
except ImportError:  # pragma: no cover
    _HumanMessage = None  # type: ignore[assignment]

# ── Module-level singletons ───────────────────────────────────────────────────

_scheduler: CronScheduler | None = None
_llm_factory: Callable[[], Any] | None = None  # () -> BaseChatModel
_job_runner: Callable[[CronJob], str] | None = None
_data_dir: pathlib.Path = pathlib.Path("data/cron")
_cron_llm_timeout_seconds: float = 120.0  # per-call LLM timeout for cron jobs

# Per-request owner ID scopes all cron operations to the calling session (#424).
# Set via set_cron_session_id() at session start; empty string = no isolation
# (single-tenant / legacy deployments).
_cron_session_id: ContextVar[str] = ContextVar("cron_session_id", default="")


def set_cron_session_id(session_id: str) -> None:
    """Bind all cron operations in the current context to *session_id*."""
    _cron_session_id.set(session_id)


def get_cron_session_id() -> str:
    """Return the session ID bound to the current execution context."""
    return _cron_session_id.get()


_CHECK_INTERVAL = 10  # seconds between scheduler tick


# ── Public configuration API ──────────────────────────────────────────────────


def configure_cron(
    data_dir: str | pathlib.Path | None = None,
    llm_factory: Callable[[], Any] | None = None,
    job_runner: Callable[[CronJob], str] | None = None,
    initial_jobs: list[dict[str, Any]] | None = None,
    llm_timeout: float | None = None,
) -> None:
    """Configure and start the cron scheduler.

    Args:
        data_dir:    Directory for job persistence (default: ``data/cron``).
        llm_factory: Zero-argument callable returning a LangChain ``BaseChatModel``.
                     Called fresh each time a job fires.
        job_runner: Optional callable used for ``context: inherit`` jobs.
        initial_jobs: Optional list of serialized cron job definitions to seed
            after the scheduler is started.
        llm_timeout: Optional per-call LLM timeout in seconds (default: 120).
    """
    global _scheduler, _llm_factory, _job_runner, _data_dir, _cron_llm_timeout_seconds
    if data_dir is not None:
        _data_dir = pathlib.Path(data_dir)
    if llm_factory is not None:
        _llm_factory = llm_factory
    if job_runner is not None:
        _job_runner = job_runner
    if llm_timeout is not None:
        _cron_llm_timeout_seconds = llm_timeout
    if _scheduler is None:
        _scheduler = CronScheduler(_data_dir)
        _scheduler.start()
    elif llm_factory is not None:
        pass  # scheduler already running; new factory will be used on next fire
    if initial_jobs:
        for job_cfg in initial_jobs:
            try:
                _scheduler.add(
                    schedule=str(job_cfg["schedule"]),
                    prompt=str(job_cfg["prompt"]),
                    name=str(job_cfg.get("name", "")),
                    context=str(job_cfg.get("context", "fresh")),
                )
            except Exception as exc:  # pragma: no cover - startup warning path
                log.warning("Skipping configured cron job %r: %s", job_cfg, exc)


def _get_scheduler() -> CronScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = CronScheduler(_data_dir)
        _scheduler.start()
    return _scheduler


# ── CronJob data object ───────────────────────────────────────────────────────


class CronJob:
    """Lightweight value object representing a scheduled cron job."""

    __slots__ = (
        "id",
        "name",
        "schedule",
        "prompt",
        "context",
        "owner_id",
        "created_at",
        "last_run",
        "next_run",
        "run_count",
    )

    def __init__(
        self,
        id: str,
        name: str,
        schedule: str,
        prompt: str,
        created_at: float,
        context: str = "fresh",
        owner_id: str = "",
        last_run: float | None = None,
        next_run: float | None = None,
        run_count: int = 0,
    ) -> None:
        self.id = id
        self.name = name
        self.schedule = schedule
        self.prompt = prompt
        self.context = context
        self.owner_id = owner_id
        self.created_at = created_at
        self.last_run = last_run
        self.next_run = next_run
        self.run_count = run_count

    def next_run_human(self) -> str:
        if self.next_run is None:
            return "unknown"
        return datetime.fromtimestamp(self.next_run, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "schedule": self.schedule,
            "prompt": self.prompt,
            "context": self.context,
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "run_count": self.run_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CronJob:
        return cls(
            id=str(d["id"]),
            name=str(d.get("name", "")),
            schedule=str(d["schedule"]),
            prompt=str(d["prompt"]),
            context=str(d.get("context", "fresh")),
            owner_id=str(d.get("owner_id", "")),
            created_at=float(d.get("created_at", time.time())),
            last_run=float(d["last_run"]) if d.get("last_run") is not None else None,
            next_run=float(d["next_run"]) if d.get("next_run") is not None else None,
            run_count=int(d.get("run_count", 0)),
        )


# ── Scheduler ─────────────────────────────────────────────────────────────────


class CronScheduler:
    """Background thread that fires scheduled cron jobs."""

    def __init__(self, data_dir: pathlib.Path) -> None:
        self._data_dir = data_dir
        self._jobs: dict[str, CronJob] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="cron-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ── Public job management ─────────────────────────────────────────────────

    def add(
        self,
        schedule: str,
        prompt: str,
        name: str = "",
        context: str = "fresh",
        owner_id: str = "",
    ) -> tuple[CronJob, bool]:
        """Add a cron job. Returns (job, is_new) where *is_new* is False if an
        identical job already existed."""
        if not _HAS_CRONITER or _croniter is None:
            raise RuntimeError(
                "croniter package is required for cron scheduling. Run: uv add croniter"
            )
        if not _croniter.is_valid(schedule):
            raise ValueError(
                f"Invalid cron expression: {schedule!r}. "
                "Use 5-field (min hr dom mon dow) or 6-field (sec min hr dom mon dow) format."
            )
        context = context.strip().lower()
        if context not in {"fresh", "inherit"}:
            raise ValueError("context must be either 'fresh' or 'inherit'")
        effective_name = name or schedule
        with self._lock:
            for existing in self._jobs.values():
                if (
                    existing.schedule == schedule
                    and existing.prompt == prompt
                    and existing.name == effective_name
                    and existing.context == context
                    and existing.owner_id == owner_id
                ):
                    return existing, False
        next_run = _croniter(schedule, time.time()).get_next(float)
        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=effective_name,
            schedule=schedule,
            prompt=prompt,
            created_at=time.time(),
            context=context,
            owner_id=owner_id,
            next_run=next_run,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._save()
        log.info(
            "Cron job %s added: owner=%r schedule=%r next=%s",
            job.id,
            owner_id or "(global)",
            schedule,
            job.next_run_human(),
        )
        return job, True

    def remove(self, job_id: str, owner_id: str = "") -> None:
        """Remove a cron job by ID.

        When *owner_id* is non-empty, raises PermissionError if the job
        belongs to a different owner — prevents cross-tenant deletion (#424).
        """
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"No cron job with ID {job_id!r}.")
            job = self._jobs[job_id]
            if owner_id and job.owner_id and job.owner_id != owner_id:
                raise PermissionError(f"Cron job {job_id!r} belongs to a different session.")
            del self._jobs[job_id]
        self._save()
        log.info("Cron job %s removed by owner=%r", job_id, owner_id or "(global)")

    def list_jobs(self, owner_id: str = "") -> list[dict]:
        """Return jobs visible to *owner_id*.

        When *owner_id* is non-empty, only jobs with a matching or empty
        owner_id are returned, preventing cross-tenant enumeration (#424).
        """
        with self._lock:
            if owner_id:
                jobs = [j for j in self._jobs.values() if not j.owner_id or j.owner_id == owner_id]
            else:
                jobs = list(self._jobs.values())
        return [{**j.to_dict(), "next_run_human": j.next_run_human()} for j in jobs]

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(_CHECK_INTERVAL)

    def _tick(self) -> None:
        now = time.time()
        with self._lock:
            due = [j for j in self._jobs.values() if j.next_run is not None and now >= j.next_run]

        if not due:
            return

        for job in due:
            self._fire(job, now)
        self._save()

    def _fire(self, job: CronJob, now: float) -> None:
        log.info("Cron job %s firing (schedule=%r run=%d)", job.id, job.schedule, job.run_count + 1)

        # Compute next_run before calling the LLM so we don't lose the schedule on error
        try:
            next_run: float | None = (
                _croniter(job.schedule, now).get_next(float) if _croniter is not None else None
            )
        except Exception:
            next_run = None

        with self._lock:
            if job.id not in self._jobs:
                return  # removed while we were about to fire
            job.last_run = now
            job.next_run = next_run
            job.run_count += 1

        self._invoke_llm(job)

    def _invoke_llm(self, job: CronJob) -> None:
        try:
            if job.context == "inherit" and _job_runner is not None:
                content = _job_runner(job) or "[no output]"
            else:
                factory = _llm_factory
                if factory is None:
                    log.warning(
                        "Cron job %s fired but no LLM factory is configured; "
                        "call configure_cron(llm_factory=...) at startup.",
                        job.id,
                    )
                    return
                if _HumanMessage is None:
                    log.warning(
                        "Cron job %s fired but langchain_core is not installed; "
                        "cannot invoke LLM without HumanMessage.",
                        job.id,
                    )
                    return
                llm = factory()
                if llm is None:
                    log.warning(
                        "Cron job %s fired but LLM factory returned None — "
                        "LLM not yet initialised; job will retry on next schedule.",
                        job.id,
                    )
                    _print_cron_output(job.name, job.prompt, "[LLM not ready — will retry]")
                    return
                if job.context == "inherit":
                    log.warning(
                        "Cron job %s requested inherited context but no runner is configured; "
                        "falling back to direct LLM invocation.",
                        job.id,
                    )
                # Mirror graph.py pattern — ThreadPoolExecutor with timeout so a
                # hung provider does not block the scheduler thread (#488).
                import concurrent.futures as _cf

                _pool = _cf.ThreadPoolExecutor(max_workers=1)
                _fut = _pool.submit(llm.invoke, [_HumanMessage(content=job.prompt)])
                _pool.shutdown(wait=False)
                try:
                    response = _fut.result(timeout=_cron_llm_timeout_seconds)
                except _cf.TimeoutError:
                    log.warning(
                        "Cron job %s LLM call timed out after %.0f seconds; "
                        "job will retry on the next schedule.",
                        job.id,
                        _cron_llm_timeout_seconds,
                    )
                    _print_cron_output(
                        job.name,
                        job.prompt,
                        f"[LLM timeout after {_cron_llm_timeout_seconds:g}s — will retry]",
                    )
                    return
                content = getattr(response, "content", str(response))
            _print_cron_output(job.name, job.prompt, content)
        except Exception as exc:
            log.warning("Cron job %s LLM call failed: %s", job.id, exc)
            _print_cron_output(job.name, job.prompt, f"[LLM error: {exc}]")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        path = self._data_dir / "jobs.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            with self._lock:
                for d in raw.get("jobs", []):
                    try:
                        job = CronJob.from_dict(d)
                        self._jobs[job.id] = job
                    except (KeyError, ValueError) as exc:
                        log.warning("Skipping malformed cron job entry: %s", exc)
            log.debug("Loaded %d cron job(s) from %s", len(self._jobs), path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            log.warning("Failed to load cron jobs from %s: %s", path, exc)

    def _save(self) -> None:
        from src.utils.atomic_write import atomic_write_json

        path = self._data_dir / "jobs.json"
        with self._lock:
            data = {"jobs": [j.to_dict() for j in self._jobs.values()]}
        try:
            with atomic_write_json(path) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            log.warning("Failed to save cron jobs to %s: %s", path, exc)


# ── Console output ────────────────────────────────────────────────────────────


def _print_cron_output(job_name: str, prompt: str, response: str) -> None:
    """Log a cron job result with a visible prefix.

    Cron output is intentionally log-only so background job activity stays out
    of the interactive terminal while still being captured in the log file.
    """
    separator = "─" * 60
    log.info(
        "\n%s\n" "[CRON] %s\n" "Prompt: %s\n" "%s\n" "%s\n" "%s\n",
        separator,
        job_name,
        prompt,
        separator,
        response,
        separator,
    )


# ── Input schemas ─────────────────────────────────────────────────────────────


class CronAddInput(BaseModel):
    schedule: str = Field(
        ...,
        description=(
            "Cron expression / pattern (5 or 6 fields). "
            "5-field: 'min hr dom mon dow' — e.g. '0 9 * * 1-5' fires at 09:00 on weekdays. "
            "6-field: 'sec min hr dom mon dow' — e.g. '0 0 9 * * 1-5' is the same. "
            "Use '*' for any value, ',' for lists, '-' for ranges, '/' for step. "
            "Also accepted as 'pattern' or 'expression'."
        ),
    )
    prompt: str = Field(
        ...,
        description="The prompt sent to the LLM when the schedule fires.",
    )
    name: str = Field(
        default="",
        description="Optional human-readable label for the job. Defaults to the cron expression.",
    )
    context: str = Field(
        default="fresh",
        description=(
            "Execution context for the scheduled job. "
            "'fresh' starts isolated; 'inherit' reuses the current session state when available."
        ),
    )


class CronListInput(BaseModel):
    pass


class CronRemoveInput(BaseModel):
    job_id: str = Field(
        ...,
        description="The 8-character job ID returned by cron_add or shown in cron_list.",
    )


# ── Tool functions ────────────────────────────────────────────────────────────


def cron_add(schedule: str, prompt: str, name: str = "", context: str = "fresh") -> str:
    """Add a recurring LLM prompt on a cron schedule."""
    owner_id = get_cron_session_id()
    try:
        job, is_new = _get_scheduler().add(
            schedule, prompt, name, context=context, owner_id=owner_id
        )
    except (ValueError, RuntimeError) as exc:
        return f"Error: {exc}"
    action = "scheduled" if is_new else "already exists"
    return (
        f"Cron job {action}.\n"
        f"  ID:       {job.id}\n"
        f"  Name:     {job.name}\n"
        f"  Schedule: {job.schedule}\n"
        f"  Context:  {job.context}\n"
        f"  Next run: {job.next_run_human()}"
    )


def cron_list() -> str:
    """List scheduled cron jobs visible to the current session."""
    jobs = _get_scheduler().list_jobs(owner_id=get_cron_session_id())
    if not jobs:
        return "No cron jobs are scheduled."
    lines: list[str] = []
    for j in jobs:
        prompt_preview = j["prompt"][:80] + ("…" if len(j["prompt"]) > 80 else "")
        lines.append(
            f"[{j['id']}] {j['name']!r}\n"
            f"  Schedule : {j['schedule']}\n"
            f"  Context  : {j.get('context', 'fresh')}\n"
            f"  Prompt   : {prompt_preview}\n"
            f"  Next run : {j['next_run_human']}\n"
            f"  Run count: {j['run_count']}"
        )
    return "\n\n".join(lines)


def cron_remove(job_id: str) -> str:
    """Remove a scheduled cron job by ID, name, or schedule expression.

    Accepts:
    - The 8-char job ID returned by cron_add.
    - The job name (case-insensitive substring match).
    - The cron schedule expression (exact match).

    Only jobs belonging to the current session can be removed (#424).
    If multiple jobs share the same name, all matching jobs are removed.
    """
    owner_id = get_cron_session_id()
    scheduler = _get_scheduler()
    jobs = scheduler.list_jobs(owner_id=owner_id)

    # Exact ID match first
    if any(j["id"] == job_id for j in jobs):
        try:
            scheduler.remove(job_id, owner_id=owner_id)
        except (KeyError, PermissionError) as exc:
            return f"Error: {exc}"
        return f"Cron job {job_id!r} removed."

    # Fallback: match by name (case-insensitive) or schedule
    needle = job_id.lower()
    matches = [
        j for j in jobs if needle in j.get("name", "").lower() or j.get("schedule") == job_id
    ]
    if not matches:
        return (
            f"No cron job found with ID, name, or schedule {job_id!r}. "
            "Use cron_list to see all active jobs and their IDs."
        )
    removed = []
    for j in matches:
        try:
            scheduler.remove(j["id"], owner_id=owner_id)
            removed.append(f"{j['id']} ({j.get('name', j.get('schedule'))})")
        except (KeyError, PermissionError):
            pass
    return f"Removed {len(removed)} cron job(s): {', '.join(removed)}."


# ── Tool registry entries ─────────────────────────────────────────────────────

TOOL_CONFIGS = [
    {
        "name": "cron_add",
        "description": (
            "Schedule a recurring prompt to be sent to the LLM at a defined time using a "
            "cron expression. The prompt is sent automatically in the background at each "
            "scheduled interval and the response is printed to the console. "
            "Use context='inherit' to run with the current session history and tools when "
            "the host process provides an inherited-context runner. "
            "Examples:\n"
            "  '0 9 * * 1-5'   — every weekday at 09:00\n"
            "  '*/30 * * * *'  — every 30 minutes\n"
            "  '0 0 1 * *'     — first day of each month at midnight"
        ),
        "input_schema": CronAddInput,
        "requires_confirmation": True,
        "function": cron_add,
    },
    {
        "name": "cron_list",
        "description": (
            "List all scheduled cron jobs with their ID, schedule expression, next run time, "
            "prompt preview, and run count."
        ),
        "input_schema": CronListInput,
        "requires_confirmation": False,
        "function": cron_list,
    },
    {
        "name": "cron_remove",
        "description": (
            "Remove a scheduled cron job. Accepts the job ID (from cron_list or cron_add), "
            "the job name (case-insensitive substring), or the cron schedule expression. "
            "If you do not remember the ID, use cron_list first or pass the name you gave it."
        ),
        "input_schema": CronRemoveInput,
        "requires_confirmation": True,
        "function": cron_remove,
    },
]

TOOL_CONFIG = TOOL_CONFIGS[0]

__all__ = [
    "CronJob",
    "CronScheduler",
    "configure_cron",
    "cron_add",
    "cron_list",
    "cron_remove",
    "CronAddInput",
    "CronListInput",
    "CronRemoveInput",
    "TOOL_CONFIG",
    "TOOL_CONFIGS",
]
