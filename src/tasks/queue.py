"""SQLite-backed background task queue.

Tasks are submitted by name (agent_name) and a prompt string.  A thread-pool
worker picks them up, calls _run_agent_task(), and writes the result back to
the database.  All DB writes are serialised by a threading.Lock; reads are
lock-free.

Module-level helpers::

    init_task_queue(db_path, log_dir, max_workers)  -> TaskQueue
    get_task_queue()                                -> TaskQueue
    submit_task(agent_name, prompt)                 -> task_id
    get_task(task_id)                               -> TaskRecord | None
    list_tasks(status, limit)                       -> list[TaskRecord]
    cancel_task(task_id)                            -> bool
"""

from __future__ import annotations

import pathlib
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum

# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class TaskRecord:
    task_id: str
    agent_name: str
    prompt: str
    status: TaskStatus
    created_at: float
    started_at: float | None
    finished_at: float | None
    result: str
    error: str
    log_path: str
    session_id: str = ""
    user_id: str = ""
    org_id: str | None = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id    TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    prompt     TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    result     TEXT NOT NULL DEFAULT '',
    error      TEXT NOT NULL DEFAULT '',
    log_path   TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    user_id    TEXT NOT NULL DEFAULT '',
    org_id     TEXT
)
"""

_MIGRATE_SESSION_ID = "ALTER TABLE tasks ADD COLUMN session_id TEXT NOT NULL DEFAULT ''"
_MIGRATE_USER_ID = "ALTER TABLE tasks ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
_MIGRATE_ORG_ID = "ALTER TABLE tasks ADD COLUMN org_id TEXT"


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------


class TaskQueue:
    def __init__(
        self,
        db_path: str | pathlib.Path,
        log_dir: str | pathlib.Path,
        max_workers: int = 4,
    ) -> None:
        self._db_path = pathlib.Path(db_path)
        self._log_dir = pathlib.Path(log_dir)
        self._max_workers = max_workers
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._runner: Callable[[TaskRecord], str] | None = None

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(_DDL)
            # Migrate: add columns if missing (pre-existing DBs)
            for migration in (_MIGRATE_SESSION_ID, _MIGRATE_USER_ID, _MIGRATE_ORG_ID):
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError:
                    pass  # column already exists

    # ── Connection helpers ─────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Runner injection ───────────────────────────────────────────────────

    def set_runner(self, runner: Callable[[TaskRecord], str]) -> None:
        """Set the function that executes agent tasks."""
        self._runner = runner

    # ── Public API ─────────────────────────────────────────────────────────

    def submit(
        self,
        agent_name: str,
        prompt: str,
        session_id: str = "",
        user_id: str = "",
        org_id: str | None = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        log_path = str(self._log_dir / f"{task_id}.log")
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO tasks
                        (task_id, agent_name, prompt, status, created_at, log_path, session_id, user_id, org_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        agent_name,
                        prompt,
                        TaskStatus.PENDING.value,
                        now,
                        log_path,
                        session_id,
                        user_id,
                        org_id,
                    ),
                )
            # Snapshot the executor under the lock so a concurrent stop()
            # cannot null/shut it down between this read and the submit below
            # (#2174). Dispatch happens outside the lock to avoid holding it
            # across executor bookkeeping.
            executor = self._executor
        if executor is not None:
            executor.submit(self._worker, task_id)
        return task_id

    def get(self, task_id: str) -> TaskRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def list(
        self,
        status: TaskStatus | None = None,
        limit: int = 50,
        session_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> list[TaskRecord]:
        with self._connect() as conn:
            where_parts: list[str] = []
            params: list = []
            if status is not None:
                where_parts.append("status = ?")
                params.append(status.value)
            if session_id is not None:
                where_parts.append("session_id = ?")
                params.append(session_id)
            if user_id is not None:
                where_parts.append("user_id = ?")
                params.append(user_id)
            if org_id is not None:
                where_parts.append("org_id = ?")
                params.append(org_id)
            where = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM tasks{where} ORDER BY created_at DESC LIMIT ?",  # nosec B608
                params,
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT status FROM tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
                if row is None:
                    return False
                if row["status"] != TaskStatus.PENDING.value:
                    return False
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE task_id = ?",
                    (TaskStatus.CANCELLED.value, task_id),
                )
        return True

    def start(self) -> None:
        with self._lock:
            # Idempotence: a second start() without an intervening stop() must
            # not leak the first executor (its worker threads would never be
            # shut down) — #2159.
            if self._executor is not None:
                return
            # Finalize tasks left RUNNING by a previous process crash. start()
            # runs before any worker in this process, so a RUNNING row here is
            # always an orphan from a prior run. Agent tasks are not idempotent,
            # so mark them FAILED rather than leaving them "running" forever or
            # blindly re-running them (#2159).
            now = time.time()
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, finished_at = ?, error = ?
                    WHERE status = ?
                    """,
                    (
                        TaskStatus.FAILED.value,
                        now,
                        "Task was interrupted by a process restart while running.",
                        TaskStatus.RUNNING.value,
                    ),
                )
            executor = ThreadPoolExecutor(max_workers=self._max_workers)
            self._executor = executor
        # Re-submit any PENDING tasks that survived a previous crash. Done
        # outside the lock so concurrent submit() traffic isn't blocked; a
        # task that is also dispatched by submit() is a no-op because
        # _worker's PENDING→RUNNING transition is a check-then-act under
        # _lock (the second worker early-returns).
        pending = self.list(status=TaskStatus.PENDING, limit=1000)
        for record in pending:
            executor.submit(self._worker, record.task_id)

    def stop(self) -> None:
        # Null the executor under the lock so a concurrent submit() observes a
        # consistent value (#2174); shut it down on the local snapshot outside
        # the lock so the blocking join doesn't stall submit()/start() traffic.
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)

    # ── Internal workers ───────────────────────────────────────────────────

    def _worker(self, task_id: str) -> None:
        now = time.time()
        # Mark RUNNING — but only if still PENDING (could have been cancelled).
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT status FROM tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
                if row is None or row["status"] != TaskStatus.PENDING.value:
                    return
                conn.execute(
                    "UPDATE tasks SET status = ?, started_at = ? WHERE task_id = ?",
                    (TaskStatus.RUNNING.value, now, task_id),
                )

        record = self.get(task_id)
        if record is None:
            return

        try:
            result = self._run_agent_task(record)
            finished = time.time()
            with self._lock:
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE tasks
                        SET status = ?, finished_at = ?, result = ?
                        WHERE task_id = ?
                        """,
                        (TaskStatus.COMPLETED.value, finished, result, task_id),
                    )
            self._append_log(task_id, f"[COMPLETED] {result}")
        except Exception as exc:  # noqa: BLE001
            finished = time.time()
            error_msg = str(exc)
            with self._lock:
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE tasks
                        SET status = ?, finished_at = ?, error = ?
                        WHERE task_id = ?
                        """,
                        (TaskStatus.FAILED.value, finished, error_msg, task_id),
                    )
            self._append_log(task_id, f"[FAILED] {error_msg}")

    def _run_agent_task(self, record: TaskRecord) -> str:
        if self._runner is not None:
            return self._runner(record)
        return (
            f"[stub] No runner configured for agent '{record.agent_name}'. "
            f"Task was submitted before the agent runner was initialized."
        )

    def _append_log(self, task_id: str, line: str) -> None:
        log_path = self._log_dir / f"{task_id}.log"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            agent_name=row["agent_name"],
            prompt=row["prompt"],
            status=TaskStatus(row["status"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            result=row["result"],
            error=row["error"],
            log_path=row["log_path"],
            session_id=row["session_id"] if "session_id" in row.keys() else "",
            user_id=row["user_id"] if "user_id" in row.keys() else "",
            org_id=row["org_id"] if "org_id" in row.keys() else None,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_queue: TaskQueue | None = None


def init_task_queue(
    db_path: str | pathlib.Path,
    log_dir: str | pathlib.Path,
    max_workers: int = 4,
) -> TaskQueue:
    global _queue
    _queue = TaskQueue(db_path=db_path, log_dir=log_dir, max_workers=max_workers)
    return _queue


def get_task_queue() -> TaskQueue:
    if _queue is None:
        raise RuntimeError("Task queue has not been initialised — call init_task_queue() first.")
    return _queue


def submit_task(
    agent_name: str,
    prompt: str,
    session_id: str = "",
    user_id: str = "",
    org_id: str | None = None,
) -> str:
    return get_task_queue().submit(
        agent_name, prompt, session_id=session_id, user_id=user_id, org_id=org_id
    )


def get_task(task_id: str) -> TaskRecord | None:
    return get_task_queue().get(task_id)


def list_tasks(
    status: TaskStatus | None = None,
    limit: int = 50,
    user_id: str | None = None,
    org_id: str | None = None,
) -> list[TaskRecord]:
    return get_task_queue().list(status=status, limit=limit, user_id=user_id, org_id=org_id)


def cancel_task(task_id: str) -> bool:
    return get_task_queue().cancel(task_id)
