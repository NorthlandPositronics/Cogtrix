"""Background task queue endpoints.

Endpoints:
    POST   /api/v1/tasks              — submit a new task
    GET    /api/v1/tasks              — list tasks (optional ?status= filter)
    GET    /api/v1/tasks/{task_id}    — get a single task by ID
    DELETE /api/v1/tasks/{task_id}    — cancel a pending task
    GET    /api/v1/tasks/{task_id}/log — stream the task log file
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse

from cogtrix_core.api.auth import TokenData, get_current_user
from cogtrix_core.api.org_context import OrgContext, assert_same_org, require_org_context
from cogtrix_core.api.schemas.common import APIResponse
from cogtrix_core.api.schemas.task import TaskCreateRequest, TaskOut

log = logging.getLogger("cogtrix.api.tasks")

router = APIRouter(prefix="/tasks", tags=["Tasks"])


def _task_to_out(task: object) -> TaskOut:
    return TaskOut(
        task_id=str(getattr(task, "task_id", "")),
        agent_name=str(getattr(task, "agent_name", "")),
        prompt=str(getattr(task, "prompt", "")),
        status=str(getattr(task, "status", "")),
        created_at=float(getattr(task, "created_at", 0.0)),
        started_at=float(v) if (v := getattr(task, "started_at", None)) is not None else None,
        finished_at=float(v) if (v := getattr(task, "finished_at", None)) is not None else None,
        result=str(getattr(task, "result", "")),
        error=str(getattr(task, "error", "")),
        log_path=str(getattr(task, "log_path", "")),
        user_id=str(getattr(task, "user_id", "")),
        org_id=getattr(task, "org_id", None),
    )


def _assert_task_ownership(record: object, current_user: TokenData, ctx: OrgContext) -> None:
    """Raise 403 FORBIDDEN if the task does not belong to the current user or org.

    Deny-by-default (#2197): previously an empty/legacy task ``user_id`` (agent-
    and CLI-spawned background tasks default ``user_id=''``) made the per-user
    check ``if task_user_id and ...`` short-circuit, so any *other* authenticated
    user could read / cancel / inspect those tasks — an IDOR (cross-user
    disclosure via ``GET``/``DELETE /tasks/{id}`` and the log endpoint). A task
    is now accessible only to its owner or an admin; unowned/legacy tasks (empty
    ``user_id``) are admin-only. (Propagating ``user_id`` onto agent/CLI-spawned
    tasks so their owner can retrieve them is tracked as the #2197 follow-up.)
    """
    task_user_id = getattr(record, "user_id", "")
    if task_user_id != current_user.user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "TASK_ACCESS_DENIED",
                "message": "You do not have permission to access this task.",
            },
        )
    assert_same_org(ctx, getattr(record, "org_id", None))


def _get_queue():
    """Return the module-level TaskQueue; raise 503 if not initialised."""
    try:
        from cogtrix_core.tasks.queue import get_task_queue

        return get_task_queue()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "TASK_QUEUE_UNAVAILABLE",
                "message": "Task queue has not been initialised.",
            },
        ) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "TASK_QUEUE_UNAVAILABLE",
                "message": "Task queue module is not available.",
            },
        ) from exc


@router.post(
    "",
    summary="Submit a background task",
    description="Submit a new task to the background queue for the named agent.",
    response_model=APIResponse[TaskOut],
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Task accepted and queued."},
        401: {"description": "Not authenticated."},
        403: {"description": "User not assigned to an organization (ORG_REQUIRED)."},
        503: {"description": "Task queue unavailable (TASK_QUEUE_UNAVAILABLE)."},
    },
)
async def create_task(
    body: TaskCreateRequest,
    current_user: TokenData = Depends(get_current_user),
    ctx: OrgContext = Depends(require_org_context),
) -> APIResponse[TaskOut]:
    """Submit a task to the background queue.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, TASK_QUEUE_UNAVAILABLE.
    """
    queue = _get_queue()
    task_id = queue.submit(
        body.agent_name, body.prompt, user_id=current_user.user_id, org_id=ctx.org_id
    )
    record = queue.get(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Task was submitted but could not be retrieved.",
            },
        )
    return APIResponse(data=_task_to_out(record))


@router.get(
    "",
    summary="List tasks",
    description="Return up to 50 recent tasks, optionally filtered by status.",
    response_model=APIResponse[list[TaskOut]],
    responses={
        200: {"description": "Task list returned."},
        401: {"description": "Not authenticated."},
        403: {"description": "User not assigned to an organization (ORG_REQUIRED)."},
        503: {"description": "Task queue unavailable (TASK_QUEUE_UNAVAILABLE)."},
    },
)
async def list_tasks(
    task_status: str | None = Query(default=None, alias="status", description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: TokenData = Depends(get_current_user),
    ctx: OrgContext = Depends(require_org_context),
) -> APIResponse[list[TaskOut]]:
    """List recent tasks.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, TASK_QUEUE_UNAVAILABLE, INVALID_STATUS.
    """
    from cogtrix_core.tasks.queue import TaskStatus

    queue = _get_queue()
    status_filter = None
    if task_status is not None:
        try:
            status_filter = TaskStatus(task_status.upper())
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_STATUS",
                    "message": f"Unknown status {task_status!r}. Valid values: {[s.value for s in TaskStatus]}",
                },
            ) from exc
    tasks = queue.list(
        status=status_filter, limit=limit, user_id=current_user.user_id, org_id=ctx.org_id
    )
    return APIResponse(data=[_task_to_out(t) for t in tasks])


@router.get(
    "/{task_id}",
    summary="Get task by ID",
    description="Return the full record for a single task.",
    response_model=APIResponse[TaskOut],
    responses={
        200: {"description": "Task returned."},
        401: {"description": "Not authenticated."},
        403: {
            "description": "Access denied — wrong org or user (TASK_ACCESS_DENIED / CROSS_ORG_ACCESS)."
        },
        404: {"description": "Task not found (TASK_NOT_FOUND)."},
        503: {"description": "Task queue unavailable (TASK_QUEUE_UNAVAILABLE)."},
    },
)
async def get_task(
    task_id: str,
    current_user: TokenData = Depends(get_current_user),
    ctx: OrgContext = Depends(require_org_context),
) -> APIResponse[TaskOut]:
    """Return a single task by ID.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, TASK_NOT_FOUND, TASK_QUEUE_UNAVAILABLE.
    """
    queue = _get_queue()
    record = queue.get(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TASK_NOT_FOUND", "message": "No task with that ID exists."},
        )
    _assert_task_ownership(record, current_user, ctx)
    return APIResponse(data=_task_to_out(record))


@router.delete(
    "/{task_id}",
    summary="Cancel a pending task",
    description="Cancel a task that is still in PENDING state. Running, completed, or failed tasks cannot be cancelled.",
    response_model=APIResponse[None],
    responses={
        200: {"description": "Task cancelled."},
        401: {"description": "Not authenticated."},
        403: {
            "description": "Access denied — wrong org or user (TASK_ACCESS_DENIED / CROSS_ORG_ACCESS)."
        },
        404: {"description": "Task not found (TASK_NOT_FOUND)."},
        409: {"description": "Task is not in a cancellable state (TASK_NOT_CANCELLABLE)."},
        503: {"description": "Task queue unavailable (TASK_QUEUE_UNAVAILABLE)."},
    },
)
async def cancel_task(
    task_id: str,
    current_user: TokenData = Depends(get_current_user),
    ctx: OrgContext = Depends(require_org_context),
) -> APIResponse[None]:
    """Cancel a pending task.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, TASK_NOT_FOUND, TASK_NOT_CANCELLABLE, TASK_QUEUE_UNAVAILABLE.
    """
    queue = _get_queue()
    # Check existence first for a meaningful 404
    record = queue.get(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TASK_NOT_FOUND", "message": "No task with that ID exists."},
        )
    _assert_task_ownership(record, current_user, ctx)
    cancelled = queue.cancel(task_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "TASK_NOT_CANCELLABLE",
                "message": "Task is not in PENDING state and cannot be cancelled.",
            },
        )
    return APIResponse(data=None)


@router.get(
    "/{task_id}/log",
    summary="Get task log",
    description="Return the raw text log for a task. Returns an empty string if no log exists yet.",
    response_class=PlainTextResponse,
    responses={
        200: {"description": "Log content returned (plain text)."},
        401: {"description": "Not authenticated."},
        404: {"description": "Task not found (TASK_NOT_FOUND)."},
        503: {"description": "Task queue unavailable (TASK_QUEUE_UNAVAILABLE)."},
    },
)
async def get_task_log(
    task_id: str,
    current_user: TokenData = Depends(get_current_user),
    ctx: OrgContext = Depends(require_org_context),
) -> PlainTextResponse:
    """Return the plain-text log for a task.

    Auth: bearer token required.
    Error codes: UNAUTHORIZED, TOKEN_EXPIRED, TASK_NOT_FOUND, TASK_QUEUE_UNAVAILABLE.
    """
    import pathlib

    queue = _get_queue()
    record = queue.get(task_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TASK_NOT_FOUND", "message": "No task with that ID exists."},
        )
    _assert_task_ownership(record, current_user, ctx)
    log_path = pathlib.Path(record.log_path).resolve()
    try:
        queue._log_dir.resolve()
        log_path.relative_to(queue._log_dir.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "Task log path is invalid."},
        ) from exc
    if not log_path.exists():
        return PlainTextResponse("")
    return PlainTextResponse(log_path.read_text(encoding="utf-8", errors="replace"))
