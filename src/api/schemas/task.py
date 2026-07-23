"""Pydantic request/response schemas for the tasks API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    """Request body for POST /api/v1/tasks."""

    agent_name: str = Field(..., min_length=1, max_length=128)
    prompt: str = Field(..., min_length=1, max_length=8192)


class TaskOut(BaseModel):
    """Serialized representation of a task queue entry."""

    task_id: str
    agent_name: str
    prompt: str
    status: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    result: str
    error: str
    log_path: str
    user_id: str = ""
    org_id: str | None = None
