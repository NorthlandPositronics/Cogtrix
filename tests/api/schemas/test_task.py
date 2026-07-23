"""Tests for src/api/schemas/task.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas.task import TaskCreateRequest, TaskOut


class TestTaskCreateRequest:
    """TaskCreateRequest schema construction and validation."""

    def test_task_create_request_valid(self) -> None:
        """Valid input constructs without error."""
        req = TaskCreateRequest(agent_name="researcher", prompt="Summarize the report.")
        assert req.agent_name == "researcher"
        assert req.prompt == "Summarize the report."

    def test_task_create_request_empty_agent_name(self) -> None:
        """Empty agent_name raises ValidationError."""
        with pytest.raises(ValidationError):
            TaskCreateRequest(agent_name="", prompt="Summarize the report.")

    def test_task_create_request_empty_prompt(self) -> None:
        """Empty prompt raises ValidationError."""
        with pytest.raises(ValidationError):
            TaskCreateRequest(agent_name="researcher", prompt="")

    def test_task_create_request_agent_name_too_long(self) -> None:
        """Agent name over 128 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            TaskCreateRequest(agent_name="x" * 129, prompt="Valid prompt.")

    def test_task_create_request_prompt_too_long(self) -> None:
        """Prompt over 8192 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            TaskCreateRequest(agent_name="researcher", prompt="x" * 8193)

    def test_task_create_request_boundary_values(self) -> None:
        """Boundary values at exactly max length pass."""
        req = TaskCreateRequest(agent_name="x" * 128, prompt="x" * 8192)
        assert req.agent_name == "x" * 128
        assert req.prompt == "x" * 8192


class TestTaskOut:
    """TaskOut schema construction and validation."""

    def test_task_out_valid(self) -> None:
        """Valid input constructs without error."""
        task = TaskOut(
            task_id="task-123",
            agent_name="researcher",
            prompt="Summarize the report.",
            status="pending",
            created_at=1234567890.0,
            started_at=None,
            finished_at=None,
            result="",
            error="",
            log_path="/tmp/task-123.log",
        )
        assert task.task_id == "task-123"
        assert task.status == "pending"
        assert task.started_at is None

    def test_task_out_with_optional_ids(self) -> None:
        """Optional user_id and org_id can be set."""
        task = TaskOut(
            task_id="task-123",
            agent_name="researcher",
            prompt="Summarize the report.",
            status="completed",
            created_at=1234567890.0,
            started_at=1234567891.0,
            finished_at=1234567900.0,
            result="Done.",
            error="",
            log_path="/tmp/task-123.log",
            user_id="user-456",
            org_id="org-789",
        )
        assert task.user_id == "user-456"
        assert task.org_id == "org-789"

    def test_task_out_defaults(self) -> None:
        """Default user_id is empty string, org_id is None."""
        task = TaskOut(
            task_id="task-123",
            agent_name="researcher",
            prompt="Test.",
            status="pending",
            created_at=0.0,
            started_at=None,
            finished_at=None,
            result="",
            error="",
            log_path="",
        )
        assert task.user_id == ""
        assert task.org_id is None

    def test_task_out_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            TaskOut(
                task_id="task-123",
                agent_name="researcher",
                prompt="Test.",
                # status missing
                created_at=0.0,
                started_at=None,
                finished_at=None,
                result="",
                error="",
                log_path="",
            )
