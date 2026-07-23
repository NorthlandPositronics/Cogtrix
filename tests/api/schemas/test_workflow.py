"""Tests for src/api/schemas/workflow.py — workflow + KB document schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas.workflow import (
    BindWorkflowRequest,
    WorkflowAutoDetectOut,
    WorkflowBindingOut,
    WorkflowCreate,
    WorkflowDocumentOut,
    WorkflowOut,
    WorkflowToolPolicyOut,
    WorkflowUpdate,
)

# ---------------------------------------------------------------------------
# WorkflowToolPolicyOut — both lists default to []
# ---------------------------------------------------------------------------


class TestWorkflowToolPolicyOut:
    def test_defaults(self) -> None:
        p = WorkflowToolPolicyOut()
        assert p.excluded_tools == []
        assert p.additional_approved_tools == []

    def test_populated(self) -> None:
        p = WorkflowToolPolicyOut(
            excluded_tools=["shell"],
            additional_approved_tools=["custom_search"],
        )
        assert p.excluded_tools == ["shell"]
        assert p.additional_approved_tools == ["custom_search"]


# ---------------------------------------------------------------------------
# WorkflowAutoDetectOut — disabled by default, configurable min_confidence
# ---------------------------------------------------------------------------


class TestWorkflowAutoDetectOut:
    def test_defaults(self) -> None:
        a = WorkflowAutoDetectOut()
        assert a.enabled is False
        assert a.keywords == []
        assert a.patterns == []
        assert a.min_confidence == 1

    def test_enabled_with_keywords(self) -> None:
        a = WorkflowAutoDetectOut(
            enabled=True, keywords=["bug", "issue"], patterns=[r"\bfix\b"], min_confidence=3
        )
        assert a.enabled is True
        assert a.min_confidence == 3


# ---------------------------------------------------------------------------
# WorkflowOut — nested defaults, optional inline / file system_prompt
# ---------------------------------------------------------------------------


class TestWorkflowOut:
    def test_valid_minimal(self) -> None:
        w = WorkflowOut(id="support", name="Customer Support")
        assert w.description == ""
        assert w.system_prompt is None
        assert w.system_prompt_file is None
        assert w.knowledge_base is False
        assert isinstance(w.tool_policy, WorkflowToolPolicyOut)
        assert isinstance(w.auto_detect, WorkflowAutoDetectOut)
        # default_factory yields independent instances.
        assert w.tool_policy.excluded_tools == []

    def test_valid_full(self) -> None:
        w = WorkflowOut(
            id="support",
            name="Customer Support",
            description="Tier-1 issue triage",
            system_prompt="You are a support agent.",
            knowledge_base=True,
            tool_policy=WorkflowToolPolicyOut(excluded_tools=["execute_shell_command"]),
            auto_detect=WorkflowAutoDetectOut(enabled=True, keywords=["ticket"]),
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-02T00:00:00Z",
        )
        assert w.knowledge_base is True
        assert w.tool_policy.excluded_tools == ["execute_shell_command"]
        assert w.auto_detect.enabled is True

    def test_inline_and_file_prompts_can_coexist(self) -> None:
        """Schema doesn't enforce mutual exclusion (resolution is route-side)."""
        w = WorkflowOut(
            id="x",
            name="x",
            system_prompt="inline",
            system_prompt_file="/etc/cogtrix/sp.md",
        )
        assert w.system_prompt == "inline"
        assert w.system_prompt_file == "/etc/cogtrix/sp.md"

    def test_missing_required_id(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowOut(name="x")  # type: ignore[call-arg]

    def test_missing_required_name(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowOut(id="x")  # type: ignore[call-arg]

    def test_nested_default_factory_independence(self) -> None:
        """Two separately-constructed defaults must not share state."""
        a = WorkflowOut(id="a", name="a")
        b = WorkflowOut(id="b", name="b")
        a.tool_policy.excluded_tools.append("shell")
        assert b.tool_policy.excluded_tools == []


# ---------------------------------------------------------------------------
# WorkflowCreate — same shape as Out without timestamps
# ---------------------------------------------------------------------------


class TestWorkflowCreate:
    def test_valid_minimal(self) -> None:
        c = WorkflowCreate(id="x", name="X")
        assert c.description == ""
        assert c.knowledge_base is False

    def test_missing_required_id(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowCreate(name="x")  # type: ignore[call-arg]

    def test_missing_required_name(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowCreate(id="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# WorkflowUpdate — all-optional
# ---------------------------------------------------------------------------


class TestWorkflowUpdate:
    def test_empty(self) -> None:
        u = WorkflowUpdate()
        assert u.name is None
        assert u.description is None
        assert u.system_prompt is None
        assert u.system_prompt_file is None
        assert u.knowledge_base is None
        assert u.tool_policy is None
        assert u.auto_detect is None

    def test_partial(self) -> None:
        u = WorkflowUpdate(
            name="New name",
            knowledge_base=True,
            tool_policy=WorkflowToolPolicyOut(excluded_tools=["shell"]),
        )
        assert u.name == "New name"
        assert u.knowledge_base is True
        assert u.tool_policy is not None
        assert u.tool_policy.excluded_tools == ["shell"]


# ---------------------------------------------------------------------------
# WorkflowBindingOut — required session_key + workflow_id
# ---------------------------------------------------------------------------


class TestWorkflowBindingOut:
    def test_valid_minimal(self) -> None:
        b = WorkflowBindingOut(session_key="telegram::123", workflow_id="support")
        assert b.assigned_at == ""
        assert b.assigned_by is None

    def test_valid_with_audit_fields(self) -> None:
        b = WorkflowBindingOut(
            session_key="x",
            workflow_id="y",
            assigned_at="2026-01-01T00:00:00Z",
            assigned_by="admin",
        )
        assert b.assigned_by == "admin"

    def test_missing_required_session_key(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowBindingOut(workflow_id="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# WorkflowDocumentOut
# ---------------------------------------------------------------------------


class TestWorkflowDocumentOut:
    def test_valid_minimal(self) -> None:
        d = WorkflowDocumentOut(doc_id="d1", filename="manual.pdf", size_bytes=12345)
        assert d.content_type is None
        assert d.status is None

    def test_valid_upload_response(self) -> None:
        d = WorkflowDocumentOut(
            doc_id="d1",
            filename="manual.pdf",
            size_bytes=12345,
            content_type="application/pdf",
            status="saved",
        )
        assert d.content_type == "application/pdf"
        assert d.status == "saved"

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowDocumentOut(filename="x", size_bytes=1)  # type: ignore[call-arg]

    def test_size_bytes_can_be_zero(self) -> None:
        d = WorkflowDocumentOut(doc_id="d", filename="empty.txt", size_bytes=0)
        assert d.size_bytes == 0


# ---------------------------------------------------------------------------
# BindWorkflowRequest
# ---------------------------------------------------------------------------


class TestBindWorkflowRequest:
    def test_valid(self) -> None:
        b = BindWorkflowRequest(workflow_id="support")
        assert b.workflow_id == "support"

    def test_missing_workflow_id(self) -> None:
        with pytest.raises(ValidationError):
            BindWorkflowRequest()  # type: ignore[call-arg]
