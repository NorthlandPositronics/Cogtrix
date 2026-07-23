"""Workflow management schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WorkflowToolPolicyOut(BaseModel):
    excluded_tools: list[str] = Field(default_factory=list)
    additional_approved_tools: list[str] = Field(default_factory=list)


class WorkflowAutoDetectOut(BaseModel):
    enabled: bool = False
    keywords: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    min_confidence: int = 1


class WorkflowOut(BaseModel):
    id: str = Field(..., description="Workflow identifier (^[a-zA-Z0-9][a-zA-Z0-9_-]*$).")
    name: str = Field(..., description="Human-readable workflow name.")
    description: str = Field(default="", description="Optional description.")
    system_prompt: str | None = Field(default=None, description="Inline system prompt override.")
    system_prompt_file: str | None = Field(
        default=None, description="Path to a system prompt file."
    )
    knowledge_base: bool = Field(
        default=False, description="Whether a per-workflow FAISS KB is enabled."
    )
    tool_policy: WorkflowToolPolicyOut = Field(default_factory=WorkflowToolPolicyOut)
    auto_detect: WorkflowAutoDetectOut = Field(default_factory=WorkflowAutoDetectOut)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")


class WorkflowCreate(BaseModel):
    id: str = Field(..., description="Workflow identifier (^[a-zA-Z0-9][a-zA-Z0-9_-]*$).")
    name: str = Field(..., description="Human-readable workflow name.")
    description: str = Field(default="")
    system_prompt: str | None = Field(default=None)
    system_prompt_file: str | None = Field(default=None)
    knowledge_base: bool = Field(default=False)
    tool_policy: WorkflowToolPolicyOut = Field(default_factory=WorkflowToolPolicyOut)
    auto_detect: WorkflowAutoDetectOut = Field(default_factory=WorkflowAutoDetectOut)


class WorkflowUpdate(BaseModel):
    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    system_prompt: str | None = Field(default=None)
    system_prompt_file: str | None = Field(default=None)
    knowledge_base: bool | None = Field(default=None)
    tool_policy: WorkflowToolPolicyOut | None = Field(default=None)
    auto_detect: WorkflowAutoDetectOut | None = Field(default=None)


class WorkflowBindingOut(BaseModel):
    session_key: str = Field(..., description="Chat session key (channel::chat_id).")
    workflow_id: str = Field(..., description="Bound workflow identifier.")
    assigned_at: str = Field(default="")
    assigned_by: str | None = Field(default=None)


class WorkflowDocumentOut(BaseModel):
    """A document stored in a workflow's knowledge base."""

    doc_id: str = Field(..., description="Unique document identifier (UUID).")
    filename: str = Field(..., description="Original uploaded filename.")
    size_bytes: int = Field(..., description="File size in bytes.")
    content_type: str | None = Field(
        default=None,
        description="MIME type inferred from filename; null if unknown.",
    )
    status: str | None = Field(
        default=None,
        description="Upload status; present on upload response only (e.g. 'saved').",
    )


class BindWorkflowRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow identifier to bind.")
