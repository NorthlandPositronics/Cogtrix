"""Tests for src/api/schemas/tool.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas.tool import (
    ToolActionRequest,
    ToolOut,
    ToolParameterSchema,
    ToolSummary,
)


class TestToolParameterSchema:
    """ToolParameterSchema construction and validation."""

    def test_tool_parameter_schema_valid(self) -> None:
        """Valid parameter constructs without error."""
        param = ToolParameterSchema(
            name="query",
            type="string",
            description="Search query.",
            required=True,
            default=None,
            enum=None,
        )
        assert param.name == "query"
        assert param.required is True
        assert param.default is None

    def test_tool_parameter_schema_with_default(self) -> None:
        """Parameter with default value."""
        param = ToolParameterSchema(
            name="limit",
            type="integer",
            description="Result limit.",
            required=False,
            default=10,
        )
        assert param.default == 10

    def test_tool_parameter_schema_with_enum(self) -> None:
        """Parameter with enum values."""
        param = ToolParameterSchema(
            name="format",
            type="string",
            description="Output format.",
            required=True,
            enum=["json", "xml", "csv"],
        )
        assert param.enum == ["json", "xml", "csv"]

    def test_tool_parameter_schema_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            ToolParameterSchema(
                name="query",
                type="string",
                description="Search query.",
                # required missing
            )


class TestToolOut:
    """ToolOut construction and validation."""

    def test_tool_out_valid(self) -> None:
        """Valid tool constructs without error."""
        tool = ToolOut(
            name="web_search",
            description="Search the web using the configured search provider.",
            short_description="Search the web.",
            status="on_demand",
            requires_confirmation=False,
            parameters=[
                ToolParameterSchema(
                    name="query",
                    type="string",
                    description="Search query.",
                    required=True,
                )
            ],
            module="web_search",
            is_mcp=False,
        )
        assert tool.name == "web_search"
        assert tool.status == "on_demand"
        assert tool.is_mcp is False
        assert tool.mcp_server is None

    def test_tool_out_mcp_tool(self) -> None:
        """MCP tool with server reference."""
        tool = ToolOut(
            name="read_file",
            description="Read a file.",
            short_description="Read file.",
            status="active",
            requires_confirmation=True,
            is_mcp=True,
            mcp_server="filesystem",
        )
        assert tool.is_mcp is True
        assert tool.mcp_server == "filesystem"

    def test_tool_out_invalid_status(self) -> None:
        """Invalid status raises ValidationError."""
        with pytest.raises(ValidationError):
            ToolOut(
                name="test",
                description="Test tool.",
                short_description="Test.",
                status="invalid",
                requires_confirmation=False,
            )

    def test_tool_out_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            ToolOut(
                name="test",
                description="Test tool.",
                short_description="Test.",
                status="active",
                # requires_confirmation missing
            )

    def test_tool_out_empty_parameters(self) -> None:
        """Empty parameters list is valid."""
        tool = ToolOut(
            name="noop",
            description="No-op tool.",
            short_description="No-op.",
            status="disabled",
            requires_confirmation=False,
            parameters=[],
        )
        assert tool.parameters == []

    def test_tool_out_all_statuses(self) -> None:
        """All valid statuses construct without error."""
        for status in ["active", "on_demand", "disabled", "auto_approved", "pinned"]:
            tool = ToolOut(
                name="test",
                description="Test.",
                short_description="Test.",
                status=status,
                requires_confirmation=False,
            )
            assert tool.status == status


class TestToolSummary:
    """ToolSummary construction and validation."""

    def test_tool_summary_valid(self) -> None:
        """Valid summary constructs without error."""
        summary = ToolSummary(
            name="web_search",
            short_description="Search the web.",
            status="on_demand",
            requires_confirmation=False,
        )
        assert summary.name == "web_search"
        assert summary.is_mcp is False

    def test_tool_summary_mcp(self) -> None:
        """MCP tool summary."""
        summary = ToolSummary(
            name="read_file",
            short_description="Read file.",
            status="active",
            requires_confirmation=True,
            is_mcp=True,
        )
        assert summary.is_mcp is True

    def test_tool_summary_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            ToolSummary(
                name="web_search",
                short_description="Search the web.",
                status="on_demand",
                # requires_confirmation missing
            )


class TestToolActionRequest:
    """ToolActionRequest construction and validation."""

    def test_tool_action_request_load(self) -> None:
        """Load action constructs without error."""
        req = ToolActionRequest(load=["web_search", "read_file"])
        assert req.load == ["web_search", "read_file"]
        assert req.unload is None

    def test_tool_action_request_unload(self) -> None:
        """Unload action constructs without error."""
        req = ToolActionRequest(unload=["read_file"])
        assert req.unload == ["read_file"]

    def test_tool_action_request_enable(self) -> None:
        """Enable action constructs without error."""
        req = ToolActionRequest(enable=["shell"])
        assert req.enable == ["shell"]

    def test_action_request_disable(self) -> None:
        """Disable action constructs without error."""
        req = ToolActionRequest(disable=["shell"])
        assert req.disable == ["shell"]

    def test_tool_action_request_auto_approve(self) -> None:
        """Auto-approve action constructs without error."""
        req = ToolActionRequest(auto_approve=["web_search"])
        assert req.auto_approve == ["web_search"]

    def test_tool_action_request_revoke_approval(self) -> None:
        """Revoke approval action constructs without error."""
        req = ToolActionRequest(revoke_approval=["web_search"])
        assert req.revoke_approval == ["web_search"]

    def test_tool_action_request_empty(self) -> None:
        """All-null request is valid (no action specified)."""
        req = ToolActionRequest()
        assert req.load is None
        assert req.unload is None
        assert req.enable is None
        assert req.disable is None
        assert req.auto_approve is None
        assert req.revoke_approval is None

    def test_tool_action_request_multiple_actions(self) -> None:
        """Multiple actions can be set simultaneously."""
        req = ToolActionRequest(
            load=["web_search"],
            unload=["read_file"],
            disable=["shell"],
        )
        assert req.load == ["web_search"]
        assert req.unload == ["read_file"]
        assert req.disable == ["shell"]
