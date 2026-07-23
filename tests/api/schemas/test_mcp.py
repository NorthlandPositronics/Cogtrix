"""Tests for src/api/schemas/mcp.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.api.schemas.mcp import (
    MCPServerAddRequest,
    MCPServerOut,
    MCPToolSummary,
)


class TestMCPToolSummary:
    """MCPToolSummary schema construction and validation."""

    def test_mcp_tool_summary_valid(self) -> None:
        """Valid input constructs without error."""
        tool = MCPToolSummary(name="read_file", description="Read a file.")
        assert tool.name == "read_file"
        assert tool.description == "Read a file."

    def test_mcp_tool_summary_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            MCPToolSummary(name="read_file")


class TestMCPServerOut:
    """MCPServerOut schema construction and validation."""

    def test_mcp_server_out_valid_stdio(self) -> None:
        """Valid stdio server constructs without error."""
        now = datetime.now(UTC)
        server = MCPServerOut(
            name="filesystem",
            status="connected",
            transport="stdio",
            command="npx",
            args=["@modelcontextprotocol/server-filesystem", "/data"],
            requires_confirmation=True,
            tools=[MCPToolSummary(name="read_file", description="Read a file.")],
            connected_at=now,
        )
        assert server.name == "filesystem"
        assert server.transport == "stdio"
        assert server.url is None
        assert len(server.tools) == 1

    def test_mcp_server_out_valid_sse(self) -> None:
        """Valid SSE server constructs without error."""
        server = MCPServerOut(
            name="remote-server",
            status="connected",
            transport="sse",
            url="http://localhost:3000/sse",
            requires_confirmation=False,
        )
        assert server.transport == "sse"
        assert server.url == "http://localhost:3000/sse"
        assert server.command is None

    def test_mcp_server_out_error_state(self) -> None:
        """Error state with error message."""
        server = MCPServerOut(
            name="broken",
            status="error",
            transport="stdio",
            command="npx",
            requires_confirmation=True,
            error="Connection refused",
        )
        assert server.status == "error"
        assert server.error == "Connection refused"

    def test_mcp_server_out_invalid_status(self) -> None:
        """Invalid status raises ValidationError."""
        with pytest.raises(ValidationError):
            MCPServerOut(
                name="test",
                status="invalid",
                transport="stdio",
                command="npx",
                requires_confirmation=True,
            )

    def test_mcp_server_out_invalid_transport(self) -> None:
        """Invalid transport raises ValidationError."""
        with pytest.raises(ValidationError):
            MCPServerOut(
                name="test",
                status="connected",
                transport="http",  # invalid
                command="npx",
                requires_confirmation=True,
            )

    def test_mcp_server_out_missing_required(self) -> None:
        """Missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            MCPServerOut(
                name="test",
                status="connected",
                transport="stdio",
                # requires_confirmation missing
            )


class TestMCPServerAddRequest:
    """MCPServerAddRequest schema construction and validation."""

    def test_mcp_server_add_request_valid_stdio(self) -> None:
        """Valid stdio add request constructs without error."""
        req = MCPServerAddRequest(
            name="filesystem",
            transport="stdio",
            command="npx",
            args=["@modelcontextprotocol/server-filesystem", "/data"],
        )
        assert req.name == "filesystem"
        assert req.transport == "stdio"
        assert req.requires_confirmation is True
        assert req.timeout == 30

    def test_mcp_server_add_request_valid_sse(self) -> None:
        """Valid SSE add request constructs without error."""
        req = MCPServerAddRequest(
            name="remote-server",
            transport="sse",
            url="http://localhost:3000/sse",
            requires_confirmation=False,
            timeout=60,
        )
        assert req.url == "http://localhost:3000/sse"
        assert req.timeout == 60

    def test_mcp_server_add_request_invalid_name(self) -> None:
        """Invalid name pattern raises ValidationError."""
        with pytest.raises(ValidationError):
            MCPServerAddRequest(
                name="invalid name!",
                transport="stdio",
                command="npx",
            )

    def test_mcp_server_add_request_name_too_long(self) -> None:
        """Name over 64 chars raises ValidationError."""
        with pytest.raises(ValidationError):
            MCPServerAddRequest(
                name="x" * 65,
                transport="stdio",
                command="npx",
            )

    def test_mcp_server_add_request_timeout_too_low(self) -> None:
        """Timeout below 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            MCPServerAddRequest(
                name="test",
                transport="stdio",
                command="npx",
                timeout=0,
            )

    def test_mcp_server_add_request_timeout_too_high(self) -> None:
        """Timeout above 300 raises ValidationError."""
        with pytest.raises(ValidationError):
            MCPServerAddRequest(
                name="test",
                transport="stdio",
                command="npx",
                timeout=301,
            )

    def test_mcp_server_add_request_env_and_headers(self) -> None:
        """Optional env and headers can be set."""
        req = MCPServerAddRequest(
            name="test",
            transport="sse",
            url="http://localhost:3000/sse",
            env={"API_KEY": "secret"},
            headers={"Authorization": "Bearer token"},
        )
        assert req.env == {"API_KEY": "secret"}
        assert req.headers == {"Authorization": "Bearer token"}

    def test_mcp_server_add_request_invalid_transport(self) -> None:
        """Invalid transport raises ValidationError."""
        with pytest.raises(ValidationError):
            MCPServerAddRequest(
                name="test",
                transport="tcp",
                command="npx",
            )
