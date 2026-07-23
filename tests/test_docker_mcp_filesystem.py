"""Regression tests for the Docker MCP filesystem bridge."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_mcp_filesystem_bridge_exposes_workspace_roots() -> None:
    text = (ROOT / "docker" / "docker-compose.yml").read_text()

    assert re.search(
        r'--stdio\s+"npx -y @modelcontextprotocol/server-filesystem /data /workspace"',
        text,
    ), "MCP filesystem bridge must expose both /data and /workspace"
    assert (
        "./workspace:/workspace:ro" in text
    ), "Docker compose must mount workspace subdir, not repo parent"
