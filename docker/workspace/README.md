# MCP Filesystem Workspace

This directory is mounted read-only into the `mcp-filesystem` container as `/workspace`.

Only files placed here are exposed to the MCP filesystem tool. The parent directory (repository root) is NOT mounted, preventing accidental exposure of `.env`, deployment configs, SSH keys, and other sensitive files.

## Usage

Place files the agent needs to read (documents, templates, reference data) in this directory before starting the compose stack.

## Security

- Mount is read-only (`:ro`) inside the container.
- Scope is limited to this directory only — not the compose project parent.
