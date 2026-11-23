"""Workspace jail. Resolved paths must stay inside MCP_ALLOWED_WORKSPACE."""

from __future__ import annotations

from pathlib import Path

from beh3m0th.mcp_server.errors import MCPPermissionError


def resolve_inside(workspace: Path, relative: str) -> Path:
    """Resolve ``relative`` and reject escapes, including symlink targets."""
    if relative == "" or "\x00" in relative:
        raise MCPPermissionError("path is empty or contains a null byte")
    root = workspace.resolve()
    candidate_path = Path(relative)
    if candidate_path.is_absolute():
        candidate = candidate_path.resolve()
    else:
        candidate = (root / candidate_path).resolve()
    if not candidate.is_relative_to(root):
        raise MCPPermissionError("path escapes the workspace")
    return candidate
