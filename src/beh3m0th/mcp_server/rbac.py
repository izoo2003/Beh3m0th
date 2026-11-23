"""Per-agent capability scopes for MCP tool calls."""

from __future__ import annotations

from beh3m0th.mcp_server.errors import MCPPermissionError
from beh3m0th.mcp_server.schemas import Scope

ROLE_SCOPES: dict[str, frozenset[Scope]] = {
    "Researcher": frozenset({Scope.READ_FS, Scope.SEARCH_CODE}),
    "Coder": frozenset(Scope),
    "Reviewer": frozenset({Scope.READ_FS, Scope.SEARCH_CODE, Scope.RUN_TESTS}),
}


def require_scope(role: str, scope: Scope) -> None:
    """Reject unknown roles and roles that lack ``scope`` before the tool runs."""
    granted = ROLE_SCOPES.get(role)
    if granted is None or scope not in granted:
        raise MCPPermissionError(f"{role} lacks {scope.value}")
