"""Failures raised by the hand-built MCP server."""

from __future__ import annotations


class MCPError(Exception):
    """Base error for the MCP server."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MCPPermissionError(MCPError):
    """RBAC or the workspace jail rejected the call. Maps to JSON-RPC -32001."""


class ToolExecutionError(MCPError):
    """The tool ran and failed. Returned as an MCP ``isError`` content block."""


class JsonRpcError(MCPError):
    """A JSON-RPC failure that the dispatcher turns into an error response."""

    def __init__(
        self,
        code: int,
        message: str,
        *,
        request_id: str | int | None = None,
        data: object | None = None,
        notification: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.data = data
        self.notification = notification
