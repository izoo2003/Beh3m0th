"""Project 6: SDK-independent JSON-RPC tool server."""

from beh3m0th.mcp_server.client import AsgiTransport, InProcessTransport, MCPClient
from beh3m0th.mcp_server.errors import (
    JsonRpcError,
    MCPError,
    MCPPermissionError,
    ToolExecutionError,
)
from beh3m0th.mcp_server.http import build_app
from beh3m0th.mcp_server.paths import resolve_inside
from beh3m0th.mcp_server.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PERMISSION_DENIED,
)
from beh3m0th.mcp_server.rbac import ROLE_SCOPES, require_scope
from beh3m0th.mcp_server.sandbox import (
    DockerSandbox,
    SandboxResult,
    SubprocessRestrictedSandbox,
    build_docker_argv,
    sandbox_from_settings,
    stripped_env,
)
from beh3m0th.mcp_server.schemas import PROTOCOL_VERSION, Scope
from beh3m0th.mcp_server.server import MCPServer, build_server

__all__ = [
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "PERMISSION_DENIED",
    "PROTOCOL_VERSION",
    "ROLE_SCOPES",
    "AsgiTransport",
    "DockerSandbox",
    "InProcessTransport",
    "JsonRpcError",
    "MCPClient",
    "MCPError",
    "MCPPermissionError",
    "MCPServer",
    "SandboxResult",
    "Scope",
    "SubprocessRestrictedSandbox",
    "ToolExecutionError",
    "build_app",
    "build_docker_argv",
    "build_server",
    "require_scope",
    "resolve_inside",
    "sandbox_from_settings",
    "stripped_env",
]
