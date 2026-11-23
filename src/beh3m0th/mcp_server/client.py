"""Async MCP client for in-process and HTTP/ASGI transports."""

from __future__ import annotations

import json
from typing import Any, Protocol

import httpx

from beh3m0th.mcp_server.errors import MCPError
from beh3m0th.mcp_server.schemas import PROTOCOL_VERSION
from beh3m0th.mcp_server.server import MCPServer


class MCPTransport(Protocol):
    async def send(self, message: dict[str, Any]) -> dict[str, Any] | None: ...


class InProcessTransport:
    """Call an ``MCPServer`` in this process. Unit tests use this path."""

    def __init__(self, server: MCPServer) -> None:
        self._server = server

    async def send(self, message: dict[str, Any]) -> dict[str, Any] | None:
        return await self._server.handle_raw(json.dumps(message))


class AsgiTransport:
    """POST JSON-RPC to a FastAPI app through ``httpx`` without opening a port."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AsgiTransport:
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self._app),
            base_url="http://mcp",
        )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if self._client is None:
            raise RuntimeError("AsgiTransport must be used as an async context manager")
        response = await self._client.post("/mcp/rpc", json=message)
        if response.status_code == 204:
            return None
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("MCP HTTP response was not an object")
        return body


class MCPClient:
    """Discover and invoke tools. Agents in later phases depend on this client."""

    def __init__(self, transport: MCPTransport) -> None:
        self._transport = transport
        self._next_id = 1

    async def initialize(self) -> dict[str, Any]:
        return await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "clientInfo": {"name": "beh3m0th-client", "version": "0.1.0"},
            },
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._request("tools/list", {})
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise RuntimeError("tools/list returned an invalid payload")
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any], role: str) -> dict[str, Any]:
        return await self._request(
            "tools/call",
            {"name": name, "arguments": arguments, "role": role},
        )

    async def ping(self) -> dict[str, Any]:
        return await self._request("ping", {})

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        message = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params,
        }
        self._next_id += 1
        response = await self._transport.send(message)
        if response is None:
            raise RuntimeError(f"{method} returned no response")
        if "error" in response:
            raise MCPError(_error_text(method, response["error"]))
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"{method} returned an invalid result")
        return result


def _error_text(method: str, error: object) -> str:
    if isinstance(error, dict):
        return f"{method} failed ({error.get('code')}): {error.get('message')}"
    return f"{method} failed"
