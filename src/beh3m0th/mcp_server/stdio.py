"""Newline-delimited JSON-RPC over stdio."""

from __future__ import annotations

import asyncio
import json
from typing import Protocol

from beh3m0th.mcp_server.server import MCPServer


class ByteWriter(Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


async def serve_stdio(server: MCPServer, reader: asyncio.StreamReader, writer: ByteWriter) -> None:
    """Read one JSON-RPC message per line and write one response line."""
    while True:
        line = await reader.readline()
        if not line:
            return
        raw = line.decode("utf-8").strip()
        if not raw:
            continue
        response = await server.handle_raw(raw)
        if response is None:
            continue
        writer.write((json.dumps(response) + "\n").encode("utf-8"))
        await writer.drain()
