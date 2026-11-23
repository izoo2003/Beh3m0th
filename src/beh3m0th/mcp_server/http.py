"""FastAPI JSON-RPC and server-sent events transports."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from beh3m0th.mcp_server.server import MCPServer


def build_app(server: MCPServer) -> FastAPI:
    """Expose ``POST /mcp/rpc`` and ``GET /mcp/sse`` for in-process agents."""
    app = FastAPI()
    app.state.mcp_server = server
    app.state.events = asyncio.Queue()

    @app.post("/mcp/rpc")
    async def rpc(request: Request) -> Response:
        raw = (await request.body()).decode("utf-8")
        response = await server.handle_raw(raw)
        if response is None:
            return Response(status_code=204)
        await app.state.events.put(response)
        return JSONResponse(response)

    @app.get("/mcp/sse")
    async def sse() -> StreamingResponse:
        return StreamingResponse(_events(app.state.events), media_type="text/event-stream")

    return app


async def _events(queue: asyncio.Queue[dict[str, object]]) -> AsyncIterator[str]:
    yield "event: endpoint\ndata: /mcp/rpc\n\n"
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), 0.05)
        except TimeoutError:
            break
        yield f"event: message\ndata: {json.dumps(item)}\n\n"
