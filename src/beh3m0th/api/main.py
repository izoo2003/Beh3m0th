"""Thin FastAPI server that runs one supervisor workflow and streams its steps."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import Field

from beh3m0th.agents.graph import build_orchestrator
from beh3m0th.agents.state import WorkflowState
from beh3m0th.core.schemas import StrictModel, TrajectoryStep
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.mcp_server.client import InProcessTransport, MCPClient
from beh3m0th.mcp_server.server import build_server


class RunCreate(StrictModel):
    """Body for ``POST /v1/runs``."""

    objective: str = Field(min_length=1)


class RunRecord:
    """Buffered steps for one run. Late SSE clients replay this list."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.steps: list[TrajectoryStep] = []
        self.status = "RUNNING"
        self.error: str | None = None


def create_app(workspace: Path | None = None) -> FastAPI:
    """Build the app. Tests pass a temp workspace so nothing is written to the repo."""
    app = FastAPI(title="Beh3m0th")
    app.state.workspace = workspace if workspace is not None else Path("data/workspace")
    app.state.runs = {}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/runs")
    async def start_run(body: RunCreate) -> dict[str, str]:
        run_id = uuid4().hex
        record = RunRecord(run_id)
        app.state.runs[run_id] = record
        await _execute(app.state.workspace, body.objective, record)
        return {"run_id": run_id, "status": record.status}

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, str | int]:
        record = _require(app, run_id)
        return {"run_id": run_id, "status": record.status, "step_count": len(record.steps)}

    @app.get("/v1/runs/{run_id}/events")
    async def events(run_id: str) -> StreamingResponse:
        record = _require(app, run_id)
        return StreamingResponse(_events(record), media_type="text/event-stream")

    return app


def _require(app: FastAPI, run_id: str) -> RunRecord:
    record = app.state.runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    return record


async def _execute(workspace: Path, objective: str, record: RunRecord) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    tracer = TrajectoryTracer(listener=record.steps.append)
    try:
        client = MCPClient(InProcessTransport(build_server(workspace, tracer=tracer)))
        orchestrator = build_orchestrator(tools=client, workspace=workspace, tracer=tracer)
        state = await orchestrator.run(
            WorkflowState(
                run_id=record.run_id,
                session_id="api",
                project_id="api",
                objective=objective,
            )
        )
    except Exception as exc:
        record.status = "FAILED"
        record.error = str(exc)
        return
    record.status = state.status


async def _events(record: RunRecord) -> AsyncIterator[str]:
    sent = 0
    while True:
        while sent < len(record.steps):
            step = record.steps[sent]
            sent += 1
            yield f"event: step\ndata: {step.model_dump_json()}\n\n"
        if record.status != "RUNNING":
            payload = json.dumps({"status": record.status, "step_count": len(record.steps)})
            yield f"event: done\ndata: {payload}\n\n"
            return
        await asyncio.sleep(0.01)


def main() -> None:
    """Serve the API on localhost. Import ``create_app`` in tests instead."""
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
