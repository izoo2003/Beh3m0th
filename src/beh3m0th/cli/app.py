"""Terminal entrypoint for one offline supervisor run."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from beh3m0th.agents.graph import build_orchestrator
from beh3m0th.agents.state import WorkflowState
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.mcp_server.client import InProcessTransport, MCPClient
from beh3m0th.mcp_server.server import build_server

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def run(
    objective: str = typer.Argument(..., help="Engineering objective for the supervisor."),
    workspace: Path = typer.Option(
        Path("data/workspace"),
        "--workspace",
        help="Workspace the MCP sandbox is allowed to touch.",
    ),
) -> None:
    """Run the typed supervisor with zero provider calls."""
    workspace.mkdir(parents=True, exist_ok=True)
    state = asyncio.run(_execute(objective, workspace))
    console.print(f"[bold]{state.status}[/bold]")
    if state.report is not None:
        console.print(state.report.summary)


def main() -> None:
    app()


async def _execute(objective: str, workspace: Path) -> WorkflowState:
    tracer = TrajectoryTracer()
    client = MCPClient(InProcessTransport(build_server(workspace, tracer=tracer)))
    orchestrator = build_orchestrator(tools=client, workspace=workspace, tracer=tracer)
    return await orchestrator.run(
        WorkflowState(
            run_id="cli",
            session_id="cli",
            project_id="local",
            objective=objective,
        )
    )
