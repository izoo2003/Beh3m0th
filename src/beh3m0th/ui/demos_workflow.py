"""Tool, memory, and agent demos. The seeded fix uses no provider."""

from __future__ import annotations

import json
from pathlib import Path

from beh3m0th.agents.graph import build_orchestrator
from beh3m0th.agents.state import Citation, FileEdit, ResearchArtifact, WorkflowState
from beh3m0th.core.llm_gateway import LLMResponse
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.mcp_server.client import InProcessTransport, MCPClient
from beh3m0th.mcp_server.errors import MCPError
from beh3m0th.mcp_server.server import build_server
from beh3m0th.memory.schemas import WorkingScratchpad
from beh3m0th.memory.service import ThreeLayerMemory
from beh3m0th.ui.embed import HashingEmbedder
from beh3m0th.ui.models import DemoResult

_WIDGET = "def answer() -> int:\n    return 0\n"
_TEST = (
    "from widget import answer\n\n\n"
    "def test_answer() -> None:\n"
    "    assert answer() == 1\n"
)
_BUG = "authentication retries failed because the pool exhausted"


class _ApprovingCompleter:
    async def complete(
        self,
        messages: list[object],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        del messages, model, temperature
        return LLMResponse(
            content=json.dumps({"verdict": "APPROVE", "confidence": 0.9, "findings": []}),
            provider="fake",
            model="fake",
            cached=False,
            cache_key="ui-agent",
        )


async def call_tool_demo(workspace: Path, action: str) -> DemoResult:
    """List or read inside the workspace. A Researcher write is rejected with -32001."""
    workspace.mkdir(parents=True, exist_ok=True)
    note = workspace / "note.txt"
    note.write_text("hello", encoding="utf-8")
    client = MCPClient(InProcessTransport(build_server(workspace)))
    if action == "list":
        result = await client.call_tool("list_directory", {"path": "."}, "Researcher")
        return _tool_result("Directory listing", result)
    if action == "read":
        result = await client.call_tool("read_file", {"path": "note.txt"}, "Researcher")
        return _tool_result("File read", result)
    if action != "write":
        raise ValueError("action must be list, read, or write")
    try:
        await client.call_tool(
            "write_file_patch",
            {"path": "note.txt", "old_text": "hello", "new_text": "nope"},
            "Researcher",
        )
    except MCPError as exc:
        unchanged = note.read_text(encoding="utf-8") == "hello"
        return DemoResult(
            title="Researcher write",
            summary=str(exc),
            rows=[{"check": "file unchanged", "value": "yes" if unchanged else "no"}],
        )
    raise RuntimeError("Researcher write was allowed")


async def recall_bugfix(root: Path) -> DemoResult:
    """Save a session-1 bug fix, then recall it as session 2."""
    root.mkdir(parents=True, exist_ok=True)
    memory = ThreeLayerMemory(root / "memory.db", HashingEmbedder())
    await memory.initialize()
    try:
        await memory.consolidator.consolidate(
            WorkingScratchpad(
                user_id="user",
                project_id="proj",
                session_id="session-1",
                objective="fix authentication retries",
                turns=[f"bug_fix: {_BUG}"],
            )
        )
        hits = await memory.recall(
            "authentication retries",
            user_id="user",
            project_id="proj",
            session_id="session-2",
        )
    finally:
        await memory.close()
    return DemoResult(
        title="Memory recall",
        summary=f"{len(hits)} memories from session 1",
        rows=[
            {
                "layer": hit.layer,
                "score": f"{hit.score:.3f}",
                "summary": hit.summary,
            }
            for hit in hits
        ],
    )


async def run_seeded_fix(workspace: Path) -> DemoResult:
    """Researcher, Coder, Reviewer, and Supervisor fix the seeded widget bug."""
    workspace.mkdir(parents=True, exist_ok=True)
    workspace.joinpath("widget.py").write_text(_WIDGET, encoding="utf-8")
    workspace.joinpath("test_widget.py").write_text(_TEST, encoding="utf-8")
    tracer = TrajectoryTracer()
    client = MCPClient(InProcessTransport(build_server(workspace, tracer=tracer)))
    state = await build_orchestrator(
        tools=client,
        workspace=workspace,
        tracer=tracer,
        completer=_ApprovingCompleter(),
        scripted_research=[_artifact()],
    ).run(
        WorkflowState(
            run_id="ui-seed",
            session_id="ui",
            project_id="demo",
            objective="make answer return 1",
        )
    )
    return _agent_result("Seeded bug fix", state.status, tracer)


async def run_abstaining_objective(workspace: Path) -> DemoResult:
    """An objective with no citations ends as ABSTAINED."""
    workspace.mkdir(parents=True, exist_ok=True)
    tracer = TrajectoryTracer()
    client = MCPClient(InProcessTransport(build_server(workspace, tracer=tracer)))
    state = await build_orchestrator(
        tools=client,
        workspace=workspace,
        tracer=tracer,
    ).run(
        WorkflowState(
            run_id="ui-abstain",
            session_id="ui",
            project_id="demo",
            objective="explain the missing module",
        )
    )
    return _agent_result("Uncited objective", state.status, tracer)


def _artifact() -> ResearchArtifact:
    return ResearchArtifact(
        artifact_id="root-cause",
        summary="answer returns 0 but the test expects 1",
        citations=[
            Citation(chunk_id="widget-answer", source_path="widget.py", line_start=1, line_end=2)
        ],
        confidence=0.95,
        edits=[FileEdit(path="widget.py", old_text="return 0", new_text="return 1")],
    )


def _agent_result(title: str, status: str, tracer: TrajectoryTracer) -> DemoResult:
    steps = [
        f"{step.step_index} {step.agent_name} {step.step_type}: {step.output_summary}"
        for step in tracer.steps
    ]
    return DemoResult(title=title, summary=status, steps=steps)


def _tool_result(title: str, result: dict[str, object]) -> DemoResult:
    content = result.get("content")
    text = ""
    if isinstance(content, list) and content and isinstance(content[0], dict):
        raw = content[0].get("text")
        text = raw if isinstance(raw, str) else ""
    return DemoResult(title=title, summary="allowed", detail=text)
