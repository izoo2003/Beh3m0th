"""Offline Phase 7 checks: seeded fix, consensus math, and forced arbitration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from beh3m0th.agents import (
    WorkflowState,
    build_orchestrator,
    consensus_score,
    panel_passes,
)
from beh3m0th.agents.state import (
    Citation,
    FileEdit,
    Finding,
    ResearchArtifact,
    ReviewBallot,
    ReviewVerdict,
)
from beh3m0th.cli.app import app
from beh3m0th.core.llm_gateway import LLMResponse
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.mcp_server.client import InProcessTransport, MCPClient
from beh3m0th.mcp_server.server import build_server
from beh3m0th.memory.schemas import ConsolidationResult, WorkingScratchpad

WIDGET = "def answer() -> int:\n    return 0\n"
TEST_FILE = (
    "from widget import answer\n\n\n"
    "def test_answer() -> None:\n"
    "    assert answer() == 1\n"
)
OBJECTIVE = "make answer return 1"


class RecordingConsolidator:
    def __init__(self) -> None:
        self.scratchpads: list[WorkingScratchpad] = []

    async def consolidate(self, scratchpad: WorkingScratchpad) -> ConsolidationResult:
        self.scratchpads.append(scratchpad)
        return ConsolidationResult(episodic=[], semantic=[])


class ScriptedCompleter:
    def __init__(self, security: str = "APPROVE") -> None:
        self.security = security

    async def complete(
        self,
        messages: list[object],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        del model, temperature
        text = str(getattr(messages[0], "content", ""))
        verdict = self.security if "security_grounding" in text else "APPROVE"
        findings = []
        if verdict == "REJECT":
            findings = [{"severity": "HIGH", "message": "security objection"}]
        return LLMResponse(
            content=json.dumps(
                {"verdict": verdict, "confidence": 0.9, "findings": findings}
            ),
            provider="fake",
            model="fake",
            cached=False,
            cache_key="agent-test",
        )


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


def _seed(workspace: Path) -> None:
    workspace.joinpath("widget.py").write_text(WIDGET, encoding="utf-8")
    workspace.joinpath("test_widget.py").write_text(TEST_FILE, encoding="utf-8")


def _ballot(
    persona: str,
    verdict: ReviewVerdict,
    *,
    weight: float,
    confidence: float,
    severity: str | None = None,
    message: str = "note",
) -> ReviewBallot:
    findings = [] if severity is None else [Finding(severity=severity, message=message)]
    return ReviewBallot(
        persona_id=persona,
        verdict=verdict,
        findings=findings,
        confidence=confidence,
        weight=weight,
    )


async def _run(
    tmp_path: Path,
    completer: ScriptedCompleter,
) -> tuple[WorkflowState, TrajectoryTracer, RecordingConsolidator]:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    _seed(workspace)
    tracer = TrajectoryTracer()
    consolidator = RecordingConsolidator()
    client = MCPClient(InProcessTransport(build_server(workspace, tracer=tracer)))
    orchestrator = build_orchestrator(
        tools=client,
        workspace=workspace,
        tracer=tracer,
        completer=completer,
        scripted_research=[_artifact()],
        consolidator=consolidator,
    )
    state = await orchestrator.run(
        WorkflowState(
            run_id="run-1",
            session_id="session-1",
            project_id="proj",
            user_id="user",
            objective=OBJECTIVE,
        )
    )
    return state, tracer, consolidator


def test_consensus_formula_and_critical_blocker() -> None:
    mixed = [
        _ballot("deterministic_gate", "APPROVE", weight=0.5, confidence=1.0),
        _ballot("correctness", "REQUEST_CHANGES", weight=0.25, confidence=0.8),
        _ballot("security_grounding", "REJECT", weight=0.25, confidence=0.5),
    ]
    numerator = 0.5 * 1 * 1 + 0.25 * 0 * 0.8 + 0.25 * -1 * 0.5
    denominator = 0.5 * 1 + 0.25 * 0.8 + 0.25 * 0.5
    assert consensus_score(mixed) == pytest.approx(numerator / denominator)

    blocked = [
        _ballot(
            "deterministic_gate",
            "APPROVE",
            weight=0.5,
            confidence=1.0,
            severity="CRITICAL",
            message="sandbox pytest failed",
        ),
        _ballot("correctness", "APPROVE", weight=0.25, confidence=1.0),
        _ballot("security_grounding", "APPROVE", weight=0.25, confidence=1.0),
    ]
    assert consensus_score(blocked) == pytest.approx(1.0)
    assert panel_passes(blocked) is False


async def test_seeded_bug_reaches_consensus_and_consolidates(tmp_path: Path) -> None:
    state, tracer, consolidator = await _run(tmp_path, ScriptedCompleter())

    assert state.status == "COMPLETED"
    assert state.report is not None
    assert state.consensus_passed is True
    assert state.code_proposals[-1].pytest_exit_code == 0
    assert (tmp_path / "repo" / "widget.py").read_text(encoding="utf-8") == WIDGET.replace(
        "return 0", "return 1"
    )
    decisions = [step.agent_name for step in tracer.steps if step.step_type == "AGENT_DECISION"]
    start = decisions.index("researcher")
    assert decisions[start : start + 4] == ["researcher", "coder", "reviewer", "supervisor"]
    assert any(step.step_type == "CONSENSUS_VOTE" for step in tracer.steps)
    assert any(step.step_type == "FINAL_OUTPUT" for step in tracer.steps)
    researcher_tools = [
        step.metadata["tool"]
        for step in tracer.steps
        if step.step_type == "MCP_TOOL_CALL" and step.agent_name == "Researcher"
    ]
    assert researcher_tools == ["read_file"]
    assert "bug_fix:" in consolidator.scratchpads[0].turns[0]


async def test_split_panel_records_conflict_and_accepts_coder(tmp_path: Path) -> None:
    state, _tracer, _consolidator = await _run(tmp_path, ScriptedCompleter(security="REJECT"))

    assert state.conflicts
    assert state.conflicts[0].issue_id == "split-panel"
    assert state.arbitration is not None
    assert state.arbitration.decision == "ACCEPT_CODER"
    assert any(step.step_type == "CONFLICT_ARBITRATION" for step in _tracer.steps)


def test_cli_help_lists_run() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
