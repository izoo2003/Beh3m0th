"""Versioned workflow state passed between supervisor, researcher, coder, and reviewer."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from beh3m0th.core.schemas import AgentStatus, StrictModel

type TaskRole = Literal["researcher", "coder", "reviewer"]
type TaskStatus = Literal["PENDING", "DONE", "SKIPPED"]
type ReviewVerdict = Literal["APPROVE", "REQUEST_CHANGES", "REJECT"]
type FindingSeverity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
type ArbitrationChoice = Literal[
    "ACCEPT_CODER",
    "ENFORCE_REVIEWER",
    "REQUEST_NEW_RESEARCH",
    "ESCALATE_ABSTAIN",
]


class TaskStep(StrictModel):
    """One planned handoff."""

    step_id: str = Field(min_length=1)
    role: TaskRole
    description: str = Field(min_length=1)
    status: TaskStatus = "PENDING"


class Citation(StrictModel):
    """A grounded source pointer attached to a research artifact."""

    chunk_id: str = Field(min_length=1)
    source_path: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)


class FileEdit(StrictModel):
    """A single replacement applied through the MCP patch tool."""

    path: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str


class ResearchArtifact(StrictModel):
    """Cited finding. Optional edits are the coder's zero-token patch plan."""

    artifact_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    edits: list[FileEdit] = Field(default_factory=list)


class CodeProposal(StrictModel):
    """Edits the coder applied, plus the sandbox pytest log."""

    proposal_id: str = Field(min_length=1)
    edits: list[FileEdit]
    diff: str = Field(min_length=1)
    pytest_exit_code: int
    pytest_stdout: str = ""
    pytest_stderr: str = ""


class Finding(StrictModel):
    """One review comment, tagged by severity."""

    severity: FindingSeverity
    message: str = Field(min_length=1)


class ReviewBallot(StrictModel):
    """One panel vote. Confidence is in ``(0, 1]`` and weight is positive."""

    persona_id: str = Field(min_length=1)
    verdict: ReviewVerdict
    findings: list[Finding] = Field(default_factory=list)
    confidence: float = Field(gt=0.0, le=1.0)
    weight: float = Field(gt=0.0)


class ConflictRecord(StrictModel):
    """Coder defense and reviewer objection for one deadlocked issue."""

    issue_id: str = Field(min_length=1)
    coder_defense: str = Field(min_length=1)
    reviewer_objection: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    rounds: int = Field(ge=1)


class ArbitrationDecision(StrictModel):
    """Binding supervisor choice for one conflict."""

    conflict_id: str = Field(min_length=1)
    decision: ArbitrationChoice
    rationale: str = Field(min_length=1)


class FinalReport(StrictModel):
    """What the supervisor returns after the graph stops."""

    summary: str = Field(min_length=1)
    status: AgentStatus
    consensus_score: float | None = None
    arbitration: ArbitrationChoice | None = None


class WorkflowState(StrictModel):
    """Typed envelope. ``schema_version`` stays at 1 for this phase."""

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    user_id: str = Field(default="local", min_length=1)
    objective: str = Field(min_length=1)
    plan: list[TaskStep] = Field(default_factory=list)
    research_findings: list[ResearchArtifact] = Field(default_factory=list)
    code_proposals: list[CodeProposal] = Field(default_factory=list)
    review_ballots: list[ReviewBallot] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    arbitration: ArbitrationDecision | None = None
    consensus_score: float | None = None
    consensus_passed: bool = False
    iteration_count: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=3, ge=1)
    status: AgentStatus = "PLANNING"
    report: FinalReport | None = None


class AgentOutput(StrictModel):
    """The next workflow state and a one-line handoff summary."""

    state: WorkflowState
    summary: str = Field(min_length=1)


def mark_done(state: WorkflowState, role: TaskRole) -> list[TaskStep]:
    """Mark the first pending step for ``role`` as done."""
    updated: list[TaskStep] = []
    flipped = False
    for step in state.plan:
        if not flipped and step.role == role and step.status == "PENDING":
            updated.append(step.model_copy(update={"status": "DONE"}))
            flipped = True
        else:
            updated.append(step)
    return updated
