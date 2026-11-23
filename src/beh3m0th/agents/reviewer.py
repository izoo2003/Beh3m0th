"""Reviewer: deterministic pytest/ruff gate plus two assembler-backed critiques."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import Field

from beh3m0th.agents.conflict import critique_split, make_conflict, repeated_request
from beh3m0th.agents.consensus import (
    CRITIQUE_WEIGHT,
    consensus_score,
    deterministic_ballot,
    panel_passes,
)
from beh3m0th.agents.errors import AgentError
from beh3m0th.agents.prompts import TextCompleter, complete_text, parse_json
from beh3m0th.agents.state import (
    AgentOutput,
    Finding,
    FindingSeverity,
    ReviewBallot,
    ReviewVerdict,
    WorkflowState,
    mark_done,
)
from beh3m0th.agents.tracing import record_step
from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.context_engine.models import ContextBucket, ContextItem
from beh3m0th.core.schemas import StrictModel
from beh3m0th.evaluation.tracer import TrajectoryTracer

_CRITIQUE = (
    "You are the {persona} reviewer. Return JSON with verdict, confidence, and findings. "
    "verdict is APPROVE, REQUEST_CHANGES, or REJECT. "
    "findings is a list of objects with severity and message. "
    "severity is CRITICAL, HIGH, MEDIUM, LOW, or INFO."
)


class _FindingDraft(StrictModel):
    severity: FindingSeverity
    message: str = Field(min_length=1)


class _BallotDraft(StrictModel):
    verdict: ReviewVerdict
    confidence: float = Field(gt=0.0, le=1.0)
    findings: list[_FindingDraft] = Field(default_factory=list)


class ReviewerAgent:
    """Score the latest proposal. A split panel or a repeated objection deadlocks."""

    def __init__(
        self,
        workspace: Path,
        *,
        completer: TextCompleter | None = None,
        assembler: ContextAssembler | None = None,
        tracer: TrajectoryTracer | None = None,
    ) -> None:
        self._workspace = workspace
        self._completer = completer
        self._assembler = assembler or ContextAssembler(tracer=tracer, agent_name="reviewer")
        self._tracer = tracer

    async def run(self, state: WorkflowState) -> AgentOutput:
        started = time.perf_counter()
        if not state.code_proposals:
            raise AgentError("reviewer has no code proposal")
        proposal = state.code_proposals[-1]
        gate = await deterministic_ballot(proposal, self._workspace)
        critiques = await self._critiques(state, gate)
        panel = [gate, *critiques]
        score = consensus_score(panel)
        split = critique_split(critiques)
        issue = "split-panel" if split else repeated_request(state.review_ballots, panel)
        passed = panel_passes(panel) and not split
        updated = self._apply(state, panel, score, passed, issue)
        record_step(
            self._tracer,
            agent_name="reviewer",
            step_type="CONSENSUS_VOTE",
            input_summary=proposal.proposal_id,
            output_summary=f"score={score:.3f} passed={passed}",
            started=started,
            metadata={"consensus_score": score, "passed": passed},
        )
        record_step(
            self._tracer,
            agent_name="reviewer",
            step_type="AGENT_DECISION",
            input_summary=state.objective,
            output_summary=updated.status,
            metadata={"status": updated.status},
        )
        return AgentOutput(state=updated, summary=updated.status)

    def _apply(
        self,
        state: WorkflowState,
        panel: list[ReviewBallot],
        score: float,
        passed: bool,
        issue: str | None,
    ) -> WorkflowState:
        ballots = [*state.review_ballots, *panel]
        base = {
            "review_ballots": ballots,
            "consensus_score": score,
            "consensus_passed": passed,
            "plan": mark_done(state, "reviewer"),
        }
        if passed:
            return state.model_copy(update={**base, "status": "COMPLETED"})
        if issue is None and state.iteration_count + 1 < state.max_iterations:
            return state.model_copy(
                update={**base, "status": "CODING", "iteration_count": state.iteration_count + 1}
            )
        conflict_id = issue or "iteration-budget"
        conflict = make_conflict(state, panel, conflict_id)
        return state.model_copy(
            update={
                **base,
                "conflicts": [*state.conflicts, conflict],
                "status": "RESOLVING_CONFLICT",
                "iteration_count": state.iteration_count + 1,
            }
        )

    async def _critiques(self, state: WorkflowState, gate: ReviewBallot) -> list[ReviewBallot]:
        if self._completer is None:
            verdict: ReviewVerdict = "APPROVE" if gate.verdict == "APPROVE" else "REJECT"
            return [
                _mirror(persona, verdict)
                for persona in ("correctness", "security_grounding")
            ]
        ballots: list[ReviewBallot] = []
        for persona in ("correctness", "security_grounding"):
            ballots.append(await self._one_critique(state, persona))
        return ballots

    async def _one_critique(self, state: WorkflowState, persona: str) -> ReviewBallot:
        if self._completer is None:
            raise AgentError("reviewer has no completer")
        proposal = state.code_proposals[-1]
        raw = await complete_text(
            self._assembler,
            self._completer,
            [
                ContextItem(
                    id=f"review-{persona}",
                    bucket=ContextBucket.SYSTEM_TOOLS,
                    priority=0,
                    content=_CRITIQUE.format(persona=persona),
                ),
                ContextItem(
                    id="review-objective",
                    bucket=ContextBucket.WORKING_MEMORY,
                    priority=1,
                    content=state.objective,
                ),
                ContextItem(
                    id="review-diff",
                    bucket=ContextBucket.RETRIEVAL,
                    priority=2,
                    content=proposal.diff,
                    truncatable=True,
                ),
            ],
        )
        draft = parse_json(raw, _BallotDraft)
        if not isinstance(draft, _BallotDraft):
            raise AgentError("review ballot was invalid")
        return ReviewBallot(
            persona_id=persona,
            verdict=draft.verdict,
            findings=[
                Finding(severity=item.severity, message=item.message) for item in draft.findings
            ],
            confidence=draft.confidence,
            weight=CRITIQUE_WEIGHT,
        )


def _mirror(persona: str, verdict: ReviewVerdict) -> ReviewBallot:
    return ReviewBallot(
        persona_id=persona,
        verdict=verdict,
        findings=[],
        confidence=0.9,
        weight=CRITIQUE_WEIGHT,
    )
