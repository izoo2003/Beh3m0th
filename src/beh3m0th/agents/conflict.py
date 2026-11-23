"""Deadlock detection and deterministic supervisor arbitration."""

from __future__ import annotations

from beh3m0th.agents.state import (
    ArbitrationDecision,
    ConflictRecord,
    ReviewBallot,
    WorkflowState,
)

_CRITIQUE_PERSONAS = ("correctness", "security_grounding")


def critique_split(ballots: list[ReviewBallot]) -> bool:
    """True when the two LLM personas did not cast the same verdict."""
    critiques = [ballot for ballot in ballots if ballot.persona_id in _CRITIQUE_PERSONAS]
    if len(critiques) < 2:
        return False
    return critiques[0].verdict != critiques[1].verdict


def repeated_request(previous: list[ReviewBallot], current: list[ReviewBallot]) -> str | None:
    """Return a finding repeated as ``REQUEST_CHANGES`` across two coder rounds."""
    seen = {
        finding.message
        for ballot in previous
        if ballot.verdict == "REQUEST_CHANGES"
        for finding in ballot.findings
    }
    for ballot in current:
        if ballot.verdict != "REQUEST_CHANGES":
            continue
        for finding in ballot.findings:
            if finding.message in seen:
                return finding.message
    return None


def make_conflict(
    state: WorkflowState,
    ballots: list[ReviewBallot],
    issue_id: str,
) -> ConflictRecord:
    """Capture the coder's diff and the reviewer's objection."""
    proposal = state.code_proposals[-1]
    objection = issue_id
    for ballot in reversed(ballots):
        if ballot.verdict != "APPROVE" and ballot.findings:
            objection = ballot.findings[-1].message
            break
    evidence = [
        citation.chunk_id
        for artifact in state.research_findings
        for citation in artifact.citations
    ]
    rounds = 1 if issue_id == "split-panel" else 2
    return ConflictRecord(
        issue_id=issue_id,
        coder_defense=proposal.diff[:500],
        reviewer_objection=objection[:500],
        evidence_ids=evidence,
        rounds=rounds,
    )


def arbitrate(state: WorkflowState) -> ArbitrationDecision:
    """Choose a binding outcome from pytest, citations, and the iteration budget."""
    conflict_id = state.conflicts[-1].issue_id if state.conflicts else "conflict"
    if state.iteration_count >= state.max_iterations:
        return ArbitrationDecision(
            conflict_id=conflict_id,
            decision="ESCALATE_ABSTAIN",
            rationale="iteration budget exhausted",
        )
    proposal = state.code_proposals[-1] if state.code_proposals else None
    citations = [
        citation
        for artifact in state.research_findings
        for citation in artifact.citations
    ]
    if proposal is not None and proposal.pytest_exit_code != 0:
        return ArbitrationDecision(
            conflict_id=conflict_id,
            decision="ENFORCE_REVIEWER",
            rationale="sandbox pytest failed",
        )
    if not citations:
        return ArbitrationDecision(
            conflict_id=conflict_id,
            decision="REQUEST_NEW_RESEARCH",
            rationale="research did not cite a source",
        )
    if proposal is not None and proposal.pytest_exit_code == 0:
        return ArbitrationDecision(
            conflict_id=conflict_id,
            decision="ACCEPT_CODER",
            rationale="sandbox pytest passed and the finding is cited",
        )
    return ArbitrationDecision(
        conflict_id=conflict_id,
        decision="ENFORCE_REVIEWER",
        rationale="no passing sandbox result",
    )
