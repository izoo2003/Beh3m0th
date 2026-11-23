"""Weighted review panel and the zero-token pytest/ruff gate."""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
from pathlib import Path

from beh3m0th.agents.errors import AgentError
from beh3m0th.agents.state import (
    CodeProposal,
    Finding,
    ReviewBallot,
    ReviewVerdict,
)

TAU_CONSENSUS = 0.6
GATE_PERSONA = "deterministic_gate"
GATE_WEIGHT = 0.5
CRITIQUE_WEIGHT = 0.25

_VERDICT_VALUE: dict[ReviewVerdict, int] = {
    "APPROVE": 1,
    "REQUEST_CHANGES": 0,
    "REJECT": -1,
}


def consensus_score(ballots: list[ReviewBallot]) -> float:
    """``sum(w * v * c) / sum(w * c)`` for one review panel."""
    if not ballots:
        raise AgentError("review panel is empty")
    denominator = sum(ballot.weight * ballot.confidence for ballot in ballots)
    if denominator == 0:
        raise AgentError("review panel has no weight")
    numerator = sum(
        ballot.weight * _VERDICT_VALUE[ballot.verdict] * ballot.confidence for ballot in ballots
    )
    return numerator / denominator


def has_critical(ballots: list[ReviewBallot]) -> bool:
    return any(finding.severity == "CRITICAL" for ballot in ballots for finding in ballot.findings)


def panel_passes(ballots: list[ReviewBallot], *, threshold: float = TAU_CONSENSUS) -> bool:
    """Pass only when the score clears the threshold and nothing is critical."""
    return consensus_score(ballots) >= threshold and not has_critical(ballots)


async def deterministic_ballot(proposal: CodeProposal, workspace: Path) -> ReviewBallot:
    """Approve when sandbox pytest, syntax, and ruff all succeed. Costs no tokens."""
    syntax_ok = await asyncio.to_thread(_syntax_ok, workspace, proposal)
    lint_ok = await asyncio.to_thread(_ruff_ok, workspace, proposal)
    tests_ok = proposal.pytest_exit_code == 0
    if tests_ok and syntax_ok and lint_ok:
        return ReviewBallot(
            persona_id=GATE_PERSONA,
            verdict="APPROVE",
            findings=[],
            confidence=1.0,
            weight=GATE_WEIGHT,
        )
    findings: list[Finding] = []
    if not tests_ok:
        findings.append(Finding(severity="CRITICAL", message="sandbox pytest failed"))
    if not syntax_ok:
        findings.append(Finding(severity="CRITICAL", message="syntax check failed"))
    if not lint_ok:
        findings.append(Finding(severity="CRITICAL", message="ruff check failed"))
    return ReviewBallot(
        persona_id=GATE_PERSONA,
        verdict="REJECT",
        findings=findings,
        confidence=1.0,
        weight=GATE_WEIGHT,
    )


def _touched_files(workspace: Path, proposal: CodeProposal) -> list[Path]:
    root = workspace.resolve()
    paths: list[Path] = []
    for edit in proposal.edits:
        path = (root / edit.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return []
        paths.append(path)
    return paths


def _syntax_ok(workspace: Path, proposal: CodeProposal) -> bool:
    paths = _touched_files(workspace, proposal)
    if not paths:
        return False
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            return False
    return True


def _ruff_ok(workspace: Path, proposal: CodeProposal) -> bool:
    paths = _touched_files(workspace, proposal)
    if not paths:
        return False
    completed = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *[str(path) for path in paths]],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0
