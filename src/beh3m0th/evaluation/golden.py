"""Load golden workflows and score trajectories with zero model calls."""

from __future__ import annotations

from pathlib import Path

import yaml

from beh3m0th.core.schemas import TrajectoryStep
from beh3m0th.evaluation.errors import EvalError
from beh3m0th.evaluation.schemas import GoldenWorkflow, TrajectoryMetrics


def load_goldens(directory: Path) -> list[GoldenWorkflow]:
    """Load ``*.yaml`` cases. The file stem must match ``case_id``."""
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise EvalError(f"no golden workflows in {directory}")
    cases: list[GoldenWorkflow] = []
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise EvalError(f"{path.name} did not contain a mapping")
        case = GoldenWorkflow.model_validate(raw)
        if case.case_id != path.stem:
            raise EvalError(f"{path.name} case_id must match the file stem")
        cases.append(case)
    return cases


def load_fixture(path: Path) -> list[TrajectoryStep]:
    """Read one JSONL trajectory. Blank lines are ignored."""
    if not path.is_file():
        raise EvalError(f"missing fixture {path}")
    steps: list[TrajectoryStep] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            steps.append(TrajectoryStep.model_validate_json(line))
    return steps


def called_tools(steps: list[TrajectoryStep]) -> list[str]:
    """Tool names recorded on MCP_TOOL_CALL steps."""
    names: list[str] = []
    for step in steps:
        if step.step_type != "MCP_TOOL_CALL":
            continue
        tool = step.metadata.get("tool")
        if isinstance(tool, str) and tool:
            names.append(tool)
    return names


def tool_precision_recall(
    steps: list[TrajectoryStep],
    required: list[str],
    forbidden: list[str],
) -> tuple[float, float]:
    """Set precision and recall. Any forbidden call forces precision to 0."""
    called = set(called_tools(steps))
    required_set = set(required)
    if called & set(forbidden):
        precision = 0.0
    elif not called:
        precision = 1.0 if not required_set else 0.0
    else:
        precision = len(called & required_set) / len(called)
    if not required_set:
        recall = 1.0
    else:
        recall = len(called & required_set) / len(required_set)
    return precision, recall


def milestone_coverage(steps: list[TrajectoryStep], milestones: list[str]) -> float:
    """Fraction of milestones hit as an in-order subsequence."""
    if not milestones:
        return 1.0
    cursor = 0
    for step in steps:
        if cursor < len(milestones) and step.step_type == milestones[cursor]:
            cursor += 1
    return cursor / len(milestones)


def efficiency_ratio(golden_steps: int, actual_steps: int) -> float:
    """``min(1, golden_steps / actual_steps)``. Extra steps never score above 1."""
    if golden_steps <= 0 or actual_steps <= 0:
        return 0.0
    return min(1.0, golden_steps / actual_steps)


def citations_present(steps: list[TrajectoryStep], expected: list[str]) -> bool:
    """True when every expected citation appears in metadata or step text."""
    if not expected:
        return True
    found: set[str] = set()
    for step in steps:
        for key in ("citation", "chunk_id"):
            value = step.metadata.get(key)
            if isinstance(value, str) and value:
                found.add(value)
        blob = f"{step.input_summary}\n{step.output_summary}"
        for citation in expected:
            if citation in blob:
                found.add(citation)
    return all(citation in found for citation in expected)


def abstained(steps: list[TrajectoryStep]) -> bool:
    """True when a step records ABSTAINED or says it abstained."""
    for step in steps:
        if step.metadata.get("status") == "ABSTAINED":
            return True
        if "abstain" in step.output_summary.lower():
            return True
    return False


def hallucinated(steps: list[TrajectoryStep], golden: GoldenWorkflow) -> bool:
    """A flagged step, or a final answer missing a required citation, is a hallucination."""
    for step in steps:
        if step.metadata.get("hallucinated") is True:
            return True
    if not golden.expected_citations:
        return False
    return not citations_present(steps, golden.expected_citations)


def score_case(golden: GoldenWorkflow, steps: list[TrajectoryStep]) -> TrajectoryMetrics:
    """Score one fixture. This path never calls a model."""
    precision, recall = tool_precision_recall(
        steps,
        golden.required_tools,
        golden.forbidden_tools,
    )
    return TrajectoryMetrics(
        case_id=golden.case_id,
        tool_precision=precision,
        tool_recall=recall,
        milestone_coverage=milestone_coverage(steps, list(golden.milestones)),
        efficiency_ratio=efficiency_ratio(golden.golden_steps, len(steps)),
        hallucination=hallucinated(steps, golden),
        abstention_correct=abstained(steps) is golden.expected_abstention,
        tokens_used=sum(step.tokens_used for step in steps),
        step_count=len(steps),
    )
