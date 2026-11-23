"""Deterministic rubric stub and an assembler-backed live judge."""

from __future__ import annotations

from pydantic import model_validator

from beh3m0th.agents.errors import AgentError
from beh3m0th.agents.prompts import TextCompleter, complete_text, parse_json
from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.context_engine.models import ContextBucket, ContextItem
from beh3m0th.core.schemas import StrictModel, TrajectoryStep
from beh3m0th.evaluation.errors import EvalError
from beh3m0th.evaluation.golden import called_tools, citations_present
from beh3m0th.evaluation.schemas import (
    RUBRIC_NAMES,
    GoldenWorkflow,
    JudgeReport,
    RubricScore,
    TrajectoryMetrics,
)

_INSTRUCTION = (
    "Grade this agent trajectory. Return a JSON object with key rubrics, a list of "
    "exactly five objects in this order: retrieval_relevance_and_grounding, "
    "context_and_memory_utilization, tool_usage_correctness, "
    "multi_agent_coordination_and_review, final_answer_correctness. "
    "Each object has name, score from 0 to 1, and rationale."
)


class _JudgePayload(StrictModel):
    rubrics: list[RubricScore]

    @model_validator(mode="after")
    def rubrics_match_catalog(self) -> _JudgePayload:
        names = [item.name for item in self.rubrics]
        if names != list(RUBRIC_NAMES):
            raise ValueError("judge rubrics must match the five catalog names in order")
        return self


def grade_deterministic(
    golden: GoldenWorkflow,
    steps: list[TrajectoryStep],
    metrics: TrajectoryMetrics,
) -> JudgeReport:
    """Map trajectory structure onto the five rubrics. No tokens are spent."""
    scores = (
        _retrieval(golden, steps, metrics),
        _memory(golden, steps),
        _tools(golden, steps, metrics),
        _coordination(golden, steps),
        _answer(golden, steps, metrics),
    )
    rubrics = [
        RubricScore(name=name, score=score, rationale=rationale)
        for name, (score, rationale) in zip(RUBRIC_NAMES, scores, strict=True)
    ]
    return JudgeReport(case_id=golden.case_id, rubrics=rubrics)


async def grade_live(
    golden: GoldenWorkflow,
    steps: list[TrajectoryStep],
    completer: TextCompleter,
    *,
    assembler: ContextAssembler | None = None,
) -> JudgeReport:
    """Grade through ContextAssembler. The completer never sees a raw concatenation."""
    engine = assembler if assembler is not None else ContextAssembler(agent_name="judge")
    try:
        raw = await complete_text(engine, completer, _items(golden, steps))
        payload = parse_json(raw, _JudgePayload)
    except AgentError as exc:
        raise EvalError("judge returned invalid JSON") from exc
    if not isinstance(payload, _JudgePayload):
        raise EvalError("judge payload was not a rubric list")
    return JudgeReport(case_id=golden.case_id, rubrics=payload.rubrics)


def _items(golden: GoldenWorkflow, steps: list[TrajectoryStep]) -> list[ContextItem]:
    lines = [
        f"{step.step_index} {step.agent_name} {step.step_type}: {step.output_summary}"
        for step in steps
    ]
    trajectory = "\n".join(lines) if lines else "empty trajectory"
    return [
        ContextItem(
            id="judge-system",
            bucket=ContextBucket.SYSTEM_TOOLS,
            priority=0,
            content=_INSTRUCTION,
        ),
        ContextItem(
            id="judge-golden",
            bucket=ContextBucket.WORKING_MEMORY,
            priority=1,
            content=golden.model_dump_json(),
        ),
        ContextItem(
            id="judge-trajectory",
            bucket=ContextBucket.RETRIEVAL,
            priority=2,
            content=trajectory,
            truncatable=True,
        ),
    ]


def _retrieval(
    golden: GoldenWorkflow,
    steps: list[TrajectoryStep],
    metrics: TrajectoryMetrics,
) -> tuple[float, str]:
    if golden.expected_abstention and metrics.abstention_correct:
        return 1.0, "abstained instead of inventing citations"
    if golden.expected_citations and citations_present(steps, golden.expected_citations):
        return 1.0, "expected citations are present"
    needs_retrieval = "RETRIEVAL" in golden.milestones
    retrieved = any(step.step_type == "RETRIEVAL" for step in steps)
    if not golden.expected_citations and (not needs_retrieval or retrieved):
        return 1.0, "retrieval matched the golden milestones"
    return 0.0, "citations or retrieval hops are missing"


def _memory(golden: GoldenWorkflow, steps: list[TrajectoryStep]) -> tuple[float, str]:
    if "MEMORY_RECALL" not in golden.milestones:
        return 1.0, "memory recall was not required"
    if any(step.step_type == "MEMORY_RECALL" for step in steps):
        return 1.0, "memory was recalled before the decision"
    return 0.0, "the golden memory milestone is missing"


def _tools(
    golden: GoldenWorkflow,
    steps: list[TrajectoryStep],
    metrics: TrajectoryMetrics,
) -> tuple[float, str]:
    if set(called_tools(steps)) & set(golden.forbidden_tools):
        return 0.0, "a forbidden tool was called"
    score = (metrics.tool_precision + metrics.tool_recall) / 2
    return score, "tool precision and recall"


def _coordination(golden: GoldenWorkflow, steps: list[TrajectoryStep]) -> tuple[float, str]:
    names = [step.agent_name for step in steps]
    types = [step.step_type for step in steps]
    sequence_ok = _is_subsequence(golden.expected_agent_sequence, names)
    vote_ok = "CONSENSUS_VOTE" not in golden.milestones or "CONSENSUS_VOTE" in types
    arbitration_ok = (
        "CONFLICT_ARBITRATION" not in golden.milestones or "CONFLICT_ARBITRATION" in types
    )
    hits = sum((sequence_ok, vote_ok, arbitration_ok))
    return hits / 3, "agent order, consensus, and arbitration"


def _answer(
    golden: GoldenWorkflow,
    steps: list[TrajectoryStep],
    metrics: TrajectoryMetrics,
) -> tuple[float, str]:
    needs_final = "FINAL_OUTPUT" in golden.milestones
    has_final = any(step.step_type == "FINAL_OUTPUT" for step in steps)
    checks = (
        metrics.abstention_correct,
        has_final if needs_final else True,
        len(steps) <= golden.max_steps,
        metrics.tokens_used <= golden.max_tokens,
    )
    score = sum(1.0 for check in checks if check) / len(checks)
    return score, "final output, budget, and abstention"


def _is_subsequence(needles: list[str], haystack: list[str]) -> bool:
    if not needles:
        return True
    cursor = 0
    for item in haystack:
        if item == needles[cursor]:
            cursor += 1
            if cursor == len(needles):
                return True
    return False
