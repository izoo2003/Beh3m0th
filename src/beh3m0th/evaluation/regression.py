"""Compare suite aggregates with the stored baseline and return a gate."""

from __future__ import annotations

from beh3m0th.evaluation.errors import EvalError
from beh3m0th.evaluation.schemas import (
    CORE_METRICS,
    RUBRIC_NAMES,
    AggregateMetrics,
    BaselineFile,
    GoldenWorkflow,
    JudgeReport,
    RegressionReport,
    TrajectoryMetrics,
)

type ScoredCase = tuple[GoldenWorkflow, TrajectoryMetrics, JudgeReport]


def aggregate(cases: list[ScoredCase]) -> AggregateMetrics:
    """Mean each core metric and each rubric across the suite."""
    if not cases:
        raise EvalError("no golden cases to aggregate")
    count = len(cases)

    def mean(values: list[float]) -> float:
        return sum(values) / count

    rubric_means: dict[str, float] = {}
    for name in RUBRIC_NAMES:
        rubric_means[name] = mean(
            [
                next(item.score for item in judged.rubrics if item.name == name)
                for _, _, judged in cases
            ]
        )
    return AggregateMetrics(
        tool_precision=mean([metrics.tool_precision for _, metrics, _ in cases]),
        tool_recall=mean([metrics.tool_recall for _, metrics, _ in cases]),
        milestone_coverage=mean([metrics.milestone_coverage for _, metrics, _ in cases]),
        efficiency_ratio=mean([metrics.efficiency_ratio for _, metrics, _ in cases]),
        hallucination_rate=mean(
            [1.0 if metrics.hallucination else 0.0 for _, metrics, _ in cases]
        ),
        abstention_accuracy=mean(
            [1.0 if metrics.abstention_correct else 0.0 for _, metrics, _ in cases]
        ),
        retrieval_relevance_and_grounding=rubric_means["retrieval_relevance_and_grounding"],
        context_and_memory_utilization=rubric_means["context_and_memory_utilization"],
        tool_usage_correctness=rubric_means["tool_usage_correctness"],
        multi_agent_coordination_and_review=rubric_means[
            "multi_agent_coordination_and_review"
        ],
        final_answer_correctness=rubric_means["final_answer_correctness"],
    )


def compare(
    current: AggregateMetrics,
    baseline: BaselineFile,
    *,
    trap_failed: bool,
) -> RegressionReport:
    """Fail on a drop past delta, a hallucination-cap breach, or a missed trap."""
    failures: list[str] = []
    for name in CORE_METRICS:
        current_value = float(getattr(current, name))
        baseline_value = float(getattr(baseline.metrics, name))
        drop = baseline_value - current_value
        if drop > baseline.delta + 1e-9:
            failures.append(f"{name} dropped by {drop:.3f}")
    if current.hallucination_rate > baseline.hallucination_cap:
        failures.append(
            f"hallucination rate {current.hallucination_rate:.3f} "
            f"exceeds cap {baseline.hallucination_cap:.3f}"
        )
    if trap_failed:
        failures.append("abstention accuracy on unanswerable traps failed")
    return RegressionReport(
        passed=not failures,
        failures=failures,
        current=current,
        baseline=baseline.metrics,
    )
