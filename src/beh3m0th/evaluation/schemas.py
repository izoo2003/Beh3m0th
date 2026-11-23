"""Strict models for golden workflows, rubrics, and regression reports."""

from __future__ import annotations

from pydantic import Field, model_validator

from beh3m0th.core.schemas import StepType, StrictModel

RUBRIC_NAMES: tuple[str, ...] = (
    "retrieval_relevance_and_grounding",
    "context_and_memory_utilization",
    "tool_usage_correctness",
    "multi_agent_coordination_and_review",
    "final_answer_correctness",
)

CORE_METRICS: tuple[str, ...] = (
    "tool_precision",
    "tool_recall",
    "milestone_coverage",
    "efficiency_ratio",
    *RUBRIC_NAMES,
    "abstention_accuracy",
)


class GoldenWorkflow(StrictModel):
    """One offline case: the expected trajectory, not the live prompt dump."""

    case_id: str = Field(min_length=1)
    task_prompt: str = Field(min_length=1)
    fixture_repo_or_docs: str = ""
    expected_agent_sequence: list[str] = Field(default_factory=list)
    milestones: list[StepType] = Field(min_length=1)
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    expected_abstention: bool = False
    max_steps: int = Field(ge=1)
    max_tokens: int = Field(ge=0)
    golden_steps: int = Field(ge=1)


class TrajectoryMetrics(StrictModel):
    """Deterministic $0 scores for a single golden case."""

    case_id: str = Field(min_length=1)
    tool_precision: float = Field(ge=0.0, le=1.0)
    tool_recall: float = Field(ge=0.0, le=1.0)
    milestone_coverage: float = Field(ge=0.0, le=1.0)
    efficiency_ratio: float = Field(ge=0.0, le=1.0)
    hallucination: bool
    abstention_correct: bool
    tokens_used: int = Field(ge=0)
    step_count: int = Field(ge=0)


class RubricScore(StrictModel):
    """One judge rubric in the closed 0-to-1 range."""

    name: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class JudgeReport(StrictModel):
    """Five rubrics, in catalog order, for one trajectory."""

    case_id: str = Field(min_length=1)
    rubrics: list[RubricScore]

    @model_validator(mode="after")
    def rubrics_match_catalog(self) -> JudgeReport:
        names = [item.name for item in self.rubrics]
        if names != list(RUBRIC_NAMES):
            raise ValueError("judge rubrics must match the five catalog names in order")
        return self


class AggregateMetrics(StrictModel):
    """Means across the golden suite. Higher is better except hallucination_rate."""

    tool_precision: float = Field(ge=0.0, le=1.0)
    tool_recall: float = Field(ge=0.0, le=1.0)
    milestone_coverage: float = Field(ge=0.0, le=1.0)
    efficiency_ratio: float = Field(ge=0.0, le=1.0)
    retrieval_relevance_and_grounding: float = Field(ge=0.0, le=1.0)
    context_and_memory_utilization: float = Field(ge=0.0, le=1.0)
    tool_usage_correctness: float = Field(ge=0.0, le=1.0)
    multi_agent_coordination_and_review: float = Field(ge=0.0, le=1.0)
    final_answer_correctness: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)
    abstention_accuracy: float = Field(ge=0.0, le=1.0)


class BaselineFile(StrictModel):
    """Stored aggregates and the regression thresholds."""

    delta: float = Field(default=0.05, gt=0.0, le=1.0)
    hallucination_cap: float = Field(default=0.0, ge=0.0, le=1.0)
    metrics: AggregateMetrics


class RegressionReport(StrictModel):
    """Gate result. ``passed`` is false when the process should exit 1."""

    passed: bool
    failures: list[str]
    current: AggregateMetrics
    baseline: AggregateMetrics
