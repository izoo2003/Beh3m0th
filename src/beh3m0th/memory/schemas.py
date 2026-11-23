"""Typed working, episodic, and semantic memory records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, model_validator

from beh3m0th.core.schemas import MemoryLayer, StrictModel, utcnow

type ImportanceKind = Literal[
    "conflict_resolution",
    "bug_fix",
    "routine_lookup",
    "repo_rule",
    "default",
]


def dump_datetime(value: datetime) -> str:
    """Store UTC timestamps so a reopened database compares cleanly."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def load_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class WorkingScratchpad(StrictModel):
    """Active session scratchpad: objective, plan, diffs, turns, and hypotheses."""

    user_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    plan_steps: list[str] = Field(default_factory=list)
    diffs: list[str] = Field(default_factory=list)
    turns: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utcnow)


class EpisodicMemory(StrictModel):
    """One completed episode, scoped to a user and project."""

    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    outcome: str = ""
    tags: list[str] = Field(default_factory=list)
    importance_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utcnow)
    last_accessed_at: datetime = Field(default_factory=utcnow)


class SemanticMemory(StrictModel):
    """A distilled fact. Inactive rows were superseded and stay out of recall."""

    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    importance_score: float = Field(ge=0.0, le=1.0)
    supersedes_id: str | None = None
    active: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_accessed_at: datetime = Field(default_factory=utcnow)


class ScoredMemory(StrictModel):
    """A recalled long-term memory and the terms of its composite score."""

    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    layer: MemoryLayer
    summary: str = Field(min_length=1)
    outcome: str = ""
    tags: list[str] = Field(default_factory=list)
    importance_score: float = Field(ge=0.0, le=1.0)
    similarity: float = Field(ge=-1.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    score: float
    created_at: datetime
    last_accessed_at: datetime
    supersedes_id: str | None = None


class RankingPolicy(StrictModel):
    """Weights for relevance, importance, and exponential recency decay."""

    relevance_weight: float = Field(default=0.5, ge=0.0)
    importance_weight: float = Field(default=0.3, ge=0.0)
    recency_weight: float = Field(default=0.2, ge=0.0)
    decay_lambda: float = Field(default=0.005, ge=0.0)
    dedup_threshold: float = Field(default=0.88, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_are_finite(self) -> RankingPolicy:
        values = {
            "relevance_weight": self.relevance_weight,
            "importance_weight": self.importance_weight,
            "recency_weight": self.recency_weight,
            "decay_lambda": self.decay_lambda,
            "dedup_threshold": self.dedup_threshold,
        }
        for name, value in values.items():
            if value != value or value in {float("inf"), float("-inf")}:
                raise ValueError(f"{name} must be finite")
        return self


class ConsolidationResult(StrictModel):
    """Episodes and facts written at the end of a session."""

    episodic: list[EpisodicMemory]
    semantic: list[SemanticMemory]
