"""Shared Pydantic models used across Beh3m0th subsystems."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

type MetadataValue = str | int | float | bool | None
type AgentStatus = Literal[
    "PLANNING",
    "RESEARCHING",
    "CODING",
    "REVIEWING",
    "RESOLVING_CONFLICT",
    "COMPLETED",
    "ABSTAINED",
    "FAILED",
]
type StepType = Literal[
    "CONTEXT_ASSEMBLE",
    "RETRIEVAL",
    "MEMORY_RECALL",
    "MCP_TOOL_CALL",
    "AGENT_DECISION",
    "CONSENSUS_VOTE",
    "CONFLICT_ARBITRATION",
    "FINAL_OUTPUT",
]
type RetrievalSource = Literal["dense", "bm25", "hybrid", "graph"]
type MemoryLayer = Literal["working", "episodic", "semantic"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentChunk(StrictModel):
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    language: str | None = None
    section_header: str | None = None
    symbol_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def line_range_is_ordered(self) -> DocumentChunk:
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


class RetrievalHit(StrictModel):
    chunk: DocumentChunk
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    source: RetrievalSource


class RetrievalResult(StrictModel):
    query: str = Field(min_length=1)
    hits: list[RetrievalHit]


class GraphTriple(StrictModel):
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    chunk_id: str | None = None
    weight: float = Field(default=1.0, gt=0.0)


class MemoryRecord(StrictModel):
    id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    layer: MemoryLayer
    summary: str = Field(min_length=1)
    outcome: str = ""
    tags: list[str] = Field(default_factory=list)
    importance_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utcnow)
    last_accessed_at: datetime = Field(default_factory=utcnow)
    supersedes_id: str | None = None


class AgentState(StrictModel):
    """Phase 0 workflow envelope. Nested research and review lists arrive in Phase 7."""

    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    status: AgentStatus = "PLANNING"
    iteration_count: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=3, ge=1)


class TrajectoryStep(StrictModel):
    step_index: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=utcnow)
    agent_name: str = Field(min_length=1)
    step_type: StepType
    input_summary: str
    output_summary: str
    tokens_used: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
