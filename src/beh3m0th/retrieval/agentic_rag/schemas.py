"""Typed subqueries, claims, citations, and abstention responses."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from beh3m0th.core.schemas import StrictModel

type Route = Literal["hybrid", "graph", "both"]
type ClaimLabel = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]
type AnswerStatus = Literal["ANSWERED", "ABSTAINED"]
type EvidenceOrigin = Literal["hybrid", "graph"]


class SubQuery(StrictModel):
    """One atomic retrieval query and where it should run."""

    text: str = Field(min_length=1)
    route: Route


class Citation(StrictModel):
    """A source pointer that must exist in the retrieved evidence."""

    chunk_id: str = Field(min_length=1)
    source_path: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def line_range_is_ordered(self) -> Citation:
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


class Claim(StrictModel):
    """One factual statement and the citations attached to it."""

    text: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)


class ClaimVerification(StrictModel):
    """Support label for one claim against the evidence set."""

    claim: str = Field(min_length=1)
    label: ClaimLabel
    citation_ids: list[str] = Field(default_factory=list)


class Evidence(StrictModel):
    """A retrieved chunk or graph fact used for grading and citation checks."""

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    origin: EvidenceOrigin = "hybrid"


class AgenticRAGResponse(StrictModel):
    """Answered claims or an explicit abstention."""

    status: AnswerStatus
    query: str = Field(min_length=1)
    claims: list[Claim]
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    found: list[str]
    missing: list[str]
    follow_ups: list[str]
    steps: int = Field(ge=1)
