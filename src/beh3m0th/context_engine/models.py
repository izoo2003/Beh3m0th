"""Typed context-assembly inputs, budgets, and reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from beh3m0th.core.schemas import MetadataValue, StrictModel


class ContextBucket(StrEnum):
    """Allocation buckets. Lower enum order is higher priority."""

    SYSTEM_TOOLS = "system_tools"
    WORKING_MEMORY = "working_memory"
    RETRIEVAL = "retrieval"
    LONG_TERM_MEMORY = "long_term_memory"
    AUXILIARY = "auxiliary"


DEFAULT_BUCKET_RATIOS: dict[ContextBucket, float] = {
    ContextBucket.SYSTEM_TOOLS: 0.20,
    ContextBucket.WORKING_MEMORY: 0.15,
    ContextBucket.RETRIEVAL: 0.45,
    ContextBucket.LONG_TERM_MEMORY: 0.20,
    ContextBucket.AUXILIARY: 0.0,
}


class ContextItem(StrictModel):
    """One candidate block for assembly."""

    id: str = Field(min_length=1)
    bucket: ContextBucket
    priority: int = Field(ge=0, le=4)
    content: str
    token_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    truncatable: bool = False


class BudgetPolicy(StrictModel):
    """Input budget and default bucket shares."""

    model_window: int = Field(gt=0)
    reserved_output: int = Field(ge=0)
    safety_margin: int = Field(default=256, ge=0)
    ratios: dict[ContextBucket, float] = Field(
        default_factory=lambda: dict(DEFAULT_BUCKET_RATIOS)
    )

    @property
    def available(self) -> int:
        return self.model_window - self.reserved_output - self.safety_margin

    @model_validator(mode="after")
    def budget_and_ratios_are_usable(self) -> BudgetPolicy:
        missing = [bucket.value for bucket in ContextBucket if bucket not in self.ratios]
        if missing:
            raise ValueError(f"ratios missing buckets: {', '.join(missing)}")
        if any(weight < 0 for weight in self.ratios.values()):
            raise ValueError("bucket ratios must be non-negative")
        total = sum(self.ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"bucket ratios must sum to 1, got {total}")
        if self.available <= 0:
            raise ValueError("available input budget must be positive")
        return self


class ContextAssemblyReport(StrictModel):
    """What the assembler kept, dropped, and moved between buckets."""

    budget_available: int = Field(ge=0)
    tokens_used: int = Field(ge=0)
    tokens_used_per_bucket: dict[str, int]
    items_included: list[str]
    items_dropped: list[str]
    overflow_actions: list[str]


class AssembledContext(StrictModel):
    """Rendered prompt plus the assembly report."""

    text: str
    items: list[ContextItem]
    report: ContextAssemblyReport
