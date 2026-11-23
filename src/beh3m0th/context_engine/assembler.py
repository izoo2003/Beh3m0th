"""Deterministic token budgeting and context assembly."""

from __future__ import annotations

import time

from beh3m0th.context_engine.allocation import (
    cover_mandatory,
    demands,
    integer_shares,
    mandatory_tokens,
    water_fill,
)
from beh3m0th.context_engine.errors import ContextBudgetExceededError
from beh3m0th.context_engine.models import (
    AssembledContext,
    BudgetPolicy,
    ContextAssemblyReport,
    ContextItem,
)
from beh3m0th.context_engine.tokenizer import TokenCounter
from beh3m0th.context_engine.trimming import (
    enforce_render_budget,
    render,
    select_items,
    tokens_per_bucket,
)
from beh3m0th.core.schemas import MetadataValue, TrajectoryStep
from beh3m0th.evaluation.tracer import TrajectoryTracer


class ContextAssembler:
    """Fit context candidates into a token budget and record the assembly."""

    def __init__(
        self,
        counter: TokenCounter | None = None,
        tracer: TrajectoryTracer | None = None,
        agent_name: str = "assembler",
    ) -> None:
        if not agent_name.strip():
            raise ValueError("agent_name must not be empty")
        self._counter = counter if counter is not None else TokenCounter()
        self._tracer = tracer
        self._agent_name = agent_name

    def assemble(self, items: list[ContextItem], policy: BudgetPolicy) -> AssembledContext:
        started = time.perf_counter()
        budget = policy.available
        counted = [self._counted(item) for item in items]
        mandatory = mandatory_tokens(counted)
        if mandatory > budget:
            raise ContextBudgetExceededError(budget, mandatory)

        shares = cover_mandatory(integer_shares(budget, policy.ratios), counted)
        caps, reallocations = water_fill(shares, demands(counted), policy.ratios)
        included, dropped, actions = select_items(counted, caps, self._counter)
        actions = [*reallocations, *actions]
        included, render_dropped, render_actions = enforce_render_budget(
            included,
            budget,
            self._counter,
        )
        dropped.extend(render_dropped)
        actions.extend(render_actions)

        text = render(included)
        tokens_used = self._counter.count(text)
        if tokens_used > budget:
            raise ContextBudgetExceededError(
                budget,
                mandatory,
                f"Rendered prompt uses {tokens_used} tokens, above the budget of {budget}",
            )
        report = ContextAssemblyReport(
            budget_available=budget,
            tokens_used=tokens_used,
            tokens_used_per_bucket=tokens_per_bucket(included),
            items_included=[item.id for item in included],
            items_dropped=dropped,
            overflow_actions=actions,
        )
        self._record(report, started, len(items))
        return AssembledContext(text=text, items=included, report=report)

    def _counted(self, item: ContextItem) -> ContextItem:
        count = self._counter.count(item.content)
        if item.token_count == count:
            return item
        return item.model_copy(update={"token_count": count})

    def _record(self, report: ContextAssemblyReport, started: float, item_count: int) -> None:
        if self._tracer is None:
            return
        metadata: dict[str, MetadataValue] = {
            "budget_available": report.budget_available,
            "included_count": len(report.items_included),
            "dropped_count": len(report.items_dropped),
        }
        for bucket, tokens in report.tokens_used_per_bucket.items():
            metadata[f"tokens_{bucket}"] = tokens
        step = TrajectoryStep(
            step_index=len(self._tracer.steps),
            agent_name=self._agent_name,
            step_type="CONTEXT_ASSEMBLE",
            input_summary=f"{item_count} items, budget {report.budget_available}",
            output_summary=(
                f"included {len(report.items_included)} dropped {len(report.items_dropped)}"
            ),
            tokens_used=report.tokens_used,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata=metadata,
        )
        self._tracer.record(step)
