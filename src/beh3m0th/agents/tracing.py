"""Trajectory steps shared by every agent handoff."""

from __future__ import annotations

import time

from beh3m0th.core.schemas import MetadataValue, StepType, TrajectoryStep
from beh3m0th.evaluation.tracer import TrajectoryTracer


def record_step(
    tracer: TrajectoryTracer | None,
    *,
    agent_name: str,
    step_type: StepType,
    input_summary: str,
    output_summary: str,
    started: float | None = None,
    metadata: dict[str, MetadataValue] | None = None,
    tokens_used: int = 0,
) -> None:
    """Append one structured step. A missing tracer is a no-op."""
    if tracer is None:
        return
    elapsed = 0.0 if started is None else (time.perf_counter() - started) * 1000
    tracer.record(
        TrajectoryStep(
            step_index=len(tracer.steps),
            agent_name=agent_name,
            step_type=step_type,
            input_summary=input_summary[:500],
            output_summary=output_summary[:500],
            tokens_used=tokens_used,
            latency_ms=elapsed,
            metadata=metadata or {},
        )
    )
