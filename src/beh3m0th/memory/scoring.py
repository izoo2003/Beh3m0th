"""Deterministic importance and composite recall ranking."""

from __future__ import annotations

import math
from datetime import datetime

from beh3m0th.memory.schemas import ImportanceKind, RankingPolicy


def importance_for(kind: ImportanceKind | str, *, verified: bool = False) -> float:
    """Map a write to a raw importance in ``[0, 1]`` without calling a model.

    A verified bug fix is ``0.85``, conflict resolution is ``0.90``, and a
    routine lookup is ``0.35``. Every other kind stays at ``0.50``.
    """
    if kind == "conflict_resolution":
        return 0.90
    if kind == "bug_fix" and verified:
        return 0.85
    if kind == "routine_lookup":
        return 0.35
    return 0.50


def recency_factor(last_accessed_at: datetime, now: datetime, decay_lambda: float) -> float:
    """Exponential decay over hours since the memory was last accessed."""
    elapsed = (now - last_accessed_at).total_seconds() / 3600.0
    hours = max(0.0, elapsed)
    factor = math.exp(-decay_lambda * hours)
    if factor > 1.0:
        return 1.0
    if factor < 0.0:
        return 0.0
    return factor


def composite_score(
    *,
    similarity: float,
    importance: float,
    recency: float,
    policy: RankingPolicy,
) -> float:
    """``w_rel * cos + w_imp * I + w_rec * exp(-lambda * hours)``."""
    return (
        policy.relevance_weight * similarity
        + policy.importance_weight * importance
        + policy.recency_weight * recency
    )


def clamp_similarity(value: float) -> float:
    return max(-1.0, min(1.0, value))
