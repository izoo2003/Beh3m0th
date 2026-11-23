"""Bucket shares and upward surplus reallocation."""

from __future__ import annotations

from beh3m0th.context_engine.errors import ContextBudgetExceededError
from beh3m0th.context_engine.models import ContextBucket, ContextItem

_BUCKET_RANK: dict[ContextBucket, int] = {
    bucket: index for index, bucket in enumerate(ContextBucket)
}


def mandatory_tokens(items: list[ContextItem]) -> int:
    return sum(item.token_count or 0 for item in items if item.priority == 0)


def demands(items: list[ContextItem]) -> dict[ContextBucket, int]:
    totals = {bucket: 0 for bucket in ContextBucket}
    for item in items:
        totals[item.bucket] += item.token_count or 0
    return totals


def integer_shares(total: int, ratios: dict[ContextBucket, float]) -> dict[ContextBucket, int]:
    buckets = list(ContextBucket)
    amounts = largest_remainder(total, [ratios[bucket] for bucket in buckets])
    return dict(zip(buckets, amounts, strict=True))


def largest_remainder(total: int, weights: list[float]) -> list[int]:
    if not weights or total <= 0:
        return [0] * len(weights)
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [0] * len(weights)
    raw = [total * weight / weight_sum for weight in weights]
    floors = [int(value) for value in raw]
    leftover = total - sum(floors)
    ranking = sorted(
        range(len(weights)),
        key=lambda index: (raw[index] - floors[index], -index),
        reverse=True,
    )
    for index in ranking[:leftover]:
        floors[index] += 1
    return floors


def cover_mandatory(
    shares: dict[ContextBucket, int],
    items: list[ContextItem],
) -> dict[ContextBucket, int]:
    """Grow a bucket share until its priority-0 items fit, taking from lower buckets."""
    updated = dict(shares)
    for bucket in ContextBucket:
        required = sum(
            item.token_count or 0
            for item in items
            if item.bucket == bucket and item.priority == 0
        )
        deficit = required - updated[bucket]
        if deficit <= 0:
            continue
        for donor in reversed(list(ContextBucket)):
            if _BUCKET_RANK[donor] <= _BUCKET_RANK[bucket]:
                continue
            take = min(deficit, updated[donor])
            if take <= 0:
                continue
            updated[donor] -= take
            updated[bucket] += take
            deficit -= take
            if deficit == 0:
                break
        if deficit > 0:
            raise ContextBudgetExceededError(sum(shares.values()), required)
    return updated


def water_fill(
    shares: dict[ContextBucket, int],
    bucket_demands: dict[ContextBucket, int],
    ratios: dict[ContextBucket, float],
) -> tuple[dict[ContextBucket, int], list[str]]:
    """Move unused share to higher-priority buckets that still have pending items."""
    assigned = {
        bucket: min(shares[bucket], bucket_demands[bucket]) for bucket in ContextBucket
    }
    actions: list[str] = []
    for donor in reversed(list(ContextBucket)):
        extra = shares[donor] - assigned[donor]
        while extra > 0:
            recipients = [
                bucket
                for bucket in ContextBucket
                if _BUCKET_RANK[bucket] < _BUCKET_RANK[donor]
                and assigned[bucket] < bucket_demands[bucket]
            ]
            if not recipients:
                break
            grants = _distribute(extra, recipients, ratios)
            moved = 0
            for bucket, grant in grants.items():
                give = min(grant, bucket_demands[bucket] - assigned[bucket])
                if give <= 0:
                    continue
                assigned[bucket] += give
                extra -= give
                moved += give
                actions.append(f"reallocated:{donor.value}->{bucket.value}:{give}")
            if moved == 0:
                break
    return assigned, actions


def _distribute(
    total: int,
    recipients: list[ContextBucket],
    ratios: dict[ContextBucket, float],
) -> dict[ContextBucket, int]:
    weights = [ratios[bucket] for bucket in recipients]
    if sum(weights) <= 0:
        weights = [1.0 for _ in recipients]
    amounts = largest_remainder(total, weights)
    return dict(zip(recipients, amounts, strict=True))
