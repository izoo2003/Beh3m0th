"""Extractive trimming and rendered-prompt budget enforcement."""

from __future__ import annotations

import re

from beh3m0th.context_engine.allocation import mandatory_tokens
from beh3m0th.context_engine.errors import ContextBudgetExceededError
from beh3m0th.context_engine.models import ContextBucket, ContextItem
from beh3m0th.context_engine.tokenizer import TokenCounter

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def select_items(
    items: list[ContextItem],
    caps: dict[ContextBucket, int],
    counter: TokenCounter,
) -> tuple[list[ContextItem], list[str], list[str]]:
    grouped: dict[ContextBucket, list[ContextItem]] = {bucket: [] for bucket in ContextBucket}
    for item in items:
        grouped[item.bucket].append(item)
    included: list[ContextItem] = []
    dropped: list[str] = []
    actions: list[str] = []
    for bucket in ContextBucket:
        kept, rejected, fit_actions = _fit_bucket(grouped[bucket], caps[bucket], counter)
        included.extend(kept)
        dropped.extend(rejected)
        actions.extend(fit_actions)
    return included, dropped, actions


def enforce_render_budget(
    items: list[ContextItem],
    budget: int,
    counter: TokenCounter,
) -> tuple[list[ContextItem], list[str], list[str]]:
    current = list(items)
    dropped: list[str] = []
    actions: list[str] = []
    while counter.count(render(current)) > budget:
        index = _lowest_droppable(current)
        if index is None:
            raise ContextBudgetExceededError(budget, mandatory_tokens(current))
        item = current[index]
        shrunk = _shrink_for_render(current, index, budget, counter)
        if shrunk is not None:
            current[index] = shrunk
            actions.append(f"trimmed:{item.id}")
            continue
        current.pop(index)
        dropped.append(item.id)
        actions.append(f"dropped:{item.id}")
    return current, dropped, actions


def render(items: list[ContextItem]) -> str:
    if not items:
        return ""
    return "\n\n".join(f"[{item.bucket.value}]\n{item.content}" for item in items)


def tokens_per_bucket(items: list[ContextItem]) -> dict[str, int]:
    totals = {bucket.value: 0 for bucket in ContextBucket}
    for item in items:
        totals[item.bucket.value] += item.token_count or 0
    return totals


def _fit_bucket(
    items: list[ContextItem],
    cap: int,
    counter: TokenCounter,
) -> tuple[list[ContextItem], list[str], list[str]]:
    ordered = [item for item in items if item.priority == 0]
    ordered.extend(item for item in items if item.priority != 0)
    remaining = cap
    included: list[ContextItem] = []
    dropped: list[str] = []
    actions: list[str] = []
    for item in ordered:
        count = item.token_count or 0
        if count <= remaining:
            included.append(item)
            remaining -= count
            continue
        if item.priority == 0:
            raise ContextBudgetExceededError(cap, count)
        trimmed = _trimmed_item(item, remaining, counter)
        if trimmed is not None:
            included.append(trimmed)
            remaining -= trimmed.token_count or 0
            actions.append(f"trimmed:{item.id}")
            continue
        dropped.append(item.id)
        actions.append(f"dropped:{item.id}")
    return included, dropped, actions


def _trimmed_item(item: ContextItem, limit: int, counter: TokenCounter) -> ContextItem | None:
    if not item.truncatable or limit <= 0:
        return None
    trimmed = _trim_to_tokens(item.content, limit, counter)
    if not trimmed:
        return None
    trimmed_count = counter.count(trimmed)
    if trimmed_count > limit or trimmed_count >= (item.token_count or 0):
        return None
    return item.model_copy(update={"content": trimmed, "token_count": trimmed_count})


def _trim_to_tokens(text: str, limit: int, counter: TokenCounter) -> str:
    if limit <= 0 or not text:
        return ""
    if counter.count(text) <= limit:
        return text
    joiner = "\n" if "\n" in text else " "
    prefix = _longest_join(_units(text), joiner, limit, counter)
    if prefix:
        return prefix
    words = text.split()
    if len(words) > 1:
        prefix = _longest_join(words, " ", limit, counter)
        if prefix:
            return prefix
    return _character_prefix(text, limit, counter)


def _units(text: str) -> list[str]:
    if "\n" in text:
        return text.split("\n")
    return [part for part in _SENTENCE_SPLIT.split(text) if part]


def _longest_join(parts: list[str], joiner: str, limit: int, counter: TokenCounter) -> str:
    best = ""
    low = 0
    high = len(parts)
    while low <= high:
        mid = (low + high) // 2
        prefix = joiner.join(parts[:mid])
        if prefix and counter.count(prefix) <= limit:
            best = prefix
            low = mid + 1
        else:
            high = mid - 1
    return best


def _character_prefix(text: str, limit: int, counter: TokenCounter) -> str:
    best = ""
    low = 0
    high = len(text)
    while low <= high:
        mid = (low + high) // 2
        prefix = text[:mid].rstrip()
        if prefix and counter.count(prefix) <= limit:
            best = prefix
            low = mid + 1
        else:
            high = mid - 1
    return best


def _lowest_droppable(items: list[ContextItem]) -> int | None:
    candidates = [index for index, item in enumerate(items) if item.priority > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda index: (items[index].priority, index))


def _shrink_for_render(
    items: list[ContextItem],
    index: int,
    budget: int,
    counter: TokenCounter,
) -> ContextItem | None:
    item = items[index]
    if not item.truncatable:
        return None
    words = item.content.split()
    if len(words) > 1:
        pieces = words
        joiner = " "
    else:
        pieces = list(item.content)
        joiner = ""
    best = ""
    low = 0
    high = len(pieces)
    while low <= high:
        mid = (low + high) // 2
        prefix = joiner.join(pieces[:mid])
        if not prefix:
            high = mid - 1
            continue
        trial = item.model_copy(update={"content": prefix, "token_count": counter.count(prefix)})
        candidate = [*items[:index], trial, *items[index + 1 :]]
        if counter.count(render(candidate)) <= budget:
            best = prefix
            low = mid + 1
        else:
            high = mid - 1
    if not best or best == item.content:
        return None
    return item.model_copy(update={"content": best, "token_count": counter.count(best)})
