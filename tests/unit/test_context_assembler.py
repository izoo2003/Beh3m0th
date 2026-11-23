"""Offline Phase 1 checks: token budgets, water-filling, and assembly traces."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from beh3m0th.context_engine import (
    BudgetPolicy,
    ContextAssembler,
    ContextBucket,
    ContextBudgetExceededError,
    ContextItem,
    TokenCounter,
)
from beh3m0th.core.schemas import TrajectoryStep
from beh3m0th.evaluation.tracer import TrajectoryTracer


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _policy(window: int, ratios: dict[ContextBucket, float] | None = None) -> BudgetPolicy:
    if ratios is None:
        return BudgetPolicy(model_window=window, reserved_output=0, safety_margin=0)
    return BudgetPolicy(
        model_window=window,
        reserved_output=0,
        safety_margin=0,
        ratios=ratios,
    )


def _retrieval_only_ratios() -> dict[ContextBucket, float]:
    ratios = {bucket: 0.0 for bucket in ContextBucket}
    ratios[ContextBucket.RETRIEVAL] = 1.0
    return ratios


def _words(count: int, label: str) -> str:
    return " ".join(f"{label}{index}" for index in range(count))


def test_available_budget_subtracts_output_reserve_and_safety_margin() -> None:
    policy = BudgetPolicy(model_window=1000, reserved_output=100)
    assert policy.available == 644


def test_token_counter_caches_calls_and_estimates_offline() -> None:
    calls = 0

    def counting(text: str) -> int:
        nonlocal calls
        calls += 1
        return word_count(text)

    counter = TokenCounter(count_fn=counting)
    assert counter.count("alpha beta") == 2
    assert counter.count("alpha beta") == 2
    assert calls == 1

    estimator = TokenCounter(count_fn=TokenCounter.estimate)
    assert estimator.count("") == 0
    assert estimator.count("a") == 1
    assert estimator.count("abcd") == 1
    assert estimator.count("abcdefgh") == 2


def test_rendered_prompt_never_exceeds_budget() -> None:
    counter = TokenCounter(count_fn=word_count)
    policy = _policy(40)
    items = [
        ContextItem(
            id="system",
            bucket=ContextBucket.SYSTEM_TOOLS,
            priority=0,
            content="Stay concise",
        ),
        *[
            ContextItem(
                id=f"hit-{index}",
                bucket=ContextBucket.RETRIEVAL,
                priority=2,
                content=_words(12, f"evidence{index}-"),
                truncatable=True,
            )
            for index in range(8)
        ],
    ]
    result = ContextAssembler(counter=counter).assemble(items, policy)
    assert counter.count(result.text) <= policy.available
    assert result.report.tokens_used == counter.count(result.text)
    assert result.report.tokens_used <= result.report.budget_available


def test_unused_memory_budget_is_reallocated_to_retrieval() -> None:
    counter = TokenCounter(count_fn=word_count)
    policy = _policy(100)
    items = [
        ContextItem(
            id="system",
            bucket=ContextBucket.SYSTEM_TOOLS,
            priority=0,
            content="You are Beh3m0th",
        ),
        ContextItem(
            id="working",
            bucket=ContextBucket.WORKING_MEMORY,
            priority=1,
            content="pool is closed",
        ),
        *[
            ContextItem(
                id=f"hit-{index}",
                bucket=ContextBucket.RETRIEVAL,
                priority=2,
                content=_words(10, f"chunk{index}-"),
            )
            for index in range(10)
        ],
    ]
    result = ContextAssembler(counter=counter).assemble(items, policy)
    base_retrieval = int(policy.available * 0.45)
    retrieval_tokens = result.report.tokens_used_per_bucket["retrieval"]
    assert retrieval_tokens == 60
    assert retrieval_tokens > base_retrieval
    assert "reallocated:long_term_memory->retrieval:20" in result.report.overflow_actions
    assert result.report.tokens_used <= policy.available
    assert result.report.items_included[:2] == ["system", "working"]
    assert result.report.items_dropped == ["hit-6", "hit-7", "hit-8", "hit-9"]


def test_mandatory_items_over_budget_raise() -> None:
    counter = TokenCounter(count_fn=word_count)
    policy = _policy(5)
    item = ContextItem(
        id="system",
        bucket=ContextBucket.SYSTEM_TOOLS,
        priority=0,
        content="one two three four five six",
    )
    with pytest.raises(ContextBudgetExceededError) as caught:
        ContextAssembler(counter=counter).assemble([item], policy)
    assert caught.value.budget == 5
    assert caught.value.mandatory_tokens == 6


def test_auxiliary_items_drop_before_retrieval() -> None:
    counter = TokenCounter(count_fn=word_count)
    policy = _policy(100)
    items = [
        ContextItem(
            id="history",
            bucket=ContextBucket.AUXILIARY,
            priority=4,
            content="old conversation turn",
        ),
        ContextItem(
            id="evidence",
            bucket=ContextBucket.RETRIEVAL,
            priority=2,
            content=_words(10, "fact-"),
        ),
    ]
    result = ContextAssembler(counter=counter).assemble(items, policy)
    assert "history" in result.report.items_dropped
    assert "evidence" in result.report.items_included
    assert "dropped:history" in result.report.overflow_actions


def test_extractive_trim_keeps_a_stable_prefix() -> None:
    counter = TokenCounter(count_fn=word_count)
    policy = _policy(3, _retrieval_only_ratios())
    content = "Alpha stays. Beta goes away."
    item = ContextItem(
        id="chunk",
        bucket=ContextBucket.RETRIEVAL,
        priority=2,
        content=content,
        truncatable=True,
    )
    result = ContextAssembler(counter=counter).assemble([item], policy)
    assert result.items[0].content == "Alpha stays."
    assert content.startswith(result.items[0].content)
    assert counter.count(result.text) <= policy.available
    assert "trimmed:chunk" in result.report.overflow_actions


def test_assembly_records_one_context_step(tmp_path: Path) -> None:
    counter = TokenCounter(count_fn=word_count)
    tracer = TrajectoryTracer(tmp_path / "trace.jsonl")
    policy = _policy(30)
    item = ContextItem(
        id="system",
        bucket=ContextBucket.SYSTEM_TOOLS,
        priority=0,
        content="You are Beh3m0th",
    )
    result = ContextAssembler(
        counter=counter,
        tracer=tracer,
        agent_name="supervisor",
    ).assemble([item], policy)
    assert len(tracer.steps) == 1
    step = tracer.steps[0]
    assert step.step_type == "CONTEXT_ASSEMBLE"
    assert step.agent_name == "supervisor"
    assert step.tokens_used == result.report.tokens_used
    assert step.metadata["budget_available"] == policy.available
    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    loaded = TrajectoryStep.model_validate_json(lines[0])
    assert loaded.tokens_used == result.report.tokens_used
    assert loaded.step_index == 0


def test_context_item_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ContextItem(
            id="system",
            bucket=ContextBucket.SYSTEM_TOOLS,
            priority=0,
            content="hello",
            unexpected=True,  # type: ignore[call-arg]
        )
