"""Offline checks for the phase demos. Streamlit itself is not imported."""

from __future__ import annotations

from pathlib import Path

import pytest

from beh3m0th.context_engine.errors import ContextBudgetExceededError
from beh3m0th.ui import (
    answer_question,
    assemble_context,
    call_tool_demo,
    home_status,
    query_graph,
    recall_bugfix,
    run_abstaining_objective,
    run_seeded_fix,
    scorecard,
    search_chunks,
)


def test_home_lists_every_phase() -> None:
    result = home_status()
    phases = {row["phase"] for row in result.rows}
    assert phases >= {str(number) for number in range(9)}
    assert all(row["mode"] != "" for row in result.rows if row["phase"].isdigit())


def test_context_stays_inside_the_budget() -> None:
    result = assemble_context(
        "You are a careful engineering assistant.",
        "The pool size is 4.",
        "Retries fail when the pool is exhausted.",
    )
    used = int(next(row["state"] for row in result.rows if row["item"] == "tokens_used"))
    budget = int(next(row["state"] for row in result.rows if row["item"] == "budget"))
    assert used <= budget
    assert "Retries fail when the pool is exhausted." in result.detail


def test_context_overflow_raises() -> None:
    with pytest.raises(ContextBudgetExceededError):
        assemble_context(
            "token " * 800,
            "notes",
            "evidence",
            model_window=900,
            reserved_output=100,
            safety_margin=100,
        )


async def test_hybrid_returns_the_auth_chunk(tmp_path: Path) -> None:
    result = await search_chunks("authentication retries pool", tmp_path)
    assert any(row["chunk"] == "auth" for row in result.rows)


async def test_graph_returns_both_hops(tmp_path: Path) -> None:
    result = await query_graph(tmp_path)
    names = {row["name"] for row in result.rows}
    assert {"answer", "helper"} <= names
    assert any("widget" in row["chunks"] for row in result.rows)


async def test_grounded_answer_and_abstention() -> None:
    answered = await answer_question(answerable=True)
    refused = await answer_question(answerable=False)
    assert answered.summary == "ANSWERED"
    assert "auth" in next(row["value"] for row in answered.rows if row["field"] == "citations")
    assert refused.summary == "ABSTAINED"


async def test_researcher_write_is_rejected(tmp_path: Path) -> None:
    listed = await call_tool_demo(tmp_path, "list")
    read = await call_tool_demo(tmp_path, "read")
    denied = await call_tool_demo(tmp_path, "write")
    assert "note.txt" in listed.detail
    assert "hello" in read.detail
    assert "-32001" in denied.summary
    assert denied.rows[0]["value"] == "yes"
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"


async def test_memory_recalls_the_session_bugfix(tmp_path: Path) -> None:
    result = await recall_bugfix(tmp_path)
    assert any("pool exhausted" in row["summary"] for row in result.rows)
    assert any(row["layer"] == "episodic" for row in result.rows)


async def test_seeded_agent_run_completes_and_uncited_run_abstains(tmp_path: Path) -> None:
    fixed = await run_seeded_fix(tmp_path / "fix")
    abstained = await run_abstaining_objective(tmp_path / "abstain")
    assert fixed.summary == "COMPLETED"
    assert any("FINAL_OUTPUT" in step for step in fixed.steps)
    assert abstained.summary == "ABSTAINED"


def test_scorecard_gate_passes() -> None:
    result = scorecard()
    assert result.summary == "passed"
    assert any(row["metric"] == "abstention_accuracy" for row in result.rows)
