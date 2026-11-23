"""Offline Phase 8 checks: tracer, golden metrics, regression exits, and SSE."""

from __future__ import annotations

import json
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from beh3m0th.api.main import create_app
from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.core.llm_gateway import LLMResponse
from beh3m0th.core.schemas import TrajectoryStep
from beh3m0th.evaluation.golden import (
    efficiency_ratio,
    load_fixture,
    milestone_coverage,
    score_case,
    tool_precision_recall,
)
from beh3m0th.evaluation.judge import grade_deterministic, grade_live
from beh3m0th.evaluation.regression import compare
from beh3m0th.evaluation.schemas import RUBRIC_NAMES, BaselineFile, GoldenWorkflow
from beh3m0th.evaluation.suite import evals_root, run_suite
from beh3m0th.evaluation.tracer import TrajectoryTracer, load_steps


def _step(
    step_type: str,
    *,
    step_index: int = 0,
    agent_name: str = "researcher",
    output_summary: str = "out",
    metadata: dict[str, str] | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=step_index,
        agent_name=agent_name,
        step_type=step_type,
        input_summary="in",
        output_summary=output_summary,
        tokens_used=1,
        latency_ms=1.0,
        metadata=metadata or {},
    )


def _golden() -> GoldenWorkflow:
    return GoldenWorkflow(
        case_id="sample",
        task_prompt="grade this trajectory",
        milestones=["FINAL_OUTPUT"],
        max_steps=4,
        max_tokens=100,
        golden_steps=1,
    )


def test_tracer_jsonl_sqlite_and_listener(tmp_path: Path) -> None:
    jsonl = tmp_path / "trace.jsonl"
    database = tmp_path / "trace.db"
    seen: list[TrajectoryStep] = []
    tracer = TrajectoryTracer(jsonl, sqlite_path=database, listener=seen.append)
    first = _step("RETRIEVAL", step_index=0, metadata={"citation": "chunk-auth"})
    second = _step("FINAL_OUTPUT", step_index=1, agent_name="supervisor")
    tracer.record(first)
    tracer.record(second)

    assert [step.step_index for step in tracer.steps] == [0, 1]
    assert [step.step_index for step in seen] == [0, 1]
    lines = [line for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert [TrajectoryStep.model_validate_json(line).step_index for line in lines] == [0, 1]
    loaded = load_steps(database)
    assert [step.step_index for step in loaded] == [0, 1]
    assert loaded[0].metadata["citation"] == "chunk-auth"
    assert loaded[1].agent_name == "supervisor"


def test_milestone_order_tools_and_efficiency() -> None:
    ordered = [
        _step("RETRIEVAL", step_index=0),
        _step("RETRIEVAL", step_index=1),
        _step("FINAL_OUTPUT", step_index=2, agent_name="supervisor"),
    ]
    swapped = [ordered[2], ordered[0], ordered[1]]
    assert milestone_coverage(ordered, ["RETRIEVAL", "RETRIEVAL", "FINAL_OUTPUT"]) == 1.0
    assert milestone_coverage(swapped, ["RETRIEVAL", "RETRIEVAL", "FINAL_OUTPUT"]) == 2 / 3

    both = [
        _step("MCP_TOOL_CALL", metadata={"tool": "read_file"}),
        _step("MCP_TOOL_CALL", step_index=1, metadata={"tool": "write_file_patch"}),
    ]
    precision, recall = tool_precision_recall(
        both,
        ["read_file", "write_file_patch"],
        [],
    )
    assert precision == 1.0
    assert recall == 1.0
    partial_precision, partial_recall = tool_precision_recall(
        both[:1],
        ["read_file", "write_file_patch"],
        [],
    )
    assert partial_precision == 1.0
    assert partial_recall == 0.5
    forbidden, _ = tool_precision_recall(
        [_step("MCP_TOOL_CALL", metadata={"tool": "write_file_patch"})],
        ["read_file"],
        ["write_file_patch"],
    )
    assert forbidden == 0.0
    assert efficiency_ratio(4, 2) == 1.0
    assert efficiency_ratio(2, 4) == 0.5


def test_deterministic_suite_exits_zero() -> None:
    assert run_suite("deterministic") == 0


def test_missed_abstention_exits_one() -> None:
    steps = load_fixture(evals_root() / "golden_workflows" / "abstention_trap.jsonl")
    degraded = [
        step.model_copy(
            update={
                "output_summary": "completed without a citation",
                "metadata": {"status": "COMPLETED"},
            }
        )
        for step in steps
    ]
    code = run_suite("deterministic", trajectories={"abstention_trap": degraded})
    assert code == 1


def test_metric_drop_past_delta_exits_one() -> None:
    steps = load_fixture(evals_root() / "golden_workflows" / "bugfix_sandbox.jsonl")
    stripped = [step for step in steps if step.step_type != "MCP_TOOL_CALL"]
    code = run_suite("deterministic", trajectories={"bugfix_sandbox": stripped})
    assert code == 1


def test_delta_threshold_allows_a_small_drop() -> None:
    baseline = BaselineFile.model_validate_json(
        (evals_root() / "baselines" / "main_baseline.json").read_text(encoding="utf-8")
    )
    slight = baseline.metrics.model_copy(update={"tool_precision": 0.96})
    steep = baseline.metrics.model_copy(update={"tool_precision": 0.94})
    assert compare(slight, baseline, trap_failed=False).passed
    assert not compare(steep, baseline, trap_failed=False).passed


def test_judge_stub_emits_five_rubrics() -> None:
    steps = load_fixture(evals_root() / "golden_workflows" / "memory_recall.jsonl")
    golden = GoldenWorkflow(
        case_id="memory_recall",
        task_prompt="recall the fix",
        expected_agent_sequence=["researcher"],
        milestones=["MEMORY_RECALL", "AGENT_DECISION"],
        max_steps=6,
        max_tokens=300,
        golden_steps=2,
    )
    report = grade_deterministic(golden, steps, score_case(golden, steps))
    assert [item.name for item in report.rubrics] == list(RUBRIC_NAMES)
    assert all(0.0 <= item.score <= 1.0 for item in report.rubrics)
    assert all(item.score == 1.0 for item in report.rubrics)


class _SpyCompleter:
    def __init__(self, tracer: TrajectoryTracer) -> None:
        self.tracer = tracer
        self.assembled_first = False
        self.calls = 0

    async def complete(
        self,
        messages: list[object],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        del model, temperature
        self.calls += 1
        self.assembled_first = any(
            step.step_type == "CONTEXT_ASSEMBLE" for step in self.tracer.steps
        )
        assert messages
        assert "retrieval_relevance_and_grounding" in str(getattr(messages[0], "content", ""))
        rubrics = [
            {"name": name, "score": 0.8, "rationale": "offline spy"} for name in RUBRIC_NAMES
        ]
        return LLMResponse(
            content=json.dumps({"rubrics": rubrics}),
            provider="fake",
            model="fake",
            cached=False,
            cache_key="judge-test",
        )


async def test_live_judge_goes_through_assembler() -> None:
    tracer = TrajectoryTracer()
    completer = _SpyCompleter(tracer)
    report = await grade_live(
        _golden(),
        [_step("FINAL_OUTPUT", agent_name="supervisor", metadata={"status": "COMPLETED"})],
        completer,
        assembler=ContextAssembler(tracer=tracer, agent_name="judge"),
    )
    assert completer.calls == 1
    assert completer.assembled_first
    assert [item.name for item in report.rubrics] == list(RUBRIC_NAMES)
    assert tracer.steps[0].step_type == "CONTEXT_ASSEMBLE"


async def test_health_run_and_sse(tmp_path: Path) -> None:
    app = create_app(workspace=tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        created = await client.post("/v1/runs", json={"objective": "explain the missing module"})
        assert created.status_code == 200
        body = created.json()
        run_id = body["run_id"]
        assert body["status"] == "ABSTAINED"
        streamed = await client.get(f"/v1/runs/{run_id}/events")
        assert streamed.status_code == 200
        assert "text/event-stream" in streamed.headers["content-type"]
        text = streamed.text
        assert "event: step" in text
        assert "event: done" in text
        status = await client.get(f"/v1/runs/{run_id}")
        assert status.status_code == 200
        assert status.json()["step_count"] >= 1
        assert status.json()["status"] == "ABSTAINED"
