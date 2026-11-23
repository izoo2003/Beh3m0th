"""Run the golden suite and print a regression scorecard."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

from rich.console import Console
from rich.table import Table

from beh3m0th.config import Settings
from beh3m0th.core.llm_gateway import LLMGateway, providers_from_settings
from beh3m0th.core.schemas import TrajectoryStep
from beh3m0th.evaluation.errors import EvalError
from beh3m0th.evaluation.golden import load_fixture, load_goldens, score_case
from beh3m0th.evaluation.judge import grade_deterministic, grade_live
from beh3m0th.evaluation.regression import aggregate, compare
from beh3m0th.evaluation.schemas import CORE_METRICS, BaselineFile, RegressionReport

_SCORECARD_FIELDS: tuple[str, ...] = (*CORE_METRICS, "hallucination_rate")


def evals_root() -> Path:
    """Repository ``evals/`` directory, independent of the process cwd."""
    return Path(__file__).resolve().parents[3] / "evals"


def run_suite(
    mode: str = "deterministic",
    *,
    root: Path | None = None,
    trajectories: Mapping[str, list[TrajectoryStep]] | None = None,
    console: Console | None = None,
) -> int:
    """Score the suite. Returns 0 when the regression gate passes."""
    out = console if console is not None else Console()
    try:
        report = asyncio.run(
            _execute(mode, root if root is not None else evals_root(), trajectories)
        )
    except EvalError as exc:
        out.print(str(exc))
        return 1
    _print_scorecard(out, report)
    return 0 if report.passed else 1


async def _execute(
    mode: str,
    root: Path,
    trajectories: Mapping[str, list[TrajectoryStep]] | None,
) -> RegressionReport:
    if mode not in {"deterministic", "live"}:
        raise EvalError(f"unknown mode {mode}")
    gateway: LLMGateway | None = None
    completer = None
    if mode == "live":
        gateway = _live_gateway()
        completer = gateway
    try:
        goldens = load_goldens(root / "golden_workflows")
        scored = []
        for golden in goldens:
            steps = _trajectory(root, golden.case_id, trajectories)
            metrics = score_case(golden, steps)
            if mode == "deterministic":
                judged = grade_deterministic(golden, steps, metrics)
            else:
                if completer is None:
                    raise EvalError("live mode requires a completer")
                judged = await grade_live(golden, steps, completer)
            scored.append((golden, metrics, judged))
        baseline = BaselineFile.model_validate_json(
            (root / "baselines" / "main_baseline.json").read_text(encoding="utf-8")
        )
        trap_failed = any(
            golden.expected_abstention and not metrics.abstention_correct
            for golden, metrics, _ in scored
        )
        return compare(aggregate(scored), baseline, trap_failed=trap_failed)
    finally:
        if gateway is not None:
            await gateway.aclose()


def _live_gateway() -> LLMGateway:
    settings = Settings.from_env()
    if not any(
        (settings.google_api_key, settings.groq_api_key, settings.openrouter_api_key)
    ):
        raise EvalError(
            "live mode requires GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY"
        )
    return LLMGateway(
        cache_path=settings.llm_cache_path,
        providers=providers_from_settings(settings),
        rpm=settings.llm_max_rpm,
    )


def _trajectory(
    root: Path,
    case_id: str,
    trajectories: Mapping[str, list[TrajectoryStep]] | None,
) -> list[TrajectoryStep]:
    if trajectories is not None and case_id in trajectories:
        return list(trajectories[case_id])
    return load_fixture(root / "golden_workflows" / f"{case_id}.jsonl")


def _print_scorecard(console: Console, report: RegressionReport) -> None:
    table = Table(title="Beh3m0th trajectory scorecard")
    table.add_column("metric")
    table.add_column("current", justify="right")
    table.add_column("baseline", justify="right")
    table.add_column("delta", justify="right")
    for name in _SCORECARD_FIELDS:
        current = float(getattr(report.current, name))
        baseline = float(getattr(report.baseline, name))
        table.add_row(name, f"{current:.3f}", f"{baseline:.3f}", f"{current - baseline:+.3f}")
    console.print(table)
    for failure in report.failures:
        console.print(failure)
    console.print("passed" if report.passed else "failed")
