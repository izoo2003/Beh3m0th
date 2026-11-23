"""Home, context-budget, and scorecard demos. No model calls."""

from __future__ import annotations

import asyncio
import io

from rich.console import Console

from beh3m0th.config import Settings
from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.context_engine.models import BudgetPolicy, ContextBucket, ContextItem
from beh3m0th.evaluation.regression import CORE_METRICS
from beh3m0th.evaluation.suite import _execute, evals_root, run_suite
from beh3m0th.ui.models import DemoResult

_PHASES: tuple[tuple[str, str], ...] = (
    ("0", "LLM gateway and local storage"),
    ("1", "Context budget"),
    ("2", "Hybrid search"),
    ("3", "GraphRAG"),
    ("4", "Grounded answers"),
    ("5", "Tool permissions"),
    ("6", "Three-layer memory"),
    ("7", "Researcher, Coder, Reviewer"),
    ("8", "Trajectory scorecard"),
)


def home_status() -> DemoResult:
    """Phase list plus whether a free-tier key is configured. The key itself stays hidden."""
    settings = Settings.from_env()
    present = bool(settings.google_api_key or settings.groq_api_key or settings.openrouter_api_key)
    return DemoResult(
        title="Beh3m0th",
        summary="Offline demos are ready. Live model calls stay off.",
        rows=[
            {"phase": number, "what": name, "mode": "offline"} for number, name in _PHASES
        ]
        + [
            {
                "phase": "keys",
                "what": "Free-tier API key",
                "mode": "present" if present else "missing",
            },
            {"phase": "profile", "what": "Storage profile", "mode": settings.profile},
        ],
    )


def assemble_context(
    system: str,
    notes: str,
    evidence: str,
    *,
    model_window: int = 4096,
    reserved_output: int = 512,
    safety_margin: int = 256,
) -> DemoResult:
    """Assemble three blocks. A mandatory block that does not fit raises."""
    assembled = ContextAssembler(agent_name="ui").assemble(
        [
            ContextItem(
                id="system",
                bucket=ContextBucket.SYSTEM_TOOLS,
                priority=0,
                content=system,
            ),
            ContextItem(
                id="notes",
                bucket=ContextBucket.WORKING_MEMORY,
                priority=1,
                content=notes,
                truncatable=True,
            ),
            ContextItem(
                id="evidence",
                bucket=ContextBucket.RETRIEVAL,
                priority=2,
                content=evidence,
                truncatable=True,
            ),
        ],
        BudgetPolicy(
            model_window=model_window,
            reserved_output=reserved_output,
            safety_margin=safety_margin,
        ),
    )
    report = assembled.report
    rows = [{"item": item, "state": "kept"} for item in report.items_included]
    rows.extend({"item": item, "state": "dropped"} for item in report.items_dropped)
    rows.append({"item": "tokens_used", "state": str(report.tokens_used)})
    rows.append({"item": "budget", "state": str(report.budget_available)})
    return DemoResult(
        title="Context assembly",
        summary=f"{report.tokens_used} of {report.budget_available} tokens",
        rows=rows,
        detail=assembled.text,
    )


def scorecard() -> DemoResult:
    """Run the deterministic golden suite and return the metric table."""
    code = run_suite(
        "deterministic",
        console=Console(file=io.StringIO(), force_terminal=False, no_color=True),
    )
    report = asyncio.run(_execute("deterministic", evals_root(), None))
    names = (*CORE_METRICS, "hallucination_rate")
    rows = [
        {"metric": name, "value": f"{float(getattr(report.current, name)):.3f}"} for name in names
    ]
    passed = code == 0 and report.passed
    return DemoResult(
        title="Trajectory scorecard",
        summary="passed" if passed else "failed",
        rows=rows,
        detail="" if passed else "\n".join(report.failures),
    )
