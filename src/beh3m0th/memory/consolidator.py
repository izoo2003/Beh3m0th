"""Turn a working scratchpad into episodic and semantic memories."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Protocol
from uuid import uuid4

from pydantic import Field, ValidationError

from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.context_engine.models import BudgetPolicy, ContextBucket, ContextItem
from beh3m0th.core.llm_gateway import LLMMessage, LLMResponse
from beh3m0th.core.schemas import StrictModel, TrajectoryStep, utcnow
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.memory.episodic import EpisodicMemoryProtocol
from beh3m0th.memory.errors import MemoryError, require_scope
from beh3m0th.memory.schemas import (
    ConsolidationResult,
    EpisodicMemory,
    SemanticMemory,
    WorkingScratchpad,
)
from beh3m0th.memory.scoring import importance_for
from beh3m0th.memory.semantic import SemanticMemoryProtocol

_BUG_FIX = re.compile(r"^(?:bug_fix|bugfix)\s*:\s*(.+)$", re.IGNORECASE)
_REPO_RULE = re.compile(r"^(?:repo_rule|rule)\s*:\s*(.+)$", re.IGNORECASE)
_INSTRUCTION = (
    "Return JSON with keys episodic and semantic. "
    "Each episodic item has summary, outcome, kind, and verified. "
    "Each semantic item has summary and kind. "
    "kind is bug_fix, conflict_resolution, routine_lookup, or repo_rule."
)


class TextCompleter(Protocol):
    """Subset of LLMGateway.complete used by optional consolidation."""

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class _EpisodeDraft(StrictModel):
    summary: str = Field(min_length=1)
    outcome: str = ""
    kind: str = "bug_fix"
    verified: bool = False


class _FactDraft(StrictModel):
    summary: str = Field(min_length=1)
    kind: str = "repo_rule"


def scratchpad_lines(scratchpad: WorkingScratchpad) -> list[str]:
    """Every non-empty scratchpad field, in the order an agent would read it."""
    raw = [
        scratchpad.objective,
        *scratchpad.plan_steps,
        *scratchpad.diffs,
        *scratchpad.turns,
        *scratchpad.hypotheses,
    ]
    return [line.strip() for line in raw if line.strip()]


def extract_entries(
    scratchpad: WorkingScratchpad,
    *,
    now: datetime,
) -> tuple[list[EpisodicMemory], list[SemanticMemory]]:
    """Tagged ``bug_fix:`` lines become episodes. Tagged ``repo_rule:`` lines become facts."""
    episodic: list[EpisodicMemory] = []
    semantic: list[SemanticMemory] = []
    for line in scratchpad_lines(scratchpad):
        bug = _BUG_FIX.match(line)
        if bug is not None:
            summary = bug.group(1).strip()
            if summary:
                episodic.append(_episode(scratchpad, summary, now, verified=True))
            continue
        rule = _REPO_RULE.match(line)
        if rule is not None:
            summary = rule.group(1).strip()
            if summary:
                semantic.append(_fact(scratchpad, summary, now, kind="repo_rule"))
    return episodic, semantic


class MemoryConsolidator:
    """Zero-token consolidation used unless a caller injects an LLM consolidator."""

    def __init__(
        self,
        episodic: EpisodicMemoryProtocol,
        semantic: SemanticMemoryProtocol,
        *,
        tracer: TrajectoryTracer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._episodic = episodic
        self._semantic = semantic
        self._tracer = tracer
        self._clock = clock or utcnow

    async def consolidate(self, scratchpad: WorkingScratchpad) -> ConsolidationResult:
        require_scope(scratchpad.user_id, scratchpad.project_id, scratchpad.session_id)
        episodic, semantic = extract_entries(scratchpad, now=self._clock())
        return await self._persist(scratchpad, episodic, semantic)

    async def _persist(
        self,
        scratchpad: WorkingScratchpad,
        episodic: list[EpisodicMemory],
        semantic: list[SemanticMemory],
    ) -> ConsolidationResult:
        saved_episodes = [await self._episodic.add(entry) for entry in episodic]
        saved_facts = [await self._semantic.upsert(fact) for fact in semantic]
        self._record(scratchpad, saved_episodes, saved_facts)
        return ConsolidationResult(episodic=saved_episodes, semantic=saved_facts)

    def _record(
        self,
        scratchpad: WorkingScratchpad,
        episodic: list[EpisodicMemory],
        semantic: list[SemanticMemory],
    ) -> None:
        if self._tracer is None:
            return
        self._tracer.record(
            TrajectoryStep(
                step_index=len(self._tracer.steps),
                timestamp=self._clock(),
                agent_name="memory_consolidator",
                step_type="AGENT_DECISION",
                input_summary=scratchpad.objective[:500],
                output_summary=f"episodic={len(episodic)} semantic={len(semantic)}",
                tokens_used=0,
                latency_ms=0.0,
                metadata={
                    "user_id": scratchpad.user_id,
                    "project_id": scratchpad.project_id,
                    "session_id": scratchpad.session_id,
                },
            )
        )


class LLMMemoryConsolidator(MemoryConsolidator):
    """Optional consolidator. The prompt is the assembler's rendered text."""

    def __init__(
        self,
        episodic: EpisodicMemoryProtocol,
        semantic: SemanticMemoryProtocol,
        assembler: ContextAssembler,
        completer: TextCompleter,
        *,
        policy: BudgetPolicy | None = None,
        tracer: TrajectoryTracer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(episodic, semantic, tracer=tracer, clock=clock)
        self._assembler = assembler
        self._completer = completer
        self._policy = policy or BudgetPolicy(
            model_window=4096,
            reserved_output=512,
            safety_margin=256,
        )

    async def consolidate(self, scratchpad: WorkingScratchpad) -> ConsolidationResult:
        require_scope(scratchpad.user_id, scratchpad.project_id, scratchpad.session_id)
        episodic, semantic = await self._from_model(scratchpad)
        return await self._persist(scratchpad, episodic, semantic)

    async def _from_model(
        self,
        scratchpad: WorkingScratchpad,
    ) -> tuple[list[EpisodicMemory], list[SemanticMemory]]:
        assembled = self._assembler.assemble(
            [
                ContextItem(
                    id="consolidate-system",
                    bucket=ContextBucket.SYSTEM_TOOLS,
                    priority=0,
                    content=_INSTRUCTION,
                ),
                ContextItem(
                    id="consolidate-scratchpad",
                    bucket=ContextBucket.WORKING_MEMORY,
                    priority=1,
                    content="\n".join(scratchpad_lines(scratchpad)),
                ),
            ],
            self._policy,
        )
        if not assembled.text.strip():
            raise MemoryError("assembled consolidation prompt was empty")
        response = await self._completer.complete(
            [LLMMessage(role="user", content=assembled.text)]
        )
        return _parse_consolidation(response.content, scratchpad, now=self._clock())


def _episode(
    scratchpad: WorkingScratchpad,
    summary: str,
    now: datetime,
    *,
    verified: bool,
    outcome: str = "",
    kind: str = "bug_fix",
) -> EpisodicMemory:
    resolved_outcome = outcome or ("verified_by_tests" if verified else "")
    return EpisodicMemory(
        id=uuid4().hex,
        user_id=scratchpad.user_id,
        project_id=scratchpad.project_id,
        session_id=scratchpad.session_id,
        summary=summary,
        outcome=resolved_outcome,
        tags=[kind],
        importance_score=importance_for(kind, verified=verified),
        created_at=now,
        last_accessed_at=now,
    )


def _fact(
    scratchpad: WorkingScratchpad,
    summary: str,
    now: datetime,
    *,
    kind: str,
) -> SemanticMemory:
    return SemanticMemory(
        id=uuid4().hex,
        user_id=scratchpad.user_id,
        project_id=scratchpad.project_id,
        session_id=scratchpad.session_id,
        summary=summary,
        tags=[kind],
        importance_score=importance_for(kind),
        created_at=now,
        updated_at=now,
        last_accessed_at=now,
    )


def _parse_consolidation(
    content: str,
    scratchpad: WorkingScratchpad,
    *,
    now: datetime,
) -> tuple[list[EpisodicMemory], list[SemanticMemory]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        raw_episodes = payload["episodic"]
        raw_facts = payload["semantic"]
        if not isinstance(raw_episodes, list) or not isinstance(raw_facts, list):
            raise TypeError("episodic and semantic must be lists")
        episodes = [_EpisodeDraft.model_validate(item) for item in raw_episodes]
        facts = [_FactDraft.model_validate(item) for item in raw_facts]
    except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise MemoryError("consolidator returned invalid JSON") from exc
    episodic = [
        _episode(
            scratchpad,
            item.summary.strip(),
            now,
            verified=item.verified,
            outcome=item.outcome,
            kind=item.kind,
        )
        for item in episodes
        if item.summary.strip()
    ]
    semantic = [
        _fact(scratchpad, item.summary.strip(), now, kind=item.kind)
        for item in facts
        if item.summary.strip()
    ]
    return episodic, semantic
