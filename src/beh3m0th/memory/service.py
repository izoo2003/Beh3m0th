"""Facade over working, episodic, and semantic memory."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from beh3m0th.core.schemas import TrajectoryStep, utcnow
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.memory.consolidator import MemoryConsolidator
from beh3m0th.memory.episodic import SqliteEpisodicMemoryStore
from beh3m0th.memory.errors import MemoryError, require_scope
from beh3m0th.memory.schemas import RankingPolicy, ScoredMemory
from beh3m0th.memory.semantic import SqliteSemanticMemoryStore
from beh3m0th.memory.working import InMemoryTTLWorkingStore
from beh3m0th.retrieval.hybrid.embedder import Embedder


class ThreeLayerMemory:
    """One SQLite file for the scratchpad snapshot and both long-term layers."""

    def __init__(
        self,
        path: Path,
        embedder: Embedder,
        *,
        tracer: TrajectoryTracer | None = None,
        ranking: RankingPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        ttl_seconds: float = 3600.0,
        agent_name: str = "memory",
    ) -> None:
        if not agent_name.strip():
            raise MemoryError("agent_name must not be empty")
        policy = ranking or RankingPolicy()
        tick = clock or utcnow
        self.path = path
        self.working = InMemoryTTLWorkingStore(path, ttl_seconds=ttl_seconds, clock=tick)
        self.episodic = SqliteEpisodicMemoryStore(path, embedder, policy=policy, clock=tick)
        self.semantic = SqliteSemanticMemoryStore(path, embedder, policy=policy, clock=tick)
        self.consolidator = MemoryConsolidator(
            self.episodic,
            self.semantic,
            tracer=tracer,
            clock=tick,
        )
        self._tracer = tracer
        self._clock = tick
        self._agent_name = agent_name

    async def initialize(self) -> None:
        await self.working.initialize()
        await self.episodic.initialize()
        await self.semantic.initialize()

    async def close(self) -> None:
        await self.semantic.close()
        await self.episodic.close()
        await self.working.close()

    async def __aenter__(self) -> ThreeLayerMemory:
        await self.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def recall(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str,
        session_id: str,
        limit: int = 5,
    ) -> list[ScoredMemory]:
        """Rank episodic and semantic hits together and record a memory-recall step."""
        require_scope(user_id, project_id, session_id)
        if limit < 1:
            raise MemoryError("limit must be positive")
        started = time.perf_counter()
        moment = self._clock()
        episodes = await self.episodic.recall(
            query,
            user_id=user_id,
            project_id=project_id,
            now=moment,
            limit=limit,
            update_access=False,
        )
        facts = await self.semantic.recall(
            query,
            user_id=user_id,
            project_id=project_id,
            now=moment,
            limit=limit,
            update_access=False,
        )
        merged = sorted([*episodes, *facts], key=lambda item: (-item.score, item.id))[:limit]
        await self.episodic.touch(
            [item.id for item in merged if item.layer == "episodic"],
            when=moment,
        )
        await self.semantic.touch(
            [item.id for item in merged if item.layer == "semantic"],
            when=moment,
        )
        recalled = [item.model_copy(update={"last_accessed_at": moment}) for item in merged]
        self._record(query, user_id, project_id, session_id, recalled, started, moment)
        return recalled

    def _record(
        self,
        query: str,
        user_id: str,
        project_id: str,
        session_id: str,
        recalled: list[ScoredMemory],
        started: float,
        moment: datetime,
    ) -> None:
        if self._tracer is None:
            return
        output = " | ".join(item.summary for item in recalled) or "none"
        metadata: dict[str, str | int | float | bool | None] = {
            "user_id": user_id,
            "project_id": project_id,
            "session_id": session_id,
            "count": len(recalled),
        }
        if recalled:
            metadata["top_score"] = recalled[0].score
        self._tracer.record(
            TrajectoryStep(
                step_index=len(self._tracer.steps),
                timestamp=moment,
                agent_name=self._agent_name,
                step_type="MEMORY_RECALL",
                input_summary=query[:500],
                output_summary=output[:500],
                tokens_used=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata=metadata,
            )
        )
