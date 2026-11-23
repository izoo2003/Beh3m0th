"""Offline Phase 6 checks: scratchpad TTL, decay ranking, dedup, and cross-session recall."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.core.llm_gateway import LLMResponse
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.memory import (
    EpisodicMemory,
    InMemoryTTLWorkingStore,
    LLMMemoryConsolidator,
    RedisWorkingStore,
    SemanticMemory,
    SqliteEpisodicMemoryStore,
    SqliteSemanticMemoryStore,
    ThreeLayerMemory,
    WorkingScratchpad,
    importance_for,
)
from beh3m0th.memory.errors import MemoryError

BUG = "authentication retries failed because the pool exhausted"
RULE = "Beh3m0th uses SQLAlchemy 2.0 async and prohibits sync DB calls"
QUERY = "what did we learn about auth and database access"
OTHER = "unrelated project convention"
FRESH = "fresh fix"
STALE = "stale lookup"
RANK_QUERY = "rank episodes"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 23, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


class MappedEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self.dimension = len(next(iter(vectors.values())))

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vectors[text]) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return list(self._vectors[text])


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int) -> None:
        self.values[key] = value

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)


class FakeCompleter:
    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    async def complete(
        self,
        messages: list[object],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        del model, temperature
        prompt = getattr(messages[0], "content")
        self.prompts.append(str(prompt))
        return LLMResponse(
            content=self.content,
            provider="fake",
            model="fake",
            cached=False,
            cache_key="memory-test",
        )


def _episode(summary: str, importance: float, *, project_id: str = "proj") -> EpisodicMemory:
    return EpisodicMemory(
        id=uuid4().hex,
        user_id="user",
        project_id=project_id,
        session_id="session-1",
        summary=summary,
        importance_score=importance,
    )


def _fact(summary: str, *, project_id: str = "proj") -> SemanticMemory:
    return SemanticMemory(
        id=uuid4().hex,
        user_id="user",
        project_id=project_id,
        session_id="session-1",
        summary=summary,
        importance_score=0.50,
    )


def _scratchpad() -> WorkingScratchpad:
    return WorkingScratchpad(
        user_id="user",
        project_id="proj",
        session_id="session-1",
        objective="Ship the auth fix",
        plan_steps=["check the pool"],
        diffs=["pool.py: release the connection after retry"],
        turns=[f"bug_fix: {BUG}", f"repo_rule: {RULE}"],
        hypotheses=["the pool never returns to idle"],
    )


def test_importance_heuristics() -> None:
    assert importance_for("conflict_resolution") == 0.90
    assert importance_for("bug_fix", verified=True) == 0.85
    assert importance_for("bug_fix") == 0.50
    assert importance_for("routine_lookup") == 0.35
    assert importance_for("repo_rule") == 0.50


async def test_working_scratchpad_reloads_until_ttl(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "memory.db"
    scratchpad = _scratchpad()
    store = InMemoryTTLWorkingStore(path, ttl_seconds=60, clock=clock)
    await store.initialize()
    try:
        await store.put(scratchpad)
        loaded = await store.get("user", "proj", "session-1")
        assert loaded is not None
        assert loaded.objective == "Ship the auth fix"
        assert loaded.plan_steps == ["check the pool"]
        assert loaded.diffs == ["pool.py: release the connection after retry"]
        assert loaded.hypotheses == ["the pool never returns to idle"]
        assert loaded.updated_at == clock.now
    finally:
        await store.close()

    reopened = InMemoryTTLWorkingStore(path, ttl_seconds=60, clock=clock)
    await reopened.initialize()
    try:
        restored = await reopened.get("user", "proj", "session-1")
        assert restored is not None
        assert restored.turns == scratchpad.turns
        clock.advance(seconds=61)
        assert await reopened.get("user", "proj", "session-1") is None
    finally:
        await reopened.close()


async def test_redis_working_store_honors_ttl() -> None:
    clock = Clock()
    client = FakeRedis()
    store = RedisWorkingStore(client, ttl_seconds=30, clock=clock)
    await store.put(_scratchpad())
    loaded = await store.get("user", "proj", "session-1")
    assert loaded is not None
    assert loaded.objective == "Ship the auth fix"
    clock.advance(seconds=31)
    assert await store.get("user", "proj", "session-1") is None
    assert client.values == {}


async def test_episodic_rank_prefers_important_recent_episode(tmp_path: Path) -> None:
    clock = Clock()
    embedder = MappedEmbedder(
        {
            FRESH: [1.0, 0.0, 0.0],
            STALE: [1.0, 0.0, 0.0],
            RANK_QUERY: [1.0, 0.0, 0.0],
        }
    )
    store = SqliteEpisodicMemoryStore(tmp_path / "memory.db", embedder, clock=clock)
    await store.initialize()
    try:
        await store.add(_episode(STALE, 0.35))
        clock.advance(hours=1000)
        await store.add(_episode(FRESH, 0.85))
        ranked = await store.recall(RANK_QUERY, user_id="user", project_id="proj")
    finally:
        await store.close()

    assert [item.summary for item in ranked] == [FRESH, STALE]
    assert ranked[0].score > ranked[1].score
    assert ranked[0].similarity == pytest.approx(1.0)
    assert ranked[1].similarity == pytest.approx(1.0)
    assert ranked[0].recency == pytest.approx(1.0)
    assert ranked[1].recency < 0.01
    assert ranked[0].importance_score == 0.85
    assert ranked[1].importance_score == 0.35


async def test_semantic_dedup_supersedes_similar_fact(tmp_path: Path) -> None:
    embedder = MappedEmbedder(
        {
            "Use sync engine calls": [1.0, 0.0, 0.0],
            "Use async engine calls": [0.99, 0.01, 0.0],
            "Format with ruff": [0.0, 1.0, 0.0],
            "which rules": [1.0, 1.0, 0.0],
        }
    )
    store = SqliteSemanticMemoryStore(tmp_path / "memory.db", embedder)
    await store.initialize()
    try:
        old = await store.upsert(_fact("Use sync engine calls"))
        new = await store.upsert(_fact("Use async engine calls"))
        other = await store.upsert(_fact("Format with ruff"))
        recalled = await store.recall("which rules", user_id="user", project_id="proj", limit=10)
    finally:
        await store.close()

    assert new.supersedes_id == old.id
    assert other.supersedes_id is None
    recalled_ids = {item.id for item in recalled}
    assert old.id not in recalled_ids
    assert {new.id, other.id} <= recalled_ids


async def test_session_two_recalls_consolidated_memories_by_score(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    embedder = MappedEmbedder(
        {
            BUG: [1.0, 0.0, 0.0],
            RULE: [0.0, 1.0, 0.0],
            OTHER: [1.0, 1.0, 0.0],
            QUERY: [1.0, 1.0, 0.0],
        }
    )
    tracer = TrajectoryTracer()
    async with ThreeLayerMemory(path, embedder, tracer=tracer) as memory:
        scratchpad = _scratchpad()
        await memory.working.put(scratchpad)
        consolidated = await memory.consolidator.consolidate(scratchpad)
        await memory.episodic.add(_episode(OTHER, 0.95, project_id="other-proj"))

    assert [item.summary for item in consolidated.episodic] == [BUG]
    assert consolidated.episodic[0].importance_score == 0.85
    assert consolidated.episodic[0].outcome == "verified_by_tests"
    assert [item.summary for item in consolidated.semantic] == [RULE]

    async with ThreeLayerMemory(path, embedder, tracer=tracer) as memory:
        ranked = await memory.recall(
            QUERY,
            user_id="user",
            project_id="proj",
            session_id="session-2",
        )

    summaries = [item.summary for item in ranked]
    assert BUG in summaries
    assert RULE in summaries
    assert OTHER not in summaries
    assert summaries == [BUG, RULE]
    assert [item.score for item in ranked] == sorted((item.score for item in ranked), reverse=True)
    assert ranked[0].score > ranked[1].score
    assert {item.project_id for item in ranked} == {"proj"}
    assert [step.step_type for step in tracer.steps] == ["AGENT_DECISION", "MEMORY_RECALL"]
    assert tracer.steps[0].agent_name == "memory_consolidator"
    assert tracer.steps[1].metadata["session_id"] == "session-2"
    assert tracer.steps[1].metadata["count"] == 2


async def test_llm_consolidator_persists_model_json(tmp_path: Path) -> None:
    payload = {
        "episodic": [
            {
                "summary": BUG,
                "outcome": "verified_by_tests",
                "kind": "bug_fix",
                "verified": True,
            }
        ],
        "semantic": [{"summary": RULE, "kind": "repo_rule"}],
    }
    embedder = MappedEmbedder({BUG: [1.0, 0.0, 0.0], RULE: [0.0, 1.0, 0.0]})
    completer = FakeCompleter(json.dumps(payload))
    async with ThreeLayerMemory(tmp_path / "memory.db", embedder) as memory:
        consolidator = LLMMemoryConsolidator(
            memory.episodic,
            memory.semantic,
            ContextAssembler(),
            completer,
        )
        result = await consolidator.consolidate(_scratchpad())

    assert completer.prompts
    assert BUG in completer.prompts[0]
    assert result.episodic[0].importance_score == 0.85
    assert result.semantic[0].summary == RULE
    assert result.semantic[0].importance_score == 0.50


async def test_recall_rejects_an_empty_query(tmp_path: Path) -> None:
    embedder = MappedEmbedder({QUERY: [1.0, 0.0, 0.0]})
    async with ThreeLayerMemory(tmp_path / "memory.db", embedder) as memory:
        with pytest.raises(MemoryError, match="query"):
            await memory.recall("  ", user_id="user", project_id="proj", session_id="session-2")
