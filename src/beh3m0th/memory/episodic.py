"""SQLite episodic memory ranked by similarity, importance, and recency."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import aiosqlite

from beh3m0th.core.schemas import utcnow
from beh3m0th.memory.errors import MemoryError, require_scope
from beh3m0th.memory.schemas import (
    EpisodicMemory,
    RankingPolicy,
    ScoredMemory,
    dump_datetime,
    load_datetime,
)
from beh3m0th.memory.scoring import clamp_similarity, composite_score, recency_factor
from beh3m0th.memory.vectors import cosine_scores, pack_vector
from beh3m0th.retrieval.hybrid.embedder import Embedder

_EPISODES = """
CREATE TABLE IF NOT EXISTS episodic_memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    outcome TEXT NOT NULL,
    tags TEXT NOT NULL,
    importance_score REAL NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodic_scope
ON episodic_memories (user_id, project_id)
"""


class EpisodicMemoryProtocol(Protocol):
    """Append-only episodes with composite recall."""

    async def add(self, entry: EpisodicMemory) -> EpisodicMemory: ...

    async def recall(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str,
        now: datetime | None = None,
        limit: int = 5,
        update_access: bool = True,
    ) -> list[ScoredMemory]: ...

    async def touch(self, ids: list[str], *, when: datetime | None = None) -> None: ...


class SqliteEpisodicMemoryStore:
    """Profile A episodes: float32 blobs in SQLite and numpy cosine search."""

    def __init__(
        self,
        path: Path,
        embedder: Embedder,
        *,
        policy: RankingPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = path
        self._embedder = embedder
        self._policy = policy or RankingPolicy()
        self._clock = clock or utcnow
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        if self._db is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.executescript(_EPISODES)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def add(self, entry: EpisodicMemory) -> EpisodicMemory:
        require_scope(entry.user_id, entry.project_id, entry.session_id)
        db = self._connection()
        cursor = await db.execute("SELECT 1 FROM episodic_memories WHERE id = ?", (entry.id,))
        if await cursor.fetchone() is not None:
            raise MemoryError(f"episodic memory already exists: {entry.id}")
        vector = await self._embed(entry.summary)
        now = self._clock()
        stored = entry.model_copy(update={"created_at": now, "last_accessed_at": now})
        await db.execute(
            """
            INSERT INTO episodic_memories (
                id, user_id, project_id, session_id, summary, outcome, tags,
                importance_score, embedding, created_at, last_accessed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stored.id,
                stored.user_id,
                stored.project_id,
                stored.session_id,
                stored.summary,
                stored.outcome,
                json.dumps(stored.tags),
                stored.importance_score,
                pack_vector(vector),
                dump_datetime(stored.created_at),
                dump_datetime(stored.last_accessed_at),
            ),
        )
        await db.commit()
        return stored

    async def recall(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str,
        now: datetime | None = None,
        limit: int = 5,
        update_access: bool = True,
    ) -> list[ScoredMemory]:
        if not query.strip():
            raise MemoryError("recall query must not be empty")
        if limit < 1:
            raise MemoryError("limit must be positive")
        if not user_id.strip() or not project_id.strip():
            raise MemoryError("user_id, project_id, and session_id are required")
        moment = now or self._clock()
        db = self._connection()
        cursor = await db.execute(
            """
            SELECT id, user_id, project_id, session_id, summary, outcome, tags,
                   importance_score, embedding, created_at, last_accessed_at
            FROM episodic_memories
            WHERE user_id = ? AND project_id = ?
            """,
            (user_id, project_id),
        )
        rows = await cursor.fetchall()
        ranked = await self._rank(query, rows, moment)
        ranked = ranked[:limit]
        if update_access and ranked:
            await self.touch([item.id for item in ranked], when=moment)
            ranked = [item.model_copy(update={"last_accessed_at": moment}) for item in ranked]
        return ranked

    async def touch(self, ids: list[str], *, when: datetime | None = None) -> None:
        if not ids:
            return
        moment = dump_datetime(when or self._clock())
        db = self._connection()
        await db.executemany(
            "UPDATE episodic_memories SET last_accessed_at = ? WHERE id = ?",
            [(moment, item_id) for item_id in ids],
        )
        await db.commit()

    async def _rank(
        self,
        query: str,
        rows: list[aiosqlite.Row],
        now: datetime,
    ) -> list[ScoredMemory]:
        if not rows:
            return []
        vector = await self._embed_query(query)
        blobs = [bytes(row["embedding"]) for row in rows]
        similarities = await cosine_scores(vector, blobs)
        if len(similarities) != len(rows):
            raise MemoryError("cosine scoring returned the wrong number of rows")
        scored: list[ScoredMemory] = []
        for row, similarity in zip(rows, similarities, strict=True):
            accessed = load_datetime(str(row["last_accessed_at"]))
            importance = float(row["importance_score"])
            recency = recency_factor(accessed, now, self._policy.decay_lambda)
            score = composite_score(
                similarity=similarity,
                importance=importance,
                recency=recency,
                policy=self._policy,
            )
            tags = cast(list[str], json.loads(str(row["tags"])))
            scored.append(
                ScoredMemory(
                    id=str(row["id"]),
                    user_id=str(row["user_id"]),
                    project_id=str(row["project_id"]),
                    session_id=str(row["session_id"]),
                    layer="episodic",
                    summary=str(row["summary"]),
                    outcome=str(row["outcome"]),
                    tags=tags,
                    importance_score=importance,
                    similarity=clamp_similarity(similarity),
                    recency=recency,
                    score=score,
                    created_at=load_datetime(str(row["created_at"])),
                    last_accessed_at=accessed,
                    supersedes_id=None,
                )
            )
        scored.sort(key=lambda item: (-item.score, item.id))
        return scored

    async def _embed(self, text: str) -> list[float]:
        vectors = await self._embedder.embed_documents([text])
        return self._checked(vectors, expected=1)[0]

    async def _embed_query(self, text: str) -> list[float]:
        vector = await self._embedder.embed_query(text)
        return self._checked([vector], expected=1)[0]

    def _checked(self, vectors: list[list[float]], *, expected: int) -> list[list[float]]:
        if len(vectors) != expected:
            raise MemoryError("embedder returned the wrong number of vectors")
        checked: list[list[float]] = []
        for vector in vectors:
            values = [float(value) for value in vector]
            if len(values) != self._embedder.dimension:
                raise MemoryError("embedder returned a vector with the wrong dimension")
            checked.append(values)
        return checked

    def _connection(self) -> aiosqlite.Connection:
        if self._db is None:
            raise MemoryError("episodic store is not initialized")
        return self._db
