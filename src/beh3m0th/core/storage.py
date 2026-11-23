"""Profile A storage: SQLite via aiosqlite and Qdrant in local disk mode."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    summary TEXT NOT NULL,
    outcome TEXT NOT NULL,
    tags TEXT NOT NULL,
    importance_score REAL NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    supersedes_id TEXT
);
CREATE TABLE IF NOT EXISTS trajectory_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_index INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    step_type TEXT NOT NULL,
    input_summary TEXT NOT NULL,
    output_summary TEXT NOT NULL,
    tokens_used INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    metadata TEXT NOT NULL
);
"""


class RelationalStore(Protocol):
    path: Path

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...


class VectorStore(Protocol):
    path: Path

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...


class SqliteStore:
    """Async SQLite database for cache rows, memories, and trajectory steps."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        if self._db is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def table_names(self) -> set[str]:
        if self._db is None:
            raise RuntimeError("SqliteStore is not initialized")
        cursor = await self._db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        rows = await cursor.fetchall()
        return {str(row[0]) for row in rows}

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


class QdrantLocalStore:
    """Qdrant embedded at a local directory. Does not download embedding models."""

    def __init__(
        self,
        path: Path,
        *,
        collection: str = "beh3m0th_chunks",
        vector_size: int = 384,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        self.path = path
        self.collection = collection
        self.vector_size = vector_size
        self._client: object | None = None

    async def initialize(self) -> None:
        await asyncio.to_thread(self._open)

    async def collection_names(self) -> list[str]:
        if self._client is None:
            raise RuntimeError("QdrantLocalStore is not initialized")
        return await asyncio.to_thread(self._collection_names)

    async def close(self) -> None:
        await asyncio.to_thread(self._close)

    def _open(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.path.mkdir(parents=True, exist_ok=True)
        if self._client is None:
            self._client = QdrantClient(path=str(self.path))
        client = self._client
        if not isinstance(client, QdrantClient):
            raise RuntimeError("Qdrant client was not created")
        existing = {item.name for item in client.get_collections().collections}
        if self.collection not in existing:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _collection_names(self) -> list[str]:
        from qdrant_client import QdrantClient

        client = self._client
        if not isinstance(client, QdrantClient):
            raise RuntimeError("QdrantLocalStore is not initialized")
        return [item.name for item in client.get_collections().collections]

    def _close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and hasattr(client, "close"):
            client.close()
