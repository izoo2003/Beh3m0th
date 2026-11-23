"""Working-memory scratchpad with a SQLite snapshot and a Redis-shaped adapter."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

import aiosqlite

from beh3m0th.core.schemas import utcnow
from beh3m0th.memory.errors import MemoryError, require_scope
from beh3m0th.memory.schemas import WorkingScratchpad, dump_datetime

_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS working_snapshots (
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, project_id, session_id)
)
"""


class WorkingMemoryProtocol(Protocol):
    """Active scratchpad for one user, project, and session."""

    async def put(self, scratchpad: WorkingScratchpad) -> None: ...

    async def get(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
    ) -> WorkingScratchpad | None: ...

    async def clear(self, user_id: str, project_id: str, session_id: str) -> None: ...


class WorkingKeyValue(Protocol):
    """Minimal Redis-shaped client. Profile B injects a real client later."""

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ex: int) -> None: ...

    async def delete(self, key: str) -> None: ...


def _scope_key(user_id: str, project_id: str, session_id: str) -> str:
    return f"beh3m0th:working:{user_id}:{project_id}:{session_id}"


def _expired(scratchpad: WorkingScratchpad, now: datetime, ttl_seconds: float) -> bool:
    age = (now - scratchpad.updated_at).total_seconds()
    return age > ttl_seconds


class InMemoryTTLWorkingStore:
    """RAM scratchpad that snapshots each write to SQLite and drops expired rows."""

    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise MemoryError("ttl_seconds must be positive")
        self.path = path
        self.ttl_seconds = ttl_seconds
        self._clock = clock or utcnow
        self._db: aiosqlite.Connection | None = None
        self._items: dict[tuple[str, str, str], WorkingScratchpad] = {}

    async def initialize(self) -> None:
        if self._db is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute(_SNAPSHOTS)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
        self._items.clear()

    async def put(self, scratchpad: WorkingScratchpad) -> None:
        require_scope(scratchpad.user_id, scratchpad.project_id, scratchpad.session_id)
        stored = scratchpad.model_copy(update={"updated_at": self._clock()})
        key = (stored.user_id, stored.project_id, stored.session_id)
        self._items[key] = stored
        db = self._connection()
        await db.execute(
            """
            INSERT INTO working_snapshots (user_id, project_id, session_id, payload, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (user_id, project_id, session_id)
            DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at
            """,
            (
                stored.user_id,
                stored.project_id,
                stored.session_id,
                stored.model_dump_json(),
                dump_datetime(stored.updated_at),
            ),
        )
        await db.commit()

    async def get(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
    ) -> WorkingScratchpad | None:
        require_scope(user_id, project_id, session_id)
        key = (user_id, project_id, session_id)
        scratchpad = self._items.get(key)
        if scratchpad is None:
            scratchpad = await self._load(key)
            if scratchpad is not None:
                self._items[key] = scratchpad
        if scratchpad is None:
            return None
        if _expired(scratchpad, self._clock(), self.ttl_seconds):
            await self.clear(user_id, project_id, session_id)
            return None
        return scratchpad

    async def clear(self, user_id: str, project_id: str, session_id: str) -> None:
        require_scope(user_id, project_id, session_id)
        self._items.pop((user_id, project_id, session_id), None)
        db = self._connection()
        await db.execute(
            """
            DELETE FROM working_snapshots
            WHERE user_id = ? AND project_id = ? AND session_id = ?
            """,
            (user_id, project_id, session_id),
        )
        await db.commit()

    async def _load(self, key: tuple[str, str, str]) -> WorkingScratchpad | None:
        db = self._connection()
        cursor = await db.execute(
            """
            SELECT payload FROM working_snapshots
            WHERE user_id = ? AND project_id = ? AND session_id = ?
            """,
            key,
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return WorkingScratchpad.model_validate_json(str(row["payload"]))

    def _connection(self) -> aiosqlite.Connection:
        if self._db is None:
            raise MemoryError("working store is not initialized")
        return self._db


class RedisWorkingStore:
    """Profile B scratchpad. The caller injects the key-value client."""

    def __init__(
        self,
        client: WorkingKeyValue,
        *,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise MemoryError("ttl_seconds must be positive")
        self._client = client
        self.ttl_seconds = ttl_seconds
        self._clock = clock or utcnow

    async def put(self, scratchpad: WorkingScratchpad) -> None:
        require_scope(scratchpad.user_id, scratchpad.project_id, scratchpad.session_id)
        stored = scratchpad.model_copy(update={"updated_at": self._clock()})
        key = _scope_key(stored.user_id, stored.project_id, stored.session_id)
        ttl = max(1, math.ceil(self.ttl_seconds))
        await self._client.set(key, stored.model_dump_json(), ex=ttl)

    async def get(
        self,
        user_id: str,
        project_id: str,
        session_id: str,
    ) -> WorkingScratchpad | None:
        require_scope(user_id, project_id, session_id)
        key = _scope_key(user_id, project_id, session_id)
        payload = await self._client.get(key)
        if payload is None:
            return None
        scratchpad = WorkingScratchpad.model_validate_json(payload)
        if _expired(scratchpad, self._clock(), self.ttl_seconds):
            await self._client.delete(key)
            return None
        return scratchpad

    async def clear(self, user_id: str, project_id: str, session_id: str) -> None:
        require_scope(user_id, project_id, session_id)
        await self._client.delete(_scope_key(user_id, project_id, session_id))
