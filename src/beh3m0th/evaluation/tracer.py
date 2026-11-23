"""In-memory trajectory capture with optional JSONL and SQLite export."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

from beh3m0th.core.schemas import TrajectoryStep

_SCHEMA = """
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


class TrajectoryTracer:
    """Append-only recorder for structured workflow steps."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        sqlite_path: Path | None = None,
        listener: Callable[[TrajectoryStep], None] | None = None,
    ) -> None:
        self.path = path
        self.sqlite_path = sqlite_path
        self._listener = listener
        self.steps: list[TrajectoryStep] = []

    def record(self, step: TrajectoryStep) -> None:
        self.steps.append(step)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(step.model_dump_json() + "\n")
        if self.sqlite_path is not None:
            _insert_step(self.sqlite_path, step)
        if self._listener is not None:
            self._listener(step)


def load_steps(sqlite_path: Path) -> list[TrajectoryStep]:
    """Read steps in insertion order from the shared trajectory table."""
    if not sqlite_path.is_file():
        return []
    with sqlite3.connect(sqlite_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT step_index, timestamp, agent_name, step_type, input_summary,
                   output_summary, tokens_used, latency_ms, metadata
            FROM trajectory_steps
            ORDER BY id
            """
        ).fetchall()
    loaded: list[TrajectoryStep] = []
    for row in rows:
        loaded.append(
            TrajectoryStep.model_validate(
                {
                    "step_index": row["step_index"],
                    "timestamp": row["timestamp"],
                    "agent_name": row["agent_name"],
                    "step_type": row["step_type"],
                    "input_summary": row["input_summary"],
                    "output_summary": row["output_summary"],
                    "tokens_used": row["tokens_used"],
                    "latency_ms": row["latency_ms"],
                    "metadata": json.loads(row["metadata"]),
                }
            )
        )
    return loaded


def _insert_step(path: Path, step: TrajectoryStep) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(_SCHEMA)
        connection.execute(
            """
            INSERT INTO trajectory_steps (
                step_index, timestamp, agent_name, step_type, input_summary,
                output_summary, tokens_used, latency_ms, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step.step_index,
                step.timestamp.isoformat(),
                step.agent_name,
                step.step_type,
                step.input_summary,
                step.output_summary,
                step.tokens_used,
                step.latency_ms,
                json.dumps(step.metadata, sort_keys=True),
            ),
        )
