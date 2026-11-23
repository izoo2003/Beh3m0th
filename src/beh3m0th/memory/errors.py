"""Failures raised by the three-layer memory stores."""

from __future__ import annotations


class MemoryError(Exception):
    """A memory scope, snapshot, or store operation was rejected."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def require_scope(user_id: str, project_id: str, session_id: str) -> None:
    """Reject blank scope keys before they reach SQLite."""
    if not user_id.strip() or not project_id.strip() or not session_id.strip():
        raise MemoryError("user_id, project_id, and session_id are required")
