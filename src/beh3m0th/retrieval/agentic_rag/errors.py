"""Failures while running the self-correcting retrieval loop."""

from __future__ import annotations


class AgenticRAGError(Exception):
    """The query was empty or the loop was configured with no work to do."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
