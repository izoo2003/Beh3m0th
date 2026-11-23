"""Failures while splitting source documents into chunks."""

from __future__ import annotations


class ChunkingError(Exception):
    """Source text could not be split into chunks."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
