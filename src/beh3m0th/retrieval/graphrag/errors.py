"""Failures while extracting or querying the knowledge graph."""

from __future__ import annotations


class GraphExtractionError(Exception):
    """Source text could not be turned into entities and relationships."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class GraphQueryError(Exception):
    """A graph query was empty or asked for an unsupported hop count."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
