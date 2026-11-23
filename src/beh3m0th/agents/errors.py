"""Failures raised by the typed multi-agent workflow."""

from __future__ import annotations


class AgentError(Exception):
    """A role, ballot, or tool call broke the workflow contract."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
