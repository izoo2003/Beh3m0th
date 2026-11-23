"""Pydantic tool registry. Schemas are exported from the input models."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from beh3m0th.mcp_server.sandbox import SandboxRunner
from beh3m0th.mcp_server.schemas import Scope, ToolDescription


class ToolContext:
    """Dependencies a tool handler may use. Constructed per call."""

    def __init__(
        self,
        *,
        workspace: Path,
        sandbox: SandboxRunner,
        role: str,
        timeout_sec: float,
    ) -> None:
        self.workspace = workspace
        self.sandbox = sandbox
        self.role = role
        self.timeout_sec = timeout_sec


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    scope: Scope
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], Awaitable[BaseModel]]


class ToolRegistry:
    """Insertion-ordered catalog of tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def add(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDescription]:
        return [
            ToolDescription(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_model.model_json_schema(),
            )
            for tool in self._tools.values()
        ]
