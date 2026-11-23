"""Pydantic models for MCP roles, tool payloads, and JSON-RPC results."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from beh3m0th.core.schemas import StrictModel

PROTOCOL_VERSION = "2024-11-05"


class Scope(StrEnum):
    READ_FS = "READ_FS"
    SEARCH_CODE = "SEARCH_CODE"
    WRITE_FS = "WRITE_FS"
    RUN_SANDBOX_CODE = "RUN_SANDBOX_CODE"
    RUN_TESTS = "RUN_TESTS"


class TextContent(StrictModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    content: list[TextContent]
    is_error: bool = Field(default=False, alias="isError")


class ClientInfo(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class InitializeParams(StrictModel):
    protocolVersion: str = Field(default=PROTOCOL_VERSION, min_length=1)
    capabilities: dict[str, Any] | None = None
    clientInfo: ClientInfo | None = None


class ToolsCapability(StrictModel):
    listChanged: bool = False


class ServerCapabilities(StrictModel):
    tools: ToolsCapability = Field(default_factory=ToolsCapability)


class ServerInfo(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class InitializeResult(StrictModel):
    protocolVersion: str
    capabilities: ServerCapabilities
    serverInfo: ServerInfo


class ToolDescription(StrictModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    inputSchema: dict[str, Any]


class ToolListResult(StrictModel):
    tools: list[ToolDescription]


class ToolCallParams(StrictModel):
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    role: str = Field(min_length=1)


class ReadFileInput(StrictModel):
    path: str = Field(min_length=1)


class ReadFileOutput(StrictModel):
    path: str
    content: str


class ListDirectoryInput(StrictModel):
    path: str = "."


class ListDirectoryOutput(StrictModel):
    path: str
    entries: list[str]


class SearchFilesInput(StrictModel):
    pattern: str = Field(min_length=1)
    path: str = "."
    max_matches: int = Field(default=50, ge=1, le=200)


class SearchMatch(StrictModel):
    path: str
    line: int = Field(ge=1)
    text: str


class SearchFilesOutput(StrictModel):
    matches: list[SearchMatch]


class WriteFilePatchInput(StrictModel):
    path: str = Field(min_length=1)
    old_text: str
    new_text: str


class WriteFilePatchOutput(StrictModel):
    path: str
    updated: bool


class RunPythonInput(StrictModel):
    code: str = Field(min_length=1, max_length=20_000)


class RunPytestInput(StrictModel):
    path: str = "."


class SandboxOutput(StrictModel):
    exit_code: int
    stdout: str
    stderr: str
    sandbox_mode: str = Field(min_length=1)
