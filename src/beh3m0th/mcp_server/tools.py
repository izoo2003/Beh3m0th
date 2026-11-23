"""Starter MCP tools. Filesystem access stays inside the workspace jail."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from beh3m0th.mcp_server.errors import ToolExecutionError
from beh3m0th.mcp_server.paths import resolve_inside
from beh3m0th.mcp_server.registry import ToolContext, ToolDefinition, ToolRegistry
from beh3m0th.mcp_server.sandbox import SandboxResult
from beh3m0th.mcp_server.schemas import (
    ListDirectoryInput,
    ListDirectoryOutput,
    ReadFileInput,
    ReadFileOutput,
    RunPytestInput,
    RunPythonInput,
    SandboxOutput,
    Scope,
    SearchFilesInput,
    SearchFilesOutput,
    SearchMatch,
    WriteFilePatchInput,
    WriteFilePatchOutput,
)

_MAX_READ_BYTES = 1_000_000


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.add(
        ToolDefinition(
            name="read_file",
            description="Read a UTF-8 text file inside the workspace jail.",
            scope=Scope.READ_FS,
            input_model=ReadFileInput,
            output_model=ReadFileOutput,
            handler=read_file,
        )
    )
    registry.add(
        ToolDefinition(
            name="list_directory",
            description="List the names in a directory inside the workspace jail.",
            scope=Scope.READ_FS,
            input_model=ListDirectoryInput,
            output_model=ListDirectoryOutput,
            handler=list_directory,
        )
    )
    registry.add(
        ToolDefinition(
            name="search_files_grep",
            description="Search file text with a Python regular expression. No shell.",
            scope=Scope.SEARCH_CODE,
            input_model=SearchFilesInput,
            output_model=SearchFilesOutput,
            handler=search_files_grep,
        )
    )
    registry.add(
        ToolDefinition(
            name="write_file_patch",
            description="Replace the first old_text match, or create a file when it is empty.",
            scope=Scope.WRITE_FS,
            input_model=WriteFilePatchInput,
            output_model=WriteFilePatchOutput,
            handler=write_file_patch,
        )
    )
    registry.add(
        ToolDefinition(
            name="run_python_sandbox",
            description="Run Python in the sandbox. The host never uses exec().",
            scope=Scope.RUN_SANDBOX_CODE,
            input_model=RunPythonInput,
            output_model=SandboxOutput,
            handler=run_python_sandbox,
        )
    )
    registry.add(
        ToolDefinition(
            name="run_pytest_sandbox",
            description="Run pytest on a path inside the workspace jail.",
            scope=Scope.RUN_TESTS,
            input_model=RunPytestInput,
            output_model=SandboxOutput,
            handler=run_pytest_sandbox,
        )
    )
    return registry


async def read_file(params: ReadFileInput, ctx: ToolContext) -> ReadFileOutput:
    path = resolve_inside(ctx.workspace, params.path)
    content = await asyncio.to_thread(_read_text, path)
    return ReadFileOutput(path=_relative(ctx.workspace, path), content=content)


async def list_directory(params: ListDirectoryInput, ctx: ToolContext) -> ListDirectoryOutput:
    path = resolve_inside(ctx.workspace, params.path)
    entries = await asyncio.to_thread(_list_names, path)
    return ListDirectoryOutput(path=_relative(ctx.workspace, path), entries=entries)


async def search_files_grep(params: SearchFilesInput, ctx: ToolContext) -> SearchFilesOutput:
    try:
        pattern = re.compile(params.pattern)
    except re.error as exc:
        raise ToolExecutionError(f"invalid pattern: {exc}") from exc
    root = resolve_inside(ctx.workspace, params.path)
    matches = await asyncio.to_thread(
        _search,
        root,
        ctx.workspace,
        pattern,
        params.max_matches,
    )
    return SearchFilesOutput(matches=matches)


async def write_file_patch(params: WriteFilePatchInput, ctx: ToolContext) -> WriteFilePatchOutput:
    path = resolve_inside(ctx.workspace, params.path)
    await asyncio.to_thread(_apply_patch, path, params.old_text, params.new_text)
    return WriteFilePatchOutput(path=_relative(ctx.workspace, path), updated=True)


async def run_python_sandbox(params: RunPythonInput, ctx: ToolContext) -> SandboxOutput:
    result = await ctx.sandbox.run_python(
        params.code,
        workspace=ctx.workspace,
        timeout_sec=ctx.timeout_sec,
    )
    return _sandbox_output(result)


async def run_pytest_sandbox(params: RunPytestInput, ctx: ToolContext) -> SandboxOutput:
    target = resolve_inside(ctx.workspace, params.path)
    result = await ctx.sandbox.run_pytest(
        target,
        workspace=ctx.workspace,
        timeout_sec=ctx.timeout_sec,
    )
    return _sandbox_output(result)


def _sandbox_output(result: SandboxResult) -> SandboxOutput:
    return SandboxOutput(
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        sandbox_mode=result.sandbox_mode,
    )


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ToolExecutionError("file not found")
    size = path.stat().st_size
    if size > _MAX_READ_BYTES:
        raise ToolExecutionError("file exceeds the read limit")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolExecutionError("file is not valid UTF-8") from exc


def _list_names(path: Path) -> list[str]:
    if not path.is_dir():
        raise ToolExecutionError("directory not found")
    return sorted(entry.name for entry in path.iterdir())


def _apply_patch(path: Path, old_text: str, new_text: str) -> None:
    if path.exists() and path.is_dir():
        raise ToolExecutionError("path is a directory")
    if not path.parent.is_dir():
        raise ToolExecutionError("parent directory does not exist")
    if not path.exists():
        if old_text != "":
            raise ToolExecutionError("file not found")
        path.write_text(new_text, encoding="utf-8")
        return
    if old_text == "":
        raise ToolExecutionError("old_text is empty")
    current = path.read_text(encoding="utf-8")
    if old_text not in current:
        raise ToolExecutionError("old_text not found")
    path.write_text(current.replace(old_text, new_text, 1), encoding="utf-8")


def _search(root: Path, workspace: Path, pattern: re.Pattern[str], limit: int) -> list[SearchMatch]:
    if not root.exists():
        raise ToolExecutionError("path not found")
    files = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
    matches: list[SearchMatch] = []
    jail = workspace.resolve()
    for file_path in files:
        resolved = file_path.resolve()
        if not resolved.is_relative_to(jail):
            continue
        if resolved.stat().st_size > _MAX_READ_BYTES:
            continue
        try:
            text = resolved.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line) is None:
                continue
            matches.append(
                SearchMatch(
                    path=_relative(workspace, resolved),
                    line=line_number,
                    text=line[:500],
                )
            )
            if len(matches) >= limit:
                return matches
    return matches


def _relative(workspace: Path, path: Path) -> str:
    relative = path.resolve().relative_to(workspace.resolve())
    rendered = relative.as_posix()
    return rendered or "."
