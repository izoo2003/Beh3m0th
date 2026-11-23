"""JSON-RPC dispatcher for the MCP tool server."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from beh3m0th.core.schemas import MetadataValue, TrajectoryStep
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.mcp_server.errors import JsonRpcError, MCPPermissionError, ToolExecutionError
from beh3m0th.mcp_server.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PERMISSION_DENIED,
    RpcRequest,
    dump_model,
    error_response,
    field_errors,
    parse_request,
    success_response,
)
from beh3m0th.mcp_server.rbac import require_scope
from beh3m0th.mcp_server.registry import ToolContext, ToolRegistry
from beh3m0th.mcp_server.sandbox import SandboxRunner, SubprocessRestrictedSandbox
from beh3m0th.mcp_server.schemas import (
    PROTOCOL_VERSION,
    InitializeParams,
    InitializeResult,
    ServerCapabilities,
    ServerInfo,
    TextContent,
    ToolCallParams,
    ToolCallResult,
    ToolListResult,
)
from beh3m0th.mcp_server.tools import build_default_registry


@dataclass(frozen=True, slots=True)
class _Invocation:
    result: dict[str, Any]
    outcome: str
    sandbox_mode: str | None


class MCPServer:
    """Dispatch ``initialize``, ``tools/list``, ``tools/call``, and ``ping``."""

    def __init__(
        self,
        workspace: Path,
        registry: ToolRegistry,
        sandbox: SandboxRunner,
        tracer: TrajectoryTracer | None = None,
        *,
        timeout_sec: float = 15.0,
        server_name: str = "beh3m0th",
        server_version: str = "0.1.0",
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        self._workspace = workspace.resolve()
        self._registry = registry
        self._sandbox = sandbox
        self._tracer = tracer
        self._timeout_sec = timeout_sec
        self._server_name = server_name
        self._server_version = server_version

    async def handle_raw(self, raw: str) -> dict[str, object] | None:
        """Handle one JSON-RPC message. Notifications return ``None``."""
        try:
            request = parse_request(raw)
        except JsonRpcError as exc:
            if exc.notification:
                return None
            return error_response(exc.request_id, exc.code, exc.message, exc.data)
        try:
            result = await self._dispatch(request)
        except JsonRpcError as exc:
            if request.notification:
                return None
            return error_response(request.id, exc.code, exc.message, exc.data)
        except Exception:
            if request.notification:
                return None
            return error_response(request.id, INTERNAL_ERROR, "Internal error")
        if request.notification:
            return None
        return success_response(request.id, result)

    async def _dispatch(self, request: RpcRequest) -> dict[str, Any]:
        if request.method == "initialize":
            return self._initialize(request.params)
        if request.method == "tools/list":
            _reject_extra_params(request.params)
            return dump_model(ToolListResult(tools=self._registry.list_tools()))
        if request.method == "tools/call":
            return await self._tools_call(request.params)
        if request.method in {"ping", "notifications/initialized"}:
            _reject_extra_params(request.params)
            return {}
        raise JsonRpcError(METHOD_NOT_FOUND, "Method not found")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            InitializeParams.model_validate(params)
        except ValidationError as exc:
            raise JsonRpcError(INVALID_PARAMS, "Invalid params", data=field_errors(exc)) from exc
        result = InitializeResult(
            protocolVersion=PROTOCOL_VERSION,
            capabilities=ServerCapabilities(),
            serverInfo=ServerInfo(name=self._server_name, version=self._server_version),
        )
        return dump_model(result)

    async def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        role = params.get("role")
        agent = role if isinstance(role, str) and role else "mcp"
        name = params.get("name")
        tool_name = name if isinstance(name, str) and name else "unknown"
        try:
            invocation = await self._invoke(params)
        except MCPPermissionError as exc:
            self._record(agent, tool_name, f"denied: {exc}", started, None)
            raise JsonRpcError(PERMISSION_DENIED, str(exc)) from exc
        except JsonRpcError:
            self._record(agent, tool_name, "invalid", started, None)
            raise
        except Exception:
            self._record(agent, tool_name, "internal", started, None)
            raise
        self._record(
            agent,
            tool_name,
            invocation.outcome,
            started,
            invocation.sandbox_mode,
        )
        return invocation.result

    async def _invoke(self, params: dict[str, Any]) -> _Invocation:
        try:
            call = ToolCallParams.model_validate(params)
        except ValidationError as exc:
            raise JsonRpcError(INVALID_PARAMS, "Invalid params", data=field_errors(exc)) from exc
        tool = self._registry.get(call.name)
        if tool is None:
            raise JsonRpcError(
                INVALID_PARAMS,
                "Invalid params",
                data=[{"field": "name", "message": "unknown tool"}],
            )
        try:
            validated = tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            raise JsonRpcError(INVALID_PARAMS, "Invalid params", data=field_errors(exc)) from exc
        require_scope(call.role, tool.scope)
        ctx = ToolContext(
            workspace=self._workspace,
            sandbox=self._sandbox,
            role=call.role,
            timeout_sec=self._timeout_sec,
        )
        try:
            output = await tool.handler(validated, ctx)
        except ToolExecutionError as exc:
            result = dump_model(
                ToolCallResult(content=[TextContent(text=exc.message)], is_error=True)
            )
            return _Invocation(result=result, outcome="error", sandbox_mode=None)
        if not isinstance(output, tool.output_model):
            raise ToolExecutionError("tool returned an invalid result")
        mode = output.sandbox_mode if hasattr(output, "sandbox_mode") else None
        sandbox_mode = mode if isinstance(mode, str) else None
        result = dump_model(
            ToolCallResult(content=[TextContent(text=output.model_dump_json())], is_error=False)
        )
        return _Invocation(result=result, outcome="ok", sandbox_mode=sandbox_mode)

    def _record(
        self,
        agent: str,
        tool: str,
        outcome: str,
        started: float,
        sandbox_mode: str | None,
    ) -> None:
        if self._tracer is None:
            return
        metadata: dict[str, MetadataValue] = {"tool": tool, "outcome": outcome[:200]}
        if sandbox_mode is not None:
            metadata["SANDBOX_MODE"] = sandbox_mode
        self._tracer.record(
            TrajectoryStep(
                step_index=len(self._tracer.steps),
                agent_name=agent,
                step_type="MCP_TOOL_CALL",
                input_summary=tool[:200],
                output_summary=outcome[:200],
                tokens_used=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata=metadata,
            )
        )


def build_server(
    workspace: Path,
    *,
    sandbox: SandboxRunner | None = None,
    tracer: TrajectoryTracer | None = None,
    timeout_sec: float = 15.0,
) -> MCPServer:
    """Offline-friendly server. The default runner never starts Docker."""
    return MCPServer(
        workspace=workspace,
        registry=build_default_registry(),
        sandbox=sandbox if sandbox is not None else SubprocessRestrictedSandbox(),
        tracer=tracer,
        timeout_sec=timeout_sec,
    )


def _reject_extra_params(params: dict[str, Any]) -> None:
    if params:
        raise JsonRpcError(
            INVALID_PARAMS,
            "Invalid params",
            data=[{"field": "params", "message": "unexpected fields"}],
        )
