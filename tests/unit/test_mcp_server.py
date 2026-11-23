"""Offline Phase 5 checks: JSON-RPC, RBAC, the workspace jail, and the sandbox fallback."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import httpx
import pytest

from beh3m0th.config import Settings
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.mcp_server import (
    PERMISSION_DENIED,
    AsgiTransport,
    DockerSandbox,
    InProcessTransport,
    MCPClient,
    SubprocessRestrictedSandbox,
    build_app,
    build_docker_argv,
    build_server,
    sandbox_from_settings,
    stripped_env,
)
from beh3m0th.mcp_server.server import MCPServer
from beh3m0th.mcp_server.stdio import serve_stdio

_MCP_ENV = (
    "MCP_SANDBOX_BACKEND",
    "MCP_SANDBOX_IMAGE",
    "MCP_SANDBOX_TIMEOUT_SEC",
    "MCP_SANDBOX_MEM_LIMIT",
    "MCP_ALLOWED_WORKSPACE",
)
_TOOL_NAMES = {
    "read_file",
    "list_directory",
    "search_files_grep",
    "write_file_patch",
    "run_python_sandbox",
    "run_pytest_sandbox",
}


def _message(method: str, params: dict[str, object] | None = None, *, request_id: int = 1) -> str:
    payload: dict[str, object] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    return json.dumps(payload)


async def _call(
    server: MCPServer,
    method: str,
    params: dict[str, object] | None = None,
    *,
    request_id: int = 1,
) -> dict[str, object]:
    response = await server.handle_raw(_message(method, params, request_id=request_id))
    assert response is not None
    return response


def _body(result: dict[str, object]) -> dict[str, object]:
    content = result["content"]
    assert isinstance(content, list)
    block = content[0]
    assert isinstance(block, dict)
    text = block["text"]
    assert isinstance(text, str)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    return parsed


def _clear_mcp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _MCP_ENV:
        monkeypatch.delenv(name, raising=False)


async def test_researcher_is_rejected_for_write_and_exec(tmp_path: Path) -> None:
    tracer = TrajectoryTracer()
    server = build_server(tmp_path, tracer=tracer)
    write = await _call(
        server,
        "tools/call",
        {
            "name": "write_file_patch",
            "arguments": {"path": "note.txt", "old_text": "", "new_text": "x"},
            "role": "Researcher",
        },
    )
    execute = await _call(
        server,
        "tools/call",
        {
            "name": "run_python_sandbox",
            "arguments": {"code": "print(1)"},
            "role": "Researcher",
        },
    )

    assert write["error"]["code"] == PERMISSION_DENIED
    assert execute["error"]["code"] == PERMISSION_DENIED
    assert not (tmp_path / "note.txt").exists()
    assert len(tracer.steps) == 2
    assert all(step.step_type == "MCP_TOOL_CALL" for step in tracer.steps)
    assert tracer.steps[0].agent_name == "Researcher"
    assert tracer.steps[0].metadata["tool"] == "write_file_patch"
    assert str(tracer.steps[0].metadata["outcome"]).startswith("denied")


async def test_path_traversal_is_blocked(tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("TOP-SECRET-SENTINEL", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    server = build_server(workspace)

    passwd = await _call(
        server,
        "tools/call",
        {
            "name": "read_file",
            "arguments": {"path": "../../etc/passwd"},
            "role": "Coder",
        },
    )
    sibling = await _call(
        server,
        "tools/call",
        {
            "name": "read_file",
            "arguments": {"path": "../secret.txt"},
            "role": "Coder",
        },
    )

    assert passwd["error"]["code"] == PERMISSION_DENIED
    assert sibling["error"]["code"] == PERMISSION_DENIED
    assert "TOP-SECRET-SENTINEL" not in json.dumps(sibling)
    assert outside.read_text(encoding="utf-8") == "TOP-SECRET-SENTINEL"


def _link_outside(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        if os.name != "nt" or not target.is_dir():
            pytest.skip("symlinks are not permitted on this machine")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not link.exists():
            pytest.skip("symlinks are not permitted on this machine")


async def test_symlink_escape_is_blocked(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET-SENTINEL", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _link_outside(workspace / "escape", outside)
    server = build_server(workspace)
    response = await _call(
        server,
        "tools/call",
        {
            "name": "read_file",
            "arguments": {"path": "escape/secret.txt"},
            "role": "Coder",
        },
    )
    assert response["error"]["code"] == PERMISSION_DENIED
    assert "TOP-SECRET-SENTINEL" not in json.dumps(response)


async def test_protocol_errors(tmp_path: Path) -> None:
    server = build_server(tmp_path)
    parse_error = await server.handle_raw("{")
    invalid = await server.handle_raw("[]")
    missing_method = await server.handle_raw('{"jsonrpc":"2.0","id":4}')
    unknown = await _call(server, "nope")
    bad_args = await _call(
        server,
        "tools/call",
        {"name": "read_file", "arguments": {}, "role": "Researcher"},
    )
    notification = await server.handle_raw(
        '{"jsonrpc":"2.0","method":"notifications/initialized"}'
    )

    assert parse_error is not None
    assert parse_error["error"]["code"] == -32700
    assert parse_error["id"] is None
    assert invalid is not None
    assert invalid["error"]["code"] == -32600
    assert missing_method is not None
    assert missing_method["error"]["code"] == -32600
    assert missing_method["id"] == 4
    assert unknown["error"]["code"] == -32601
    assert bad_args["error"]["code"] == -32602
    assert notification is None


async def test_lifecycle_and_coder_file_tools(tmp_path: Path) -> None:
    server = build_server(tmp_path)
    client = MCPClient(InProcessTransport(server))

    initialized = await client.initialize()
    tools = await client.list_tools()
    ping = await client.ping()
    written = await client.call_tool(
        "write_file_patch",
        {"path": "note.txt", "old_text": "", "new_text": "hello"},
        "Coder",
    )
    read = await client.call_tool("read_file", {"path": "note.txt"}, "Coder")
    listed = await client.call_tool("list_directory", {"path": "."}, "Researcher")
    found = await client.call_tool(
        "search_files_grep",
        {"pattern": "hello", "path": "."},
        "Researcher",
    )

    assert initialized["protocolVersion"] == "2024-11-05"
    assert "tools" in initialized["capabilities"]
    assert initialized["serverInfo"]["name"] == "beh3m0th"
    assert ping == {}
    names = {tool["name"] for tool in tools}
    assert names == _TOOL_NAMES
    read_schema = next(tool for tool in tools if tool["name"] == "read_file")
    assert "path" in read_schema["inputSchema"]["properties"]
    assert written["isError"] is False
    assert _body(written)["updated"] is True
    assert _body(read)["content"] == "hello"
    assert "note.txt" in _body(listed)["entries"]
    assert _body(found)["matches"][0]["text"] == "hello"
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"


async def test_python_sandbox_uses_fallback(tmp_path: Path) -> None:
    tracer = TrajectoryTracer()
    server = build_server(tmp_path, tracer=tracer)
    client = MCPClient(InProcessTransport(server))

    result = await client.call_tool("run_python_sandbox", {"code": "print(2)"}, "Coder")

    body = _body(result)
    assert result["isError"] is False
    assert body["stdout"].strip() == "2"
    assert body["exit_code"] == 0
    assert body["sandbox_mode"] == "fallback"
    assert tracer.steps[-1].step_type == "MCP_TOOL_CALL"
    assert tracer.steps[-1].metadata["SANDBOX_MODE"] == "fallback"


async def test_pytest_sandbox_uses_fallback(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    server = build_server(workspace)
    client = MCPClient(InProcessTransport(server))

    result = await client.call_tool(
        "run_pytest_sandbox",
        {"path": "test_ok.py"},
        "Reviewer",
    )

    body = _body(result)
    assert body["exit_code"] == 0, body
    assert body["sandbox_mode"] == "fallback"


def test_docker_argv_is_locked_down(tmp_path: Path) -> None:
    argv = build_docker_argv(
        image="beh3m0th-sandbox:latest",
        workspace=tmp_path,
        command=["python", "-c", "print(1)"],
        memory="256m",
        cpus="0.5",
        pids_limit=64,
        container_name="beh3m0th-sbx-test",
    )
    assert "--network=none" in argv
    assert "--read-only" in argv
    assert "--memory=256m" in argv
    assert "--cpus=0.5" in argv
    assert "--pids-limit=64" in argv
    assert "--tmpfs" in argv
    assert "beh3m0th-sandbox:latest" in argv


async def test_http_rpc_sse_and_client(tmp_path: Path) -> None:
    app = build_app(build_server(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        posted = await http.post(
            "/mcp/rpc",
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        )
        streamed = await http.get("/mcp/sse")
    assert posted.status_code == 200
    assert posted.json()["result"] == {}
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert "event: endpoint" in streamed.text
    assert "event: message" in streamed.text

    async with AsgiTransport(app) as asgi:
        assert await MCPClient(asgi).ping() == {}


async def test_stdio_round_trip(tmp_path: Path) -> None:
    server = build_server(tmp_path)
    reader = asyncio.StreamReader()
    reader.feed_data(b'{"jsonrpc":"2.0","id":7,"method":"ping"}\n')
    reader.feed_eof()
    chunks: list[bytes] = []

    class _Writer:
        def write(self, data: bytes) -> None:
            chunks.append(data)

        async def drain(self) -> None:
            return None

    await serve_stdio(server, reader, _Writer())
    payload = json.loads(b"".join(chunks).decode("utf-8"))
    assert payload["id"] == 7
    assert payload["result"] == {}


def test_settings_select_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_mcp_env(monkeypatch)
    defaults = Settings.from_env(env_file=tmp_path / "missing.env")
    assert defaults.mcp_sandbox_backend == "subprocess_restricted"
    assert defaults.mcp_sandbox_timeout_sec == 15.0
    assert isinstance(sandbox_from_settings(defaults), SubprocessRestrictedSandbox)

    monkeypatch.setenv("MCP_SANDBOX_BACKEND", "docker")
    monkeypatch.setenv("MCP_SANDBOX_IMAGE", "example:latest")
    monkeypatch.setenv("MCP_SANDBOX_TIMEOUT_SEC", "9")
    monkeypatch.setenv("MCP_SANDBOX_MEM_LIMIT", "128m")
    monkeypatch.setenv("MCP_ALLOWED_WORKSPACE", str(tmp_path))
    selected = Settings.from_env(env_file=tmp_path / "missing.env")
    sandbox = sandbox_from_settings(selected)
    assert isinstance(sandbox, DockerSandbox)
    assert sandbox.image == "example:latest"
    assert sandbox.memory == "128m"
    assert selected.mcp_sandbox_timeout_sec == 9.0
    assert selected.mcp_allowed_workspace == tmp_path

    monkeypatch.setenv("MCP_SANDBOX_BACKEND", "kubernetes")
    with pytest.raises(ValueError, match="MCP_SANDBOX_BACKEND"):
        Settings.from_env(env_file=tmp_path / "missing.env")


def test_stripped_env_drops_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--cov")
    cleaned = stripped_env()
    assert "GROQ_API_KEY" not in cleaned
    assert "PYTEST_ADDOPTS" not in cleaned
    assert "PATH" in cleaned
