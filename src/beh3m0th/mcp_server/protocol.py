"""Raw JSON-RPC 2.0 parsing and response envelopes. No MCP SDK."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from beh3m0th.mcp_server.errors import JsonRpcError

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
PERMISSION_DENIED = -32001

_ALLOWED_KEYS = frozenset({"jsonrpc", "method", "params", "id"})


@dataclass(frozen=True, slots=True)
class RpcRequest:
    """One JSON-RPC request object. A missing ``id`` is a notification."""

    id: str | int | None
    method: str
    params: dict[str, Any]
    notification: bool


def parse_request(raw: str) -> RpcRequest:
    """Parse a single JSON-RPC request. Batch arrays are rejected."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JsonRpcError(PARSE_ERROR, "Parse error", data=str(exc)) from exc
    if not isinstance(payload, dict):
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request")
    request_id, notification = _request_id(payload)
    if set(payload) - _ALLOWED_KEYS or payload.get("jsonrpc") != "2.0":
        raise JsonRpcError(
            INVALID_REQUEST,
            "Invalid Request",
            request_id=request_id,
            notification=notification,
        )
    method = payload.get("method")
    if not isinstance(method, str) or method == "":
        raise JsonRpcError(
            INVALID_REQUEST,
            "Invalid Request",
            request_id=request_id,
            notification=notification,
        )
    params = payload.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise JsonRpcError(
            INVALID_PARAMS,
            "Invalid params",
            request_id=request_id,
            notification=notification,
        )
    return RpcRequest(
        id=request_id,
        method=method,
        params=params,
        notification=notification,
    )


def error_response(
    request_id: str | int | None,
    code: int,
    message: str,
    data: object | None = None,
) -> dict[str, object]:
    error: dict[str, object] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def success_response(request_id: str | int | None, result: dict[str, Any]) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def dump_model(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True)


def field_errors(exc: ValidationError) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item["loc"])
        errors.append({"field": location or "params", "message": str(item["msg"])})
    return errors


def _request_id(payload: dict[str, Any]) -> tuple[str | int | None, bool]:
    if "id" not in payload:
        return None, True
    value = payload["id"]
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request")
    return value, False
