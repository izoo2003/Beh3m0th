"""Manual supervisor wiring. Exits before any provider call when no free-tier key is set."""

from __future__ import annotations

import sys

from beh3m0th.agents.graph import build_orchestrator
from beh3m0th.config import Settings
from beh3m0th.mcp_server.client import InProcessTransport, MCPClient
from beh3m0th.mcp_server.server import build_server


def main() -> int:
    settings = Settings.from_env()
    if not _has_key(settings):
        print("No free-tier key set. Refusing to call a provider.")
        return 2
    workspace = settings.mcp_allowed_workspace
    workspace.mkdir(parents=True, exist_ok=True)
    client = MCPClient(InProcessTransport(build_server(workspace)))
    build_orchestrator(tools=client, workspace=workspace)
    print("Orchestrator ready. No provider call was made.")
    return 0


def _has_key(settings: Settings) -> bool:
    return bool(settings.google_api_key or settings.groq_api_key or settings.openrouter_api_key)


if __name__ == "__main__":
    sys.exit(main())
