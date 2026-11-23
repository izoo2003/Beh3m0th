"""Coder: apply cited edits through MCP and run sandbox pytest."""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from pydantic import Field

from beh3m0th.agents.errors import AgentError
from beh3m0th.agents.prompts import TextCompleter, complete_text, parse_json
from beh3m0th.agents.researcher import ToolClient
from beh3m0th.agents.state import (
    AgentOutput,
    CodeProposal,
    FileEdit,
    WorkflowState,
    mark_done,
)
from beh3m0th.agents.tracing import record_step
from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.context_engine.models import ContextBucket, ContextItem
from beh3m0th.core.schemas import StrictModel
from beh3m0th.evaluation.tracer import TrajectoryTracer

_EDIT_INSTRUCTION = (
    "Return JSON with key edits. Each edit has path, old_text, and new_text. "
    "old_text must be copied from the cited source."
)


class _EditDraft(StrictModel):
    edits: list[FileEdit] = Field(min_length=1)


class CoderAgent:
    """Patch the workspace and record the sandbox pytest result."""

    def __init__(
        self,
        tools: ToolClient,
        *,
        completer: TextCompleter | None = None,
        assembler: ContextAssembler | None = None,
        tracer: TrajectoryTracer | None = None,
    ) -> None:
        self._tools = tools
        self._completer = completer
        self._assembler = assembler or ContextAssembler(tracer=tracer, agent_name="coder")
        self._tracer = tracer

    async def run(self, state: WorkflowState) -> AgentOutput:
        started = time.perf_counter()
        edits = await self._edits(state)
        for edit in edits:
            await _call(
                self._tools,
                "write_file_patch",
                {"path": edit.path, "old_text": edit.old_text, "new_text": edit.new_text},
            )
        result = await _call(self._tools, "run_pytest_sandbox", {"path": "."})
        proposal = CodeProposal(
            proposal_id=uuid4().hex,
            edits=edits,
            diff=_render_diff(edits),
            pytest_exit_code=int(result["exit_code"]),
            pytest_stdout=str(result.get("stdout", "")),
            pytest_stderr=str(result.get("stderr", "")),
        )
        updated = state.model_copy(
            update={
                "code_proposals": [*state.code_proposals, proposal],
                "status": "REVIEWING",
                "plan": mark_done(state, "coder"),
            }
        )
        summary = f"pytest exit {proposal.pytest_exit_code}"
        record_step(
            self._tracer,
            agent_name="coder",
            step_type="AGENT_DECISION",
            input_summary=state.objective,
            output_summary=summary,
            started=started,
            metadata={"pytest_exit_code": proposal.pytest_exit_code},
        )
        return AgentOutput(state=updated, summary=summary)

    async def _edits(self, state: WorkflowState) -> list[FileEdit]:
        for artifact in reversed(state.research_findings):
            if artifact.edits:
                return list(artifact.edits)
        if self._completer is None:
            raise AgentError("coder has no edits")
        raw = await complete_text(
            self._assembler,
            self._completer,
            [
                ContextItem(
                    id="coder-system",
                    bucket=ContextBucket.SYSTEM_TOOLS,
                    priority=0,
                    content=_EDIT_INSTRUCTION,
                ),
                ContextItem(
                    id="coder-objective",
                    bucket=ContextBucket.WORKING_MEMORY,
                    priority=1,
                    content=state.objective,
                ),
            ],
        )
        draft = parse_json(raw, _EditDraft)
        if not isinstance(draft, _EditDraft):
            raise AgentError("coder edit payload was invalid")
        return list(draft.edits)


def _render_diff(edits: list[FileEdit]) -> str:
    blocks = [
        f"--- {edit.path}\n+++ {edit.path}\n-{edit.old_text}\n+{edit.new_text}" for edit in edits
    ]
    return "\n".join(blocks)


async def _call(tools: ToolClient, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await tools.call_tool(name, arguments, "Coder")
    if result.get("isError") or result.get("is_error"):
        raise AgentError(_text(result) or f"{name} failed")
    parsed = json.loads(_text(result))
    if not isinstance(parsed, dict):
        raise AgentError(f"{name} returned an invalid payload")
    return parsed


def _text(result: dict[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return ""
    first = content[0]
    if isinstance(first, dict) and isinstance(first.get("text"), str):
        return str(first["text"])
    return ""
