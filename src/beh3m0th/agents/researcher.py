"""Researcher: cited findings only. This role never writes or executes."""

from __future__ import annotations

import time
from typing import Any, Protocol

from beh3m0th.agents.errors import AgentError
from beh3m0th.agents.state import (
    AgentOutput,
    Citation,
    ResearchArtifact,
    WorkflowState,
    mark_done,
)
from beh3m0th.agents.tracing import record_step
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.memory.schemas import ScoredMemory
from beh3m0th.retrieval.agentic_rag.schemas import AgenticRAGResponse

_READ_TOOLS = frozenset({"read_file", "list_directory", "search_files_grep"})


class ToolClient(Protocol):
    """Subset of MCPClient used by the agents."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        role: str,
    ) -> dict[str, Any]: ...


class MemoryReader(Protocol):
    """Subset of ThreeLayerMemory.recall."""

    async def recall(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str,
        session_id: str,
        limit: int = 5,
    ) -> list[ScoredMemory]: ...


class ResearchAnswerer(Protocol):
    """Subset of SelfCorrectingRAG.answer."""

    async def answer(self, query: str) -> AgenticRAGResponse: ...


class ResearcherAgent:
    """Produce a grounded artifact. Write and exec tools are rejected here."""

    def __init__(
        self,
        *,
        tools: ToolClient | None = None,
        memory: MemoryReader | None = None,
        rag: ResearchAnswerer | None = None,
        scripted: list[ResearchArtifact] | None = None,
        tracer: TrajectoryTracer | None = None,
    ) -> None:
        self._tools = tools
        self._memory = memory
        self._rag = rag
        self._scripted = scripted
        self._tracer = tracer

    async def run(self, state: WorkflowState) -> AgentOutput:
        started = time.perf_counter()
        await self._recall(state)
        if self._scripted is not None:
            artifacts = list(self._scripted)
        elif self._rag is None:
            artifacts = []
        else:
            artifacts = await self._from_rag(state.objective)
        if not artifacts or not any(artifact.citations for artifact in artifacts):
            updated = state.model_copy(
                update={"status": "ABSTAINED", "plan": mark_done(state, "researcher")}
            )
            summary = "research abstained"
        else:
            await self._read_sources(artifacts)
            updated = state.model_copy(
                update={
                    "research_findings": [*state.research_findings, *artifacts],
                    "status": "CODING",
                    "plan": mark_done(state, "researcher"),
                }
            )
            summary = artifacts[0].summary
        record_step(
            self._tracer,
            agent_name="researcher",
            step_type="AGENT_DECISION",
            input_summary=state.objective,
            output_summary=summary,
            started=started,
            metadata={"status": updated.status},
        )
        return AgentOutput(state=updated, summary=summary)

    async def _recall(self, state: WorkflowState) -> None:
        if self._memory is None:
            return
        started = time.perf_counter()
        hits = await self._memory.recall(
            state.objective,
            user_id=state.user_id,
            project_id=state.project_id,
            session_id=state.session_id,
            limit=3,
        )
        record_step(
            self._tracer,
            agent_name="researcher",
            step_type="MEMORY_RECALL",
            input_summary=state.objective,
            output_summary=" | ".join(hit.summary for hit in hits) or "none",
            started=started,
            metadata={"count": len(hits)},
        )

    async def _from_rag(self, objective: str) -> list[ResearchArtifact]:
        if self._rag is None:
            return []
        response = await self._rag.answer(objective)
        if response.status != "ANSWERED" or not response.citations:
            return []
        citations = [
            Citation(
                chunk_id=item.chunk_id,
                source_path=item.source_path,
                line_start=item.line_start,
                line_end=item.line_end,
            )
            for item in response.citations
        ]
        summary = response.claims[0].text if response.claims else objective
        return [
            ResearchArtifact(
                artifact_id="rag",
                summary=summary,
                citations=citations,
                confidence=response.confidence,
            )
        ]

    async def _read_sources(self, artifacts: list[ResearchArtifact]) -> None:
        if self._tools is None:
            return
        seen: set[str] = set()
        for artifact in artifacts:
            for citation in artifact.citations:
                path = citation.source_path
                if path is None or path in seen:
                    continue
                seen.add(path)
                await self._read(path)

    async def _read(self, path: str) -> None:
        await _call_read(self._tools, "read_file", {"path": path})


async def _call_read(
    tools: ToolClient | None,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if name not in _READ_TOOLS:
        raise AgentError(f"researcher cannot call {name}")
    if tools is None:
        raise AgentError("researcher has no tool client")
    result = await tools.call_tool(name, arguments, "Researcher")
    if result.get("isError") or result.get("is_error"):
        raise AgentError(_content_text(result) or f"{name} failed")
    return result


def _content_text(result: dict[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return ""
    first = content[0]
    if isinstance(first, dict):
        text = first.get("text")
        return text if isinstance(text, str) else ""
    return ""
