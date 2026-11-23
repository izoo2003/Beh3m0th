"""LangGraph supervisor. Agent modules never import langgraph."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from beh3m0th.agents.coder import CoderAgent
from beh3m0th.agents.prompts import TextCompleter
from beh3m0th.agents.researcher import MemoryReader, ResearchAnswerer, ResearcherAgent, ToolClient
from beh3m0th.agents.reviewer import ReviewerAgent
from beh3m0th.agents.state import AgentOutput, ResearchArtifact, WorkflowState
from beh3m0th.agents.supervisor import Consolidator, SupervisorAgent
from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.evaluation.tracer import TrajectoryTracer

_ROUTE = {
    "RESEARCHING": "researcher",
    "CODING": "coder",
    "REVIEWING": "reviewer",
}


class SupervisorOrchestrator:
    """Compile the typed graph once and return the final workflow state."""

    def __init__(
        self,
        researcher: ResearcherAgent,
        coder: CoderAgent,
        reviewer: ReviewerAgent,
        supervisor: SupervisorAgent,
    ) -> None:
        graph: StateGraph[WorkflowState] = StateGraph(WorkflowState)
        graph.add_node("supervisor", _node(supervisor))
        graph.add_node("researcher", _node(researcher))
        graph.add_node("coder", _node(coder))
        graph.add_node("reviewer", _node(reviewer))
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            _route,
            {"researcher": "researcher", "coder": "coder", "reviewer": "reviewer", END: END},
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("coder", "supervisor")
        graph.add_edge("reviewer", "supervisor")
        self._compiled = graph.compile()

    async def run(self, state: WorkflowState) -> WorkflowState:
        result = await self._compiled.ainvoke(
            state.model_dump(mode="json"),
            config={"recursion_limit": 30},
        )
        if isinstance(result, WorkflowState):
            return result
        return WorkflowState.model_validate(dict(result))


def build_orchestrator(
    *,
    tools: ToolClient,
    workspace: Path,
    tracer: TrajectoryTracer | None = None,
    completer: TextCompleter | None = None,
    scripted_research: list[ResearchArtifact] | None = None,
    consolidator: Consolidator | None = None,
    memory: MemoryReader | None = None,
    rag: ResearchAnswerer | None = None,
    assembler: ContextAssembler | None = None,
    supervisor_completer: TextCompleter | None = None,
) -> SupervisorOrchestrator:
    """Wire the four roles. The review completer is not used for arbitration."""
    return SupervisorOrchestrator(
        ResearcherAgent(
            tools=tools,
            memory=memory,
            rag=rag,
            scripted=scripted_research,
            tracer=tracer,
        ),
        CoderAgent(tools, completer=completer, assembler=assembler, tracer=tracer),
        ReviewerAgent(workspace, completer=completer, assembler=assembler, tracer=tracer),
        SupervisorAgent(
            consolidator=consolidator,
            completer=supervisor_completer,
            assembler=assembler,
            tracer=tracer,
        ),
    )


def _node(agent: ResearcherAgent | CoderAgent | ReviewerAgent | SupervisorAgent):
    async def run(state: WorkflowState | Mapping[str, Any]) -> dict[str, Any]:
        output: AgentOutput = await agent.run(_as_state(state))
        return output.state.model_dump(mode="json")

    return run


def _route(state: WorkflowState | Mapping[str, Any]) -> str:
    current = _as_state(state)
    if current.status in {"COMPLETED", "ABSTAINED", "FAILED"}:
        return END
    return _ROUTE.get(current.status, END)


def _as_state(state: WorkflowState | Mapping[str, Any]) -> WorkflowState:
    if isinstance(state, WorkflowState):
        return state
    return WorkflowState.model_validate(dict(state))
