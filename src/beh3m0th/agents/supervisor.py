"""Supervisor: plan the run, arbitrate deadlocks, and write the final report."""

from __future__ import annotations

import time
from typing import Protocol

from pydantic import Field

from beh3m0th.agents.conflict import arbitrate
from beh3m0th.agents.errors import AgentError
from beh3m0th.agents.prompts import TextCompleter, complete_text, parse_json
from beh3m0th.agents.state import (
    AgentOutput,
    ArbitrationChoice,
    ArbitrationDecision,
    FinalReport,
    TaskStep,
    WorkflowState,
)
from beh3m0th.agents.tracing import record_step
from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.context_engine.models import ContextBucket, ContextItem
from beh3m0th.core.schemas import StrictModel
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.memory.schemas import ConsolidationResult, WorkingScratchpad

_DECISION_INSTRUCTION = (
    "Return JSON with decision and rationale. decision is ACCEPT_CODER, "
    "ENFORCE_REVIEWER, REQUEST_NEW_RESEARCH, or ESCALATE_ABSTAIN."
)
_STATUS_FOR: dict[ArbitrationChoice, str] = {
    "ACCEPT_CODER": "COMPLETED",
    "ENFORCE_REVIEWER": "CODING",
    "REQUEST_NEW_RESEARCH": "RESEARCHING",
    "ESCALATE_ABSTAIN": "ABSTAINED",
}


class Consolidator(Protocol):
    """Subset of MemoryConsolidator.consolidate."""

    async def consolidate(self, scratchpad: WorkingScratchpad) -> ConsolidationResult: ...


class _DecisionDraft(StrictModel):
    decision: ArbitrationChoice
    rationale: str = Field(min_length=1)


class SupervisorAgent:
    """Zero-token planner and arbitrator. A completer can override the decision."""

    def __init__(
        self,
        *,
        consolidator: Consolidator | None = None,
        completer: TextCompleter | None = None,
        assembler: ContextAssembler | None = None,
        tracer: TrajectoryTracer | None = None,
    ) -> None:
        self._consolidator = consolidator
        self._completer = completer
        self._assembler = assembler or ContextAssembler(tracer=tracer, agent_name="supervisor")
        self._tracer = tracer

    async def run(self, state: WorkflowState) -> AgentOutput:
        if state.status == "PLANNING":
            return self._plan(state)
        if state.status == "RESOLVING_CONFLICT":
            return await self._resolve(state)
        if state.status in {"COMPLETED", "ABSTAINED"} and state.report is None:
            return await self._finalize(state)
        return AgentOutput(state=state, summary=f"dispatch {state.status}")

    def _plan(self, state: WorkflowState) -> AgentOutput:
        started = time.perf_counter()
        plan = [
            TaskStep(
                step_id="research",
                role="researcher",
                description=f"Investigate: {state.objective}",
            ),
            TaskStep(
                step_id="code",
                role="coder",
                description="Apply the cited fix and run pytest",
            ),
            TaskStep(
                step_id="review",
                role="reviewer",
                description="Vote on the sandbox result",
            ),
        ]
        updated = state.model_copy(update={"plan": plan, "status": "RESEARCHING"})
        summary = "planned research, code, and review"
        record_step(
            self._tracer,
            agent_name="supervisor",
            step_type="AGENT_DECISION",
            input_summary=state.objective,
            output_summary=summary,
            started=started,
            metadata={"status": updated.status},
        )
        return AgentOutput(state=updated, summary=summary)

    async def _resolve(self, state: WorkflowState) -> AgentOutput:
        started = time.perf_counter()
        decision = await self._decision(state)
        status = _STATUS_FOR[decision.decision]
        updated = state.model_copy(update={"arbitration": decision, "status": status})
        record_step(
            self._tracer,
            agent_name="supervisor",
            step_type="CONFLICT_ARBITRATION",
            input_summary=decision.conflict_id,
            output_summary=decision.decision,
            started=started,
            metadata={"decision": decision.decision},
        )
        if updated.status in {"COMPLETED", "ABSTAINED"}:
            return await self._finalize(updated)
        record_step(
            self._tracer,
            agent_name="supervisor",
            step_type="AGENT_DECISION",
            input_summary=state.objective,
            output_summary=decision.decision,
            metadata={"status": updated.status},
        )
        return AgentOutput(state=updated, summary=decision.decision)

    async def _decision(self, state: WorkflowState) -> ArbitrationDecision:
        if self._completer is None:
            return arbitrate(state)
        raw = await complete_text(
            self._assembler,
            self._completer,
            [
                ContextItem(
                    id="arbitrate-system",
                    bucket=ContextBucket.SYSTEM_TOOLS,
                    priority=0,
                    content=_DECISION_INSTRUCTION,
                ),
                ContextItem(
                    id="arbitrate-objective",
                    bucket=ContextBucket.WORKING_MEMORY,
                    priority=1,
                    content=state.objective,
                ),
            ],
        )
        draft = parse_json(raw, _DecisionDraft)
        if not isinstance(draft, _DecisionDraft):
            raise AgentError("arbitration payload was invalid")
        conflict_id = state.conflicts[-1].issue_id if state.conflicts else "conflict"
        return ArbitrationDecision(
            conflict_id=conflict_id,
            decision=draft.decision,
            rationale=draft.rationale,
        )

    async def _finalize(self, state: WorkflowState) -> AgentOutput:
        started = time.perf_counter()
        if state.status == "COMPLETED":
            summary = _success_summary(state)
            await self._consolidate(state)
        else:
            summary = f"abstained: {state.objective}"
        report = FinalReport(
            summary=summary,
            status=state.status,
            consensus_score=state.consensus_score,
            arbitration=None if state.arbitration is None else state.arbitration.decision,
        )
        updated = state.model_copy(update={"report": report})
        record_step(
            self._tracer,
            agent_name="supervisor",
            step_type="AGENT_DECISION",
            input_summary=state.objective,
            output_summary=summary,
            started=started,
            metadata={"status": updated.status},
        )
        record_step(
            self._tracer,
            agent_name="supervisor",
            step_type="FINAL_OUTPUT",
            input_summary=state.objective,
            output_summary=summary,
            metadata={"status": updated.status},
        )
        return AgentOutput(state=updated, summary=summary)

    async def _consolidate(self, state: WorkflowState) -> None:
        if self._consolidator is None:
            return
        diffs = [proposal.diff for proposal in state.code_proposals]
        scratchpad = WorkingScratchpad(
            user_id=state.user_id,
            project_id=state.project_id,
            session_id=state.session_id,
            objective=state.objective,
            plan_steps=[step.description for step in state.plan],
            diffs=diffs,
            turns=[f"bug_fix: {state.objective}; tests passed"],
        )
        await self._consolidator.consolidate(scratchpad)


def _success_summary(state: WorkflowState) -> str:
    exit_code = state.code_proposals[-1].pytest_exit_code if state.code_proposals else 1
    return f"completed: {state.objective}; pytest exit {exit_code}"
