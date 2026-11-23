"""Assembler-backed JSON completions shared by the critique and edit prompts."""

from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import ValidationError

from beh3m0th.agents.errors import AgentError
from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.context_engine.models import BudgetPolicy, ContextItem
from beh3m0th.core.llm_gateway import LLMMessage, LLMResponse
from beh3m0th.core.schemas import StrictModel


class TextCompleter(Protocol):
    """Subset of LLMGateway.complete used by optional critique and edits."""

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


def default_budget() -> BudgetPolicy:
    return BudgetPolicy(model_window=4096, reserved_output=512, safety_margin=256)


def parse_json(content: str, model: type[StrictModel]) -> StrictModel:
    """Parse a model JSON object, including a fenced block."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        return model.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise AgentError("model returned invalid JSON") from exc


async def complete_text(
    assembler: ContextAssembler,
    completer: TextCompleter,
    items: list[ContextItem],
    policy: BudgetPolicy | None = None,
) -> str:
    """Assemble the prompt, then ask the injected completer. No raw concatenation."""
    assembled = assembler.assemble(items, policy or default_budget())
    if not assembled.text.strip():
        raise AgentError("assembled prompt was empty")
    response = await completer.complete([LLMMessage(role="user", content=assembled.text)])
    return response.content
