"""Optional prose extraction through the context assembler and LLM gateway."""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import ValidationError

from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.context_engine.models import BudgetPolicy, ContextBucket, ContextItem
from beh3m0th.core.llm_gateway import LLMMessage, LLMResponse
from beh3m0th.core.schemas import DocumentChunk
from beh3m0th.retrieval.graphrag.errors import GraphExtractionError
from beh3m0th.retrieval.graphrag.schemas import GraphExtractionResult

_INSTRUCTION = (
    "Extract typed entities and directed relationships as JSON with keys "
    "'entities' and 'relationships'. Entity kinds are Concept, Service, Component, "
    "API, Policy, or Error. Predicates are DEPENDS_ON, CAUSES, MITIGATES, "
    "CONFIGURES, READS, or CALLS. Include chunk_id on every relationship."
)


class TextCompleter(Protocol):
    """Subset of LLMGateway.complete used by optional prose extraction."""

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class LLMProseExtractor:
    """Optional prose extractor. Prompts go through the context assembler."""

    def __init__(
        self,
        assembler: ContextAssembler,
        completer: TextCompleter,
        policy: BudgetPolicy | None = None,
    ) -> None:
        self._assembler = assembler
        self._completer = completer
        self._policy = policy or BudgetPolicy(
            model_window=4096,
            reserved_output=512,
            safety_margin=256,
        )

    async def extract(self, chunk: DocumentChunk) -> GraphExtractionResult:
        assembled = self._assembler.assemble(
            [
                ContextItem(
                    id="graph-extract-system",
                    bucket=ContextBucket.SYSTEM_TOOLS,
                    priority=0,
                    content=_INSTRUCTION,
                ),
                ContextItem(
                    id=chunk.chunk_id,
                    bucket=ContextBucket.RETRIEVAL,
                    priority=2,
                    content=chunk.text,
                    truncatable=True,
                    metadata={"chunk_id": chunk.chunk_id},
                ),
            ],
            self._policy,
        )
        if not assembled.text.strip():
            raise GraphExtractionError("assembled extraction prompt was empty")
        response = await self._completer.complete(
            [LLMMessage(role="user", content=assembled.text)]
        )
        return _parse_payload(response.content)


def _parse_payload(content: str) -> GraphExtractionResult:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return GraphExtractionResult.model_validate_json(text)
    except ValidationError as exc:
        raise GraphExtractionError("prose extractor returned invalid JSON") from exc
