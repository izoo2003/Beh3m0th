"""Deterministic query splits and an optional assembler-backed rewriter."""

from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import ValidationError

from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.context_engine.models import BudgetPolicy, ContextBucket, ContextItem
from beh3m0th.core.llm_gateway import LLMMessage, LLMResponse
from beh3m0th.retrieval.agentic_rag.errors import AgenticRAGError
from beh3m0th.retrieval.agentic_rag.schemas import Route, SubQuery
from beh3m0th.retrieval.hybrid.bm25_index import tokenize

_SPLIT = re.compile(r"\s+and\s+|[;?]|\n+", re.IGNORECASE)
_SYMBOL = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*\b"
    r"|\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+\b"
)
_RELATIONAL = frozenset({"call", "calls", "read", "reads", "depend", "depends"})
_INSTRUCTION = (
    "Return JSON with key subqueries. Each item has text and route. "
    "route is hybrid, graph, or both. Emit at most 3 items."
)


class TextCompleter(Protocol):
    """Subset of LLMGateway.complete used by optional rewriting and judging."""

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class QueryRewriter(Protocol):
    """Splits a query and writes a follow-up when evidence is weak."""

    async def decompose(
        self,
        query: str,
        evidence: list[str] | None = None,
    ) -> list[SubQuery]: ...

    async def gap(self, query: str, missing: list[str]) -> SubQuery: ...


def route_for(text: str) -> Route:
    """Send relational questions to the graph, symbols to both, and the rest to hybrid."""
    if set(tokenize(text)) & _RELATIONAL:
        return "graph"
    if _SYMBOL.search(text):
        return "both"
    return "hybrid"


def decompose_query(query: str) -> list[SubQuery]:
    """Split on conjunctions and question marks into at most three subqueries."""
    parts = [part.strip(" .") for part in _SPLIT.split(query) if part.strip(" .")]
    if not parts:
        parts = [query.strip()]
    return [SubQuery(text=part, route=route_for(part)) for part in parts[:3]]


def gap_query(query: str, missing: list[str]) -> SubQuery:
    """Name the missing evidence without calling a model."""
    focus = missing[0].strip() if missing else "supporting evidence"
    text = f"{query.strip()} evidence for {focus}"
    return SubQuery(text=text[:300], route=route_for(query))


class DeterministicRewriter:
    """Zero-token rewriter used unless a caller injects an LLM rewriter."""

    async def decompose(
        self,
        query: str,
        evidence: list[str] | None = None,
    ) -> list[SubQuery]:
        del evidence
        return decompose_query(query)

    async def gap(self, query: str, missing: list[str]) -> SubQuery:
        return gap_query(query, missing)


class LLMQueryRewriter:
    """Optional rewriter. The prompt is the assembler's rendered text."""

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

    async def decompose(
        self,
        query: str,
        evidence: list[str] | None = None,
    ) -> list[SubQuery]:
        payload = await self._complete(query, evidence or [])
        return _parse_subqueries(payload)

    async def gap(self, query: str, missing: list[str]) -> SubQuery:
        payload = await self._complete(query, missing)
        return _parse_subqueries(payload)[0]

    async def _complete(self, query: str, evidence: list[str]) -> str:
        items = [
            ContextItem(
                id="rewrite-system",
                bucket=ContextBucket.SYSTEM_TOOLS,
                priority=0,
                content=_INSTRUCTION,
            ),
            ContextItem(
                id="rewrite-query",
                bucket=ContextBucket.WORKING_MEMORY,
                priority=1,
                content=query,
            ),
        ]
        for index, snippet in enumerate(evidence[:3]):
            if not snippet.strip():
                continue
            items.append(
                ContextItem(
                    id=f"rewrite-evidence-{index}",
                    bucket=ContextBucket.RETRIEVAL,
                    priority=2,
                    content=snippet,
                    truncatable=True,
                )
            )
        assembled = self._assembler.assemble(items, self._policy)
        if not assembled.text.strip():
            raise AgenticRAGError("assembled rewrite prompt was empty")
        response = await self._completer.complete(
            [LLMMessage(role="user", content=assembled.text)]
        )
        return response.content


def _parse_subqueries(content: str) -> list[SubQuery]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        raw = payload["subqueries"]
        found = [SubQuery.model_validate(item) for item in raw[:3]]
    except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise AgenticRAGError("rewriter returned invalid JSON") from exc
    if not found:
        raise AgenticRAGError("rewriter returned no subqueries")
    return found
