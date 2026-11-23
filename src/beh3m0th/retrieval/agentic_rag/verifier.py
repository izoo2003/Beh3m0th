"""Citation checks and claim-level groundedness."""

from __future__ import annotations

import json
import re
from typing import Protocol

from pydantic import ValidationError

from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.context_engine.models import BudgetPolicy, ContextBucket, ContextItem
from beh3m0th.core.llm_gateway import LLMMessage
from beh3m0th.retrieval.agentic_rag.errors import AgenticRAGError
from beh3m0th.retrieval.agentic_rag.rewriter import TextCompleter
from beh3m0th.retrieval.agentic_rag.schemas import (
    Claim,
    ClaimLabel,
    ClaimVerification,
    Evidence,
)
from beh3m0th.retrieval.hybrid.bm25_index import tokenize

_CHUNK = re.compile(r"\[chunk:([^\]]+)\]")
_SPAN = re.compile(r"\[([^\[\]:]+\.[A-Za-z0-9]+):(\d+)-(\d+)\]")
_SUPPORT_RATIO = 0.5
_INSTRUCTION = (
    "Label each claim SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED. "
    "Return JSON with key verifications. Each item has claim, label, and citation_ids."
)


class ClaimVerifier(Protocol):
    """Scores claims against retrieved evidence."""

    async def verify(
        self,
        claims: list[Claim],
        evidence: list[Evidence],
    ) -> list[ClaimVerification]: ...


def groundedness(verifications: list[ClaimVerification]) -> float:
    """Share of claims labeled SUPPORTED. An empty set scores 0."""
    if not verifications:
        return 0.0
    supported = sum(1 for item in verifications if item.label == "SUPPORTED")
    return supported / len(verifications)


def verify_claims(claims: list[Claim], evidence: list[Evidence]) -> list[ClaimVerification]:
    """Label claims from citation membership and token overlap. No model call."""
    return [_label_claim(claim, evidence) for claim in claims]


class DeterministicVerifier:
    """Zero-token citation and overlap checker."""

    async def verify(
        self,
        claims: list[Claim],
        evidence: list[Evidence],
    ) -> list[ClaimVerification]:
        return verify_claims(claims, evidence)


class LLMClaimVerifier:
    """Optional judge. Missing citations stay UNSUPPORTED even if the model agrees."""

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
        self._deterministic = DeterministicVerifier()

    async def verify(
        self,
        claims: list[Claim],
        evidence: list[Evidence],
    ) -> list[ClaimVerification]:
        fallback = verify_claims(claims, evidence)
        if not claims:
            return []
        items = [
            ContextItem(
                id="verify-system",
                bucket=ContextBucket.SYSTEM_TOOLS,
                priority=0,
                content=_INSTRUCTION,
            )
        ]
        for index, claim in enumerate(claims):
            items.append(
                ContextItem(
                    id=f"claim-{index}",
                    bucket=ContextBucket.WORKING_MEMORY,
                    priority=1,
                    content=claim.text,
                )
            )
        for index, item in enumerate(evidence[:6]):
            items.append(
                ContextItem(
                    id=f"evidence-{item.chunk_id}-{index}",
                    bucket=ContextBucket.RETRIEVAL,
                    priority=2,
                    content=f"{item.chunk_id}: {item.text}",
                    truncatable=True,
                )
            )
        assembled = self._assembler.assemble(items, self._policy)
        if not assembled.text.strip():
            raise AgenticRAGError("assembled verification prompt was empty")
        response = await self._completer.complete(
            [LLMMessage(role="user", content=assembled.text)]
        )
        judged = _parse_verdicts(response.content)
        return _prefer_valid_judgments(claims, evidence, judged, fallback)


def _label_claim(claim: Claim, evidence: list[Evidence]) -> ClaimVerification:
    citation_ids, valid = _citation_ids(claim, evidence)
    if not valid:
        return ClaimVerification(
            claim=claim.text,
            label="UNSUPPORTED",
            citation_ids=citation_ids,
        )
    cited = [item for item in evidence if item.chunk_id in set(citation_ids)]
    ratio = _overlap_ratio(claim.text, " ".join(item.text for item in cited))
    return ClaimVerification(
        claim=claim.text,
        label=_label_for_ratio(ratio),
        citation_ids=citation_ids,
    )


def _citation_ids(claim: Claim, evidence: list[Evidence]) -> tuple[list[str], bool]:
    found: list[str] = []
    valid = True
    known = {item.chunk_id: item for item in evidence}
    for citation in claim.citations:
        found.append(citation.chunk_id)
        if citation.chunk_id not in known:
            valid = False
    for match in _CHUNK.finditer(claim.text):
        chunk_id = match.group(1).strip()
        found.append(chunk_id)
        if chunk_id not in known:
            valid = False
    for match in _SPAN.finditer(claim.text):
        path = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        matched = _match_span(evidence, path, start, end)
        if matched is None:
            valid = False
        else:
            found.append(matched)
    if not found:
        return [], False
    return _unique(found), valid


def _match_span(evidence: list[Evidence], path: str, start: int, end: int) -> str | None:
    for item in evidence:
        if item.source_path != path or item.line_start is None or item.line_end is None:
            continue
        if item.line_start <= start and end <= item.line_end:
            return item.chunk_id
    return None


def _overlap_ratio(claim: str, evidence_text: str) -> float:
    body = _SPAN.sub(" ", _CHUNK.sub(" ", claim))
    claim_tokens = set(tokenize(body))
    evidence_tokens = set(tokenize(evidence_text))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def _label_for_ratio(ratio: float) -> ClaimLabel:
    if ratio >= _SUPPORT_RATIO:
        return "SUPPORTED"
    if ratio > 0.0:
        return "PARTIALLY_SUPPORTED"
    return "UNSUPPORTED"


def _parse_verdicts(content: str) -> list[ClaimVerification]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        return [ClaimVerification.model_validate(item) for item in payload["verifications"]]
    except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise AgenticRAGError("verifier returned invalid JSON") from exc


def _prefer_valid_judgments(
    claims: list[Claim],
    evidence: list[Evidence],
    judged: list[ClaimVerification],
    fallback: list[ClaimVerification],
) -> list[ClaimVerification]:
    by_claim = {item.claim: item for item in judged}
    chosen: list[ClaimVerification] = []
    for claim, default in zip(claims, fallback, strict=True):
        citation_ids, valid = _citation_ids(claim, evidence)
        verdict = by_claim.get(claim.text)
        if verdict is None or not valid:
            chosen.append(default)
            continue
        chosen.append(
            verdict.model_copy(update={"citation_ids": citation_ids or verdict.citation_ids})
        )
    return chosen


def _unique(values: list[str]) -> list[str]:
    found: list[str] = []
    for value in values:
        if value not in found:
            found.append(value)
    return found
