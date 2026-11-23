"""Extractive claims and score math for the self-correcting loop."""

from __future__ import annotations

from beh3m0th.retrieval.agentic_rag.schemas import (
    AgenticRAGResponse,
    AnswerStatus,
    Citation,
    Claim,
    ClaimVerification,
    Evidence,
)
from beh3m0th.retrieval.graphrag.schemas import SubgraphResult
from beh3m0th.retrieval.hybrid.schemas import HybridSearchResult


class ExtractiveSynthesizer:
    """Cite the first line of each chunk that passed the relevance gate."""

    def synthesize(self, query: str, evidence: list[Evidence]) -> list[Claim]:
        del query
        claims: list[Claim] = []
        for item in evidence:
            sentence = item.text.strip().splitlines()[0][:240]
            if not sentence:
                continue
            claims.append(
                Claim(
                    text=f"{sentence} [chunk:{item.chunk_id}]",
                    citations=[
                        Citation(
                            chunk_id=item.chunk_id,
                            source_path=item.source_path,
                            line_start=item.line_start,
                            line_end=item.line_end,
                        )
                    ],
                )
            )
        return claims


def from_hybrid(result: HybridSearchResult) -> list[Evidence]:
    """Copy hybrid hits into evidence rows."""
    return [
        Evidence(
            chunk_id=hit.chunk.chunk_id,
            text=hit.chunk.text,
            source_path=hit.chunk.source_path,
            line_start=hit.chunk.line_start,
            line_end=hit.chunk.line_end,
            score=clamp(hit.score),
            origin="hybrid",
        )
        for hit in result.hits
    ]


def from_graph(result: SubgraphResult) -> list[Evidence]:
    """Turn graph edges and entity chunk ids into citable evidence."""
    names = {entity.entity_id: entity.name for entity in result.entities}
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for relationship in result.relationships:
        if not relationship.chunk_id or relationship.chunk_id in seen:
            continue
        seen.add(relationship.chunk_id)
        source = names.get(relationship.source_id, relationship.source_id)
        target = names.get(relationship.target_id, relationship.target_id)
        evidence.append(
            Evidence(
                chunk_id=relationship.chunk_id,
                text=f"{source} {relationship.predicate} {target}",
                source_path="graph",
                origin="graph",
            )
        )
    for entity in result.entities:
        for chunk_id in entity.chunk_ids:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            evidence.append(
                Evidence(chunk_id=chunk_id, text=entity.name, source_path="graph", origin="graph")
            )
    return evidence


def merge_evidence(current: list[Evidence], incoming: list[Evidence]) -> list[Evidence]:
    """Keep the higher score when the same chunk is retrieved again."""
    merged = {item.chunk_id: item for item in current}
    for item in incoming:
        previous = merged.get(item.chunk_id)
        if previous is None or item.score > previous.score:
            merged[item.chunk_id] = item
    return list(merged.values())


def mean_top(evidence: list[Evidence], top_k: int) -> float:
    """Average the strongest rerank scores."""
    if not evidence:
        return 0.0
    ordered = sorted(evidence, key=lambda item: item.score, reverse=True)
    chosen = ordered[: max(top_k, 1)]
    return sum(item.score for item in chosen) / len(chosen)


def confidence(mean_score: float, score: float, graph_ran: bool, evidence: list[Evidence]) -> float:
    """Blend rerank, groundedness, and graph coverage when a graph query ran."""
    if graph_ran:
        coverage = 1.0 if any(item.origin == "graph" for item in evidence) else 0.0
        return clamp((mean_score + coverage + score) / 3)
    return clamp((mean_score + score) / 2)


def gaps(query: str, verifications: list[ClaimVerification], passed: list[Evidence]) -> list[str]:
    """Describe what the next retrieval pass should look for."""
    unsupported = [item.claim for item in verifications if item.label != "SUPPORTED"]
    if unsupported:
        return unsupported
    del passed
    return [query.strip()]


def response(
    *,
    status: AnswerStatus,
    query: str,
    claims: list[Claim],
    confidence: float,
    groundedness: float,
    found: list[str],
    missing: list[str],
    follow_ups: list[str],
    steps: int,
) -> AgenticRAGResponse:
    """Build the public response and flatten citations from the kept claims."""
    citations: list[Citation] = []
    for claim in claims:
        citations.extend(claim.citations)
    return AgenticRAGResponse(
        status=status,
        query=query,
        claims=claims,
        citations=citations,
        confidence=confidence,
        groundedness=groundedness,
        found=unique(found),
        missing=missing,
        follow_ups=follow_ups,
        steps=steps,
    )


def unique(values: list[str]) -> list[str]:
    found: list[str] = []
    for value in values:
        if value not in found:
            found.append(value)
    return found


def clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
