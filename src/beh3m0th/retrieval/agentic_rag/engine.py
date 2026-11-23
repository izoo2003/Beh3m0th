"""Bounded retrieve, grade, synthesize, verify, and abstain loop."""

from __future__ import annotations

import time
from typing import Protocol

from beh3m0th.core.schemas import MetadataValue, TrajectoryStep
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.retrieval.agentic_rag.errors import AgenticRAGError
from beh3m0th.retrieval.agentic_rag.rewriter import DeterministicRewriter, QueryRewriter
from beh3m0th.retrieval.agentic_rag.schemas import (
    AgenticRAGResponse,
    Claim,
    ClaimVerification,
    Evidence,
    SubQuery,
)
from beh3m0th.retrieval.agentic_rag.synthesis import (
    ExtractiveSynthesizer,
    clamp,
    confidence,
    from_graph,
    from_hybrid,
    gaps,
    mean_top,
    merge_evidence,
    response,
)
from beh3m0th.retrieval.agentic_rag.verifier import (
    ClaimVerifier,
    DeterministicVerifier,
    groundedness,
)
from beh3m0th.retrieval.graphrag.schemas import SubgraphResult
from beh3m0th.retrieval.hybrid.schemas import HybridSearchResult, MetadataFilter


class HybridSearcher(Protocol):
    """Subset of HybridRetriever.search used by the loop."""

    async def search(
        self,
        query: str,
        filters: MetadataFilter | None = None,
        top_k: int = 6,
        fetch_k: int = 20,
    ) -> HybridSearchResult: ...


class GraphSearcher(Protocol):
    """Subset of GraphRAGPipeline.query used by the loop."""

    async def query(self, text: str, mode: str = "local", hops: int = 2) -> SubgraphResult: ...


class ClaimSynthesizer(Protocol):
    """Turns passing evidence into cited claims."""

    def synthesize(self, query: str, evidence: list[Evidence]) -> list[Claim]: ...


class RerankScorer(Protocol):
    """Scores query-document pairs. Higher means more relevant."""

    async def score(self, query: str, texts: list[str]) -> list[float]: ...


class SelfCorrectingRAG:
    """Retry retrieval until claims are grounded, or abstain."""

    def __init__(
        self,
        *,
        retriever: HybridSearcher | None = None,
        graph: GraphSearcher | None = None,
        reranker: RerankScorer | None = None,
        rewriter: QueryRewriter | None = None,
        verifier: ClaimVerifier | None = None,
        synthesizer: ClaimSynthesizer | None = None,
        tracer: TrajectoryTracer | None = None,
        agent_name: str = "agentic_rag",
        max_steps: int = 3,
        tau_rel: float = 0.60,
        tau_ground: float = 0.80,
        tau_abstain: float = 0.55,
        top_k: int = 4,
    ) -> None:
        if retriever is None and graph is None:
            raise AgenticRAGError("a retriever or graph pipeline is required")
        if not agent_name.strip():
            raise ValueError("agent_name must not be empty")
        if max_steps < 1:
            raise AgenticRAGError("max_steps must be positive")
        self._retriever = retriever
        self._graph = graph
        self._reranker = reranker
        self._rewriter = rewriter if rewriter is not None else DeterministicRewriter()
        self._verifier = verifier if verifier is not None else DeterministicVerifier()
        self._synthesizer = synthesizer if synthesizer is not None else ExtractiveSynthesizer()
        self._tracer = tracer
        self._agent_name = agent_name
        self._max_steps = max_steps
        self._tau_rel = tau_rel
        self._tau_ground = tau_ground
        self._tau_abstain = tau_abstain
        self._top_k = top_k

    async def answer(self, query: str) -> AgenticRAGResponse:
        if not query.strip():
            raise AgenticRAGError("query must not be empty")
        subqueries = await self._rewriter.decompose(query.strip())
        evidence: list[Evidence] = []
        graded: list[Evidence] = []
        claims: list[Claim] = []
        verifications: list[ClaimVerification] = []
        missing: list[str] = []
        graph_ran = False
        steps = 0
        mean_score = 0.0
        while steps < self._max_steps:
            steps += 1
            started = time.perf_counter()
            batch = await self._retrieve(subqueries)
            graph_ran = graph_ran or any(item.origin == "graph" for item in batch)
            evidence = merge_evidence(evidence, batch)
            graded = await self._grade(query, evidence)
            mean_score = mean_top(graded, self._top_k)
            self._record(
                input_summary=query,
                output_summary=",".join(item.chunk_id for item in graded[:8]) or "no hits",
                latency_ms=(time.perf_counter() - started) * 1000,
                metadata={
                    "step": steps,
                    "mean_rerank": mean_score,
                    "hit_count": len(graded),
                    "chunk_ids": ",".join(item.chunk_id for item in graded),
                },
            )
            passed = [item for item in graded if item.score >= self._tau_rel]
            claims = self._synthesizer.synthesize(query, passed)
            verify_started = time.perf_counter()
            verifications = await self._verifier.verify(claims, evidence)
            score = groundedness(verifications)
            self._record(
                input_summary=query,
                output_summary=f"groundedness {score:.2f}",
                latency_ms=(time.perf_counter() - verify_started) * 1000,
                metadata={
                    "step": steps,
                    "groundedness": score,
                    "claim_count": len(verifications),
                    "supported": sum(1 for item in verifications if item.label == "SUPPORTED"),
                },
            )
            if mean_score >= self._tau_rel and score >= self._tau_ground:
                break
            missing = gaps(query, verifications, passed)
            if steps >= self._max_steps:
                break
            subqueries = [await self._rewriter.gap(query, missing)]
        score = groundedness(verifications)
        confidence_score = confidence(mean_score, score, graph_ran, graded)
        supported = [
            claim
            for claim, verdict in zip(claims, verifications, strict=True)
            if verdict.label == "SUPPORTED"
        ]
        found = [item.chunk_id for item in graded]
        if confidence_score < self._tau_abstain or not supported:
            gaps_found = missing or [f"no relevant evidence for {query.strip()}"]
            return response(
                status="ABSTAINED",
                query=query.strip(),
                claims=[],
                confidence=confidence_score,
                groundedness=score,
                found=found,
                missing=gaps_found,
                follow_ups=[f"What evidence supports {gap}?" for gap in gaps_found[:3]],
                steps=steps,
            )
        return response(
            status="ANSWERED",
            query=query.strip(),
            claims=supported,
            confidence=confidence_score,
            groundedness=score,
            found=found,
            missing=[],
            follow_ups=[],
            steps=steps,
        )

    async def _retrieve(self, subqueries: list[SubQuery]) -> list[Evidence]:
        found: list[Evidence] = []
        for subquery in subqueries:
            if _uses_hybrid(subquery, self._retriever, self._graph):
                assert self._retriever is not None
                result = await self._retriever.search(
                    subquery.text,
                    None,
                    self._top_k,
                    max(self._top_k * 2, 8),
                )
                found.extend(from_hybrid(result))
            if _uses_graph(subquery, self._retriever, self._graph):
                assert self._graph is not None
                result = await self._graph.query(subquery.text, mode="local", hops=2)
                found.extend(from_graph(result))
        return found

    async def _grade(self, query: str, evidence: list[Evidence]) -> list[Evidence]:
        if not evidence or self._reranker is None:
            return evidence
        scores = await self._reranker.score(query, [item.text for item in evidence])
        if len(scores) != len(evidence):
            raise AgenticRAGError("reranker returned the wrong number of scores")
        return [
            item.model_copy(update={"score": clamp(score)})
            for item, score in zip(evidence, scores, strict=True)
        ]

    def _record(
        self,
        *,
        input_summary: str,
        output_summary: str,
        latency_ms: float,
        metadata: dict[str, MetadataValue],
    ) -> None:
        if self._tracer is None:
            return
        self._tracer.record(
            TrajectoryStep(
                step_index=len(self._tracer.steps),
                agent_name=self._agent_name,
                step_type="RETRIEVAL",
                input_summary=input_summary[:200],
                output_summary=output_summary[:200] or "empty",
                tokens_used=0,
                latency_ms=latency_ms,
                metadata=metadata,
            )
        )


def _uses_hybrid(
    subquery: SubQuery,
    retriever: HybridSearcher | None,
    graph: GraphSearcher | None,
) -> bool:
    if subquery.route == "graph" and graph is not None:
        return False
    if subquery.route in ("hybrid", "both") and retriever is not None:
        return True
    return retriever is not None and graph is None


def _uses_graph(
    subquery: SubQuery,
    retriever: HybridSearcher | None,
    graph: GraphSearcher | None,
) -> bool:
    if graph is None:
        return False
    if subquery.route in ("graph", "both"):
        return True
    return retriever is None

