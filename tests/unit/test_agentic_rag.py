"""Offline Phase 4 checks: grounding, retries, and abstention."""

from __future__ import annotations

import pytest

from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.core.llm_gateway import LLMResponse
from beh3m0th.core.schemas import DocumentChunk, RetrievalHit
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.retrieval.agentic_rag import (
    AgenticRAGError,
    Citation,
    Claim,
    Evidence,
    LLMClaimVerifier,
    LLMQueryRewriter,
    SelfCorrectingRAG,
    route_for,
    verify_claims,
)
from beh3m0th.retrieval.agentic_rag.engine import ExtractiveSynthesizer
from beh3m0th.retrieval.graphrag.schemas import (
    Entity,
    Relationship,
    SubgraphResult,
)
from beh3m0th.retrieval.hybrid.schemas import HybridSearchResult, RetrievalMetrics

ANSWER = "Users cannot sign in when the connection pool is exhausted."
DISTRACTOR = "The login button uses a blue css class."


def _chunk(chunk_id: str, text: str, source_path: str = "docs/auth.md") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        text=text,
        source_path=source_path,
    )


def _metrics() -> RetrievalMetrics:
    return RetrievalMetrics(
        embed_ms=0.0,
        bm25_ms=0.0,
        vector_ms=0.0,
        rerank_ms=0.0,
        total_ms=0.0,
    )


class ScriptedRetriever:
    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    async def search(
        self,
        query: str,
        filters: object = None,
        top_k: int = 6,
        fetch_k: int = 20,
    ) -> HybridSearchResult:
        del filters, top_k, fetch_k
        self.queries.append(query)
        hits = [
            RetrievalHit(chunk=chunk, score=0.5, rank=index, source="hybrid")
            for index, chunk in enumerate(self.chunks, start=1)
        ]
        return HybridSearchResult(query=query, hits=hits, metrics=_metrics())


class ScriptedGraph:
    def __init__(self, chunk_id: str = "graph-chunk") -> None:
        self.chunk_id = chunk_id
        self.queries: list[str] = []

    async def query(self, text: str, mode: str = "local", hops: int = 2) -> SubgraphResult:
        del mode, hops
        self.queries.append(text)
        service = Entity(
            entity_id="service",
            name="ServiceA",
            canonical_name="servicea",
            kind="Class",
            chunk_ids=[self.chunk_id],
        )
        auth = Entity(
            entity_id="auth",
            name="AuthModule",
            canonical_name="authmodule",
            kind="Module",
            chunk_ids=[self.chunk_id],
        )
        return SubgraphResult(
            query=text,
            mode="local",
            entities=[service, auth],
            relationships=[
                Relationship(
                    source_id="service",
                    target_id="auth",
                    predicate="CALLS",
                    chunk_id=self.chunk_id,
                )
            ],
            chunk_ids=[self.chunk_id],
            score=1.0,
        )


class SequenceReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls = 0

    async def score(self, query: str, texts: list[str]) -> list[float]:
        del query
        value = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return [value for _ in texts]


class ScriptedCompleter:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.prompts: list[str] = []

    async def complete(
        self,
        messages,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        del model, temperature
        self.calls += 1
        self.prompts.append(messages[0].content)
        return LLMResponse(
            content=self.content,
            provider="fake",
            model="fake",
            cached=False,
            cache_key="scripted",
        )


class MixedSynthesizer:
    def synthesize(self, query: str, evidence: list[Evidence]) -> list[Claim]:
        claims = ExtractiveSynthesizer().synthesize(query, evidence)
        claims.append(
            Claim(
                text="The outage was invented [chunk:missing]",
                citations=[Citation(chunk_id="missing")],
            )
        )
        return claims


@pytest.mark.asyncio
async def test_relevant_evidence_is_answered_with_real_chunk_ids() -> None:
    retriever = ScriptedRetriever([_chunk("auth", ANSWER)])
    tracer = TrajectoryTracer()
    engine = SelfCorrectingRAG(
        retriever=retriever,
        reranker=SequenceReranker([0.95]),
        tracer=tracer,
    )
    result = await engine.answer("why does login fail")
    assert result.status == "ANSWERED"
    assert result.confidence >= 0.55
    assert result.groundedness >= 0.80
    assert {citation.chunk_id for citation in result.citations} <= {"auth"}
    assert result.citations
    assert [step.step_type for step in tracer.steps] == ["RETRIEVAL", "RETRIEVAL"]


@pytest.mark.asyncio
async def test_unanswerable_query_abstains() -> None:
    retriever = ScriptedRetriever([_chunk("distractor", DISTRACTOR)])
    engine = SelfCorrectingRAG(retriever=retriever, reranker=SequenceReranker([0.05]))
    result = await engine.answer("why does login fail")
    assert result.status == "ABSTAINED"
    assert result.confidence < 0.55
    assert result.claims == []
    assert set(result.found) <= {"distractor"}
    assert result.follow_ups
    assert len(retriever.queries) == 3


@pytest.mark.asyncio
async def test_low_rerank_retries_and_stops_at_three_steps() -> None:
    always_low = ScriptedRetriever([_chunk("distractor", DISTRACTOR)])
    low_engine = SelfCorrectingRAG(
        retriever=always_low,
        reranker=SequenceReranker([0.20, 0.20, 0.20]),
    )
    low_result = await low_engine.answer("why does login fail")
    assert len(always_low.queries) == 3
    assert low_result.steps == 3
    assert low_result.status == "ABSTAINED"

    recovered = ScriptedRetriever([_chunk("auth", ANSWER)])
    recovered_engine = SelfCorrectingRAG(
        retriever=recovered,
        reranker=SequenceReranker([0.20, 0.95]),
    )
    recovered_result = await recovered_engine.answer("why does login fail")
    assert len(recovered.queries) == 2
    assert recovered_result.status == "ANSWERED"


def test_unknown_citation_is_unsupported() -> None:
    evidence = [Evidence(chunk_id="auth", text=ANSWER, source_path="docs/auth.md", score=0.95)]
    claim = Claim(
        text="The pool is exhausted [chunk:missing]",
        citations=[Citation(chunk_id="missing")],
    )
    verdict = verify_claims([claim], evidence)[0]
    assert verdict.label == "UNSUPPORTED"


def test_path_span_citation_matches_retrieved_chunk() -> None:
    evidence = [
        Evidence(
            chunk_id="auth",
            text=ANSWER,
            source_path="src/auth.py",
            line_start=45,
            line_end=68,
            score=0.95,
        )
    ]
    claim = Claim(text=f"{ANSWER} [src/auth.py:45-68]")
    verdict = verify_claims([claim], evidence)[0]
    assert verdict.label == "SUPPORTED"
    assert verdict.citation_ids == ["auth"]


@pytest.mark.asyncio
async def test_unsupported_claim_is_removed_from_answered_response() -> None:
    engine = SelfCorrectingRAG(
        retriever=ScriptedRetriever([_chunk("auth", ANSWER)]),
        reranker=SequenceReranker([0.95]),
        synthesizer=MixedSynthesizer(),
    )
    result = await engine.answer("why does login fail")
    assert result.status == "ANSWERED"
    assert {citation.chunk_id for citation in result.citations} == {"auth"}
    assert all("missing" not in claim.text for claim in result.claims)


@pytest.mark.asyncio
async def test_relational_query_uses_graph_chunk_id() -> None:
    hybrid = ScriptedRetriever([_chunk("auth", ANSWER)])
    graph = ScriptedGraph()
    engine = SelfCorrectingRAG(
        retriever=hybrid,
        graph=graph,
        reranker=SequenceReranker([0.95]),
    )
    assert route_for("ServiceA calls AuthModule") == "graph"
    result = await engine.answer("ServiceA calls AuthModule")
    assert hybrid.queries == []
    assert graph.queries
    assert result.status == "ANSWERED"
    assert {citation.chunk_id for citation in result.citations} == {"graph-chunk"}


@pytest.mark.asyncio
async def test_optional_models_run_only_when_injected() -> None:
    idle = ScriptedCompleter('{"subqueries": [{"text": "unused", "route": "hybrid"}]}')
    engine = SelfCorrectingRAG(
        retriever=ScriptedRetriever([_chunk("auth", ANSWER)]),
        reranker=SequenceReranker([0.95]),
    )
    await engine.answer("why does login fail")
    assert idle.calls == 0

    tracer = TrajectoryTracer()
    rewriter = LLMQueryRewriter(ContextAssembler(tracer=tracer), idle)
    rewritten = await rewriter.decompose("why does login fail", evidence=["pool exhausted marker"])
    assert idle.calls == 1
    assert rewritten[0].text == "unused"
    assert "pool exhausted marker" in idle.prompts[0]
    assert any(step.step_type == "CONTEXT_ASSEMBLE" for step in tracer.steps)

    judge = ScriptedCompleter(
        '{"verifications": [{"claim": "unmatched", "label": "UNSUPPORTED", "citation_ids": []}]}'
    )
    judge_tracer = TrajectoryTracer()
    judged = SelfCorrectingRAG(
        retriever=ScriptedRetriever([_chunk("auth", ANSWER)]),
        reranker=SequenceReranker([0.95]),
        verifier=LLMClaimVerifier(ContextAssembler(tracer=judge_tracer), judge),
        tracer=judge_tracer,
    )
    judged_result = await judged.answer("why does login fail")
    assert judge.calls == 1
    assert ANSWER in judge.prompts[0]
    assert judged_result.status == "ANSWERED"
    kinds = {step.step_type for step in judge_tracer.steps}
    assert {"RETRIEVAL", "CONTEXT_ASSEMBLE"} <= kinds


@pytest.mark.asyncio
async def test_empty_query_and_unconfigured_loop_are_rejected() -> None:
    with pytest.raises(AgenticRAGError):
        SelfCorrectingRAG()
    engine = SelfCorrectingRAG(retriever=ScriptedRetriever([_chunk("auth", ANSWER)]))
    with pytest.raises(AgenticRAGError):
        await engine.answer("   ")
    with pytest.raises(AgenticRAGError):
        SelfCorrectingRAG(retriever=ScriptedRetriever([_chunk("auth", ANSWER)]), max_steps=0)
