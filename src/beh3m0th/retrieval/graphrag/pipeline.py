"""Ingest chunks into the knowledge graph and answer local or global queries."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

from beh3m0th.core.schemas import DocumentChunk, MetadataValue, TrajectoryStep
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.retrieval.graphrag.community import detect_communities, hashing_vector
from beh3m0th.retrieval.graphrag.errors import GraphQueryError
from beh3m0th.retrieval.graphrag.extractor import extract_prose, extract_python, merge_results
from beh3m0th.retrieval.graphrag.graph_store import GraphStoreProtocol
from beh3m0th.retrieval.graphrag.llm_extract import LLMProseExtractor
from beh3m0th.retrieval.graphrag.resolve import merge_by_cosine
from beh3m0th.retrieval.graphrag.schemas import (
    CommunitySummary,
    Entity,
    GraphExtractionResult,
    GraphTraversal,
    QueryMode,
    SubgraphResult,
)
from beh3m0th.retrieval.hybrid.bm25_index import tokenize
from beh3m0th.retrieval.hybrid.vector_store import HybridVectorStore


class NameEmbedder(Protocol):
    """Embeds canonical names for optional cosine deduplication."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class GraphRAGPipeline:
    """Build a knowledge graph and search it by hops or by community."""

    def __init__(
        self,
        store: GraphStoreProtocol,
        *,
        tracer: TrajectoryTracer | None = None,
        agent_name: str = "graphrag",
        prose_extractor: LLMProseExtractor | None = None,
        embedder: NameEmbedder | None = None,
        vector_store: HybridVectorStore | None = None,
    ) -> None:
        if not agent_name.strip():
            raise ValueError("agent_name must not be empty")
        self._store = store
        self._tracer = tracer
        self._agent_name = agent_name
        self._prose = prose_extractor
        self._embedder = embedder
        self._vector_store = vector_store

    async def ingest(self, chunks: list[DocumentChunk]) -> GraphExtractionResult:
        started = time.perf_counter()
        parts = [await self._extract(chunk) for chunk in chunks]
        merged = merge_results(parts)
        if self._embedder is not None and merged.entities:
            vectors = await self._embedder.embed_documents(
                [entity.canonical_name for entity in merged.entities]
            )
            merged = merge_by_cosine(merged, vectors)
        await self._store.upsert(merged)
        entities = await self._store.entities()
        relationships = await self._store.relationships()
        communities = await asyncio.to_thread(detect_communities, entities, relationships)
        await self._store.add_communities(communities)
        if self._vector_store is not None and communities:
            await self._index_communities(communities)
        elapsed = (time.perf_counter() - started) * 1000
        self._record(
            input_summary=",".join(chunk.chunk_id for chunk in chunks)[:200] or "empty",
            output_summary=(
                f"{len(merged.entities)} entities, "
                f"{len(merged.relationships)} edges, "
                f"{len(communities)} communities"
            ),
            latency_ms=elapsed,
            metadata={
                "entity_count": len(merged.entities),
                "relationship_count": len(merged.relationships),
                "community_count": len(communities),
            },
        )
        return merged

    async def query(
        self,
        text: str,
        mode: QueryMode = "local",
        hops: int = 2,
    ) -> SubgraphResult:
        started = time.perf_counter()
        if not text.strip():
            raise GraphQueryError("query must not be empty")
        if mode not in ("local", "global", "hybrid"):
            raise GraphQueryError("mode must be local, global, or hybrid")
        if hops not in (1, 2, 3):
            raise GraphQueryError("hops must be between 1 and 3")
        local = None
        communities: list[CommunitySummary] = []
        if mode in ("local", "hybrid"):
            entities = await self._store.entities()
            local = await self._store.traverse(_anchors(text, entities), hops)
        if mode in ("global", "hybrid"):
            communities = await self._rank_communities(text)
        result = _assemble(text, mode, local, communities)
        self._record(
            input_summary=text[:200],
            output_summary=",".join(result.chunk_ids[:8]) or "no chunks",
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={
                "mode": mode,
                "hops": hops,
                "entity_count": len(result.entities),
                "chunk_ids": ",".join(result.chunk_ids),
            },
        )
        return result

    async def _extract(self, chunk: DocumentChunk) -> GraphExtractionResult:
        if chunk.language == "python":
            return await asyncio.to_thread(extract_python, chunk)
        if self._prose is not None:
            return await self._prose.extract(chunk)
        return await asyncio.to_thread(extract_prose, chunk)

    async def _index_communities(self, communities: list[CommunitySummary]) -> None:
        store = self._vector_store
        if store is None:
            return
        await store.initialize()
        chunks = [
            DocumentChunk(
                chunk_id=summary.community_id,
                doc_id=summary.community_id,
                text=summary.summary,
                source_path="graph://communities",
                language="community",
                tags=["community"],
            )
            for summary in communities
        ]
        vectors = [hashing_vector(chunk.text, store.vector_size) for chunk in chunks]
        await store.upsert(chunks, vectors)

    async def _rank_communities(self, query: str) -> list[CommunitySummary]:
        communities = await self._store.communities()
        overlap = {
            summary.community_id: _overlap(query, summary.summary) for summary in communities
        }
        vector_scores: dict[str, float] = {}
        if self._vector_store is not None and communities:
            vector = hashing_vector(query, self._vector_store.vector_size)
            hits = await self._vector_store.search(vector, None, limit=len(communities))
            vector_scores = {hit.chunk_id: score for hit, score in hits}
        return sorted(
            communities,
            key=lambda summary: (
                -vector_scores.get(summary.community_id, 0.0),
                -overlap[summary.community_id],
                summary.community_id,
            ),
        )

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
                input_summary=input_summary,
                output_summary=output_summary,
                tokens_used=0,
                latency_ms=latency_ms,
                metadata=metadata,
            )
        )


def _anchors(query: str, entities: list[Entity]) -> list[str]:
    tokens = set(tokenize(query))
    if not tokens:
        return []
    found: list[str] = []
    for entity in entities:
        if entity.kind == "Community":
            continue
        labels = [entity.name, entity.canonical_name, *entity.aliases]
        if any(tokens & set(tokenize(label)) for label in labels):
            found.append(entity.entity_id)
    return found


def _overlap(query: str, summary: str) -> float:
    query_tokens = set(tokenize(query))
    summary_tokens = set(tokenize(summary))
    if not query_tokens or not summary_tokens:
        return 0.0
    return len(query_tokens & summary_tokens) / len(query_tokens)


def _assemble(
    text: str,
    mode: QueryMode,
    local: GraphTraversal | None,
    communities: list[CommunitySummary],
) -> SubgraphResult:
    entities = [] if local is None else list(local.entities)
    relationships = [] if local is None else list(local.relationships)
    chunk_ids = [] if local is None else list(local.chunk_ids)
    score = 0.0 if local is None else local.score
    if communities:
        score += _overlap(text, communities[0].summary)
    return SubgraphResult(
        query=text,
        mode=mode,
        entities=entities,
        relationships=relationships,
        chunk_ids=chunk_ids,
        communities=communities,
        score=score,
    )
