"""BM25 + Qdrant hybrid search with reciprocal rank fusion and reranking."""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict

from beh3m0th.core.schemas import DocumentChunk, RetrievalHit, TrajectoryStep
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.retrieval.hybrid.bm25_index import BM25Index
from beh3m0th.retrieval.hybrid.embedder import Embedder
from beh3m0th.retrieval.hybrid.reranker import Reranker, normalize_scores
from beh3m0th.retrieval.hybrid.schemas import (
    HybridSearchResult,
    MetadataFilter,
    RetrievalMetrics,
)
from beh3m0th.retrieval.hybrid.vector_store import HybridVectorStore

type _ScoredChunk = tuple[DocumentChunk, float]


class HybridRetriever:
    """Index chunks locally and search with RRF plus a cross-encoder."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        reranker: Reranker,
        vector_store: HybridVectorStore,
        bm25: BM25Index,
        tracer: TrajectoryTracer | None = None,
        agent_name: str = "retriever",
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        cache_size: int = 128,
    ) -> None:
        if not agent_name.strip():
            raise ValueError("agent_name must not be empty")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self._embedder = embedder
        self._reranker = reranker
        self._store = vector_store
        self._bm25 = bm25
        self._tracer = tracer
        self._agent_name = agent_name
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight
        self._query_vectors: _LRU = _LRU(cache_size)
        self._results: _LRU = _LRU(cache_size)

    async def index(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        await self._store.initialize()
        vectors = await self._embedder.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedder returned the wrong number of vectors")
        await self._store.upsert(chunks, vectors)
        await asyncio.to_thread(self._bm25.upsert, chunks)
        self._query_vectors.clear()
        self._results.clear()

    async def search(
        self,
        query: str,
        filters: MetadataFilter | None = None,
        top_k: int = 6,
        fetch_k: int = 20,
    ) -> HybridSearchResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1 or fetch_k < 1:
            raise ValueError("top_k and fetch_k must be positive")
        started = time.perf_counter()
        cache_key = _cache_key(query, filters, top_k, fetch_k)
        cached = self._results.get(cache_key)
        if isinstance(cached, HybridSearchResult):
            result = cached.model_copy(
                update={"metrics": cached.metrics.model_copy(update={"cache_hit": True})}
            )
            self._record(result)
            return result
        if not self._bm25.chunks:
            result = _empty_result(query, (time.perf_counter() - started) * 1000)
            self._record(result)
            return result

        candidate_k = max(top_k, fetch_k)
        dense_outcome, lexical_outcome = await asyncio.gather(
            self._dense(query, filters, candidate_k),
            self._lexical(query, filters, candidate_k),
        )
        dense_hits, embed_ms, vector_ms = dense_outcome
        lexical_hits, bm25_ms = lexical_outcome
        fused = _reciprocal_rank_fusion(
            dense_hits,
            lexical_hits,
            k=self._rrf_k,
            dense_weight=self._dense_weight,
            bm25_weight=self._bm25_weight,
        )[:candidate_k]
        rerank_started = time.perf_counter()
        hits = await self._rerank(query, fused, top_k)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        result = HybridSearchResult(
            query=query,
            hits=hits,
            metrics=RetrievalMetrics(
                embed_ms=embed_ms,
                bm25_ms=bm25_ms,
                vector_ms=vector_ms,
                rerank_ms=rerank_ms,
                total_ms=(time.perf_counter() - started) * 1000,
            ),
        )
        self._results.put(cache_key, result)
        self._record(result)
        return result

    async def _dense(
        self,
        query: str,
        filters: MetadataFilter | None,
        limit: int,
    ) -> tuple[list[_ScoredChunk], float, float]:
        started = time.perf_counter()
        vector = await self._embed_query(query)
        embed_ms = (time.perf_counter() - started) * 1000
        search_started = time.perf_counter()
        hits = await self._store.search(vector, filters, limit)
        vector_ms = (time.perf_counter() - search_started) * 1000
        return hits, embed_ms, vector_ms

    async def _lexical(
        self,
        query: str,
        filters: MetadataFilter | None,
        limit: int,
    ) -> tuple[list[_ScoredChunk], float]:
        started = time.perf_counter()
        hits = await asyncio.to_thread(self._bm25.search, query, filters, limit)
        return hits, (time.perf_counter() - started) * 1000

    async def _embed_query(self, query: str) -> list[float]:
        cached = self._query_vectors.get(query)
        if isinstance(cached, list):
            return [float(value) for value in cached]
        vector = [float(value) for value in await self._embedder.embed_query(query)]
        self._query_vectors.put(query, vector)
        return vector

    async def _rerank(
        self,
        query: str,
        fused: list[_ScoredChunk],
        top_k: int,
    ) -> list[RetrievalHit]:
        if not fused:
            return []
        raw = await self._reranker.score(query, [chunk.text for chunk, _ in fused])
        if len(raw) != len(fused):
            raise ValueError("reranker returned the wrong number of scores")
        normalized = normalize_scores([float(score) for score in raw])
        order = sorted(range(len(fused)), key=lambda index: (-normalized[index], index))
        hits: list[RetrievalHit] = []
        for rank, index in enumerate(order[:top_k], start=1):
            chunk, _score = fused[index]
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=normalized[index],
                    rank=rank,
                    source="hybrid",
                )
            )
        return hits

    def _record(self, result: HybridSearchResult) -> None:
        if self._tracer is None:
            return
        metrics = result.metrics
        preview = ", ".join(hit.chunk.chunk_id for hit in result.hits[:5])
        self._tracer.record(
            TrajectoryStep(
                step_index=len(self._tracer.steps),
                agent_name=self._agent_name,
                step_type="RETRIEVAL",
                input_summary=result.query[:200],
                output_summary=preview or "no hits",
                tokens_used=0,
                latency_ms=metrics.total_ms,
                metadata={
                    "embed_ms": metrics.embed_ms,
                    "bm25_ms": metrics.bm25_ms,
                    "vector_ms": metrics.vector_ms,
                    "rerank_ms": metrics.rerank_ms,
                    "total_ms": metrics.total_ms,
                    "cache_hit": metrics.cache_hit,
                    "hit_count": len(result.hits),
                },
            )
        )


class _LRU:
    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str) -> object | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: object) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


def _reciprocal_rank_fusion(
    dense_hits: list[_ScoredChunk],
    lexical_hits: list[_ScoredChunk],
    *,
    k: int,
    dense_weight: float,
    bm25_weight: float,
) -> list[_ScoredChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, DocumentChunk] = {}
    _add_ranks(dense_hits, scores, chunks, weight=dense_weight, k=k)
    _add_ranks(lexical_hits, scores, chunks, weight=bm25_weight, k=k)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [(chunks[chunk_id], scores[chunk_id]) for chunk_id in ordered]


def _add_ranks(
    hits: list[_ScoredChunk],
    scores: dict[str, float],
    chunks: dict[str, DocumentChunk],
    *,
    weight: float,
    k: int,
) -> None:
    for rank, (chunk, _score) in enumerate(hits, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
        chunks[chunk.chunk_id] = chunk


def _cache_key(
    query: str,
    filters: MetadataFilter | None,
    top_k: int,
    fetch_k: int,
) -> str:
    payload = filters.model_dump(mode="json") if filters is not None else None
    return json.dumps(
        {"query": query, "filters": payload, "top_k": top_k, "fetch_k": fetch_k},
        sort_keys=True,
        default=str,
    )


def _empty_result(query: str, total_ms: float) -> HybridSearchResult:
    return HybridSearchResult(
        query=query,
        hits=[],
        metrics=RetrievalMetrics(
            embed_ms=0.0,
            bm25_ms=0.0,
            vector_ms=0.0,
            rerank_ms=0.0,
            total_ms=total_ms,
        ),
    )
