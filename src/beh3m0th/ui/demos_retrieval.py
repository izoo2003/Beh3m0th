"""Hybrid search, GraphRAG, and grounded-answer demos."""

from __future__ import annotations

from pathlib import Path

from beh3m0th.core.schemas import DocumentChunk, RetrievalHit
from beh3m0th.retrieval.agentic_rag.engine import SelfCorrectingRAG
from beh3m0th.retrieval.graphrag.graph_store import NetworkXGraphStore
from beh3m0th.retrieval.graphrag.pipeline import GraphRAGPipeline
from beh3m0th.retrieval.hybrid.bm25_index import BM25Index
from beh3m0th.retrieval.hybrid.embedder import FastEmbedEmbedder
from beh3m0th.retrieval.hybrid.reranker import FastEmbedReranker
from beh3m0th.retrieval.hybrid.retriever import HybridRetriever
from beh3m0th.retrieval.hybrid.schemas import HybridSearchResult, RetrievalMetrics
from beh3m0th.ui.embed import HashingEmbedder, OverlapReranker
from beh3m0th.ui.models import DemoResult

_SAMPLE = (
    ("auth", "Authentication retries fail when the connection pool is exhausted."),
    ("ui", "The login button uses a blue css class."),
    ("cache", "The prompt cache stores identical requests in SQLite."),
)

_GRAPH = "\n".join(
    [
        "def helper() -> int:",
        "    return 1",
        "",
        "def answer() -> int:",
        "    return helper()",
    ]
)


class _FixedRetriever:
    """Return the same chunks for every subquery. Used by the grounded-answer demo."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = chunks

    async def search(
        self,
        query: str,
        filters: object = None,
        top_k: int = 6,
        fetch_k: int = 20,
    ) -> HybridSearchResult:
        del filters, top_k, fetch_k
        hits = [
            RetrievalHit(chunk=chunk, score=0.5, rank=index, source="hybrid")
            for index, chunk in enumerate(self._chunks, start=1)
        ]
        return HybridSearchResult(
            query=query,
            hits=hits,
            metrics=RetrievalMetrics(
                embed_ms=0.0,
                bm25_ms=0.0,
                vector_ms=0.0,
                rerank_ms=0.0,
                total_ms=0.0,
            ),
        )


class _FixedReranker:
    def __init__(self, score: float) -> None:
        self._score = score

    async def score(self, query: str, texts: list[str]) -> list[float]:
        del query
        return [self._score for _ in texts]


async def search_chunks(query: str, root: Path, *, use_onnx: bool = False) -> DemoResult:
    """Index three built-in chunks and return the fused hits."""
    root.mkdir(parents=True, exist_ok=True)
    embedder = FastEmbedEmbedder() if use_onnx else HashingEmbedder()
    reranker = FastEmbedReranker() if use_onnx else OverlapReranker()
    store = None
    try:
        from beh3m0th.retrieval.hybrid.vector_store import HybridVectorStore

        store = HybridVectorStore(path=root / "qdrant", vector_size=embedder.dimension)
        retriever = HybridRetriever(
            embedder=embedder,
            reranker=reranker,
            vector_store=store,
            bm25=BM25Index(root / "bm25.json"),
        )
        await retriever.index(
            [
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id="demo",
                    text=text,
                    source_path="docs/demo.md",
                )
                for chunk_id, text in _SAMPLE
            ]
        )
        found = await retriever.search(query)
    finally:
        if store is not None:
            await store.close()
    rows = [
        {
            "rank": str(hit.rank),
            "chunk": hit.chunk.chunk_id,
            "source": hit.source,
            "text": hit.chunk.text,
        }
        for hit in found.hits
    ]
    return DemoResult(
        title="Hybrid search",
        summary=f"{len(found.hits)} hits for {query}",
        rows=rows,
    )


async def query_graph(root: Path) -> DemoResult:
    """Ingest a function that calls another and return the two-hop subgraph."""
    pipeline = GraphRAGPipeline(NetworkXGraphStore(root / "graph.json"))
    chunk = DocumentChunk(
        chunk_id="widget",
        doc_id="widget",
        text=_GRAPH,
        source_path="widget.py",
        language="python",
    )
    await pipeline.ingest([chunk])
    found = await pipeline.query("answer helper", hops=2)
    names = ", ".join(entity.name for entity in found.entities)
    chunks = ", ".join(found.chunk_ids)
    return DemoResult(
        title="GraphRAG",
        summary=f"{len(found.entities)} nodes across {chunks or 'no chunks'}",
        rows=[
            {
                "name": entity.name,
                "kind": entity.kind,
                "chunks": ",".join(entity.chunk_ids),
            }
            for entity in found.entities
        ],
        detail=names,
    )


async def answer_question(*, answerable: bool) -> DemoResult:
    """Answer from a real chunk, or abstain when the only chunk is off topic."""
    if answerable:
        chunks = [
            DocumentChunk(
                chunk_id="auth",
                doc_id="demo",
                text="Authentication retries fail when the connection pool is exhausted.",
                source_path="docs/auth.md",
            )
        ]
        score = 0.95
        query = "why do authentication retries fail"
    else:
        chunks = [
            DocumentChunk(
                chunk_id="ui",
                doc_id="demo",
                text="The login button uses a blue css class.",
                source_path="docs/ui.md",
            )
        ]
        score = 0.05
        query = "why do authentication retries fail"
    result = await SelfCorrectingRAG(
        retriever=_FixedRetriever(chunks),
        reranker=_FixedReranker(score),
    ).answer(query)
    citations = ", ".join(item.chunk_id for item in result.citations) or "none"
    return DemoResult(
        title="Grounded answer",
        summary=result.status,
        rows=[{"field": "citations", "value": citations}, {"field": "query", "value": query}],
        detail=result.claims[0].text if result.claims else "",
    )
