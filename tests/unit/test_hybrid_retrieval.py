"""Offline Phase 2 checks: chunking, BM25, Qdrant, and hybrid ranking."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from beh3m0th.core.schemas import DocumentChunk, TrajectoryStep
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.retrieval.hybrid import (
    BM25Index,
    FastEmbedEmbedder,
    FastEmbedReranker,
    HybridRetriever,
    HybridVectorStore,
    MetadataFilter,
    tokenize,
)
from beh3m0th.retrieval.hybrid.retriever import _reciprocal_rank_fusion
from beh3m0th.retrieval.ingestion import ChunkingError, chunk_markdown, chunk_python

PROSE = (
    "Users cannot sign in when the connection pool is exhausted "
    "during authentication retries."
)
DISTRACTOR = "login login login form css class login-button styling for the login page"
SYMBOL_TEXT = "def checkout(self):\n    return self.pool.acquire()"
NEIGHBOR = "The connection pool waits before giving a session to the caller."
CONCEPTUAL = "why login fail"
SYMBOL_QUERY = "AuthPool.checkout"


class MappedEmbedder:
    dimension = 4

    def __init__(
        self,
        documents: dict[str, list[float]],
        queries: dict[str, list[float]],
    ) -> None:
        self._documents = documents
        self._queries = queries
        self.query_calls = 0
        self.document_calls = 0

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [list(self._documents[text]) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return list(self._queries[text])


class ScriptedReranker:
    async def score(self, query: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        for text in texts:
            if query == CONCEPTUAL and text == PROSE:
                scores.append(0.95)
            elif query == SYMBOL_QUERY and "def checkout" in text:
                scores.append(0.95)
            else:
                scores.append(0.05)
        return scores


class FlatEmbedder:
    dimension = 4

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class FlatReranker:
    async def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.5 for _ in texts]


def _chunk(
    chunk_id: str,
    text: str,
    source_path: str,
    *,
    symbol_name: str | None = None,
    language: str = "markdown",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        text=text,
        source_path=source_path,
        language=language,
        symbol_name=symbol_name,
        created_at=datetime(2026, 11, 23, tzinfo=UTC),
    )


def test_tokenize_splits_camel_case_and_snake_case() -> None:
    assert tokenize("AuthPool retry_auth") == ["auth", "pool", "retry", "auth"]


def test_markdown_chunks_keep_header_path_and_overlap() -> None:
    guide = "\n".join(
        [
            "# Guide",
            "",
            "Intro paragraph lives here.",
            "",
            "## Auth",
            "",
            "Users sign in here.",
            "",
            "### Pool",
            "",
            "The pool is exhausted.",
        ]
    )
    chunks = chunk_markdown(guide, source_path="docs/guide.md", doc_id="guide")
    headers = [chunk.section_header for chunk in chunks]
    assert headers == ["Guide", "Guide > Auth", "Guide > Auth > Pool"]
    assert all(chunk.language == "markdown" for chunk in chunks)
    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [(3, 3), (7, 7), (11, 11)]

    notes = "\n".join(
        [
            "# Notes",
            "",
            "alpha beta gamma delta",
            "",
            "epsilon zeta eta theta",
        ]
    )
    overlapped = chunk_markdown(
        notes,
        source_path="docs/notes.md",
        doc_id="notes",
        max_chars=30,
        overlap_lines=1,
    )
    assert len(overlapped) == 2
    assert overlapped[0].section_header == "Notes"
    assert overlapped[1].section_header == "Notes"
    assert overlapped[0].text == "alpha beta gamma delta"
    assert overlapped[1].text.startswith("alpha beta gamma delta")
    assert "epsilon zeta eta theta" in overlapped[1].text


def test_python_ast_chunks_symbols_and_rejects_invalid_syntax() -> None:
    source = "\n".join(
        [
            "class AuthPool:",
            "    def checkout(self):",
            "        return 1",
            "",
            "async def retry_auth():",
            "    return 2",
        ]
    )
    chunks = chunk_python(source, source_path="src/auth.py", doc_id="auth")
    symbols = [chunk.symbol_name for chunk in chunks]
    assert symbols == ["AuthPool", "AuthPool.checkout", "retry_auth"]
    assert all(chunk.language == "python" for chunk in chunks)
    assert all(
        chunk.line_start is not None
        and chunk.line_end is not None
        and chunk.line_end >= chunk.line_start
        for chunk in chunks
    )
    checkout = next(chunk for chunk in chunks if chunk.symbol_name == "AuthPool.checkout")
    assert "def checkout" in checkout.text
    with pytest.raises(ChunkingError):
        chunk_python("def (\n", source_path="src/broken.py", doc_id="broken")


def test_reciprocal_rank_fusion_uses_rank_not_raw_score() -> None:
    first = _chunk("a", "alpha", "src/a.py")
    second = _chunk("b", "beta", "src/b.py")
    fused = _reciprocal_rank_fusion(
        [(second, 0.99), (first, 0.01)],
        [(first, 3.0)],
        k=60,
        dense_weight=1.0,
        bm25_weight=1.0,
    )
    scores = {chunk.chunk_id: score for chunk, score in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 61)


def test_bm25_index_reloads_from_json(tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    chunk = _chunk(
        "symbol",
        SYMBOL_TEXT,
        "src/auth/pool.py",
        symbol_name="AuthPool.checkout",
        language="python",
    )
    index = BM25Index(path)
    index.upsert([chunk])
    loaded = BM25Index(path)
    loaded.load()
    hits = loaded.search(SYMBOL_QUERY, None, limit=5)
    assert hits[0][0].chunk_id == "symbol"


def test_fastembed_wrappers_do_not_load_weights() -> None:
    embedder = FastEmbedEmbedder()
    reranker = FastEmbedReranker()
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"
    assert embedder.dimension == 384
    assert embedder._model is None
    assert reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert reranker._model is None


@pytest.mark.asyncio
async def test_path_prefix_filter_drops_other_files(tmp_path: Path) -> None:
    auth = _chunk("auth", "login handler in the auth pool", "src/auth/pool.py")
    ui = _chunk("ui", "login button css", "src/ui/login.py")
    store = HybridVectorStore(path=tmp_path / "qdrant", vector_size=4)
    retriever = HybridRetriever(
        embedder=FlatEmbedder(),
        reranker=FlatReranker(),
        vector_store=store,
        bm25=BM25Index(tmp_path / "bm25.json"),
    )
    try:
        await retriever.index([auth, ui])
        result = await retriever.search(
            "login",
            filters=MetadataFilter(path_prefix="src/auth"),
            top_k=5,
        )
    finally:
        await store.close()
    assert [hit.chunk.chunk_id for hit in result.hits] == ["auth"]


@pytest.mark.asyncio
async def test_hybrid_beats_bm25_only_and_dense_only(tmp_path: Path) -> None:
    prose = _chunk("prose", PROSE, "docs/auth.md")
    distractor = _chunk("distractor", DISTRACTOR, "docs/ui.md")
    symbol = _chunk(
        "symbol",
        SYMBOL_TEXT,
        "src/auth/pool.py",
        symbol_name="AuthPool.checkout",
        language="python",
    )
    neighbor = _chunk("neighbor", NEIGHBOR, "docs/pool.md")
    chunks = [prose, distractor, symbol, neighbor]
    embedder = MappedEmbedder(
        documents={
            PROSE: [1.0, 0.0, 0.0, 0.0],
            DISTRACTOR: [0.0, 1.0, 0.0, 0.0],
            SYMBOL_TEXT: [0.0, 0.0, 1.0, 0.0],
            NEIGHBOR: [0.0, 0.0, 0.0, 1.0],
        },
        queries={
            CONCEPTUAL: [1.0, 0.0, 0.0, 0.0],
            SYMBOL_QUERY: [0.0, 0.0, 0.0, 1.0],
        },
    )
    store = HybridVectorStore(path=tmp_path / "qdrant", vector_size=4)
    bm25 = BM25Index(tmp_path / "bm25.json")
    tracer = TrajectoryTracer(tmp_path / "trace.jsonl")
    retriever = HybridRetriever(
        embedder=embedder,
        reranker=ScriptedReranker(),
        vector_store=store,
        bm25=bm25,
        tracer=tracer,
        agent_name="researcher",
    )
    try:
        await retriever.index(chunks)
        conceptual = await retriever.search(CONCEPTUAL, top_k=4)
        assert embedder.query_calls == 1
        repeated = await retriever.search(CONCEPTUAL, top_k=4)
        assert embedder.query_calls == 1
        assert repeated.metrics.cache_hit is True
        symbol_hit = await retriever.search(SYMBOL_QUERY, top_k=4)
        conceptual_vector = await embedder.embed_query(CONCEPTUAL)
        symbol_vector = await embedder.embed_query(SYMBOL_QUERY)
        dense_conceptual = await store.search(conceptual_vector, None, limit=4)
        dense_symbol = await store.search(symbol_vector, None, limit=4)
    finally:
        await store.close()

    bm25_conceptual = bm25.search(CONCEPTUAL, None, limit=4)
    bm25_symbol = bm25.search(SYMBOL_QUERY, None, limit=4)
    assert conceptual.hits[0].chunk.chunk_id == "prose"
    assert bm25_conceptual[0][0].chunk_id == "distractor"
    assert dense_conceptual[0][0].chunk_id == "prose"
    assert symbol_hit.hits[0].chunk.chunk_id == "symbol"
    assert bm25_symbol[0][0].chunk_id == "symbol"
    assert dense_symbol[0][0].chunk_id == "neighbor"
    hybrid_wins_both = (
        conceptual.hits[0].chunk.chunk_id == "prose"
        and symbol_hit.hits[0].chunk.chunk_id == "symbol"
    )
    bm25_wins_both = (
        bm25_conceptual[0][0].chunk_id == "prose" and bm25_symbol[0][0].chunk_id == "symbol"
    )
    dense_wins_both = (
        dense_conceptual[0][0].chunk_id == "prose" and dense_symbol[0][0].chunk_id == "symbol"
    )
    assert hybrid_wins_both
    assert not bm25_wins_both
    assert not dense_wins_both

    assert len(tracer.steps) == 3
    step = tracer.steps[0]
    assert step.step_type == "RETRIEVAL"
    assert step.agent_name == "researcher"
    for field in ("embed_ms", "bm25_ms", "vector_ms", "rerank_ms", "total_ms"):
        assert field in step.metadata
        assert float(step.metadata[field]) >= 0
    loaded = TrajectoryStep.model_validate_json(
        (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert loaded.step_type == "RETRIEVAL"
