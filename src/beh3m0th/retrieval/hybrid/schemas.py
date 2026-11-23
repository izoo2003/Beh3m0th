"""Filter, metrics, and search result models for hybrid retrieval."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from beh3m0th.core.schemas import DocumentChunk, RetrievalHit, StrictModel


class MetadataFilter(StrictModel):
    """Constraints applied before BM25 and Qdrant ranking."""

    language: str | None = None
    symbol_name: str | None = None
    tags: list[str] | None = None
    path_prefix: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None

    @model_validator(mode="after")
    def range_is_ordered(self) -> MetadataFilter:
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_before < self.created_after
        ):
            raise ValueError("created_before must be greater than or equal to created_after")
        return self


class RetrievalMetrics(StrictModel):
    """Per-stage latency for one search."""

    embed_ms: float = Field(ge=0.0)
    bm25_ms: float = Field(ge=0.0)
    vector_ms: float = Field(ge=0.0)
    rerank_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)
    cache_hit: bool = False


class HybridSearchResult(StrictModel):
    """Reranked hits plus timing for the query that produced them."""

    query: str = Field(min_length=1)
    hits: list[RetrievalHit]
    metrics: RetrievalMetrics


def chunk_matches(chunk: DocumentChunk, metadata_filter: MetadataFilter | None) -> bool:
    """Return whether a chunk satisfies the filter."""
    if metadata_filter is None:
        return True
    if metadata_filter.language is not None and chunk.language != metadata_filter.language:
        return False
    if (
        metadata_filter.symbol_name is not None
        and chunk.symbol_name != metadata_filter.symbol_name
    ):
        return False
    if metadata_filter.tags is not None and not any(
        tag in metadata_filter.tags for tag in chunk.tags
    ):
        return False
    if metadata_filter.path_prefix is not None and not chunk.source_path.startswith(
        metadata_filter.path_prefix
    ):
        return False
    if (
        metadata_filter.created_after is not None
        and chunk.created_at < metadata_filter.created_after
    ):
        return False
    if (
        metadata_filter.created_before is not None
        and chunk.created_at > metadata_filter.created_before
    ):
        return False
    return True
