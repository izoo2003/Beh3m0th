"""Qdrant upsert and filtered search for local path or remote URL."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from beh3m0th.core.schemas import DocumentChunk
from beh3m0th.retrieval.hybrid.schemas import MetadataFilter, chunk_matches

_NAMESPACE = uuid.UUID("6b0f0b3e-7c1a-4d5e-9a2b-1c3d5e7f90ab")


class HybridVectorStore:
    """Dense chunk store. Embedded `path=` is the default; `url=` targets Docker or cloud."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        url: str | None = None,
        collection: str = "beh3m0th_chunks",
        vector_size: int = 384,
    ) -> None:
        if (path is None) == (url is None):
            raise ValueError("pass exactly one of path or url")
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        self.path = path
        self.url = url
        self.collection = collection
        self.vector_size = vector_size
        self._client: object | None = None

    async def initialize(self) -> None:
        await asyncio.to_thread(self._open)

    async def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        await asyncio.to_thread(self._upsert, chunks, vectors)

    async def search(
        self,
        vector: list[float],
        metadata_filter: MetadataFilter | None,
        limit: int,
    ) -> list[tuple[DocumentChunk, float]]:
        if limit < 1:
            return []
        return await asyncio.to_thread(self._search, vector, metadata_filter, limit)

    async def close(self) -> None:
        await asyncio.to_thread(self._close)

    def _open(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, KeywordIndexParams, VectorParams

        if self._client is None:
            if self.path is not None:
                self.path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(self.path))
            else:
                self._client = QdrantClient(url=self.url)
        client = self._require_client()
        existing = {item.name for item in client.get_collections().collections}
        if self.collection not in existing:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
        if self.url is not None:
            schema = client.get_collection(self.collection).payload_schema
            if "source_path" not in schema:
                client.create_payload_index(
                    collection_name=self.collection,
                    field_name="source_path",
                    field_schema=KeywordIndexParams(type="keyword", prefix=True),
                )

    def _upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        client = self._require_client()
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.vector_size:
                raise ValueError(
                    f"vector for {chunk.chunk_id} has size {len(vector)}, "
                    f"expected {self.vector_size}"
                )
            points.append(
                PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=[float(value) for value in vector],
                    payload=_payload(chunk),
                )
            )
        if points:
            client.upsert(collection_name=self.collection, points=points, wait=True)

    def _search(
        self,
        vector: list[float],
        metadata_filter: MetadataFilter | None,
        limit: int,
    ) -> list[tuple[DocumentChunk, float]]:
        client = self._require_client()
        response = client.query_points(
            collection_name=self.collection,
            query=[float(value) for value in vector],
            query_filter=_qdrant_filter(metadata_filter),
            limit=limit,
        )
        hits: list[tuple[DocumentChunk, float]] = []
        for point in response.points:
            payload = point.payload or {}
            raw = payload.get("chunk_json")
            if not isinstance(raw, str):
                continue
            chunk = DocumentChunk.model_validate_json(raw)
            if not chunk_matches(chunk, metadata_filter):
                continue
            hits.append((chunk, max(0.0, float(point.score))))
        return hits

    def _close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and hasattr(client, "close"):
            client.close()

    def _require_client(self) -> object:
        from qdrant_client import QdrantClient

        client = self._client
        if not isinstance(client, QdrantClient):
            raise RuntimeError("HybridVectorStore is not initialized")
        return client


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _payload(chunk: DocumentChunk) -> dict[str, object]:
    return {
        "chunk_json": chunk.model_dump_json(),
        "language": chunk.language or "",
        "symbol_name": chunk.symbol_name or "",
        "source_path": chunk.source_path,
        "tags": list(chunk.tags),
        "created_at": chunk.created_at.timestamp(),
    }


def _qdrant_filter(metadata_filter: MetadataFilter | None) -> object | None:
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchAny,
        MatchPrefix,
        MatchValue,
        Range,
    )

    if metadata_filter is None:
        return None
    must: list[object] = []
    if metadata_filter.language is not None:
        must.append(
            FieldCondition(key="language", match=MatchValue(value=metadata_filter.language))
        )
    if metadata_filter.symbol_name is not None:
        must.append(
            FieldCondition(
                key="symbol_name",
                match=MatchValue(value=metadata_filter.symbol_name),
            )
        )
    if metadata_filter.tags is not None:
        must.append(FieldCondition(key="tags", match=MatchAny(any=list(metadata_filter.tags))))
    if metadata_filter.path_prefix is not None:
        must.append(
            FieldCondition(
                key="source_path",
                match=MatchPrefix(prefix=metadata_filter.path_prefix),
            )
        )
    if metadata_filter.created_after is not None or metadata_filter.created_before is not None:
        must.append(
            FieldCondition(
                key="created_at",
                range=Range(
                    gte=(
                        metadata_filter.created_after.timestamp()
                        if metadata_filter.created_after is not None
                        else None
                    ),
                    lte=(
                        metadata_filter.created_before.timestamp()
                        if metadata_filter.created_before is not None
                        else None
                    ),
                ),
            )
        )
    if not must:
        return None
    return Filter(must=must)
