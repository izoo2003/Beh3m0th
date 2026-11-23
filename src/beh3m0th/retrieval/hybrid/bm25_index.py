"""Code-aware BM25 index persisted as JSON."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from beh3m0th.core.schemas import DocumentChunk
from beh3m0th.retrieval.hybrid.schemas import MetadataFilter, chunk_matches

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, splitting snake_case and camelCase identifiers."""
    tokens: list[str] = []
    for raw in _TOKEN.findall(text):
        for part in raw.split("_"):
            if not part:
                continue
            pieces = _CAMEL.findall(part)
            if pieces:
                tokens.extend(piece.lower() for piece in pieces)
            else:
                tokens.append(part.lower())
    return tokens


class BM25Index:
    """Lexical index. The saved file stores chunks and tokens, not the BM25 object."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._chunks: list[DocumentChunk] = []
        self._tokens: list[list[str]] = []
        self._model: BM25Okapi | None = None

    @property
    def chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        by_id = {chunk.chunk_id: index for index, chunk in enumerate(self._chunks)}
        for chunk in chunks:
            tokens = tokenize(_indexed_text(chunk)) or ["empty"]
            existing = by_id.get(chunk.chunk_id)
            if existing is None:
                by_id[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
                self._tokens.append(tokens)
            else:
                self._chunks[existing] = chunk
                self._tokens[existing] = tokens
        self._rebuild()
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
            "tokens": self._tokens,
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self) -> None:
        if not self.path.is_file():
            self._chunks = []
            self._tokens = []
            self._model = None
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._chunks = [DocumentChunk.model_validate(item) for item in payload["chunks"]]
        self._tokens = [list(tokens) for tokens in payload["tokens"]]
        self._rebuild()

    def search(
        self,
        query: str,
        metadata_filter: MetadataFilter | None,
        limit: int,
    ) -> list[tuple[DocumentChunk, float]]:
        if limit < 1 or self._model is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_terms = set(query_tokens)
        scores = self._model.get_scores(query_tokens)
        ranked: list[tuple[DocumentChunk, float]] = []
        for chunk, tokens, score in zip(self._chunks, self._tokens, scores, strict=True):
            if not chunk_matches(chunk, metadata_filter):
                continue
            if query_terms.isdisjoint(tokens):
                continue
            ranked.append((chunk, float(score)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    def _rebuild(self) -> None:
        if not self._tokens:
            self._model = None
            return
        self._model = BM25Okapi(self._tokens)


def _indexed_text(chunk: DocumentChunk) -> str:
    parts = [chunk.symbol_name or "", chunk.section_header or "", chunk.text]
    return "\n".join(part for part in parts if part)
