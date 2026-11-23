"""Deterministic stand-ins so the UI never downloads an ONNX model by default."""

from __future__ import annotations

import hashlib

from beh3m0th.retrieval.hybrid.bm25_index import tokenize


class HashingEmbedder:
    """384-d hashed vectors. Same text always maps to the same vector."""

    dimension: int = 384

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vector(text, self.dimension) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return _vector(text, self.dimension)


class OverlapReranker:
    """Score a passage by how many query tokens it contains."""

    async def score(self, query: str, texts: list[str]) -> list[float]:
        needed = set(tokenize(query))
        scores: list[float] = []
        for text in texts:
            seen = set(tokenize(text))
            if not needed or not seen:
                scores.append(0.0)
            else:
                scores.append(len(needed & seen) / len(needed))
        return scores


def _vector(text: str, dimension: int) -> list[float]:
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dimension:
        seed = hashlib.sha256(seed).digest()
        for byte in seed:
            values.append((byte / 127.5) - 1.0)
            if len(values) == dimension:
                break
    return values
