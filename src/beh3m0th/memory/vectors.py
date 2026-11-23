"""Float32 embedding blobs and batched cosine similarity."""

from __future__ import annotations

import asyncio

import numpy as np

from beh3m0th.memory.errors import MemoryError


def pack_vector(values: list[float]) -> bytes:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise MemoryError("embedding must be a non-empty vector")
    return array.tobytes()


def unpack_vector(blob: bytes) -> np.ndarray:
    if len(blob) == 0 or len(blob) % 4 != 0:
        raise MemoryError("embedding blob is empty or misaligned")
    return np.frombuffer(blob, dtype=np.float32).astype(np.float32, copy=True)


def cosine_against(query: list[float], blobs: list[bytes]) -> list[float]:
    """Cosine of one query against many stored float32 blobs."""
    if not blobs:
        return []
    matrix = np.stack([unpack_vector(blob) for blob in blobs])
    query_vector = np.asarray(query, dtype=np.float32)
    if matrix.shape[1] != query_vector.shape[0]:
        raise MemoryError("query embedding dimension does not match stored memories")
    query_norm = float(np.linalg.norm(query_vector))
    row_norms = np.linalg.norm(matrix, axis=1)
    dots = matrix @ query_vector
    scores = np.zeros(len(blobs), dtype=np.float64)
    if query_norm > 0.0:
        denom = row_norms * query_norm
        mask = denom > 0.0
        scores[mask] = dots[mask] / denom[mask]
    return [float(value) for value in scores]


async def cosine_scores(query: list[float], blobs: list[bytes]) -> list[float]:
    """Run the vectorized cosine off the event loop."""
    return await asyncio.to_thread(cosine_against, query, blobs)
