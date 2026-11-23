"""Lazy FastEmbed cross-encoder and score normalization."""

from __future__ import annotations

import asyncio
from typing import Protocol


class Reranker(Protocol):
    """Scores query-document pairs. Higher means more relevant."""

    async def score(self, query: str, texts: list[str]) -> list[float]: ...


def normalize_scores(scores: list[float]) -> list[float]:
    """Map scores into 0..1. A flat list becomes all ones."""
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0 for _ in scores]
    span = high - low
    return [(score - low) / span for score in scores]


class FastEmbedReranker:
    """Local CPU cross-encoder. The ONNX model loads on the first score call."""

    model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        if model_name is not None:
            self.model_name = model_name
        self._model: object | None = None

    async def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return await asyncio.to_thread(self._score, query, texts)

    def _score(self, query: str, texts: list[str]) -> list[float]:
        model = self._load()
        raw = [float(value) for value in model.rerank(query, texts)]
        return normalize_scores(raw)

    def _load(self) -> object:
        if self._model is None:
            from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name, lazy_load=True)
        model = self._model
        if not hasattr(model, "rerank"):
            raise RuntimeError("FastEmbed reranker is not loaded")
        return model
