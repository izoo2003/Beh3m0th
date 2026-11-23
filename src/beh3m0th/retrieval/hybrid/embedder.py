"""Lazy FastEmbed ONNX embedder and an injectable embedder protocol."""

from __future__ import annotations

import asyncio
from typing import Protocol


class Embedder(Protocol):
    """Dense vectors for documents and queries."""

    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local CPU embedder. The ONNX model loads on the first embed call."""

    model_name: str = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str | None = None) -> None:
        if model_name is not None:
            self.model_name = model_name
        self.dimension = 384
        self._model: object | None = None

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed, texts)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        embedded = model.embed(texts)
        return [[float(value) for value in vector] for vector in embedded]

    def _load(self) -> object:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        model = self._model
        if not hasattr(model, "embed"):
            raise RuntimeError("FastEmbed model is not loaded")
        return model
