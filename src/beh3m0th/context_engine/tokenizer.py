"""Offline token counting with a tiktoken path and a local estimator."""

from __future__ import annotations

from collections.abc import Callable


class TokenCounter:
    """Count tokens, caching each distinct string.

    A caller-supplied ``count_fn`` wins. Otherwise the counter uses tiktoken
    ``cl100k_base`` when that encoding loads without network access, and a
    character estimator when it does not.
    """

    def __init__(
        self,
        count_fn: Callable[[str], int] | None = None,
        *,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self._count_fn = count_fn
        self._cache: dict[str, int] = {}
        self._encoding: object | None = None
        if count_fn is not None:
            return
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        if self._count_fn is not None:
            counted = self._count_fn(text)
        elif self._encoding is not None:
            encode = getattr(self._encoding, "encode", None)
            if not callable(encode):
                counted = self.estimate(text)
            else:
                counted = len(encode(text))
        else:
            counted = self.estimate(text)
        if counted < 0:
            raise ValueError("token count must be non-negative")
        self._cache[text] = counted
        return counted

    @staticmethod
    def estimate(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)
