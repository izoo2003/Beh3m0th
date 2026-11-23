"""Free-tier LLM gateway: SQLite cache, token bucket, and provider fallback."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import aiosqlite
import httpx
from pydantic import BaseModel, ConfigDict, Field

from beh3m0th.config import (
    DEFAULT_GOOGLE_FALLBACK_1,
    DEFAULT_GOOGLE_FALLBACK_2,
    DEFAULT_GOOGLE_MODEL,
    DEFAULT_GROQ_FALLBACK_1,
    DEFAULT_GROQ_FALLBACK_2,
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENROUTER_FALLBACK_1,
    DEFAULT_OPENROUTER_FALLBACK_2,
    DEFAULT_OPENROUTER_MODEL,
    Settings,
)

_CACHE_SQL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class LLMMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    provider: str
    model: str
    cached: bool
    cache_key: str


class LLMProviderExhaustedError(Exception):
    """Every configured free provider failed or had no API key."""

    def __init__(self, attempts: list[str]) -> None:
        self.attempts = attempts
        detail = "; ".join(attempts) if attempts else "no providers configured"
        super().__init__(f"All free LLM providers failed: {detail}")


class ProviderCallError(Exception):
    """One provider returned an error. The gateway tries the next provider."""


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    model: str
    api_key: str
    base_url: str

    def __post_init__(self) -> None:
        if not self.name or not self.model or not self.base_url:
            raise ValueError("provider name, model, and base_url are required")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))


class TokenBucket:
    """Async token bucket. Capacity and refill rate follow requests per minute."""

    def __init__(self, rpm: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.capacity = float(rpm)
        self.refill_per_sec = float(rpm) / 60.0
        self._tokens = float(rpm)
        self._clock = clock
        self._updated = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self._updated)
                self._updated = now
                self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_for = (1.0 - self._tokens) / self.refill_per_sec
            await asyncio.sleep(wait_for)


def cache_key(model: str, messages: list[LLMMessage], temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [message.model_dump() for message in messages],
        "params": {"temperature": temperature},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_GOOGLE_API = "https://generativelanguage.googleapis.com/v1beta"
_GROQ_API = "https://api.groq.com/openai/v1"
_OPENROUTER_API = "https://openrouter.ai/api/v1"


def _is_google(provider: Provider) -> bool:
    return "generativelanguage.googleapis.com" in provider.base_url


def _append_model_family(
    providers: list[Provider],
    *,
    family: str,
    api_key: str,
    base_url: str,
    models: tuple[str, ...],
) -> None:
    """Append a base model and optional fallbacks. Empty or duplicate model IDs are skipped."""
    seen: set[str] = set()
    index = 0
    for raw_name in models:
        model_name = raw_name.strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        name = family if index == 0 else f"{family}-fallback-{index}"
        providers.append(
            Provider(
                name=name,
                model=model_name,
                api_key=api_key,
                base_url=base_url,
            )
        )
        index += 1


def default_providers(
    *,
    google_api_key: str = "",
    groq_api_key: str = "",
    openrouter_api_key: str = "",
    google_model: str = DEFAULT_GOOGLE_MODEL,
    google_fallback_models: tuple[str, ...] = (
        DEFAULT_GOOGLE_FALLBACK_1,
        DEFAULT_GOOGLE_FALLBACK_2,
    ),
    groq_model: str = DEFAULT_GROQ_MODEL,
    groq_fallback_models: tuple[str, ...] = (
        DEFAULT_GROQ_FALLBACK_1,
        DEFAULT_GROQ_FALLBACK_2,
    ),
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
    openrouter_fallback_models: tuple[str, ...] = (
        DEFAULT_OPENROUTER_FALLBACK_1,
        DEFAULT_OPENROUTER_FALLBACK_2,
    ),
) -> list[Provider]:
    """Google models, then Groq, then OpenRouter. Each family can have base + fallbacks."""
    providers: list[Provider] = []
    _append_model_family(
        providers,
        family="google",
        api_key=google_api_key,
        base_url=_GOOGLE_API,
        models=(google_model, *google_fallback_models),
    )
    _append_model_family(
        providers,
        family="groq",
        api_key=groq_api_key,
        base_url=_GROQ_API,
        models=(groq_model, *groq_fallback_models),
    )
    _append_model_family(
        providers,
        family="openrouter",
        api_key=openrouter_api_key,
        base_url=_OPENROUTER_API,
        models=(openrouter_model, *openrouter_fallback_models),
    )
    return providers


def providers_from_settings(settings: Settings) -> list[Provider]:
    return default_providers(
        google_api_key=settings.google_api_key,
        groq_api_key=settings.groq_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        google_model=settings.google_model,
        google_fallback_models=(
            settings.google_fallback_model_1,
            settings.google_fallback_model_2,
        ),
        groq_model=settings.groq_model,
        groq_fallback_models=(
            settings.groq_fallback_model_1,
            settings.groq_fallback_model_2,
        ),
        openrouter_model=settings.openrouter_model,
        openrouter_fallback_models=(
            settings.openrouter_fallback_model_1,
            settings.openrouter_fallback_model_2,
        ),
    )


class LLMGateway:
    """Async chat client. Identical prompts return the cached SQLite body and make no HTTP call."""

    def __init__(
        self,
        *,
        cache_path: Path,
        providers: list[Provider],
        rpm: float = 12,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.cache_path = cache_path
        self._providers = list(providers)
        bucket_clock = clock if clock is not None else time.monotonic
        self._buckets = {
            provider.name: TokenBucket(rpm, clock=bucket_clock) for provider in self._providers
        }
        self._client = httpx.AsyncClient(transport=transport, timeout=30.0)
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages must not be empty")
        logical_model = model or "default"
        key = cache_key(logical_model, messages, temperature)
        async with self._lock:
            cached = await self._cache_get(key)
            if cached is not None:
                provider_name, used_model, content = cached
                return LLMResponse(
                    content=content,
                    provider=provider_name,
                    model=used_model,
                    cached=True,
                    cache_key=key,
                )
            errors: list[str] = []
            for provider in self._providers:
                if not provider.api_key:
                    errors.append(f"{provider.name}: empty API key")
                    continue
                await self._buckets[provider.name].acquire()
                chosen = model if model and provider.name == "google" else provider.model
                try:
                    content = await self._call(provider, messages, temperature, chosen)
                except ProviderCallError as exc:
                    errors.append(f"{provider.name}: {exc}")
                    continue
                await self._cache_put(key, provider.name, chosen, content)
                return LLMResponse(
                    content=content,
                    provider=provider.name,
                    model=chosen,
                    cached=False,
                    cache_key=key,
                )
            raise LLMProviderExhaustedError(errors)

    async def _call(
        self,
        provider: Provider,
        messages: list[LLMMessage],
        temperature: float,
        model: str,
    ) -> str:
        if _is_google(provider):
            url = f"{provider.base_url}/models/{model}:generateContent"
            headers = {"x-goog-api-key": provider.api_key}
            body: dict[str, object] = _google_body(messages, temperature)
        else:
            url = f"{provider.base_url}/chat/completions"
            headers = {"authorization": f"Bearer {provider.api_key}"}
            body = {
                "model": model,
                "messages": [message.model_dump() for message in messages],
                "temperature": temperature,
            }
        try:
            response = await self._client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise ProviderCallError(str(exc)) from exc
        if response.status_code != 200:
            raise ProviderCallError(f"HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderCallError("response was not JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderCallError("response JSON was not an object")
        family = "google" if _is_google(provider) else provider.name
        return _extract_text(family, payload)

    async def _cache_get(self, key: str) -> tuple[str, str, str] | None:
        await self._ensure_cache()
        async with aiosqlite.connect(self.cache_path) as db:
            cursor = await db.execute(
                "SELECT provider, model, content FROM llm_cache WHERE cache_key = ?",
                (key,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return str(row[0]), str(row[1]), str(row[2])

    async def _cache_put(self, key: str, provider: str, model: str, content: str) -> None:
        await self._ensure_cache()
        created_at = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self.cache_path) as db:
            await db.execute(
                """
                INSERT INTO llm_cache (cache_key, provider, model, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    provider = excluded.provider,
                    model = excluded.model,
                    content = excluded.content,
                    created_at = excluded.created_at
                """,
                (key, provider, model, content, created_at),
            )
            await db.commit()

    async def _ensure_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.cache_path) as db:
            await db.execute(_CACHE_SQL)
            await db.commit()


def _google_body(messages: list[LLMMessage], temperature: float) -> dict[str, object]:
    system_parts = [message.content for message in messages if message.role == "system"]
    contents: list[dict[str, object]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "model" if message.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message.content}]})
    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Continue."}]})
    body: dict[str, object] = {
        "contents": contents,
        "generationConfig": {"temperature": temperature},
    }
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    return body


def _extract_text(provider_name: str, payload: dict[str, object]) -> str:
    try:
        if provider_name == "google":
            candidates = payload["candidates"]
            if not isinstance(candidates, list) or not candidates:
                raise ProviderCallError("Google response had no candidates")
            first = candidates[0]
            if not isinstance(first, dict):
                raise ProviderCallError("Google candidate was not an object")
            content = first["content"]
            if not isinstance(content, dict):
                raise ProviderCallError("Google content was not an object")
            parts = content["parts"]
            if not isinstance(parts, list) or not parts:
                raise ProviderCallError("Google response had no parts")
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                    return part["text"]
            raise ProviderCallError("Google part had no text")
        choices = payload["choices"]
        if not isinstance(choices, list) or not choices:
            raise ProviderCallError("chat response had no choices")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ProviderCallError("chat choice was not an object")
        message = first_choice["message"]
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderCallError("chat message had no content")
        return message["content"]
    except KeyError as exc:
        raise ProviderCallError(f"missing field {exc}") from exc
