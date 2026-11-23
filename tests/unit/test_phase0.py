"""Offline Phase 0 checks: schemas, LLM cache, and embedded storage."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

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
from beh3m0th.core.llm_gateway import (
    LLMGateway,
    LLMMessage,
    Provider,
    TokenBucket,
    cache_key,
    providers_from_settings,
)
from beh3m0th.core.schemas import (
    AgentState,
    DocumentChunk,
    GraphTriple,
    MemoryRecord,
    RetrievalHit,
    RetrievalResult,
    TrajectoryStep,
)
from beh3m0th.core.storage import QdrantLocalStore, SqliteStore


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text="Authentication retries fail when the pool is exhausted.",
        source_path="src/auth.py",
        language="python",
        symbol_name="login",
        tags=["auth"],
        line_start=10,
        line_end=24,
        created_at=datetime(2026, 11, 23, tzinfo=UTC),
    )


def test_schemas_round_trip_and_reject_invalid_values() -> None:
    chunk = _chunk()
    result = RetrievalResult(
        query="why do auth retries fail",
        hits=[RetrievalHit(chunk=chunk, score=0.8, rank=1, source="hybrid")],
    )
    triple = GraphTriple(
        subject="AuthModule",
        predicate="CALLS",
        object="TokenValidator",
        chunk_id="chunk-1",
    )
    memory = MemoryRecord(
        id="mem-1",
        project_id="beh3m0th",
        session_id="session-1",
        layer="episodic",
        summary="Retries failed because the pool was closed.",
        outcome="patched",
        tags=["bugfix"],
        importance_score=0.85,
        supersedes_id=None,
    )
    state = AgentState(
        run_id="run-1",
        session_id="session-1",
        project_id="beh3m0th",
        objective="Fix authentication retries under load.",
    )
    step = TrajectoryStep(
        step_index=0,
        agent_name="researcher",
        step_type="RETRIEVAL",
        input_summary="auth retries",
        output_summary="pool exhaustion",
        tokens_used=0,
        latency_ms=12.5,
        metadata={"source": "hybrid"},
    )

    assert RetrievalResult.model_validate(result.model_dump()) == result
    assert GraphTriple.model_validate(triple.model_dump()) == triple
    assert MemoryRecord.model_validate(memory.model_dump()) == memory
    assert AgentState.model_validate(state.model_dump()) == state
    assert TrajectoryStep.model_validate(step.model_dump()) == step
    assert triple.weight == 1.0
    assert state.status == "PLANNING"
    assert state.max_iterations == 3

    with pytest.raises(ValidationError):
        DocumentChunk(chunk_id="c", doc_id="d", text="", source_path="a.py")
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="c",
            doc_id="d",
            text="hello",
            source_path="a.py",
            extra_field="nope",
        )
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="c",
            doc_id="d",
            text="hello",
            source_path="a.py",
            line_start=30,
            line_end=10,
        )
    with pytest.raises(ValidationError):
        TrajectoryStep(
            step_index=0,
            agent_name="researcher",
            step_type="NOT_A_STEP",
            input_summary="",
            output_summary="",
            tokens_used=0,
            latency_ms=0,
        )
    with pytest.raises(ValidationError):
        MemoryRecord(
            id="mem-2",
            project_id="beh3m0th",
            session_id="session-1",
            layer="episodic",
            summary="too important",
            importance_score=1.5,
        )
    with pytest.raises(ValidationError):
        AgentState(
            run_id="run-1",
            session_id="session-1",
            project_id="beh3m0th",
            objective="Fix retries.",
            status="SHIPPED",
        )


def test_settings_reads_profile_and_gemini_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_MODEL",
        "GOOGLE_FALLBACK_MODEL_1",
        "GOOGLE_FALLBACK_MODEL_2",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "GROQ_FALLBACK_MODEL_1",
        "GROQ_FALLBACK_MODEL_2",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_FALLBACK_MODEL_1",
        "OPENROUTER_FALLBACK_MODEL_2",
        "PROFILE",
        "STORAGE_PROFILE",
        "LLM_CACHE_PATH",
        "LLM_MAX_RPM",
        "SQLITE_PATH",
        "QDRANT_LOCAL_PATH",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "GEMINI_API_KEY=gemini-free",
                "GROQ_API_KEY=groq-free",
                "OPENROUTER_API_KEY=openrouter-free",
                "PROFILE=embedded",
                "",
            ]
        ),
        encoding="utf-8",
    )
    settings = Settings.from_env(env_file=env_file)
    assert settings.google_api_key == "gemini-free"
    assert settings.groq_api_key == "groq-free"
    assert settings.openrouter_api_key == "openrouter-free"
    assert settings.profile == "embedded"
    assert settings.google_model == DEFAULT_GOOGLE_MODEL
    assert settings.google_fallback_model_1 == DEFAULT_GOOGLE_FALLBACK_1
    assert settings.google_fallback_model_2 == DEFAULT_GOOGLE_FALLBACK_2
    assert settings.groq_model == DEFAULT_GROQ_MODEL
    assert settings.groq_fallback_model_1 == DEFAULT_GROQ_FALLBACK_1
    assert settings.groq_fallback_model_2 == DEFAULT_GROQ_FALLBACK_2
    assert settings.openrouter_model == DEFAULT_OPENROUTER_MODEL
    assert settings.openrouter_fallback_model_1 == DEFAULT_OPENROUTER_FALLBACK_1
    assert settings.openrouter_fallback_model_2 == DEFAULT_OPENROUTER_FALLBACK_2
    providers = providers_from_settings(settings)
    google = [item for item in providers if item.name.startswith("google")]
    groq = [item for item in providers if item.name.startswith("groq")]
    openrouter = [item for item in providers if item.name.startswith("openrouter")]
    assert [item.model for item in google] == [
        DEFAULT_GOOGLE_MODEL,
        DEFAULT_GOOGLE_FALLBACK_1,
        DEFAULT_GOOGLE_FALLBACK_2,
    ]
    assert [item.model for item in groq] == [DEFAULT_GROQ_MODEL, DEFAULT_GROQ_FALLBACK_1]
    assert [item.model for item in openrouter] == [
        DEFAULT_OPENROUTER_MODEL,
        DEFAULT_OPENROUTER_FALLBACK_1,
        DEFAULT_OPENROUTER_FALLBACK_2,
    ]
    assert {item.api_key for item in google} == {"gemini-free"}
    assert {item.api_key for item in groq} == {"groq-free"}
    assert {item.api_key for item in openrouter} == {"openrouter-free"}
    assert settings.llm_max_rpm == 12
    assert settings.qdrant_local_path == Path("./data/qdrant")


def test_cache_key_is_sha256_and_stable() -> None:
    messages = [LLMMessage(role="user", content="hello")]
    first = cache_key("default", messages, 0.0)
    second = cache_key("default", messages, 0.0)
    changed = cache_key("default", [LLMMessage(role="user", content="hello!")], 0.0)
    assert first == second
    assert first != changed
    assert len(first) == 64


def _google_ok(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"candidates": [{"content": {"parts": [{"text": text}]}}]},
    )


def _chat_ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


async def test_cache_hit_skips_the_second_http_call(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        return _google_ok("cached-body")

    gateway = LLMGateway(
        cache_path=tmp_path / "cache.db",
        providers=[
            Provider(
                name="google",
                model="gemini-2.5-flash-lite",
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            )
        ],
        transport=httpx.MockTransport(handler),
    )
    try:
        messages = [LLMMessage(role="user", content="same prompt")]
        first = await gateway.complete(messages)
        second = await gateway.complete(messages)
        third = await gateway.complete([LLMMessage(role="user", content="different prompt")])
    finally:
        await gateway.aclose()

    assert first.cached is False
    assert first.content == "cached-body"
    assert second.cached is True
    assert second.content == "cached-body"
    assert second.cache_key == first.cache_key
    assert third.cached is False
    assert calls == [
        "generativelanguage.googleapis.com",
        "generativelanguage.googleapis.com",
    ]


async def test_fallback_caches_the_successful_provider(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if "generativelanguage" in request.url.host:
            return httpx.Response(429, json={"error": "rate limit"})
        if "groq" in request.url.host:
            return _chat_ok("from-groq")
        return httpx.Response(500, json={"error": "down"})

    gateway = LLMGateway(
        cache_path=tmp_path / "cache.db",
        providers=[
            Provider(
                name="google",
                model="gemini-2.5-flash-lite",
                api_key="google-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            ),
            Provider(
                name="groq",
                model="llama-3.3-70b-versatile",
                api_key="groq-key",
                base_url="https://api.groq.com/openai/v1",
            ),
            Provider(
                name="openrouter",
                model="deepseek/deepseek-r1:free",
                api_key="",
                base_url="https://openrouter.ai/api/v1",
            ),
        ],
        transport=httpx.MockTransport(handler),
    )
    try:
        messages = [LLMMessage(role="user", content="need a fallback")]
        first = await gateway.complete(messages)
        second = await gateway.complete(messages)
    finally:
        await gateway.aclose()

    assert first.cached is False
    assert first.provider == "groq"
    assert first.content == "from-groq"
    assert second.cached is True
    assert second.content == "from-groq"
    assert calls == ["generativelanguage.googleapis.com", "api.groq.com"]


async def test_google_key_falls_through_base_then_two_fallback_models(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if DEFAULT_GOOGLE_MODEL in request.url.path:
            return httpx.Response(429, json={"error": "rate limit"})
        if DEFAULT_GOOGLE_FALLBACK_1 in request.url.path:
            return httpx.Response(503, json={"error": "unavailable"})
        if DEFAULT_GOOGLE_FALLBACK_2 in request.url.path:
            return _google_ok("from-second-fallback")
        return httpx.Response(500, json={"error": "unexpected"})

    settings = Settings(
        profile="embedded",
        google_api_key="google-key",
        google_model=DEFAULT_GOOGLE_MODEL,
        google_fallback_model_1=DEFAULT_GOOGLE_FALLBACK_1,
        google_fallback_model_2=DEFAULT_GOOGLE_FALLBACK_2,
        groq_api_key="",
        groq_model=DEFAULT_GROQ_MODEL,
        groq_fallback_model_1=DEFAULT_GROQ_FALLBACK_1,
        groq_fallback_model_2=DEFAULT_GROQ_FALLBACK_2,
        openrouter_api_key="",
        openrouter_model=DEFAULT_OPENROUTER_MODEL,
        openrouter_fallback_model_1=DEFAULT_OPENROUTER_FALLBACK_1,
        openrouter_fallback_model_2=DEFAULT_OPENROUTER_FALLBACK_2,
        llm_cache_path=tmp_path / "cache.db",
        llm_max_rpm=12,
        sqlite_path=tmp_path / "beh.db",
        qdrant_local_path=tmp_path / "qdrant",
    )
    gateway = LLMGateway(
        cache_path=tmp_path / "cache.db",
        providers=providers_from_settings(settings),
        transport=httpx.MockTransport(handler),
    )
    try:
        messages = [LLMMessage(role="user", content="use the google fallbacks")]
        first = await gateway.complete(messages)
        second = await gateway.complete(messages)
    finally:
        await gateway.aclose()

    assert first.cached is False
    assert first.provider == "google-fallback-2"
    assert first.model == DEFAULT_GOOGLE_FALLBACK_2
    assert first.content == "from-second-fallback"
    assert second.cached is True
    assert len(calls) == 3


async def test_groq_key_falls_through_base_then_fallback_model(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("utf-8")
        calls.append(body)
        if DEFAULT_GROQ_MODEL in body:
            return httpx.Response(429, json={"error": "rate limit"})
        if DEFAULT_GROQ_FALLBACK_1 in body:
            return _chat_ok("from-groq-fallback")
        return httpx.Response(500, json={"error": "unexpected"})

    settings = Settings(
        profile="embedded",
        google_api_key="",
        google_model=DEFAULT_GOOGLE_MODEL,
        google_fallback_model_1=DEFAULT_GOOGLE_FALLBACK_1,
        google_fallback_model_2=DEFAULT_GOOGLE_FALLBACK_2,
        groq_api_key="groq-key",
        groq_model=DEFAULT_GROQ_MODEL,
        groq_fallback_model_1=DEFAULT_GROQ_FALLBACK_1,
        groq_fallback_model_2="",
        openrouter_api_key="",
        openrouter_model=DEFAULT_OPENROUTER_MODEL,
        openrouter_fallback_model_1=DEFAULT_OPENROUTER_FALLBACK_1,
        openrouter_fallback_model_2=DEFAULT_OPENROUTER_FALLBACK_2,
        llm_cache_path=tmp_path / "cache.db",
        llm_max_rpm=12,
        sqlite_path=tmp_path / "beh.db",
        qdrant_local_path=tmp_path / "qdrant",
    )
    gateway = LLMGateway(
        cache_path=tmp_path / "cache.db",
        providers=providers_from_settings(settings),
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await gateway.complete([LLMMessage(role="user", content="groq fallbacks")])
    finally:
        await gateway.aclose()

    assert result.provider == "groq-fallback-1"
    assert result.model == DEFAULT_GROQ_FALLBACK_1
    assert result.content == "from-groq-fallback"
    assert len(calls) == 2


async def test_token_bucket_waits_only_after_capacity_is_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    clock = Clock()
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    bucket = TokenBucket(60, clock=clock)
    for _ in range(60):
        await bucket.acquire()
    assert sleeps == []
    await bucket.acquire()
    assert sleeps == [pytest.approx(1.0)]


async def test_sqlite_initialize_creates_tables(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "nested" / "beh3m0th.db")
    await store.initialize()
    try:
        assert store.path.is_file()
        names = await store.table_names()
        assert {"llm_cache", "memories", "trajectory_steps"} <= names
        await store.initialize()
        assert await store.table_names() >= names
    finally:
        await store.close()


async def test_qdrant_local_initialize_creates_collection(tmp_path: Path) -> None:
    store = QdrantLocalStore(tmp_path / "qdrant")
    await store.initialize()
    try:
        assert store.path.is_dir()
        assert "beh3m0th_chunks" in await store.collection_names()
    finally:
        await store.close()
