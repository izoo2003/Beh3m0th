"""Offline Phase 3 checks: extraction, hops, Leiden, and graph stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from beh3m0th.context_engine.assembler import ContextAssembler
from beh3m0th.core.llm_gateway import LLMResponse
from beh3m0th.core.schemas import DocumentChunk
from beh3m0th.evaluation.tracer import TrajectoryTracer
from beh3m0th.retrieval.graphrag import (
    Entity,
    GraphExtractionError,
    GraphExtractionResult,
    GraphQueryError,
    GraphRAGPipeline,
    LLMProseExtractor,
    Neo4jGraphStore,
    NetworkXGraphStore,
    Relationship,
    cosine_similarity,
    extract_prose,
    extract_python,
    hashing_vector,
    merge_by_cosine,
)
from beh3m0th.retrieval.graphrag.extractor import canonical_name, entity_id_for, infer_kind
from beh3m0th.retrieval.hybrid.vector_store import HybridVectorStore

CODE = "\n".join(
    [
        "from pkg.policy import PolicyDoc",
        "",
        "class ServiceA(RemoteClient):",
        "    def run(self):",
        "        AuthModule().validate()",
    ]
)
PROSE = "AuthModule checks tokens. AuthModule reads PolicyDoc."
CLUSTERS = "AlphaService depends on BetaService. GammaPolicy configures DeltaPolicy."


def _chunk(chunk_id: str, text: str, source_path: str, language: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        text=text,
        source_path=source_path,
        language=language,
    )


def _edge(result: GraphExtractionResult, predicate: str, target_name: str) -> Relationship:
    names = {entity.entity_id: entity.name for entity in result.entities}
    matches = [
        relationship
        for relationship in result.relationships
        if relationship.predicate == predicate and names.get(relationship.target_id) == target_name
    ]
    assert matches
    return matches[0]


class ScriptedCompleter:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.prompts: list[str] = []

    async def complete(
        self,
        messages,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        del model, temperature
        self.calls += 1
        self.prompts.append(messages[0].content)
        return LLMResponse(
            content=self.content,
            provider="fake",
            model="fake",
            cached=False,
            cache_key="scripted",
        )


class FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    async def data(self) -> list[dict[str, object]]:
        return self._rows


class FakeSession:
    def __init__(self, queries: list[tuple[str, dict[str, object]]]) -> None:
        self.queries = queries

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def run(
        self,
        query: str,
        parameters: dict[str, object] | None = None,
    ) -> FakeResult:
        self.queries.append((query, parameters or {}))
        if "RELATED*1.." in query:
            return FakeResult(
                [
                    {
                        "source_id": "service",
                        "target_id": "auth",
                        "predicate": "CALLS",
                        "chunk_id": "code",
                        "weight": 1.0,
                        "source_name": "ServiceA",
                        "source_canonical": "servicea",
                        "source_kind": "Class",
                        "target_name": "AuthModule",
                        "target_canonical": "authmodule",
                        "target_kind": "Module",
                    }
                ]
            )
        return FakeResult([])


class FakeDriver:
    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, object]]] = []
        self._session = FakeSession(self.queries)

    def session(self) -> FakeSession:
        return self._session


def test_python_ast_extracts_defines_imports_inherits_and_calls() -> None:
    result = extract_python(_chunk("code", CODE, "src/service.py", "python"))
    names = {entity.name for entity in result.entities}
    assert {"ServiceA", "ServiceA.run", "AuthModule", "RemoteClient", "PolicyDoc"} <= names
    call = _edge(result, "CALLS", "AuthModule")
    assert call.chunk_id == "code"
    assert _edge(result, "INHERITS", "RemoteClient").chunk_id == "code"
    assert _edge(result, "IMPORTS", "PolicyDoc").chunk_id == "code"
    assert _edge(result, "DEFINES", "ServiceA.run").source_id != call.source_id
    defines = _edge(result, "DEFINES", "ServiceA")
    assert any(entity.entity_id == defines.source_id for entity in result.entities)


def test_prose_hashes_repeated_mentions_into_one_reads_edge() -> None:
    result = extract_prose(_chunk("prose", PROSE, "docs/policy.md", "markdown"))
    auth = [entity for entity in result.entities if entity.name == "AuthModule"]
    assert len(auth) == 1
    expected = entity_id_for(infer_kind("AuthModule"), canonical_name("AuthModule"))
    assert auth[0].entity_id == expected
    reads = _edge(result, "READS", "PolicyDoc")
    assert reads.chunk_id == "prose"
    assert reads.source_id == auth[0].entity_id


def test_invalid_python_raises_extraction_error() -> None:
    with pytest.raises(GraphExtractionError):
        extract_python(_chunk("bad", "def (\n", "src/bad.py", "python"))


def test_cosine_merge_collapses_same_kind_near_duplicates() -> None:
    validator = Entity(
        entity_id="bbbbbbbbbbbbbbbb",
        name="Token Validator",
        canonical_name="token validator",
        kind="Service",
        chunk_ids=["left"],
    )
    compact = Entity(
        entity_id="aaaaaaaaaaaaaaaa",
        name="TokenValidator",
        canonical_name="tokenvalidator",
        kind="Service",
        chunk_ids=["right"],
    )
    store = Entity(
        entity_id="cccccccccccccccc",
        name="Store",
        canonical_name="store",
        kind="Component",
        chunk_ids=["right"],
    )
    extraction = GraphExtractionResult(
        entities=[validator, compact, store],
        relationships=[
            Relationship(
                source_id=compact.entity_id,
                target_id=store.entity_id,
                predicate="CALLS",
                chunk_id="right",
            )
        ],
    )
    merged = merge_by_cosine(extraction, [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert len(merged.entities) == 2
    survivor = next(entity for entity in merged.entities if entity.kind == "Service")
    assert survivor.entity_id == compact.entity_id
    assert "Token Validator" in survivor.aliases
    assert merged.relationships[0].source_id == survivor.entity_id


@pytest.mark.asyncio
async def test_multihop_query_returns_subgraph_and_source_chunks(tmp_path: Path) -> None:
    pipeline = GraphRAGPipeline(NetworkXGraphStore(tmp_path / "graph.json"))
    await pipeline.ingest(
        [
            _chunk("code", CODE, "src/service.py", "python"),
            _chunk("prose", "AuthModule reads PolicyDoc.", "docs/policy.md", "markdown"),
        ]
    )
    result = await pipeline.query("ServiceA", mode="local", hops=3)
    names = {entity.name for entity in result.entities}
    predicates = {relationship.predicate for relationship in result.relationships}
    assert {"ServiceA", "AuthModule", "PolicyDoc"} <= names
    assert {"CALLS", "READS"} <= predicates
    assert {"code", "prose"} <= set(result.chunk_ids)


@pytest.mark.asyncio
async def test_query_rejects_empty_text_and_hops_outside_range(tmp_path: Path) -> None:
    pipeline = GraphRAGPipeline(NetworkXGraphStore(tmp_path / "graph.json"))
    with pytest.raises(GraphQueryError):
        await pipeline.query("   ", hops=2)
    with pytest.raises(GraphQueryError):
        await pipeline.query("ServiceA", hops=0)
    with pytest.raises(GraphQueryError):
        await pipeline.query("ServiceA", hops=4)


@pytest.mark.asyncio
async def test_networkx_store_reloads_node_link_json(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    pipeline = GraphRAGPipeline(NetworkXGraphStore(path))
    await pipeline.ingest([_chunk("code", CODE, "src/service.py", "python")])
    loaded = NetworkXGraphStore(path)
    names = {entity.name for entity in await loaded.entities()}
    predicates = {relationship.predicate for relationship in await loaded.relationships()}
    assert "ServiceA" in names
    assert "AuthModule" in names
    assert "CALLS" in predicates


@pytest.mark.asyncio
async def test_leiden_global_query_prefers_matching_community(tmp_path: Path) -> None:
    store = NetworkXGraphStore(tmp_path / "graph.json")
    pipeline = GraphRAGPipeline(store)
    await pipeline.ingest([_chunk("notes", CLUSTERS, "docs/arch.md", "markdown")])
    entities = {entity.name: entity for entity in await store.entities()}
    assert entities["AlphaService"].community_id != entities["GammaPolicy"].community_id
    assert entities["AlphaService"].community_id is not None
    result = await pipeline.query("GammaPolicy", mode="global", hops=1)
    assert result.communities
    assert "GammaPolicy" in result.communities[0].summary
    assert "AlphaService" not in result.communities[0].summary


@pytest.mark.asyncio
async def test_neo4j_adapter_merges_and_bounds_traversal() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)
    extraction = extract_python(_chunk("code", CODE, "src/service.py", "python"))
    await store.upsert(extraction)
    traversal = await store.traverse(["service"], 2)
    queries = [query for query, _parameters in driver.queries]
    assert any("MERGE" in query for query in queries)
    assert any("*1..2" in query for query in queries)
    assert traversal.relationships[0].predicate == "CALLS"
    assert traversal.chunk_ids == ["code"]


@pytest.mark.asyncio
async def test_llm_extractor_is_optional_and_parses_assembled_json(tmp_path: Path) -> None:
    payload = GraphExtractionResult(
        entities=[
            Entity(
                entity_id="policy-doc",
                name="PolicyDoc",
                canonical_name="policydoc",
                kind="Policy",
                chunk_ids=["llm"],
            )
        ],
        relationships=[],
    )
    completer = ScriptedCompleter(payload.model_dump_json())
    pipeline = GraphRAGPipeline(NetworkXGraphStore(tmp_path / "graph.json"))
    await pipeline.ingest(
        [_chunk("llm", "PolicyDoc mitigates Outage.", "docs/p.md", "markdown")]
    )
    assert completer.calls == 0
    extractor = LLMProseExtractor(ContextAssembler(), completer)
    parsed = await extractor.extract(
        _chunk("llm", "PolicyDoc mitigates Outage.", "docs/p.md", "markdown")
    )
    assert completer.calls == 1
    assert "PolicyDoc mitigates Outage." in completer.prompts[0]
    assert parsed.entities[0].name == "PolicyDoc"
    with pytest.raises(GraphExtractionError):
        await LLMProseExtractor(ContextAssembler(), ScriptedCompleter("not-json")).extract(
            _chunk("llm", "PolicyDoc mitigates Outage.", "docs/p.md", "markdown")
        )


@pytest.mark.asyncio
async def test_ingest_and_query_record_retrieval_steps(tmp_path: Path) -> None:
    tracer = TrajectoryTracer()
    pipeline = GraphRAGPipeline(
        NetworkXGraphStore(tmp_path / "graph.json"),
        tracer=tracer,
    )
    await pipeline.ingest(
        [
            _chunk("code", CODE, "src/service.py", "python"),
            _chunk("prose", "AuthModule reads PolicyDoc.", "docs/policy.md", "markdown"),
        ]
    )
    await pipeline.query("ServiceA", mode="hybrid", hops=3)
    assert [step.step_type for step in tracer.steps] == ["RETRIEVAL", "RETRIEVAL"]
    assert tracer.steps[0].tokens_used == 0
    assert "code" in str(tracer.steps[1].metadata["chunk_ids"])
    assert "prose" in str(tracer.steps[1].metadata["chunk_ids"])


@pytest.mark.asyncio
async def test_community_summaries_are_indexed_with_hashing_vectors(tmp_path: Path) -> None:
    vector_store = HybridVectorStore(
        path=tmp_path / "qdrant",
        vector_size=32,
        collection="beh3m0th_communities",
    )
    pipeline = GraphRAGPipeline(
        NetworkXGraphStore(tmp_path / "graph.json"),
        vector_store=vector_store,
    )
    try:
        await pipeline.ingest([_chunk("notes", CLUSTERS, "docs/arch.md", "markdown")])
        hits = await vector_store.search(hashing_vector("GammaPolicy", 32), None, limit=4)
    finally:
        await vector_store.close()
    assert hits
    assert all(hit.language == "community" for hit, _score in hits)
    assert any(hit.chunk_id.startswith("community:") for hit, _score in hits)
