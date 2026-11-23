"""Optional Neo4j adapter. The driver is injected so unit tests stay offline."""

from __future__ import annotations

from typing import Protocol

from beh3m0th.retrieval.graphrag.errors import GraphQueryError
from beh3m0th.retrieval.graphrag.schemas import (
    CommunitySummary,
    Entity,
    EntityKind,
    GraphExtractionResult,
    GraphTraversal,
    RelationKind,
    Relationship,
)

type CypherValue = str | int | float | bool | None | list[str]

_MERGE_ENTITY = """
MERGE (n:Entity {entity_id: $entity_id})
SET n.name = $name,
    n.canonical_name = $canonical_name,
    n.kind = $kind,
    n.aliases = $aliases,
    n.chunk_ids = $chunk_ids,
    n.community_id = $community_id
"""
_MERGE_EDGE = """
MATCH (s:Entity {entity_id: $source_id})
MATCH (t:Entity {entity_id: $target_id})
MERGE (s)-[r:RELATED {predicate: $predicate}]->(t)
SET r.chunk_id = $chunk_id, r.weight = $weight
"""
_MATCH_ENTITIES = """
MATCH (n:Entity)
RETURN n.entity_id AS entity_id, n.name AS name, n.canonical_name AS canonical_name,
       n.kind AS kind, n.aliases AS aliases, n.chunk_ids AS chunk_ids,
       n.community_id AS community_id
"""
_MATCH_EDGES = """
MATCH (s:Entity)-[r:RELATED]->(t:Entity)
RETURN s.entity_id AS source_id, t.entity_id AS target_id, r.predicate AS predicate,
       r.chunk_id AS chunk_id, r.weight AS weight
"""
_MATCH_COMMUNITIES = """
MATCH (c:Community)
RETURN c.community_id AS community_id, c.member_ids AS member_ids,
       c.member_names AS member_names, c.summary AS summary, c.edge_counts AS edge_counts
"""
_TRAVERSAL = """
MATCH (s:Entity)-[rels:RELATED*1..{hops}]->(t:Entity)
WHERE s.entity_id IN $anchors
UNWIND rels AS r
WITH startNode(r) AS a, endNode(r) AS b, r
RETURN a.entity_id AS source_id, b.entity_id AS target_id, r.predicate AS predicate,
       r.chunk_id AS chunk_id, r.weight AS weight,
       a.name AS source_name, a.canonical_name AS source_canonical, a.kind AS source_kind,
       b.name AS target_name, b.canonical_name AS target_canonical, b.kind AS target_kind
"""


class _Neo4jResult(Protocol):
    async def data(self) -> list[dict[str, object]]: ...


class _Neo4jSession(Protocol):
    async def run(
        self,
        query: str,
        parameters: dict[str, CypherValue] | None = None,
    ) -> _Neo4jResult: ...

    async def __aenter__(self) -> _Neo4jSession: ...

    async def __aexit__(self, *args: object) -> None: ...


class _Neo4jDriver(Protocol):
    def session(self) -> _Neo4jSession: ...


class Neo4jGraphStore:
    """Profile B graph store. MERGE writes and bounded Cypher traversals."""

    def __init__(
        self,
        driver: _Neo4jDriver | None = None,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        if driver is None and not uri:
            raise ValueError("Neo4j driver or uri is required")
        self._driver = driver
        self._uri = uri
        self._user = user or "neo4j"
        self._password = password or ""

    async def upsert(self, extraction: GraphExtractionResult) -> None:
        for entity in extraction.entities:
            await self._run(
                _MERGE_ENTITY,
                {
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "canonical_name": entity.canonical_name,
                    "kind": entity.kind,
                    "aliases": list(entity.aliases),
                    "chunk_ids": list(entity.chunk_ids),
                    "community_id": entity.community_id,
                },
            )
        for relationship in extraction.relationships:
            await self._run(
                _MERGE_EDGE,
                {
                    "source_id": relationship.source_id,
                    "target_id": relationship.target_id,
                    "predicate": relationship.predicate,
                    "chunk_id": relationship.chunk_id,
                    "weight": relationship.weight,
                },
            )

    async def add_communities(self, communities: list[CommunitySummary]) -> None:
        for summary in communities:
            await self._run(
                """
                MERGE (c:Community {community_id: $community_id})
                SET c.member_ids = $member_ids, c.member_names = $member_names,
                    c.summary = $summary, c.edge_counts = $edge_counts
                """,
                {
                    "community_id": summary.community_id,
                    "member_ids": list(summary.member_ids),
                    "member_names": list(summary.member_names),
                    "summary": summary.summary,
                    "edge_counts": _edge_count_pairs(summary),
                },
            )

    async def traverse(self, anchor_ids: list[str], max_hops: int) -> GraphTraversal:
        if max_hops not in (1, 2, 3):
            raise GraphQueryError("hops must be between 1 and 3")
        rows = await self._run(_TRAVERSAL.format(hops=max_hops), {"anchors": list(anchor_ids)})
        return _traversal(rows)

    async def entities(self) -> list[Entity]:
        rows = await self._run(_MATCH_ENTITIES, {})
        return [entity for row in rows if (entity := _entity_row(row)) is not None]

    async def relationships(self) -> list[Relationship]:
        rows = await self._run(_MATCH_EDGES, {})
        return [relationship for row in rows if (relationship := _edge_row(row)) is not None]

    async def communities(self) -> list[CommunitySummary]:
        rows = await self._run(_MATCH_COMMUNITIES, {})
        found: list[CommunitySummary] = []
        for row in rows:
            summary = _community_row(row)
            if summary is not None:
                found.append(summary)
        return found

    async def _connect(self) -> _Neo4jDriver:
        if self._driver is None:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
        return self._driver

    async def _run(self, query: str, parameters: dict[str, CypherValue]) -> list[dict[str, object]]:
        driver = await self._connect()
        session = driver.session()
        async with session:
            result = await session.run(query, parameters)
            return [dict(row) for row in await result.data()]


def _edge_count_pairs(summary: CommunitySummary) -> list[str]:
    return [f"{name}={count}" for name, count in sorted(summary.edge_counts.items())]


def _entity_row(row: dict[str, object]) -> Entity | None:
    entity_id = row.get("entity_id")
    name = row.get("name")
    canonical = row.get("canonical_name")
    kind = row.get("kind")
    if not all(isinstance(value, str) and value for value in (entity_id, name, canonical, kind)):
        return None
    community_id = row.get("community_id")
    return Entity(
        entity_id=str(entity_id),
        name=str(name),
        canonical_name=str(canonical),
        kind=_kind(kind),
        aliases=_strings(row.get("aliases")),
        chunk_ids=_strings(row.get("chunk_ids")),
        community_id=str(community_id) if isinstance(community_id, str) else None,
    )


def _edge_row(row: dict[str, object]) -> Relationship | None:
    source_id = row.get("source_id")
    target_id = row.get("target_id")
    predicate = row.get("predicate")
    if not all(isinstance(value, str) and value for value in (source_id, target_id, predicate)):
        return None
    chunk_id = row.get("chunk_id")
    weight = row.get("weight", 1.0)
    return Relationship(
        source_id=str(source_id),
        target_id=str(target_id),
        predicate=_predicate(predicate),
        chunk_id=str(chunk_id) if isinstance(chunk_id, str) else None,
        weight=float(weight) if isinstance(weight, int | float) else 1.0,
    )


def _community_row(row: dict[str, object]) -> CommunitySummary | None:
    community_id = row.get("community_id")
    summary = row.get("summary")
    if not isinstance(community_id, str) or not isinstance(summary, str):
        return None
    return CommunitySummary(
        community_id=community_id,
        member_ids=_strings(row.get("member_ids")),
        member_names=_strings(row.get("member_names")),
        summary=summary,
        edge_counts=_counts(row.get("edge_counts")),
    )


def _traversal(rows: list[dict[str, object]]) -> GraphTraversal:
    entities: dict[str, Entity] = {}
    relationships: dict[tuple[str, str, str], Relationship] = {}
    for row in rows:
        relationship = _edge_row(row)
        if relationship is None:
            continue
        relationships[(relationship.source_id, relationship.target_id, relationship.predicate)] = (
            relationship
        )
        _remember_end(entities, row, "source")
        _remember_end(entities, row, "target")
    entity_list = list(entities.values())
    relationship_list = list(relationships.values())
    chunk_ids: list[str] = []
    for entity in entity_list:
        chunk_ids.extend(chunk_id for chunk_id in entity.chunk_ids if chunk_id not in chunk_ids)
    for relationship in relationship_list:
        if relationship.chunk_id and relationship.chunk_id not in chunk_ids:
            chunk_ids.append(relationship.chunk_id)
    weight = sum(relationship.weight for relationship in relationship_list)
    return GraphTraversal(
        entities=entity_list,
        relationships=relationship_list,
        chunk_ids=sorted(chunk_ids),
        score=weight,
    )


def _remember_end(entities: dict[str, Entity], row: dict[str, object], prefix: str) -> None:
    entity_id = row.get(f"{prefix}_id")
    name = row.get(f"{prefix}_name")
    canonical = row.get(f"{prefix}_canonical")
    kind = row.get(f"{prefix}_kind")
    if not all(isinstance(value, str) and value for value in (entity_id, name, canonical, kind)):
        return
    if str(entity_id) in entities:
        return
    entities[str(entity_id)] = Entity(
        entity_id=str(entity_id),
        name=str(name),
        canonical_name=str(canonical),
        kind=_kind(kind),
    )


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _counts(value: object) -> dict[str, int]:
    if not isinstance(value, list):
        return {}
    counts: dict[str, int] = {}
    for item in value:
        if not isinstance(item, str) or "=" not in item:
            continue
        name, raw = item.split("=", 1)
        if raw.isdigit():
            counts[name] = int(raw)
    return counts


def _kind(value: object) -> EntityKind:
    text = str(value)
    allowed: tuple[EntityKind, ...] = (
        "File",
        "Class",
        "Function",
        "Module",
        "Concept",
        "Service",
        "Component",
        "API",
        "Policy",
        "Error",
        "Community",
    )
    if text not in allowed:
        return "Concept"
    for kind in allowed:
        if kind == text:
            return kind
    return "Concept"


def _predicate(value: object) -> RelationKind:
    text = str(value)
    allowed: tuple[RelationKind, ...] = (
        "DEFINES",
        "IMPORTS",
        "CALLS",
        "INHERITS",
        "DEPENDS_ON",
        "CAUSES",
        "MITIGATES",
        "CONFIGURES",
        "READS",
    )
    for predicate in allowed:
        if predicate == text:
            return predicate
    return "DEPENDS_ON"
