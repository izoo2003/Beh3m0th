"""NetworkX graph store with JSON node-link persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Protocol, cast

import networkx as nx
from networkx.readwrite.json_graph import node_link_data, node_link_graph

from beh3m0th.retrieval.graphrag.errors import GraphQueryError
from beh3m0th.retrieval.graphrag.extractor import canonical_name
from beh3m0th.retrieval.graphrag.schemas import (
    CommunitySummary,
    Entity,
    GraphExtractionResult,
    GraphTraversal,
    RelationKind,
    Relationship,
)

_DEFAULT_PATH = Path("./data/knowledge_graph.json")


class GraphStoreProtocol(Protocol):
    """Swap-ready knowledge graph. NetworkX is the $0 default."""

    async def upsert(self, extraction: GraphExtractionResult) -> None: ...

    async def add_communities(self, communities: list[CommunitySummary]) -> None: ...

    async def traverse(self, anchor_ids: list[str], max_hops: int) -> GraphTraversal: ...

    async def entities(self) -> list[Entity]: ...

    async def relationships(self) -> list[Relationship]: ...

    async def communities(self) -> list[CommunitySummary]: ...


class NetworkXGraphStore:
    """Directed multigraph persisted as networkx node-link JSON."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = _DEFAULT_PATH if path is None else path
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._communities: list[CommunitySummary] = []
        self._aliases: dict[str, str] = {}
        if self.path.is_file():
            self._load()

    async def upsert(self, extraction: GraphExtractionResult) -> None:
        await asyncio.to_thread(self._upsert, extraction)

    async def add_communities(self, communities: list[CommunitySummary]) -> None:
        await asyncio.to_thread(self._add_communities, communities)

    async def traverse(self, anchor_ids: list[str], max_hops: int) -> GraphTraversal:
        if max_hops not in (1, 2, 3):
            raise GraphQueryError("hops must be between 1 and 3")
        return await asyncio.to_thread(self._traverse, anchor_ids, max_hops)

    async def entities(self) -> list[Entity]:
        return await asyncio.to_thread(self._entities)

    async def relationships(self) -> list[Relationship]:
        return await asyncio.to_thread(self._relationships)

    async def communities(self) -> list[CommunitySummary]:
        return list(self._communities)

    def _upsert(self, extraction: GraphExtractionResult) -> None:
        id_map: dict[str, str] = {}
        for entity in extraction.entities:
            resolved = self._resolve(entity)
            id_map[entity.entity_id] = resolved
            if resolved in self._graph:
                self._merge_node(resolved, entity)
            else:
                self._graph.add_node(resolved, **_node_attrs(entity))
                self._register(resolved, entity)
        for relationship in extraction.relationships:
            source_id = id_map.get(relationship.source_id, relationship.source_id)
            target_id = id_map.get(relationship.target_id, relationship.target_id)
            if source_id not in self._graph or target_id not in self._graph:
                continue
            if source_id == target_id:
                continue
            self._add_edge(source_id, target_id, relationship)
        self._save()

    def _add_communities(self, communities: list[CommunitySummary]) -> None:
        stale = [
            node
            for node, data in self._graph.nodes(data=True)
            if data.get("kind") == "Community"
        ]
        self._graph.remove_nodes_from(stale)
        self._communities = list(communities)
        assigned = {
            member_id: summary.community_id
            for summary in communities
            for member_id in summary.member_ids
        }
        for node, data in self._graph.nodes(data=True):
            data["community_id"] = assigned.get(str(node))
        for summary in communities:
            entity = Entity(
                entity_id=summary.community_id,
                name=summary.community_id,
                canonical_name=canonical_name(summary.community_id) or summary.community_id,
                kind="Community",
                chunk_ids=[],
                community_id=summary.community_id,
            )
            self._graph.add_node(entity.entity_id, **_node_attrs(entity), summary=summary.summary)
        self._graph.graph["communities"] = [summary.model_dump() for summary in communities]
        self._save()

    def _traverse(self, anchor_ids: list[str], max_hops: int) -> GraphTraversal:
        anchors = [anchor for anchor in anchor_ids if anchor in self._graph]
        visited = set(anchors)
        frontier = list(anchors)
        relationships: list[Relationship] = []
        seen: set[tuple[str, str, str]] = set()
        for _ in range(max_hops):
            nxt: list[str] = []
            for node in frontier:
                for _, target, key, data in self._graph.out_edges(node, keys=True, data=True):
                    predicate = str(data.get("predicate", key))
                    marker = (str(node), str(target), predicate)
                    if marker not in seen:
                        seen.add(marker)
                        relationships.append(_relationship(str(node), str(target), data))
                    if target not in visited:
                        visited.add(target)
                        nxt.append(target)
            frontier = nxt
            if not frontier:
                break
        entities = [self._entity(node) for node in sorted(visited, key=str)]
        chunk_ids = _chunk_ids(entities, relationships)
        degree = sum(self._graph.degree(node) for node in visited)
        weight = sum(relationship.weight for relationship in relationships)
        return GraphTraversal(
            entities=entities,
            relationships=relationships,
            chunk_ids=chunk_ids,
            score=float(degree) + weight,
        )

    def _entities(self) -> list[Entity]:
        return [self._entity(node) for node in sorted(self._graph.nodes, key=str)]

    def _relationships(self) -> list[Relationship]:
        relationships: list[Relationship] = []
        for source, target, data in self._graph.edges(data=True):
            relationships.append(_relationship(str(source), str(target), data))
        relationships.sort(key=lambda item: (item.source_id, item.target_id, item.predicate))
        return relationships

    def _resolve(self, entity: Entity) -> str:
        if entity.entity_id in self._graph:
            return entity.entity_id
        for label in (entity.canonical_name, entity.name, *entity.aliases):
            found = self._aliases.get(_alias_key(entity.kind, label))
            if found is not None:
                return found
        return entity.entity_id

    def _merge_node(self, node_id: str, entity: Entity) -> None:
        data = self._graph.nodes[node_id]
        aliases = list(data.get("aliases") or [])
        for alias in [entity.name, *entity.aliases]:
            if alias != data.get("name") and alias not in aliases:
                aliases.append(alias)
        data["aliases"] = aliases
        chunk_ids = list(data.get("chunk_ids") or [])
        for chunk_id in entity.chunk_ids:
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
        data["chunk_ids"] = chunk_ids
        self._register(node_id, entity)

    def _register(self, node_id: str, entity: Entity) -> None:
        kind = str(self._graph.nodes[node_id].get("kind", entity.kind))
        for label in (entity.canonical_name, entity.name, *entity.aliases):
            self._aliases[_alias_key(kind, label)] = node_id

    def _add_edge(self, source_id: str, target_id: str, relationship: Relationship) -> None:
        existing = self._graph.get_edge_data(source_id, target_id, key=relationship.predicate)
        if existing is not None:
            if relationship.weight > float(existing.get("weight", 0.0)):
                existing["weight"] = relationship.weight
            if existing.get("chunk_id") in (None, "") and relationship.chunk_id:
                existing["chunk_id"] = relationship.chunk_id
            return
        self._graph.add_edge(
            source_id,
            target_id,
            key=relationship.predicate,
            predicate=relationship.predicate,
            chunk_id=relationship.chunk_id,
            weight=relationship.weight,
        )

    def _entity(self, node: object) -> Entity:
        data = self._graph.nodes[node]
        community_id = data.get("community_id")
        return Entity(
            entity_id=str(node),
            name=str(data["name"]),
            canonical_name=str(data["canonical_name"]),
            kind=data["kind"],
            aliases=list(data.get("aliases") or []),
            chunk_ids=list(data.get("chunk_ids") or []),
            community_id=str(community_id) if community_id else None,
        )

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._graph = node_link_graph(payload, edges="edges")
        raw = self._graph.graph.get("communities") or []
        self._communities = [CommunitySummary.model_validate(item) for item in raw]
        self._aliases = {}
        for node in self._graph.nodes:
            data = self._graph.nodes[node]
            entity = self._entity(node)
            self._register(str(node), entity)
            if data.get("kind") != entity.kind:
                data["kind"] = entity.kind

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = node_link_data(self._graph, edges="edges")
        self.path.write_text(json.dumps(payload), encoding="utf-8")


def _alias_key(kind: str, label: str) -> str:
    return f"{kind}:{canonical_name(label)}"


def _node_attrs(entity: Entity) -> dict[str, object]:
    return {
        "name": entity.name,
        "canonical_name": entity.canonical_name,
        "kind": entity.kind,
        "aliases": list(entity.aliases),
        "chunk_ids": list(entity.chunk_ids),
        "community_id": entity.community_id,
    }


def _relationship(source_id: str, target_id: str, data: dict[str, object]) -> Relationship:
    chunk_id = data.get("chunk_id")
    predicate = cast(RelationKind, data.get("predicate"))
    return Relationship(
        source_id=source_id,
        target_id=target_id,
        predicate=predicate,
        chunk_id=str(chunk_id) if chunk_id else None,
        weight=float(data.get("weight", 1.0)),
    )


def _chunk_ids(entities: list[Entity], relationships: list[Relationship]) -> list[str]:
    found: list[str] = []
    for entity in entities:
        for chunk_id in entity.chunk_ids:
            if chunk_id not in found:
                found.append(chunk_id)
    for relationship in relationships:
        if relationship.chunk_id and relationship.chunk_id not in found:
            found.append(relationship.chunk_id)
    return sorted(found)
