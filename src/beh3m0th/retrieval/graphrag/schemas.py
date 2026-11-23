"""Typed entities, relationships, communities, and subgraph results."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from beh3m0th.core.schemas import StrictModel

type EntityKind = Literal[
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
]
type RelationKind = Literal[
    "DEFINES",
    "IMPORTS",
    "CALLS",
    "INHERITS",
    "DEPENDS_ON",
    "CAUSES",
    "MITIGATES",
    "CONFIGURES",
    "READS",
]
type QueryMode = Literal["local", "global", "hybrid"]


class Entity(StrictModel):
    """One code symbol, prose concept, or community node."""

    entity_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    kind: EntityKind
    aliases: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    community_id: str | None = None


class Relationship(StrictModel):
    """A directed edge with provenance back to a source chunk."""

    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    predicate: RelationKind
    chunk_id: str | None = None
    weight: float = Field(default=1.0, gt=0.0)


class GraphExtractionResult(StrictModel):
    """Entities and relationships produced from one or more chunks."""

    entities: list[Entity]
    relationships: list[Relationship]


class CommunitySummary(StrictModel):
    """Leiden cluster and its deterministic structural summary."""

    community_id: str = Field(min_length=1)
    member_ids: list[str]
    member_names: list[str]
    summary: str = Field(min_length=1)
    edge_counts: dict[str, int]


class GraphTraversal(StrictModel):
    """Bounded neighborhood collected from anchor nodes."""

    entities: list[Entity]
    relationships: list[Relationship]
    chunk_ids: list[str]
    score: float = Field(default=0.0, ge=0.0)


class SubgraphResult(StrictModel):
    """Local hops, community summaries, or both, for one query."""

    query: str = Field(min_length=1)
    mode: QueryMode
    entities: list[Entity]
    relationships: list[Relationship]
    chunk_ids: list[str]
    communities: list[CommunitySummary] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0)
