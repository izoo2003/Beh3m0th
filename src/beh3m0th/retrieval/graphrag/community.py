"""Leiden communities and deterministic structural summaries."""

from __future__ import annotations

import hashlib
import math

import igraph as ig
import leidenalg

from beh3m0th.retrieval.graphrag.schemas import CommunitySummary, Entity, Relationship
from beh3m0th.retrieval.hybrid.bm25_index import tokenize


def detect_communities(
    entities: list[Entity],
    relationships: list[Relationship],
) -> list[CommunitySummary]:
    """Partition non-community nodes with Leiden and describe each cluster."""
    members = [entity for entity in entities if entity.kind != "Community"]
    if not members:
        return []
    index = {entity.entity_id: position for position, entity in enumerate(members)}
    undirected: dict[tuple[int, int], float] = {}
    for relationship in relationships:
        if relationship.source_id not in index or relationship.target_id not in index:
            continue
        left = index[relationship.source_id]
        right = index[relationship.target_id]
        if left == right:
            continue
        key = (min(left, right), max(left, right))
        undirected[key] = undirected.get(key, 0.0) + relationship.weight
    edges = list(undirected)
    graph = ig.Graph(n=len(members), edges=edges, directed=False)
    weights = [undirected[edge] for edge in edges] if edges else None
    if weights is not None:
        graph.es["weight"] = weights
    partition = leidenalg.find_partition(
        graph,
        leidenalg.ModularityVertexPartition,
        weights="weight" if weights is not None else None,
        seed=7,
    )
    grouped: dict[int, list[Entity]] = {}
    for entity, community in zip(members, partition.membership, strict=True):
        grouped.setdefault(int(community), []).append(entity)
    ordered = sorted(
        grouped.values(),
        key=lambda group: min(entity.canonical_name for entity in group),
    )
    summaries: list[CommunitySummary] = []
    for position, group in enumerate(ordered):
        member_ids = [entity.entity_id for entity in sorted(group, key=lambda item: item.name)]
        member_names = [entity.name for entity in sorted(group, key=lambda item: item.name)]
        member_set = set(member_ids)
        counts: dict[str, int] = {}
        for relationship in relationships:
            if relationship.source_id in member_set and relationship.target_id in member_set:
                counts[relationship.predicate] = counts.get(relationship.predicate, 0) + 1
        community_id = f"community:{position}"
        count_text = ", ".join(f"{name}={counts[name]}" for name in sorted(counts)) or "none"
        names = ", ".join(member_names)
        summaries.append(
            CommunitySummary(
                community_id=community_id,
                member_ids=member_ids,
                member_names=member_names,
                summary=f"Community {community_id}: {names}. Edges: {count_text}.",
                edge_counts=counts,
            )
        )
    return summaries


def hashing_vector(text: str, dimension: int) -> list[float]:
    """Bag-of-tokens hashing trick. No model weights are loaded."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    vector = [0.0] * dimension
    tokens = tokenize(text) or ["empty"]
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
