"""Optional embedding merge for entities that share a kind."""

from __future__ import annotations

import math

from beh3m0th.retrieval.graphrag.schemas import Entity, GraphExtractionResult, Relationship


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return the cosine of two equal-length vectors, or 0 when undefined."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def merge_by_cosine(
    extraction: GraphExtractionResult,
    vectors: list[list[float]],
    threshold: float = 0.90,
) -> GraphExtractionResult:
    """Collapse same-kind entities whose name vectors exceed the threshold."""
    if len(vectors) != len(extraction.entities):
        raise ValueError("vectors and entities must have the same length")
    parent = list(range(len(extraction.entities)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(extraction.entities)):
        for right in range(left + 1, len(extraction.entities)):
            if extraction.entities[left].kind != extraction.entities[right].kind:
                continue
            if cosine_similarity(vectors[left], vectors[right]) <= threshold:
                continue
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                continue
            survivor, absorbed = _pair(extraction, left_root, right_root)
            parent[absorbed] = survivor

    groups: dict[int, list[int]] = {}
    for index in range(len(extraction.entities)):
        groups.setdefault(find(index), []).append(index)
    id_map: dict[str, str] = {}
    entities: list[Entity] = []
    for indexes in groups.values():
        ordered = sorted(indexes, key=lambda item: extraction.entities[item].entity_id)
        survivor = extraction.entities[ordered[0]].model_copy(deep=True)
        for index in ordered[1:]:
            incoming = extraction.entities[index]
            id_map[incoming.entity_id] = survivor.entity_id
            _absorb(survivor, incoming)
        id_map.setdefault(survivor.entity_id, survivor.entity_id)
        entities.append(survivor)
    return GraphExtractionResult(
        entities=entities,
        relationships=_remap(extraction.relationships, id_map),
    )


def _pair(
    extraction: GraphExtractionResult,
    left_root: int,
    right_root: int,
) -> tuple[int, int]:
    left_id = extraction.entities[left_root].entity_id
    right_id = extraction.entities[right_root].entity_id
    if left_id <= right_id:
        return left_root, right_root
    return right_root, left_root


def _absorb(survivor: Entity, incoming: Entity) -> None:
    for chunk_id in incoming.chunk_ids:
        if chunk_id not in survivor.chunk_ids:
            survivor.chunk_ids.append(chunk_id)
    for alias in [incoming.name, *incoming.aliases]:
        if alias != survivor.name and alias not in survivor.aliases:
            survivor.aliases.append(alias)


def _remap(
    relationships: list[Relationship],
    id_map: dict[str, str],
) -> list[Relationship]:
    remapped: dict[tuple[str, str, str], Relationship] = {}
    for relationship in relationships:
        source_id = id_map.get(relationship.source_id, relationship.source_id)
        target_id = id_map.get(relationship.target_id, relationship.target_id)
        if source_id == target_id:
            continue
        key = (source_id, target_id, relationship.predicate)
        remapped.setdefault(
            key,
            relationship.model_copy(update={"source_id": source_id, "target_id": target_id}),
        )
    return list(remapped.values())
