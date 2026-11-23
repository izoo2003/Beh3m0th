"""Zero-token code and prose extraction."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from beh3m0th.core.schemas import DocumentChunk
from beh3m0th.retrieval.graphrag.errors import GraphExtractionError
from beh3m0th.retrieval.graphrag.schemas import (
    Entity,
    EntityKind,
    GraphExtractionResult,
    RelationKind,
    Relationship,
)

_NAME = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b")
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
_STOP_NAMES = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "When",
        "And",
        "For",
    }
)
_VERBS: tuple[tuple[re.Pattern[str], RelationKind], ...] = (
    (re.compile(r"\bdepend(?:s|ed)? on\b", re.IGNORECASE), "DEPENDS_ON"),
    (re.compile(r"\brequires?\b", re.IGNORECASE), "DEPENDS_ON"),
    (re.compile(r"\bcauses?\b", re.IGNORECASE), "CAUSES"),
    (re.compile(r"\btriggers?\b", re.IGNORECASE), "CAUSES"),
    (re.compile(r"\bmitigates?\b", re.IGNORECASE), "MITIGATES"),
    (re.compile(r"\bprevents?\b", re.IGNORECASE), "MITIGATES"),
    (re.compile(r"\bconfigures?\b", re.IGNORECASE), "CONFIGURES"),
    (re.compile(r"\bsets?\b", re.IGNORECASE), "CONFIGURES"),
    (re.compile(r"\breads?\b", re.IGNORECASE), "READS"),
    (re.compile(r"\bcalls?\b", re.IGNORECASE), "CALLS"),
)
def canonical_name(name: str) -> str:
    """Lowercase a name, keep dots, and collapse every other separator."""
    cleaned = re.sub(r"[^A-Za-z0-9.]+", " ", name).strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def entity_id_for(kind: EntityKind, canonical: str) -> str:
    """Content hash of kind plus canonical name. Sixteen hex chars stay stable."""
    digest = hashlib.sha256(f"{kind}\n{canonical}".encode()).hexdigest()
    return digest[:16]


def infer_kind(name: str) -> EntityKind:
    """Map a surface name onto a shared code or prose kind."""
    compact = re.sub(r"[^A-Za-z0-9]", "", name).lower()
    if compact.endswith("service"):
        return "Service"
    if compact.endswith("module"):
        return "Module"
    if compact.endswith("api"):
        return "API"
    if "policy" in compact:
        return "Policy"
    if compact.endswith("error") or compact.endswith("exception"):
        return "Error"
    if compact.endswith("component"):
        return "Component"
    return "Concept"


def make_entity(name: str, kind: EntityKind, chunk_id: str) -> Entity:
    """Build an entity whose id is the content hash of its kind and name."""
    canonical = canonical_name(name) or "unknown"
    return Entity(
        entity_id=entity_id_for(kind, canonical),
        name=name,
        canonical_name=canonical,
        kind=kind,
        chunk_ids=[chunk_id],
    )


def extract_python(chunk: DocumentChunk) -> GraphExtractionResult:
    """Extract file, class, function, and module nodes without using an LLM."""
    try:
        tree = ast.parse(chunk.text)
    except SyntaxError as exc:
        location = exc.lineno or 1
        raise GraphExtractionError(
            f"invalid Python in {chunk.source_path} at line {location}: {exc.msg}"
        ) from exc
    source = Path(chunk.source_path).as_posix()
    file_entity = make_entity(source, "File", chunk.chunk_id)
    module_entity = make_entity(Path(chunk.source_path).stem or "module", "Module", chunk.chunk_id)
    entities = {
        file_entity.entity_id: file_entity,
        module_entity.entity_id: module_entity,
    }
    relationships = [
        Relationship(
            source_id=file_entity.entity_id,
            target_id=module_entity.entity_id,
            predicate="DEFINES",
            chunk_id=chunk.chunk_id,
        )
    ]
    _walk_body(tree.body, module_entity, [], entities, relationships, chunk)
    return GraphExtractionResult(
        entities=list(entities.values()),
        relationships=relationships,
    )


def extract_prose(chunk: DocumentChunk) -> GraphExtractionResult:
    """Hash capitalized mentions and verb patterns into a typed subgraph."""
    entities: dict[str, Entity] = {}
    relationships: list[Relationship] = []
    seen: set[tuple[str, str, str]] = set()
    for sentence in _SENTENCE.findall(chunk.text):
        spans = [
            (match.start(), match.end(), match.group(0))
            for match in _NAME.finditer(sentence)
            if match.group(0) not in _STOP_NAMES
        ]
        resolved = [
            _ensure(name, infer_kind(name), chunk.chunk_id, entities) for _, _, name in spans
        ]
        for left in range(len(spans)):
            for right in range(left + 1, len(spans)):
                if resolved[left].entity_id == resolved[right].entity_id:
                    continue
                between = sentence[spans[left][1] : spans[right][0]]
                predicate = _predicate_between(between)
                if predicate is None:
                    continue
                key = (resolved[left].entity_id, resolved[right].entity_id, predicate)
                if key in seen:
                    continue
                seen.add(key)
                relationships.append(
                    Relationship(
                        source_id=resolved[left].entity_id,
                        target_id=resolved[right].entity_id,
                        predicate=predicate,
                        chunk_id=chunk.chunk_id,
                    )
                )
    return GraphExtractionResult(entities=list(entities.values()), relationships=relationships)


def merge_results(parts: list[GraphExtractionResult]) -> GraphExtractionResult:
    """Union entities that share an id and keep the first copy of each edge."""
    entities: dict[str, Entity] = {}
    relationships: dict[tuple[str, str, str], Relationship] = {}
    for part in parts:
        for entity in part.entities:
            current = entities.get(entity.entity_id)
            if current is None:
                entities[entity.entity_id] = entity.model_copy(deep=True)
            else:
                _absorb(current, entity)
        for relationship in part.relationships:
            key = (relationship.source_id, relationship.target_id, relationship.predicate)
            relationships.setdefault(key, relationship)
    return GraphExtractionResult(
        entities=list(entities.values()),
        relationships=list(relationships.values()),
    )


def _walk_body(
    body: list[ast.stmt],
    parent: Entity,
    stack: list[str],
    entities: dict[str, Entity],
    relationships: list[Relationship],
    chunk: DocumentChunk,
) -> None:
    for child in body:
        if isinstance(child, ast.ClassDef):
            _add_class(child, parent, stack, entities, relationships, chunk)
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            _add_function(child, parent, stack, entities, relationships, chunk)
        elif isinstance(child, ast.Import | ast.ImportFrom) and not stack:
            _add_imports(child, parent, entities, relationships, chunk)


def _add_class(
    node: ast.ClassDef,
    parent: Entity,
    stack: list[str],
    entities: dict[str, Entity],
    relationships: list[Relationship],
    chunk: DocumentChunk,
) -> None:
    qualified = ".".join([*stack, node.name])
    entity = _ensure(qualified, "Class", chunk.chunk_id, entities)
    _link(parent, entity, "DEFINES", chunk.chunk_id, relationships)
    for base in node.bases:
        if isinstance(base, ast.Name):
            base_entity = _ensure(base.id, infer_kind(base.id), chunk.chunk_id, entities)
            _link(entity, base_entity, "INHERITS", chunk.chunk_id, relationships)
    _walk_body(node.body, entity, [*stack, node.name], entities, relationships, chunk)


def _add_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    parent: Entity,
    stack: list[str],
    entities: dict[str, Entity],
    relationships: list[Relationship],
    chunk: DocumentChunk,
) -> None:
    qualified = ".".join([*stack, node.name])
    entity = _ensure(qualified, "Function", chunk.chunk_id, entities)
    _link(parent, entity, "DEFINES", chunk.chunk_id, relationships)
    for call in _calls(node):
        target = _call_target(call.func)
        if target is None:
            continue
        callee = _ensure(target, infer_kind(target), chunk.chunk_id, entities)
        if callee.entity_id != entity.entity_id:
            _link(entity, callee, "CALLS", chunk.chunk_id, relationships)
    _walk_body(node.body, entity, [*stack, node.name], entities, relationships, chunk)


def _add_imports(
    node: ast.Import | ast.ImportFrom,
    parent: Entity,
    entities: dict[str, Entity],
    relationships: list[Relationship],
    chunk: DocumentChunk,
) -> None:
    if isinstance(node, ast.Import):
        names = [(alias.name, "Module") for alias in node.names]
    else:
        names = []
        if node.module:
            names.append((node.module, "Module"))
        imported = [
            (alias.name, infer_kind(alias.name))
            for alias in node.names
            if alias.name != "*"
        ]
        names.extend(imported)
    for name, kind in names:
        target = _ensure(name, kind, chunk.chunk_id, entities)
        _link(parent, target, "IMPORTS", chunk.chunk_id, relationships)


def _calls(node: ast.AST) -> list[ast.Call]:
    found: list[ast.Call] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if isinstance(child, ast.Call):
            found.append(child)
        found.extend(_calls(child))
    return found


def _call_target(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in {"self", "cls"}:
            return None
        return node.id
    if isinstance(node, ast.Call):
        return _call_target(node.func)
    if isinstance(node, ast.Attribute):
        parent = _call_target(node.value)
        if parent is not None:
            return parent
        if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            return node.attr
        return node.attr
    return None


def _ensure(
    name: str,
    kind: EntityKind,
    chunk_id: str,
    entities: dict[str, Entity],
) -> Entity:
    entity = make_entity(name, kind, chunk_id)
    current = entities.get(entity.entity_id)
    if current is None:
        entities[entity.entity_id] = entity
        return entity
    if chunk_id not in current.chunk_ids:
        current.chunk_ids.append(chunk_id)
    if name != current.name and name not in current.aliases:
        current.aliases.append(name)
    return current


def _link(
    source: Entity,
    target: Entity,
    predicate: RelationKind,
    chunk_id: str,
    relationships: list[Relationship],
) -> None:
    for relationship in relationships:
        if (
            relationship.source_id == source.entity_id
            and relationship.target_id == target.entity_id
            and relationship.predicate == predicate
        ):
            return
    relationships.append(
        Relationship(
            source_id=source.entity_id,
            target_id=target.entity_id,
            predicate=predicate,
            chunk_id=chunk_id,
        )
    )


def _predicate_between(text: str) -> RelationKind | None:
    for pattern, predicate in _VERBS:
        if pattern.search(text):
            return predicate
    return None


def _absorb(current: Entity, incoming: Entity) -> None:
    for chunk_id in incoming.chunk_ids:
        if chunk_id not in current.chunk_ids:
            current.chunk_ids.append(chunk_id)
    for alias in [incoming.name, *incoming.aliases]:
        if alias != current.name and alias not in current.aliases:
            current.aliases.append(alias)
