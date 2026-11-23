"""Python function and class chunking with the standard library parser."""

from __future__ import annotations

import ast

from beh3m0th.core.schemas import DocumentChunk
from beh3m0th.retrieval.ingestion.errors import ChunkingError


def chunk_python(
    text: str,
    *,
    source_path: str,
    doc_id: str,
    tags: list[str] | None = None,
) -> list[DocumentChunk]:
    """Emit one chunk per class, function, and async function."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        location = exc.lineno or 1
        raise ChunkingError(
            f"invalid Python in {source_path} at line {location}: {exc.msg}"
        ) from exc
    lines = text.splitlines()
    tag_list = list(tags or [])
    chunks: list[DocumentChunk] = []
    _visit(tree, [], lines, source_path, doc_id, tag_list, chunks)
    return chunks


def _visit(
    node: ast.AST,
    stack: list[str],
    lines: list[str],
    source_path: str,
    doc_id: str,
    tags: list[str],
    chunks: list[DocumentChunk],
) -> None:
    body = getattr(node, "body", [])
    for child in body:
        if isinstance(child, ast.ClassDef):
            symbol = ".".join([*stack, child.name])
            chunks.append(_chunk(child, symbol, lines, source_path, doc_id, tags, len(chunks)))
            _visit(child, [*stack, child.name], lines, source_path, doc_id, tags, chunks)
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            symbol = ".".join([*stack, child.name])
            chunks.append(_chunk(child, symbol, lines, source_path, doc_id, tags, len(chunks)))
            _visit(child, [*stack, child.name], lines, source_path, doc_id, tags, chunks)


def _chunk(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    symbol: str,
    lines: list[str],
    source_path: str,
    doc_id: str,
    tags: list[str],
    serial: int,
) -> DocumentChunk:
    start = node.lineno
    end = node.end_lineno or node.lineno
    body = "\n".join(lines[start - 1 : end]).strip()
    return DocumentChunk(
        chunk_id=f"{doc_id}:py:{symbol}:{start}:{serial}",
        doc_id=doc_id,
        text=body or symbol,
        source_path=source_path,
        language="python",
        symbol_name=symbol,
        tags=tags,
        line_start=start,
        line_end=end,
    )
