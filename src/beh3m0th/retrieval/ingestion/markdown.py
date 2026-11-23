"""Header-aware Markdown chunking with paragraph overlap."""

from __future__ import annotations

import re

from beh3m0th.core.schemas import DocumentChunk

_HEADER = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
type _Line = tuple[int, str]


def chunk_markdown(
    text: str,
    *,
    source_path: str,
    doc_id: str,
    max_chars: int = 800,
    overlap_lines: int = 1,
    tags: list[str] | None = None,
) -> list[DocumentChunk]:
    """Split Markdown on ATX headers, then on blank lines when a section is long."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")
    if not text.strip():
        return []
    tag_list = list(tags or [])
    chunks: list[DocumentChunk] = []
    for header, content in _sections(text):
        for block in _windows(content, max_chars=max_chars, overlap_lines=overlap_lines):
            body = "\n".join(line for _, line in block).strip()
            if not body:
                continue
            line_start = block[0][0]
            line_end = block[-1][0]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}:md:{line_start}:{len(chunks)}",
                    doc_id=doc_id,
                    text=body,
                    source_path=source_path,
                    language="markdown",
                    section_header=header,
                    tags=tag_list,
                    line_start=line_start,
                    line_end=line_end,
                )
            )
    return chunks


def _sections(text: str) -> list[tuple[str | None, list[_Line]]]:
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str | None, list[_Line]]] = []
    header: str | None = None
    content: list[_Line] = []

    def flush() -> None:
        nonlocal content
        if any(line.strip() for _, line in content):
            sections.append((header, content))
        content = []

    for number, line in enumerate(text.splitlines(), start=1):
        match = _HEADER.match(line)
        if match is None:
            content.append((number, line))
            continue
        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        header = " > ".join(item for _, item in stack)
    flush()
    return sections


def _windows(content: list[_Line], *, max_chars: int, overlap_lines: int) -> list[list[_Line]]:
    paragraphs = _paragraphs(content)
    if not paragraphs:
        return []
    windows: list[list[_Line]] = []
    current: list[_Line] = []
    current_len = 0
    for paragraph in paragraphs:
        paragraph_len = _length(paragraph)
        if paragraph_len > max_chars:
            if current:
                windows.append(current)
                current = []
                current_len = 0
            windows.extend(_split_long_paragraph(paragraph, max_chars, overlap_lines))
            continue
        if current and current_len + paragraph_len + 1 > max_chars:
            windows.append(current)
            current = _overlap(current, overlap_lines)
            current_len = _length(current)
        if current:
            current_len += 1
        current.extend(paragraph)
        current_len += paragraph_len
    if current:
        windows.append(current)
    return windows


def _paragraphs(content: list[_Line]) -> list[list[_Line]]:
    paragraphs: list[list[_Line]] = []
    current: list[_Line] = []
    for item in content:
        if item[1].strip():
            current.append(item)
            continue
        if current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    return paragraphs


def _split_long_paragraph(
    paragraph: list[_Line],
    max_chars: int,
    overlap_lines: int,
) -> list[list[_Line]]:
    windows: list[list[_Line]] = []
    current: list[_Line] = []
    current_len = 0
    for item in paragraph:
        size = len(item[1])
        if current and current_len + size + 1 > max_chars:
            windows.append(current)
            current = _overlap(current, overlap_lines)
            current_len = _length(current)
        if current:
            current_len += 1
        current.append(item)
        current_len += size
    if current:
        windows.append(current)
    return windows


def _overlap(block: list[_Line], overlap_lines: int) -> list[_Line]:
    if overlap_lines <= 0:
        return []
    return list(block[-overlap_lines:])


def _length(block: list[_Line]) -> int:
    if not block:
        return 0
    return sum(len(line) for _, line in block) + len(block) - 1
