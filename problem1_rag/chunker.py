"""Token-aware recursive text splitter with deterministic chunk IDs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import tiktoken

from problem1_rag.parsers import ParsedDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


@dataclass
class DocumentChunk:
    """Chunk record ready for embedding and vector store insertion."""

    chunk_id: str
    source_id: str
    doc_hash: str
    file_type: str
    topic: str
    section_header: str
    chunk_index: int
    token_count: int
    text: str


def count_tokens(text: str) -> int:
    """Count BPE tokens using cl100k_base."""
    return len(_ENCODING.encode(text))


def make_chunk_id(source_id: str, chunk_index: int) -> str:
    """Generate a deterministic, human-readable chunk ID from source_id + chunk_index."""
    digest = hashlib.sha256(f"{source_id}:{chunk_index}".encode()).hexdigest()[:10]
    stem = source_id.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    safe_stem = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem)[:28]
    return f"{safe_stem}__c{chunk_index:03d}__{digest}"


def _split_recursive(text: str, chunk_size_tokens: int, sep_idx: int = 0) -> list[str]:
    """Split text recursively down separator hierarchy until every piece fits chunk_size."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if count_tokens(cleaned) <= chunk_size_tokens:
        return [cleaned]

    if sep_idx >= len(_SEPARATORS) - 1:
        # Hard token-slice fallback when a single token sequence exceeds chunk_size_tokens
        tokens = _ENCODING.encode(cleaned)
        return [
            _ENCODING.decode(tokens[i : i + chunk_size_tokens]).strip()
            for i in range(0, len(tokens), chunk_size_tokens)
            if tokens[i : i + chunk_size_tokens]
        ]

    sep = _SEPARATORS[sep_idx]
    raw_parts = cleaned.split(sep) if sep else list(cleaned)
    pieces: list[str] = []
    for i, part in enumerate(raw_parts):
        suffix = "." if sep == ". " and i < len(raw_parts) - 1 and not part.endswith(".") else ""
        candidate = (part + suffix).strip()
        if not candidate:
            continue
        if count_tokens(candidate) <= chunk_size_tokens:
            pieces.append(candidate)
        else:
            pieces.extend(_split_recursive(candidate, chunk_size_tokens, sep_idx + 1))
    return pieces


def _pack_with_overlap(
    pieces: list[str],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[str]:
    """Merge fine-grained pieces into token-bounded chunks preserving trailing token overlap."""
    chunks: list[str] = []
    current_parts: list[str] = []

    for piece in pieces:
        candidate = " ".join([*current_parts, piece]).strip()
        if current_parts and count_tokens(candidate) > chunk_size_tokens:
            merged = " ".join(current_parts).strip()
            chunks.append(merged)

            # Carry trailing overlap tokens into the next chunk
            if chunk_overlap_tokens > 0:
                tail_tokens = _ENCODING.encode(merged)[-chunk_overlap_tokens:]
                overlap_text = _ENCODING.decode(tail_tokens).strip()
                current_parts = [overlap_text, piece] if overlap_text else [piece]
                if count_tokens(" ".join(current_parts)) > chunk_size_tokens:
                    current_parts = [piece]
            else:
                current_parts = [piece]
        else:
            current_parts.append(piece)

    if current_parts:
        chunks.append(" ".join(current_parts).strip())
    return chunks


def chunk_document(
    doc: ParsedDocument,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
) -> list[DocumentChunk]:
    """Split a ParsedDocument into overlapping token-bounded DocumentChunks."""
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must satisfy 0 <= overlap < chunk_size_tokens")

    section_units: list[tuple[str, str]] = []
    for sec in doc.sections:
        labeled = f"[{sec.section_header}] {sec.text}".strip()
        for piece in _split_recursive(labeled, chunk_size_tokens):
            section_units.append((sec.section_header, piece))

    if not section_units:
        return []

    # Pack adjacent units into chunks up to chunk_size_tokens while tracking dominant section
    packed_texts = _pack_with_overlap(
        [text for _, text in section_units],
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )

    chunks: list[DocumentChunk] = []
    for idx, text in enumerate(packed_texts):
        matched_header = doc.sections[0].section_header
        for header, _ in section_units:
            if f"[{header}]" in text:
                matched_header = header
                break
        chunks.append(
            DocumentChunk(
                chunk_id=make_chunk_id(doc.source_id, idx),
                source_id=doc.source_id,
                doc_hash=doc.doc_hash,
                file_type=doc.file_type,
                topic=doc.topic,
                section_header=matched_header,
                chunk_index=idx,
                token_count=count_tokens(text),
                text=text,
            )
        )
    return chunks
