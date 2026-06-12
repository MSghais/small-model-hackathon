from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    ordinal: int
    text: str


def _approx_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def chunk_text(
    text: str,
    *,
    doc_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> list[TextChunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[TextChunk] = []
    start = 0
    ordinal = 0
    step = max(1, chunk_size - chunk_overlap)

    while start < len(words):
        end = min(len(words), start + chunk_size)
        piece = " ".join(words[start:end]).strip()
        if piece:
            digest = hashlib.sha256(f"{doc_id}:{ordinal}:{piece}".encode()).hexdigest()[:16]
            chunks.append(TextChunk(chunk_id=f"{doc_id}_{ordinal}_{digest}", ordinal=ordinal, text=piece))
            ordinal += 1
        if end >= len(words):
            break
        start += step

    return chunks
