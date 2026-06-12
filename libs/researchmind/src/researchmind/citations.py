from __future__ import annotations

from dataclasses import dataclass

from researchmind.store import StoredChunk


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: str
    doc_title: str
    doc_uri: str
    excerpt: str


def format_context_block(chunks: list[StoredChunk]) -> tuple[str, list[Citation]]:
    citations: list[Citation] = []
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        excerpt = chunk.text[:400] + ("..." if len(chunk.text) > 400 else "")
        citations.append(
            Citation(
                index=i,
                chunk_id=chunk.id,
                doc_title=chunk.doc_title,
                doc_uri=chunk.doc_uri,
                excerpt=excerpt,
            )
        )
        blocks.append(f"[{i}] ({chunk.doc_title})\n{chunk.text}")

    context = "\n\n---\n\n".join(blocks)
    return context, citations


def format_references(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["**References**"]
    for c in citations:
        lines.append(f"- [{c.index}] {c.doc_title} — {c.doc_uri}")
    return "\n".join(lines)
