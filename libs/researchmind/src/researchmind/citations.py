from __future__ import annotations

import re
from dataclasses import dataclass

from inference.response_clean import looks_like_reasoning_only, strip_reasoning_output

from researchmind.store import StoredChunk

_EXCERPT_LEN = 400
_PASSAGE_LEN = 700
_CITATION_RUN = re.compile(r"(?:\[\d{1,4}\]\s*){3,}")


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: str
    doc_title: str
    doc_uri: str
    excerpt: str


def _clean_passage(text: str) -> str:
    """Collapse long runs of in-text [n] markers from scraped papers."""
    cleaned = _CITATION_RUN.sub("[…] ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > _PASSAGE_LEN:
        return cleaned[:_PASSAGE_LEN] + "…"
    return cleaned


def format_context_block(chunks: list[StoredChunk]) -> tuple[str, list[Citation]]:
    """Build LLM context with one citation index per source document."""
    groups: list[tuple[str, str, list[StoredChunk]]] = []
    seen_uris: set[str] = set()
    for chunk in chunks:
        if chunk.doc_uri in seen_uris:
            for uri, _title, group in groups:
                if uri == chunk.doc_uri:
                    group.append(chunk)
                    break
        else:
            seen_uris.add(chunk.doc_uri)
            groups.append((chunk.doc_uri, chunk.doc_title, [chunk]))

    citations: list[Citation] = []
    blocks: list[str] = []
    for i, (uri, title, doc_chunks) in enumerate(groups, start=1):
        passages = [_clean_passage(c.text) for c in doc_chunks if c.text.strip()]
        merged = "\n\n".join(passages)
        excerpt = merged[:_EXCERPT_LEN] + ("..." if len(merged) > _EXCERPT_LEN else "")
        citations.append(
            Citation(
                index=i,
                chunk_id=doc_chunks[0].id,
                doc_title=title,
                doc_uri=uri,
                excerpt=excerpt,
            )
        )
        blocks.append(f"[{i}] **{title}**\n{uri}\n\n{merged}")

    context = "\n\n---\n\n".join(blocks)
    return context, citations


def format_references(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["**References**"]
    for c in citations:
        lines.append(f"- [{c.index}] {c.doc_title} — {c.doc_uri}")
    return "\n".join(lines)


def clean_model_answer(answer: str) -> str:
    """Remove thinking traces, duplicate references, and citation spam from model output."""
    text = strip_reasoning_output(answer)
    if "**References**" in text:
        text = text.split("**References**", maxsplit=1)[0].rstrip()
    if "\nReferences\n" in text:
        text = text.split("\nReferences\n", maxsplit=1)[0].rstrip()
    text = _CITATION_RUN.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if not text or looks_like_reasoning_only(text):
        return (
            "The model returned planning text without a final answer. "
            "Try asking again or switch to a non-reasoning model preset."
        )
    return text