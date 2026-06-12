from __future__ import annotations

from researchmind.citations import (
    clean_model_answer,
    format_context_block,
    format_references,
)
from researchmind.store import StoredChunk


def _chunk(chunk_id: str, doc_uri: str, text: str) -> StoredChunk:
    return StoredChunk(
        id=chunk_id,
        doc_id="doc1",
        ordinal=0,
        text=text,
        doc_title="AI Agents Review",
        doc_uri=doc_uri,
        metadata={},
    )


def test_format_context_groups_chunks_by_document():
    chunks = [
        _chunk("c1", "https://example.com/paper", "First passage about agents."),
        _chunk("c2", "https://example.com/paper", "Second passage about planning."),
    ]
    context, citations = format_context_block(chunks)
    assert context.count("[1]") == 1
    assert "[2]" not in context
    assert len(citations) == 1
    assert "First passage" in context
    assert "Second passage" in context


def test_format_references_one_line_per_source():
    _, citations = format_context_block(
        [
            _chunk("c1", "https://a.test", "alpha"),
            _chunk("c2", "https://a.test", "beta"),
        ]
    )
    refs = format_references(citations)
    assert refs.count("https://a.test") == 1


def test_clean_passage_collapses_citation_runs():
    chunks = [_chunk("c1", "https://a.test", "[1] [2] [3] [4] [5] actual content")]
    context, _ = format_context_block(chunks)
    assert "[1] [2] [3] [4] [5]" not in context
    assert "actual content" in context


def test_clean_model_answer_strips_reference_spam():
    raw = "Summary here [1][2][3][4][5].\n\n**References**\n- [1] dup"
    cleaned = clean_model_answer(raw)
    assert "**References**" not in cleaned
    assert "[1][2][3]" not in cleaned
    assert "Summary here" in cleaned
