from __future__ import annotations

from researchmind.chunking import chunk_text


def test_chunk_text_splits_long_document():
    words = ["word"] * 600
    text = " ".join(words)
    chunks = chunk_text(text, doc_id="doc1", chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[0].ordinal == 0


def test_chunk_text_empty():
    assert chunk_text("", doc_id="x") == []
