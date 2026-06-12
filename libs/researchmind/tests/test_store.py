from __future__ import annotations

import numpy as np

from researchmind.config import ResearchMindConfig
from researchmind.store import MemRAGStore


def test_store_dedup_and_chunks(tmp_path):
    cfg = ResearchMindConfig(
        data_dir=tmp_path,
        embed_model="test",
        auto_search=False,
        top_k=3,
        max_context_chunks=8,
        chunk_size=512,
        chunk_overlap=128,
    )
    store = MemRAGStore(cfg)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    chunks = [("c1", 0, "hello world", emb, {})]
    doc_id, is_new = store.add_document(
        source_type="test",
        uri="test://a",
        title="A",
        text="hello world",
        chunks=chunks,
    )
    assert is_new
    doc_id2, is_new2 = store.add_document(
        source_type="test",
        uri="test://a",
        title="A",
        text="hello world",
        chunks=chunks,
    )
    assert not is_new2
    assert doc_id == doc_id2
    assert store.count_chunks() == 1


def test_session_messages(tmp_path):
    cfg = ResearchMindConfig(
        data_dir=tmp_path,
        embed_model="test",
        auto_search=False,
        top_k=3,
        max_context_chunks=8,
        chunk_size=512,
        chunk_overlap=128,
    )
    store = MemRAGStore(cfg)
    session = store.create_session(topic="test topic")
    store.add_message(session.id, "user", "hi", [])
    msgs = store.get_messages(session.id)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
