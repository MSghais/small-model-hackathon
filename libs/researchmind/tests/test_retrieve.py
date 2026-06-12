from __future__ import annotations

import numpy as np

from researchmind.config import ResearchMindConfig
from researchmind.retrieve import retrieve
from researchmind.store import MemRAGStore


def _fake_embed(monkeypatch):
    def fake_embed_texts(texts, *, model_name):
        out = []
        for t in texts:
            if "photosynthesis" in t.lower():
                out.append(np.array([1.0, 0.0], dtype=np.float32))
            else:
                out.append(np.array([0.0, 1.0], dtype=np.float32))
        return np.stack(out)

    monkeypatch.setattr("researchmind.retrieve.embed_texts", fake_embed_texts)


def test_retrieve_ranks_by_similarity(tmp_path, monkeypatch):
    _fake_embed(monkeypatch)
    cfg = ResearchMindConfig(
        data_dir=tmp_path,
        embed_model="test",
        auto_search=False,
        top_k=1,
        chunk_size=512,
        chunk_overlap=128,
    )
    store = MemRAGStore(cfg)
    store.set_embed_dim(2)
    store.add_document(
        source_type="test",
        uri="a",
        title="A",
        text="photosynthesis in plants",
        chunks=[("c1", 0, "photosynthesis in plants", np.array([1.0, 0.0], dtype=np.float32), {})],
    )
    store.add_document(
        source_type="test",
        uri="b",
        title="B",
        text="fractions math",
        chunks=[("c2", 0, "fractions math", np.array([0.0, 1.0], dtype=np.float32), {})],
    )
    hits = retrieve("photosynthesis", store, config=cfg, top_k=1, expand_neighbors=False)
    assert len(hits) == 1
    assert "photosynthesis" in hits[0].text


def test_retrieve_filters_by_session(tmp_path, monkeypatch):
    _fake_embed(monkeypatch)
    cfg = ResearchMindConfig(
        data_dir=tmp_path,
        embed_model="test",
        auto_search=False,
        top_k=2,
        chunk_size=512,
        chunk_overlap=128,
    )
    store = MemRAGStore(cfg)
    store.set_embed_dim(2)
    sid_a = store.create_session(topic="a").id
    sid_b = store.create_session(topic="b").id
    store.add_document(
        source_type="test",
        uri="a",
        title="Plants",
        text="photosynthesis in plants",
        chunks=[("c1", 0, "photosynthesis in plants", np.array([1.0, 0.0], dtype=np.float32), {})],
        session_id=sid_a,
    )
    store.add_document(
        source_type="test",
        uri="b",
        title="Math",
        text="fractions math",
        chunks=[("c2", 0, "fractions math", np.array([0.0, 1.0], dtype=np.float32), {})],
        session_id=sid_b,
    )
    scoped = retrieve(
        "photosynthesis",
        store,
        config=cfg,
        top_k=2,
        expand_neighbors=False,
        session_id=sid_a,
    )
    assert len(scoped) == 1
    assert "photosynthesis" in scoped[0].text
