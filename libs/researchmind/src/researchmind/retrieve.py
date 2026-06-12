from __future__ import annotations

import numpy as np

from researchmind.config import ResearchMindConfig, get_config
from researchmind.embeddings import embed_texts
from researchmind.store import MemRAGStore, StoredChunk


def retrieve(
    query: str,
    store: MemRAGStore,
    *,
    config: ResearchMindConfig | None = None,
    top_k: int | None = None,
    expand_neighbors: bool = True,
    session_id: str | None = None,
    doc_ids: list[str] | None = None,
) -> list[StoredChunk]:
    cfg = config or get_config()
    k = top_k if top_k is not None else cfg.top_k
    all_chunks = store.get_chunks_with_embeddings(
        session_id=session_id,
        doc_ids=doc_ids,
    )
    if not all_chunks:
        return []

    q_vec = embed_texts([query], model_name=cfg.embed_model)[0]
    scored: list[tuple[float, StoredChunk]] = []
    for chunk, emb in all_chunks:
        sim = float(np.dot(q_vec, emb))
        scored.append((sim, chunk))

    max_chunks = cfg.max_context_chunks
    scored.sort(key=lambda x: x[0], reverse=True)
    selected: list[StoredChunk] = []
    seen_ids: set[str] = set()

    for _, chunk in scored[:k]:
        if len(selected) >= max_chunks:
            break
        if chunk.id not in seen_ids:
            selected.append(chunk)
            seen_ids.add(chunk.id)
        if expand_neighbors and len(selected) < max_chunks:
            for nid in store.get_neighbor_chunk_ids(chunk.id)[:1]:
                if len(selected) >= max_chunks:
                    break
                if nid not in seen_ids:
                    neighbors = store.get_chunks_by_ids([nid])
                    for n in neighbors:
                        selected.append(n)
                        seen_ids.add(n.id)
                        break

    return selected[:max_chunks]
