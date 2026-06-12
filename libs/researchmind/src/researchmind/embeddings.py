from __future__ import annotations

import numpy as np

_embedder = None
_embedder_model_name: str | None = None


def get_embedder(model_name: str):
    global _embedder, _embedder_model_name
    if _embedder is None or _embedder_model_name != model_name:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(model_name)
        _embedder_model_name = model_name
    return _embedder


def embed_texts(texts: list[str], *, model_name: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_embedder(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embedding_to_bytes(vector: np.ndarray) -> bytes:
    return vector.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).reshape(dim)
