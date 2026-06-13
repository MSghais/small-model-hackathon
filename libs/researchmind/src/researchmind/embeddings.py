from __future__ import annotations

import os

import numpy as np

from inference.device_utils import clear_cuda_cache, is_cuda_oom

_embedder = None
_embedder_model_name: str | None = None
_embedder_device: str | None = None


def _embed_device_preference() -> str:
    return os.environ.get("RESEARCHMIND_EMBED_DEVICE", "cpu").strip().lower()


def _embed_device_candidates() -> list[str]:
    pref = _embed_device_preference()
    if pref == "cpu":
        return ["cpu"]
    if pref == "cuda":
        return ["cuda", "cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            return ["cuda", "cpu"]
    except ImportError:
        pass
    return ["cpu"]


def get_embedder(model_name: str):
    global _embedder, _embedder_model_name, _embedder_device
    if (
        _embedder is not None
        and _embedder_model_name == model_name
        and _embedder_device is not None
    ):
        return _embedder

    from sentence_transformers import SentenceTransformer

    last_error: Exception | None = None
    for device in _embed_device_candidates():
        try:
            _embedder = SentenceTransformer(model_name, device=device)
            _embedder_model_name = model_name
            _embedder_device = device
            print(f"[researchmind] Embedding model on {device}", flush=True)
            return _embedder
        except Exception as exc:
            if device != "cpu" and is_cuda_oom(exc):
                last_error = exc
                clear_cuda_cache()
                print(
                    "[researchmind] CUDA OOM loading embedder; falling back to CPU…",
                    flush=True,
                )
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load embedding model {model_name!r}")


def embed_texts(texts: list[str], *, model_name: str) -> np.ndarray:
    global _embedder, _embedder_device

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = get_embedder(model_name)
    try:
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except RuntimeError as exc:
        if _embedder_device != "cpu" and is_cuda_oom(exc):
            clear_cuda_cache()
            _embedder = None
            _embedder_device = None
            model = get_embedder(model_name)
            vectors = model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        else:
            raise
    return np.asarray(vectors, dtype=np.float32)


def embedding_to_bytes(vector: np.ndarray) -> bytes:
    return vector.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).reshape(dim)
