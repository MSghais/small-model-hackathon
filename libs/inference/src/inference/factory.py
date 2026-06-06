import os
from functools import lru_cache

from inference.base import InferenceBackend
from inference.llama_cpp import LlamaCppBackend


@lru_cache(maxsize=1)
def get_backend() -> InferenceBackend:
    backend_name = os.environ.get("INFERENCE_BACKEND", "llama_cpp").lower()

    if backend_name == "llama_cpp":
        return LlamaCppBackend()

    if backend_name == "transformers":
        from inference.transformers import TransformersBackend

        return TransformersBackend()

    raise ValueError(
        f"Unknown INFERENCE_BACKEND={backend_name!r}. "
        "Expected 'llama_cpp' or 'transformers'."
    )
