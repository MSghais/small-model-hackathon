from inference.base import InferenceBackend
from inference.config import ModelConfig, get_model_config
from inference.llama_cpp import LlamaCppBackend

_backend: InferenceBackend | None = None
_backend_key: tuple | None = None


def _create_backend(config: ModelConfig) -> InferenceBackend:
    if config.backend == "llama_cpp":
        return LlamaCppBackend(config)

    if config.backend == "transformers":
        from inference.transformers import TransformersBackend

        return TransformersBackend(config)

    raise ValueError(
        f"Unknown backend {config.backend!r} for preset {config.key!r}. "
        "Expected 'llama_cpp' or 'transformers'."
    )


def get_backend(model_key: str | None = None) -> InferenceBackend:
    global _backend, _backend_key

    config = get_model_config(model_key)
    cache_key = config.cache_key()

    if _backend is None or _backend_key != cache_key:
        _backend = _create_backend(config)
        _backend_key = cache_key

    return _backend


def reset_backend() -> None:
    global _backend, _backend_key
    if _backend is not None and hasattr(_backend, "unload"):
        _backend.unload()
    _backend = None
    _backend_key = None
