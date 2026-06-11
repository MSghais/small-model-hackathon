from inference.config import get_app_config, get_model_config
from inference.factory import get_backend, reset_backend

_app_config = get_app_config()
_current_model_key: str | None = None
_load_state: dict[str, bool] = {}
_load_errors: dict[str, str] = {}


def get_active_model_key() -> str:
    return _app_config.active_model


def ensure_model_loaded(model_key: str) -> str | None:
    global _current_model_key

    if model_key != _current_model_key:
        reset_backend()
        _current_model_key = model_key

    if _load_state.get(model_key):
        return None

    if model_key in _load_errors:
        return _load_errors[model_key]

    try:
        get_backend(model_key).load()
        _load_state[model_key] = True
        return None
    except Exception as exc:  # noqa: BLE001 — surface model load failures in the UI
        message = f"Failed to load model: {exc}"
        _load_errors[model_key] = message
        return message


def runtime_device_hint(model_key: str) -> str:
    model = get_model_config(model_key)
    if model.backend == "transformers":
        try:
            import torch

            if torch.cuda.is_available():
                return f"GPU ({torch.cuda.get_device_name(0)})"
        except ImportError:
            pass
        return "CPU"
    if model.n_gpu_layers > 0:
        return f"llama.cpp GPU offload ({model.n_gpu_layers} layers)"
    return "CPU"


def warmup(model_key: str | None = None) -> str:
    key = model_key or _app_config.active_model
    model = get_model_config(key)

    if _load_state.get(key):
        backend = get_backend(key)
        device = (
            backend.device_label
            if hasattr(backend, "device_label")
            else runtime_device_hint(key)
        )
        return f"Model ready: {model.label} on {device}"

    if key in _load_errors:
        return _load_errors[key]

    device_hint = runtime_device_hint(key)
    return (
        f"Preset `{key}` selected ({model.backend}, {device_hint}). "
        "Weights load on the first request."
    )


def model_status(model_key: str) -> str:
    model = get_model_config(model_key)
    return f"**{model.label}**\n\n- Backend: `{model.backend}`\n- {warmup(model_key)}"
