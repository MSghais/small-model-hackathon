from inference.config import get_app_config, get_model_config
from inference.factory import get_backend, reset_backend
from inference.response_clean import strip_reasoning_output

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
        "Loading weights…"
    )


def reload_model(model_key: str) -> str:
    """Clear cached backend and reload weights for settings panel."""
    global _current_model_key

    key = model_key or _app_config.active_model
    reset_backend()
    _current_model_key = None
    _load_state.pop(key, None)
    _load_errors.pop(key, None)
    error = ensure_model_loaded(key)
    if error:
        return error
    return warmup(key)


def preload_active_model() -> str:
    """Load the active preset at startup so the first request is fast."""
    key = get_active_model_key()
    print(f"[startup] Loading model preset `{key}`…", flush=True)
    error = ensure_model_loaded(key)
    if error:
        print(f"[startup] {error}", flush=True)
        return error
    status = warmup(key)
    print(f"[startup] {status}", flush=True)
    return status


def model_status(model_key: str) -> str:
    model = get_model_config(model_key)
    return f"**{model.label}**\n\n- Backend: `{model.backend}`\n- {warmup(model_key)}"


def _history_to_messages(history: list) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    return messages


def chat(message: str, history: list, model_key: str) -> str:
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return load_error

    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": message})
    reply = get_backend(model_key).chat(messages)
    return strip_reasoning_output(reply)
