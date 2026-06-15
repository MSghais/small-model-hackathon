from gradio_space.spaces_runtime import gpu_task
from inference.config import get_app_config, get_model_config
from inference.factory import get_backend, reset_backend
from inference.response_clean import strip_reasoning_output

_app_config = get_app_config()
_runtime_model_key: str | None = None
_current_model_key: str | None = None
_load_state: dict[str, bool] = {}
_load_errors: dict[str, str] = {}


def get_active_model_key() -> str:
    return _runtime_model_key or _app_config.active_model


def set_runtime_model_key(key: str) -> str:
    """Pin the active preset for all tabs until process restart."""
    global _runtime_model_key, _current_model_key

    model = get_model_config(key)
    if key != get_active_model_key():
        reset_backend()
        _current_model_key = None
        if _runtime_model_key:
            _load_state.pop(_runtime_model_key, None)
            _load_errors.pop(_runtime_model_key, None)
    _runtime_model_key = key
    return model.label


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
    key = model_key or get_active_model_key()
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


@gpu_task(duration=120)
def reload_model(model_key: str) -> str:
    """Clear cached backend and reload weights for settings panel."""
    global _current_model_key

    key = model_key or get_active_model_key()
    set_runtime_model_key(key)
    reset_backend()
    _current_model_key = None
    _load_state.pop(key, None)
    _load_errors.pop(key, None)
    error = ensure_model_loaded(key)
    if error:
        return error
    return warmup(key)


def select_and_reload_model(model_key: str) -> str:
    """Switch runtime preset and load weights (Settings dropdown)."""
    return reload_model(model_key)


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
    notes = ""
    if model.backend == "llama_cpp" and model.multimodal:
        notes = (
            "\n- Note: text-only on llama.cpp; use transformers preset for image/video input."
        )
    return (
        f"**{model.label}**\n\n"
        f"- Backend: `{model.backend}`\n"
        f"- {warmup(model_key)}{notes}"
    )


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


@gpu_task(duration=60)
def chat(message: str, history: list, model_key: str) -> str:
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return load_error

    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": message})
    reply = get_backend(model_key).chat(messages)
    return strip_reasoning_output(reply)
