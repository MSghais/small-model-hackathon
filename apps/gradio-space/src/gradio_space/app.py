import os

import gradio as gr

from inference.config import get_app_config, get_model_config
from inference.factory import get_backend, reset_backend

_app_config = get_app_config()
_current_model_key: str | None = None
_load_state: dict[str, bool] = {}
_load_errors: dict[str, str] = {}


def _ensure_model_loaded(model_key: str) -> str | None:
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
    load_error = _ensure_model_loaded(model_key)
    if load_error:
        return load_error

    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": message})
    return get_backend(model_key).chat(messages)


def warmup(model_key: str | None = None) -> str:
    key = model_key or _app_config.active_model
    model = get_model_config(key)

    if _load_state.get(key):
        return f"Model ready: {model.label}"

    if key in _load_errors:
        return _load_errors[key]

    return (
        f"Preset `{key}` selected ({model.backend}). "
        "Weights load on the first chat message — this can take a few minutes on CPU."
    )


def model_status(model_key: str) -> str:
    model = get_model_config(model_key)
    return f"**{model.label}**\n\n- Backend: `{model.backend}`\n- {warmup(model_key)}"


def build_demo() -> gr.Blocks:
    active = _app_config.active
    presets_note = (
        f"Presets file: `{_app_config.presets_path}`"
        if _app_config.presets_path
        else "Using built-in presets (models.yaml not found)."
    )

    with gr.Blocks(title="Small Model Hackathon") as demo:
        gr.Markdown(
            f"""
# Small Model Chat

Local inference with preset-based configuration.

- **Default preset:** `{active.key}` — {active.label}
- **Backend:** `{active.backend}`
- {presets_note}

Part of the [Build Small Hackathon](https://huggingface.co/build-small-hackathon).
"""
        )

        model_key = gr.State(_app_config.active_model)

        if _app_config.allow_model_switch and len(_app_config.models) > 1:
            model_dropdown = gr.Dropdown(
                choices=_app_config.model_choices(),
                value=_app_config.active_model,
                label="Model preset",
                info="Switch presets for local testing. Each preset loads on first use.",
            )
            status = gr.Markdown(model_status(_app_config.active_model))

            model_dropdown.change(
                fn=model_status,
                inputs=model_dropdown,
                outputs=status,
            ).then(
                fn=lambda key: key,
                inputs=model_dropdown,
                outputs=model_key,
            )

            gr.ChatInterface(
                fn=chat,
                additional_inputs=[model_dropdown],
                examples=[
                    ["Hello! What can you help me with?", _app_config.active_model],
                    ["Explain llama.cpp in one sentence.", _app_config.active_model],
                ],
            )
        else:
            status = gr.Markdown(model_status(_app_config.active_model))
            gr.ChatInterface(
                fn=lambda message, history: chat(message, history, _app_config.active_model),
                examples=["Hello! What can you help me with?", "Explain llama.cpp in one sentence."],
            )
            demo.load(lambda: warmup(_app_config.active_model), outputs=status)

    return demo


demo = build_demo()


def main() -> None:
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
