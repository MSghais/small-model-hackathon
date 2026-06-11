import gradio as gr

from gradio_space.model_loading import (
    chat as chat_fn,
    ensure_model_loaded,
    get_active_model_key,
    model_status,
    warmup,
)
from inference.config import get_app_config

_app_config = get_app_config()


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
    return chat_fn(message, history, model_key)


def build_chat_tab() -> None:
    gr.Markdown(
        """
### Model chat (debug)

Test the active local model with a simple chat interface.
"""
    )

    model_key = get_active_model_key()

    if _app_config.allow_model_switch and len(_app_config.models) > 1:
        model_dropdown = gr.Dropdown(
            choices=_app_config.model_choices(),
            value=_app_config.active_model,
            label="Model preset",
        )
        status = gr.Markdown(model_status(model_key))
        model_dropdown.change(fn=model_status, inputs=model_dropdown, outputs=status)
        gr.ChatInterface(
            fn=chat,
            additional_inputs=[model_dropdown],
            examples=[
                ["Hello! What can you help me with?", _app_config.active_model],
                ["Explain photosynthesis in one sentence.", _app_config.active_model],
            ],
        )
    else:
        status = gr.Markdown(model_status(model_key))
        gr.ChatInterface(
            fn=lambda message, history: chat(message, history, model_key),
            examples=[
                "Hello! What can you help me with?",
                "Explain photosynthesis in one sentence.",
            ],
        )
        gr.on(fn=lambda: warmup(model_key), outputs=status)
