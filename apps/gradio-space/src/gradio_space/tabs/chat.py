import gradio as gr

from gradio_space.model_loading import chat, model_status
from inference.config import get_app_config

_app_config = get_app_config()


def build_chat_tab() -> None:
    gr.Markdown(
        """
### Model chat (debug)

Test the active local model with a simple chat interface.
"""
    )

    model_key = _app_config.active_model

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
