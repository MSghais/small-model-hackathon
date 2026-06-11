import os

import gradio as gr

from gradio_space.model_loading import warmup
from gradio_space.tabs import build_chat_tab, build_education_pptx_tab
from inference.config import get_app_config

_app_config = get_app_config()


def build_demo() -> gr.Blocks:
    active = _app_config.active
    presets_note = (
        f"Presets file: `{_app_config.presets_path}`"
        if _app_config.presets_path
        else "Using built-in presets (models.yaml not found)."
    )

    with gr.Blocks(title="Lesson Agent — Build Small Hackathon") as demo:
        gr.Markdown(
            f"""
# Lesson Agent

Local skill-based agent for teachers — **topic in, PowerPoint out**.

- **Model:** `{active.key}` — {active.label}
- **Backend:** `{active.backend}`
- {presets_note}

Part of the [Build Small Hackathon](https://huggingface.co/build-small-hackathon).
"""
        )

        with gr.Tabs():
            with gr.Tab("Lesson slides"):
                build_education_pptx_tab()
            with gr.Tab("Chat (debug)"):
                build_chat_tab()

        demo.load(lambda: warmup(_app_config.active_model))

    return demo


demo = build_demo()


def main() -> None:
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
