import os

import gradio as gr

from gradio_space.model_loading import preload_active_model
from gradio_space.tabs import (
    build_chat_tab,
    build_education_pptx_tab,
    build_echo_coach_tab,
    build_research_mind_tab,
)
from gradio_space.tabs.education_pptx import gradio_allowed_paths
from gradio_space.tabs.echo_coach import echo_coach_allowed_paths
from gradio_space.tabs.research_mind import researchmind_allowed_paths
from inference.config import get_app_config

_app_config = get_app_config()


def build_demo() -> gr.Blocks:
    active = _app_config.active
    presets_note = (
        f"Presets file: `{_app_config.presets_path}`"
        if _app_config.presets_path
        else "Using built-in presets (models.yaml not found)."
    )

    with gr.Blocks(title="Lesson Agent + ResearchMind — Build Small Hackathon") as demo:
        gr.Markdown(
            f"""
# Lesson Agent + ResearchMind + EchoCoach

Local skill-based agents — **lesson slides**, **research with MemRAG**, and **voice practice coaching** (offline).

- **Model:** `{active.key}` — {active.label}
- **Backend:** `{active.backend}`
- {presets_note}

Part of the [Build Small Hackathon](https://huggingface.co/build-small-hackathon).
"""
        )

        with gr.Tabs():
            with gr.Tab("Lesson slides"):
                build_education_pptx_tab()
            with gr.Tab("ResearchMind"):
                build_research_mind_tab()
            with gr.Tab("EchoCoach"):
                build_echo_coach_tab()
            with gr.Tab("Chat (debug)"):
                build_chat_tab()

    return demo


def main() -> None:
    preload_active_model()
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        allowed_paths=[
            *gradio_allowed_paths(),
            *researchmind_allowed_paths(),
            *echo_coach_allowed_paths(),
        ],
    )


if __name__ == "__main__":
    main()
