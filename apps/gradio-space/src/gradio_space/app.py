import gradio as gr

from gradio_space.tabs import (
    build_chat_tab,
    build_education_pptx_tab,
    build_echo_coach_tab,
    build_research_mind_tab,
    build_teacher_voice_tab,
)
from gradio_space.ui.settings_panel import build_settings_panel


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Build Small — Lesson Agent") as demo:
        with gr.Row(elem_classes=["app-header"]):
            gr.HTML(
                """
<div class="brand-block">
  <h1>Build Small</h1>
  <p>Local lesson slides, research, voice coaching — offline on small models.
  <a href="/">Studio UI</a> ·
  <a href="https://huggingface.co/build-small-hackathon" target="_blank">Hackathon</a></p>
</div>
"""
            )
            settings_toggle = gr.Button("⚙ Settings", size="sm", variant="secondary")

        with gr.Accordion("Settings", open=False, elem_id="settings-panel") as settings_acc:
            build_settings_panel()

        settings_open = gr.State(False)

        def _toggle_settings(is_open: bool) -> tuple[bool, dict]:
            new_open = not is_open
            return new_open, gr.update(open=new_open)

        settings_toggle.click(
            fn=_toggle_settings,
            inputs=[settings_open],
            outputs=[settings_open, settings_acc],
        )

        with gr.Tabs():
            with gr.Tab("Lesson slides"):
                build_education_pptx_tab()
            with gr.Tab("ResearchMind"):
                build_research_mind_tab()
            with gr.Tab("EchoCoach"):
                build_echo_coach_tab()
            with gr.Tab("TeacherVoice"):
                build_teacher_voice_tab()
            with gr.Tab("Chat (debug)"):
                build_chat_tab()

    return demo


def main() -> None:
    from gradio_space.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
