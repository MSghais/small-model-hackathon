import gradio as gr

from agent.runner import AgentRunner
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key, model_status
from inference.factory import get_backend


def generate_lesson_slides(
    topic: str,
    grade: str,
    slide_count: int,
) -> tuple[str, str | None, str, str]:
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return load_error, None, "", load_error

    if not topic.strip():
        message = "Please enter a lesson topic."
        return message, None, "", message

    try:
        runner = AgentRunner()
        result = runner.run_education_pptx(
            topic=topic,
            grade=grade,
            slide_count=int(slide_count),
            model_key=model_key,
            backend=get_backend(model_key),
        )
    except Exception as exc:  # noqa: BLE001 — show agent errors in UI
        message = f"Agent error: {exc}"
        return message, None, "", message

    trace_summary = (
        f"Run `{result.trace.run_id}` · skill `{result.trace.skill}` · "
        f"model `{result.trace.model}`\n\n"
        f"Trace saved: `{result.trace_path}`"
    )
    return result.markdown_preview, result.pptx_path, trace_summary, result.trace.to_json()


def build_education_pptx_tab() -> None:
    model_key = get_active_model_key()

    gr.Markdown(
        """
### Lesson slide builder

Enter a topic and grade level. A **local small model** drafts the outline;
the agent then builds a downloadable PowerPoint — no cloud LLM API.
"""
    )
    gr.Markdown(model_status(model_key))

    with gr.Row():
        topic = gr.Textbox(
            label="Lesson topic",
            placeholder="e.g. Photosynthesis, Fractions, The water cycle",
        )
        grade = gr.Dropdown(
            label="Grade level",
            choices=["K", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "Adult"],
            value="6",
        )
        slide_count = gr.Slider(
            minimum=3,
            maximum=8,
            step=1,
            value=5,
            label="Content slides",
        )

    generate_btn = gr.Button("Generate lesson slides", variant="primary")

    outline_preview = gr.Markdown(label="Outline preview")
    pptx_file = gr.File(label="Download PowerPoint", interactive=False)
    trace_box = gr.Textbox(
        label="Agent trace (JSON)",
        lines=12,
        max_lines=20,
        interactive=False,
    )

    with gr.Accordion("Trace summary", open=False):
        trace_summary = gr.Markdown()

    generate_btn.click(
        fn=generate_lesson_slides,
        inputs=[topic, grade, slide_count],
        outputs=[outline_preview, pptx_file, trace_summary, trace_box],
    )
