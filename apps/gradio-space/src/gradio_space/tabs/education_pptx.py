import gradio as gr

from agent.runner import AgentRunner
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key, model_status
from inference.factory import get_backend

def generate_lesson_slides(
    topic: str,
    grade: str,
    slide_count: int,
) -> tuple[str, str, list[tuple[str, str]], str | None, str | None, str | None, str, str]:
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return load_error, "", [], None, None, None, load_error, load_error

    if not topic.strip():
        message = "Please enter a lesson topic."
        return message, "", [], None, None, None, message, message

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
        return message, "", [], None, None, None, message, message

    gallery = [(path, f"Slide {i}") for i, path in enumerate(result.preview_images)]
    trace_summary = (
        f"Run `{result.trace.run_id}` · skill `{result.trace.skill}` · "
        f"model `{result.trace.model}`\n\n"
        f"Trace saved: `{result.trace_path}`"
    )
    return (
        result.markdown_preview,
        result.html_preview,
        gallery,
        result.pptx_path,
        result.docx_path,
        result.html_export_path,
        trace_summary,
        result.trace.to_json(),
    )


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

    with gr.Tabs():
        with gr.Tab("Slide preview"):
            slide_preview = gr.HTML(label="Slides")
            slide_gallery = gr.Gallery(
                label="Slide thumbnails",
                columns=2,
                height="auto",
                object_fit="contain",
            )
        with gr.Tab("Outline"):
            outline_preview = gr.Markdown(label="Outline (markdown)")

    with gr.Row():
        pptx_file = gr.File(label="Download PowerPoint (.pptx)", interactive=False)
        docx_file = gr.File(
            label="Download Word / Google Docs (.docx)",
            interactive=False,
        )
        html_file = gr.File(
            label="Download HTML (import to Google Docs)",
            interactive=False,
        )

    gr.Markdown(
        """
**Open in Google Docs:** download the `.docx` file, upload it to [Google Drive](https://drive.google.com),
then choose **Open with → Google Docs**. You can also upload the `.html` file via
**Google Docs → File → Open → Upload**.
"""
    )

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
        outputs=[
            outline_preview,
            slide_preview,
            slide_gallery,
            pptx_file,
            docx_file,
            html_file,
            trace_summary,
            trace_box,
        ],
    )
