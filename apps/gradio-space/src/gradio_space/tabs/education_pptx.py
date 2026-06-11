from pathlib import Path

import gradio as gr

from agent.runner import AgentRunner
from agent.tools.pptx import get_outputs_dir
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key, model_status
from inference.factory import get_backend

def _error_html(message: str) -> str:
    safe = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<div style="padding:12px;border:1px solid #c44;border-radius:8px;'
        f'background:#fff5f5;color:#8a1f1f;">{safe}</div>'
    )


def generate_lesson_slides(
    topic: str,
    grade: str,
    slide_count: int,
) -> tuple[str, str, list[str], str | None, str | None, str | None, str, str]:
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return load_error, _error_html(load_error), [], None, None, None, load_error, load_error

    if not topic.strip():
        message = "Please enter a lesson topic."
        return message, _error_html(message), [], None, None, None, message, message

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
        return message, _error_html(message), [], None, None, None, message, message

    gallery = [str(Path(p).resolve()) for p in result.preview_images]
    trace_summary = (
        f"Run `{result.trace.run_id}` · skill `{result.trace.skill}` · "
        f"model `{result.trace.model}`\n\n"
        f"Trace saved: `{result.trace_path}`"
    )
    return (
        result.markdown_preview,
        result.html_preview,
        gallery,
        str(Path(result.pptx_path).resolve()),
        str(Path(result.docx_path).resolve()),
        str(Path(result.html_export_path).resolve()),
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
                height=420,
                object_fit="contain",
                preview=True,
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


def gradio_allowed_paths() -> list[str]:
    """Paths Gradio must be allowed to read for previews and downloads."""
    root = get_outputs_dir().resolve()
    return [str(root)]
