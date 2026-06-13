from pathlib import Path

import gradio as gr

from agent.runner import AgentRunner
from agent.tools.pptx import get_outputs_dir
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key
from gradio_space.research_helpers import (
    list_session_choices,
    merge_lesson_urls,
    refresh_doc_choices,
    refresh_sessions,
)
from gradio_space.ui.components import build_advanced_panel, tab_hero
from inference.factory import get_backend
from researchmind.config import get_config

SOURCE_MODES = [
    ("None (model only)", "none"),
    ("Web search", "web"),
    ("RAG (indexed sources)", "rag"),
]

SEARCH_WORKFLOWS = [
    ("Two-step (discover & confirm)", "two_step"),
    ("Auto search & ingest", "auto"),
]

_SOURCE_LABEL_TO_VALUE = {label: value for label, value in SOURCE_MODES}
_WORKFLOW_LABEL_TO_VALUE = {label: value for label, value in SEARCH_WORKFLOWS}


def _source_mode_value(label: str) -> str:
    return _SOURCE_LABEL_TO_VALUE.get(label, "none")


def _search_workflow_value(label: str) -> str:
    return _WORKFLOW_LABEL_TO_VALUE.get(label, "two_step")


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


def _empty_outputs(message: str) -> tuple:
    return (
        message,
        _error_html(message),
        [],
        None,
        None,
        None,
        message,
        message,
        message,
    )


def update_source_visibility(source_mode_label: str, search_workflow_label: str):
    mode = _source_mode_value(source_mode_label)
    workflow = _search_workflow_value(search_workflow_label)
    is_web = mode == "web"
    is_rag = mode == "rag"
    is_sources = is_web or is_rag
    is_two_step = is_web and workflow == "two_step"
    is_auto = is_web and workflow == "auto"
    return (
        gr.update(visible=is_web),
        gr.update(visible=is_two_step),
        gr.update(visible=is_two_step),
        gr.update(visible=is_sources),
        gr.update(visible=is_sources),
        gr.update(visible=is_rag),
        gr.update(visible=is_rag),
        gr.update(visible=is_rag),
        gr.update(visible=is_rag),
        gr.update(
            value="Search web & generate" if is_auto else "Generate lesson slides",
        ),
    )


def discover_lesson_sources(
    topic: str,
    session_id: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, object, object]:
    progress(0, desc="Discovering sources…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return load_error, gr.update(choices=[], value=[]), refresh_sessions(session_id)

    if not topic.strip():
        msg = "Enter a lesson topic to discover sources."
        return msg, gr.update(choices=[], value=[]), refresh_sessions(session_id)

    try:
        runner = AgentRunner()
        discover = runner.run_researchmind_discover(
            topic=topic,
            auto_search=False,
            session_id=session_id or None,
            model_key=model_key,
            backend=get_backend(model_key),
        )
        choices = discover.suggested_urls
        if not choices:
            summary = (
                "No verified URLs found. Try a more specific topic, paste URLs manually, "
                "or switch to **Auto search & ingest**."
            )
        else:
            summary = (
                f"Found **{len(choices)}** verified URL(s). Select sources, then click "
                "**Generate lesson slides**."
            )
        progress(1.0, desc="Done")
        return (
            summary,
            gr.update(choices=choices, value=choices),
            refresh_sessions(discover.session_id),
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Discover error: {exc}"
        return msg, gr.update(choices=[], value=[]), refresh_sessions(session_id)


def generate_lesson_slides(
    topic: str,
    grade: str,
    slide_count: int,
    source_mode_label: str,
    search_workflow_label: str,
    urls_text: str,
    selected_urls: list[str],
    upload_files: list[str] | None,
    session_id: str,
    doc_ids: list[str] | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, list[str], str | None, str | None, str | None, str, str, str]:
    progress(0, desc="Loading model…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return _empty_outputs(load_error)

    if not topic.strip():
        message = "Please enter a lesson topic."
        return _empty_outputs(message)

    source_mode = _source_mode_value(source_mode_label)
    search_workflow = _search_workflow_value(search_workflow_label)
    merged_urls = merge_lesson_urls(urls_text, selected_urls)
    files = [Path(p) for p in (upload_files or [])]

    try:
        progress(0.1, desc="Generating lesson slides…")
        runner = AgentRunner()
        result = runner.run_education_pptx(
            topic=topic,
            grade=grade,
            slide_count=int(slide_count),
            model_key=model_key,
            backend=get_backend(model_key),
            source_mode=source_mode,  # type: ignore[arg-type]
            search_workflow=search_workflow,  # type: ignore[arg-type]
            urls=merged_urls,
            files=files,
            session_id=session_id or None,
            doc_ids=doc_ids or [],
        )
    except Exception as exc:  # noqa: BLE001
        message = f"Agent error: {exc}"
        return _empty_outputs(message)

    progress(1.0, desc="Done")
    gallery = [str(Path(p).resolve()) for p in result.preview_images]
    trace_summary = (
        f"Run `{result.trace.run_id}` · skill `{result.trace.skill}` · "
        f"model `{result.trace.model}`\n\n"
        f"Trace saved: `{result.trace_path}`"
    )
    source_status = result.source_summary or "_No external sources used (model only)._"
    return (
        result.markdown_preview,
        result.html_preview,
        gallery,
        str(Path(result.pptx_path).resolve()),
        str(Path(result.docx_path).resolve()),
        str(Path(result.html_export_path).resolve()),
        trace_summary,
        result.trace.to_json(),
        source_status,
    )


def build_education_pptx_tab() -> None:
    tab_hero(
        "Draft lesson slides locally — add web or RAG sources optionally.",
        steps=["Lesson details", "Sources", "Generate", "Preview"],
        active_step=0,
    )

    with gr.Row():
        topic = gr.Textbox(
            label="Lesson topic",
            placeholder="e.g. Photosynthesis, Fractions, The water cycle",
            scale=3,
        )
        grade = gr.Dropdown(
            label="Grade level",
            choices=["K", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "Adult"],
            value="6",
            scale=1,
        )
        slide_count = gr.Slider(
            minimum=3,
            maximum=8,
            step=1,
            value=5,
            label="Content slides",
            scale=1,
        )

    with gr.Accordion("Add research sources (optional)", open=False):
        source_mode = gr.Radio(
            label="Source mode",
            choices=[m[0] for m in SOURCE_MODES],
            value=SOURCE_MODES[0][0],
        )
        search_workflow = gr.Radio(
            label="Web search workflow",
            choices=[m[0] for m in SEARCH_WORKFLOWS],
            value=SEARCH_WORKFLOWS[0][0],
            visible=False,
        )
        discover_btn = gr.Button("Discover sources", variant="secondary", visible=False)
        with gr.Row():
            session_dd = gr.Dropdown(
                label="ResearchMind session",
                choices=list_session_choices(),
                value="",
                visible=False,
            )
            refresh_sess_btn = gr.Button("↻", size="sm", visible=False, min_width=40)
        url_choices = gr.CheckboxGroup(
            label="Suggested URLs to use",
            choices=[],
            visible=False,
        )
        urls_text = gr.Textbox(
            label="URLs (one per line, optional)",
            lines=3,
            placeholder="https://en.wikipedia.org/wiki/...",
            visible=False,
        )
        upload_files = gr.File(
            label="Upload PDF or DOCX",
            file_count="multiple",
            file_types=[".pdf", ".docx"],
            visible=False,
        )
        doc_dd = gr.CheckboxGroup(
            label="Documents in session (RAG scope)",
            choices=[],
            value=[],
            visible=False,
        )

    generate_btn = gr.Button("Generate lesson slides", variant="primary", elem_classes=["primary-cta"])
    source_status = gr.Markdown(value="_No sources gathered yet._")

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

    with gr.Accordion("Export help — open in Google Docs", open=False):
        gr.Markdown(
            """
Download the `.docx` file, upload it to [Google Drive](https://drive.google.com),
then choose **Open with → Google Docs**. You can also upload the `.html` file via
**Google Docs → File → Open → Upload**.
"""
        )

    advanced = build_advanced_panel()

    source_controls = [
        search_workflow,
        discover_btn,
        url_choices,
        urls_text,
        upload_files,
        session_dd,
        refresh_sess_btn,
        doc_dd,
        generate_btn,
    ]

    def _refresh_visibility(mode_label: str, workflow_label: str):
        return update_source_visibility(mode_label, workflow_label)

    source_mode.change(
        fn=_refresh_visibility,
        inputs=[source_mode, search_workflow],
        outputs=source_controls,
    )
    search_workflow.change(
        fn=_refresh_visibility,
        inputs=[source_mode, search_workflow],
        outputs=source_controls,
    )

    refresh_sess_btn.click(fn=refresh_sessions, inputs=[session_dd], outputs=[session_dd])
    session_dd.change(
        fn=refresh_doc_choices,
        inputs=[session_dd, doc_dd],
        outputs=[doc_dd],
    )

    discover_btn.click(
        fn=discover_lesson_sources,
        inputs=[topic, session_dd],
        outputs=[source_status, url_choices, session_dd],
    )

    generate_btn.click(
        fn=generate_lesson_slides,
        inputs=[
            topic,
            grade,
            slide_count,
            source_mode,
            search_workflow,
            urls_text,
            url_choices,
            upload_files,
            session_dd,
            doc_dd,
        ],
        outputs=[
            outline_preview,
            slide_preview,
            slide_gallery,
            pptx_file,
            docx_file,
            html_file,
            advanced.trace_summary,
            advanced.trace_box,
            source_status,
        ],
    )


def gradio_allowed_paths() -> list[str]:
    """Paths Gradio must be allowed to read for previews and downloads."""
    root = get_outputs_dir().resolve()
    cfg = get_config()
    rm_root = cfg.data_dir.resolve()
    rm_root.mkdir(parents=True, exist_ok=True)
    return [str(root), str(rm_root)]
