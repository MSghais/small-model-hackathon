from html import escape
from pathlib import Path

import gradio as gr

from agent.progress import SlideGenerationProgress
from agent.runner import AgentResult, AgentRunner
from agent.tools.pptx import get_outputs_dir
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key
from gradio_space.research_helpers import (
    list_session_choices,
    merge_lesson_urls,
    refresh_doc_choices,
    refresh_sessions,
    resolve_doc_ids,
    resolve_session,
    resolve_topic,
)
from gradio_space.ui.components import build_advanced_panel, DOC_CHOICE_LIST_CLASSES, WorkspaceWidgets
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
    log_html = (
        f'<div class="slide-gen-log"><div class="slide-gen-log-banner error">'
        f"{message}</div></div>"
    )
    return (
        message,
        _error_html(message),
        [],
        None,
        None,
        None,
        log_html,
        message,
        message,
        message,
    )


def _running_preview_html(step_label: str = "Generating slides…") -> str:
    safe = (
        step_label.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        '<div class="lesson-running-preview">'
        '<div class="lesson-running-spinner" aria-hidden="true"></div>'
        f"<p><strong>{safe}</strong></p>"
        "<p class=\"lesson-running-hint\">Local models can take 30–90s on CPU. "
        "Steps update live below.</p>"
        "</div>"
    )


def _interim_outputs(
    slide_progress: SlideGenerationProgress,
    *,
    status: str = "_Generating lesson slides…_",
    step_label: str = "Generating slides…",
) -> tuple:
    log_html = slide_progress.format_log_html(running=True)
    return (
        "",
        _running_preview_html(step_label),
        [],
        None,
        None,
        None,
        log_html,
        "",
        "",
        status,
    )


def _format_processing_log(
    progress: SlideGenerationProgress,
    *,
    trace_summary: str = "",
    source_status: str = "",
) -> str:
    footer_parts: list[str] = []
    if source_status:
        footer_parts.append(
            f"<p><strong>Sources:</strong> {escape(strip_md_inline(source_status))}</p>"
        )
    if trace_summary:
        footer_parts.append(
            f'<pre class="slide-gen-log-trace">{escape(trace_summary)}</pre>'
        )
    footer_html = "".join(footer_parts)
    return progress.format_log_html(running=False, footer_html=footer_html)


def strip_md_inline(text: str) -> str:
    return str(text).replace("**", "").replace("_", "").replace("`", "")


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
    workspace_topic: str,
    workspace_session: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, object, object]:
    topic = resolve_topic(topic, workspace_topic)
    session_id = resolve_session(session_id, workspace_session)
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
    workspace_topic: str = "",
    workspace_session: str = "",
    workspace_doc_ids: list[str] | None = None,
    progress: gr.Progress = gr.Progress(),
    *,
    skip_preview_images: bool = False,
):
    topic = resolve_topic(topic, workspace_topic)
    session_id = resolve_session(session_id, workspace_session)
    doc_ids = resolve_doc_ids(doc_ids, workspace_doc_ids)
    slide_progress = SlideGenerationProgress(
        on_update=lambda fraction, desc: progress(fraction, desc=desc),
    )
    slide_progress.begin("load_model", "Load language model")

    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        yield _empty_outputs(load_error)
        return

    if not topic.strip():
        message = "Please enter a lesson topic."
        yield _empty_outputs(message)
        return

    source_mode = _source_mode_value(source_mode_label)
    search_workflow = _search_workflow_value(search_workflow_label)
    merged_urls = merge_lesson_urls(urls_text, selected_urls)
    files = [Path(p) for p in (upload_files or [])]

    current_step = "Load language model"
    yield _interim_outputs(slide_progress, step_label=current_step)

    result = None
    try:
        runner = AgentRunner()
        for item in runner.iter_education_pptx(
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
            progress=slide_progress,
            skip_preview_images=skip_preview_images,
        ):
            if isinstance(item, AgentResult):
                result = item
                break
            current_step = item.steps[-1].label if item.steps else current_step
            yield _interim_outputs(slide_progress, step_label=current_step)
    except Exception as exc:  # noqa: BLE001
        message = f"Agent error: {exc}"
        slide_progress.finish()
        yield (
            message,
            _error_html(message),
            [],
            None,
            None,
            None,
            slide_progress.format_log_html(running=False),
            message,
            message,
            message,
        )
        return

    if result is None:
        message = "Agent error: generation finished without a result."
        yield _empty_outputs(message)
        return

    progress(1.0, desc="Done")
    gallery = [str(Path(p).resolve()) for p in result.preview_images]
    trace_summary = (
        f"Run `{result.trace.run_id}` · skill `{result.trace.skill}` · "
        f"model `{result.trace.model}`\n\n"
        f"Trace saved: `{result.trace_path}`"
    )
    source_status = result.source_summary or "_No external sources used (model only)._"
    processing_log = _format_processing_log(
        slide_progress,
        trace_summary=trace_summary,
        source_status=source_status,
    )
    yield (
        result.markdown_preview,
        result.html_preview,
        gallery,
        str(Path(result.pptx_path).resolve()),
        str(Path(result.docx_path).resolve()),
        str(Path(result.html_export_path).resolve()),
        processing_log,
        trace_summary,
        result.trace.to_json(),
        source_status,
    )


def build_education_pptx_tab(workspace: WorkspaceWidgets) -> None:
    gr.Markdown("### Create lesson slides", elem_classes=["lesson-tab-heading"])
    gr.HTML(
        '<p class="tab-subtitle">Enter your topic below, adjust grade and length if needed, then generate.</p>'
    )

    with gr.Column(elem_classes=["lesson-form-primary"]):
        topic = gr.Textbox(
            label="What are you teaching?",
            placeholder="e.g. Photosynthesis, Fractions, The water cycle, AI agents…",
            lines=2,
            max_lines=3,
            elem_classes=["lesson-topic-input"],
        )

    with gr.Row(elem_classes=["lesson-form-secondary"]):
        grade = gr.Dropdown(
            label="Grade",
            choices=["K", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "Adult"],
            value="6",
            scale=1,
            min_width=100,
        )
        slide_count = gr.Slider(
            minimum=3,
            maximum=8,
            step=1,
            value=5,
            label="Slides",
            scale=2,
        )

    with gr.Accordion("Research sources (optional)", open=False, elem_classes=["lesson-optional-accordion"]):
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
            elem_classes=DOC_CHOICE_LIST_CLASSES,
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
            elem_classes=DOC_CHOICE_LIST_CLASSES,
        )

    with gr.Row(elem_classes=["lesson-generate-row"]):
        generate_btn = gr.Button(
            "Generate lesson slides",
            variant="primary",
            elem_classes=["primary-cta"],
            scale=1,
        )

    source_status = gr.Markdown(value="_Ready to generate._", elem_classes=["lesson-status"])
    processing_log = gr.HTML(
        value=(
            '<div class="slide-gen-log slide-gen-log-idle">'
            "<p>Generation steps and timings appear here when you run.</p>"
            "</div>"
        ),
        elem_classes=["lesson-processing-log"],
    )

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
        inputs=[topic, session_dd, workspace.topic, workspace.session_dd],
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
            workspace.topic,
            workspace.session_dd,
            workspace.doc_dd,
        ],
        outputs=[
            outline_preview,
            slide_preview,
            slide_gallery,
            pptx_file,
            docx_file,
            html_file,
            processing_log,
            advanced.trace_summary,
            advanced.trace_box,
            source_status,
        ],
        show_progress="hidden",
    )

    def _sync_topic_from_workspace(ws_topic: str, local_topic: str) -> str:
        if not (local_topic or "").strip():
            return ws_topic
        return local_topic

    def _sync_session_from_workspace(ws_session: str, local_session: str) -> str:
        if not (local_session or "").strip():
            return ws_session
        return local_session

    workspace.topic.change(
        fn=_sync_topic_from_workspace,
        inputs=[workspace.topic, topic],
        outputs=[topic],
    )
    workspace.session_dd.change(
        fn=_sync_session_from_workspace,
        inputs=[workspace.session_dd, session_dd],
        outputs=[session_dd],
    ).then(
        fn=refresh_doc_choices,
        inputs=[session_dd, doc_dd],
        outputs=[doc_dd],
    )


def gradio_allowed_paths() -> list[str]:
    """Paths Gradio must be allowed to read for previews and downloads."""
    root = get_outputs_dir().resolve()
    cfg = get_config()
    rm_root = cfg.data_dir.resolve()
    rm_root.mkdir(parents=True, exist_ok=True)
    return [str(root), str(rm_root)]
