from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from agent.runner import AgentRunner
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key
from gradio_space.research_helpers import (
    format_citations_markdown,
    format_ingest_status,
    list_session_choices,
    load_trace_json,
    memory_summary,
    rag_scope_hint,
    refresh_doc_choices,
    refresh_sessions,
    run_research_question,
    trace_summary_markdown,
)
from gradio_space.ui.components import build_advanced_panel, tab_hero
from inference.factory import get_backend

logger = logging.getLogger(__name__)


def discover_sources(
    topic: str,
    session_id: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, object, str, str, str, str, object]:
    progress(0, desc="Searching web…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return (
            load_error,
            gr.update(choices=[], value=[]),
            session_id,
            load_error,
            load_error,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
        )

    if not topic.strip():
        msg = "Enter a topic to discover sources."
        return (
            msg,
            gr.update(choices=[], value=[]),
            session_id,
            msg,
            msg,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
        )

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
                "or use **Auto search & ingest**."
            )
        else:
            summary = (
                f"Found **{len(choices)} verified URL(s)**. Select sources and click "
                "**Ingest selected**."
            )
        trace_json = load_trace_json(discover.trace_path)
        progress(1.0, desc="Done")
        return (
            summary,
            gr.update(choices=choices, value=choices),
            discover.session_id,
            trace_summary_markdown(discover.trace_path),
            trace_json,
            memory_summary(discover.session_id),
            refresh_doc_choices(discover.session_id, []),
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Discover error: {exc}"
        return (
            msg,
            gr.update(choices=[], value=[]),
            session_id,
            msg,
            msg,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
        )


def auto_search_ingest(
    topic: str,
    session_id: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, object, str, str, str, str, object]:
    progress(0, desc="Auto search & ingest…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return (
            load_error,
            gr.update(choices=[], value=[]),
            session_id,
            load_error,
            load_error,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
        )

    if not topic.strip():
        msg = "Enter a topic for auto search."
        return (
            msg,
            gr.update(choices=[], value=[]),
            session_id,
            msg,
            msg,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
        )

    try:
        runner = AgentRunner()
        result = runner.run_researchmind_ingest(
            topic=topic,
            urls=[],
            files=[],
            auto_search=True,
            session_id=session_id or None,
            model_key=model_key,
            backend=get_backend(model_key),
        )
        trace_json = load_trace_json(result.trace_path)
        progress(1.0, desc="Done")
        return (
            format_ingest_status(result),
            gr.update(choices=[], value=[]),
            result.session_id,
            trace_summary_markdown(result.trace_path),
            trace_json,
            memory_summary(result.session_id),
            refresh_doc_choices(result.session_id, []),
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Auto ingest error: {exc}"
        return (
            msg,
            gr.update(choices=[], value=[]),
            session_id,
            msg,
            msg,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
        )


def ingest_selected(
    topic: str,
    urls_text: str,
    selected_urls: list[str],
    upload_files: list[str] | None,
    session_id: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str, object, object]:
    progress(0, desc="Ingesting sources…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return (
            load_error,
            memory_summary(session_id),
            load_error,
            load_error,
            refresh_sessions(session_id),
            refresh_doc_choices(session_id, []),
        )

    direct_urls = [ln.strip() for ln in urls_text.splitlines() if ln.strip()]
    all_urls = list(dict.fromkeys([*direct_urls, *(selected_urls or [])]))
    files = [Path(p) for p in (upload_files or [])]

    if not all_urls and not files:
        msg = "Provide URLs, select suggested sources, or upload a file."
        return (
            msg,
            memory_summary(session_id),
            msg,
            msg,
            refresh_sessions(session_id),
            refresh_doc_choices(session_id, []),
        )

    try:
        logger.info("Ingesting %d URL(s) and %d file(s)", len(all_urls), len(files))
        runner = AgentRunner()
        result = runner.run_researchmind_ingest(
            topic=topic or None,
            urls=all_urls,
            files=files,
            auto_search=False,
            session_id=session_id or None,
            model_key=model_key,
            backend=get_backend(model_key),
        )
        trace_json = load_trace_json(result.trace_path)
        progress(1.0, desc="Done")
        return (
            format_ingest_status(result),
            memory_summary(result.session_id),
            trace_json,
            trace_summary_markdown(result.trace_path),
            refresh_sessions(result.session_id),
            refresh_doc_choices(result.session_id, []),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest failed")
        msg = f"**Ingest error:** {exc}"
        return (
            msg,
            memory_summary(session_id),
            msg,
            msg,
            refresh_sessions(session_id),
            refresh_doc_choices(session_id, []),
        )


def ask_question(
    question: str,
    session_id: str,
    doc_ids: list[str] | None,
    chat_history: list[dict],
    progress: gr.Progress = gr.Progress(),
) -> tuple[list[dict], str, str, str, str]:
    if not question.strip():
        return chat_history or [], "Enter a question.", "", rag_scope_hint(session_id, doc_ids), question

    try:
        progress(0, desc="Searching corpus…")
        answer, trace_json, trace_summary = run_research_question(
            question,
            session_id=session_id,
            doc_ids=doc_ids,
        )
        citations = format_citations_markdown(trace_json)
        if citations:
            answer = f"{answer}\n{citations}"
        history = list(chat_history or [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        progress(1.0, desc="Done")
        return history, trace_json, trace_summary, rag_scope_hint(session_id, doc_ids), ""
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research chat failed")
        history = list(chat_history or [])
        history.append({"role": "user", "content": question})
        err = f"Chat error: {exc}"
        history.append({"role": "assistant", "content": err})
        return history, err, err, rag_scope_hint(session_id, doc_ids), question


def build_research_mind_tab() -> None:
    """ResearchMind UI — ingest library + corpus chat side by side."""
    tab_hero(
        "Index sources once, then ask questions offline with citations.",
        steps=["Ingest", "Ask"],
        active_step=0,
    )

    with gr.Row():
        session_dd = gr.Dropdown(
            label="Session",
            choices=list_session_choices(),
            value="",
            scale=4,
        )
        refresh_btn = gr.Button("↻", size="sm", scale=0, min_width=40)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-card"><h4>Build library</h4></div>')
            topic = gr.Textbox(
                label="Topic (optional)",
                placeholder="e.g. Photosynthesis, American Revolution",
            )
            urls_text = gr.Textbox(
                label="URLs (one per line, optional)",
                lines=3,
                placeholder="https://en.wikipedia.org/wiki/...",
            )
            upload_files = gr.File(
                label="Upload PDF or DOCX",
                file_count="multiple",
                file_types=[".pdf", ".docx"],
            )
            url_choices = gr.CheckboxGroup(label="Suggested URLs to ingest", choices=[])
            with gr.Row():
                discover_btn = gr.Button("Discover sources", variant="secondary")
                auto_btn = gr.Button("Auto search & ingest", variant="secondary")
            ingest_btn = gr.Button("Ingest selected", variant="primary", elem_classes=["primary-cta"])
            ingest_status = gr.Markdown()

            with gr.Accordion("Indexed documents", open=False):
                memory_md = gr.Markdown(value=memory_summary(""))
                refresh_memory_btn = gr.Button("Refresh", size="sm")

            advanced = build_advanced_panel()

        with gr.Column(scale=1):
            gr.HTML('<div class="panel-card"><h4>Ask your corpus</h4></div>')
            with gr.Accordion("Limit to documents", open=False):
                doc_dd = gr.CheckboxGroup(
                    label="Documents (empty = all in session)",
                    choices=[],
                    value=[],
                )
            rag_hint = gr.Markdown(value=rag_scope_hint("", []))
            chatbot = gr.Chatbot(label="Research chat", height=400)
            question = gr.Textbox(
                label="Question",
                placeholder="What do these sources say about AI agents?",
            )
            ask_btn = gr.Button("Ask", variant="primary", elem_classes=["primary-cta"])

    refresh_btn.click(fn=refresh_sessions, inputs=[session_dd], outputs=[session_dd])
    refresh_memory_btn.click(fn=memory_summary, inputs=[session_dd], outputs=[memory_md])
    session_dd.change(fn=memory_summary, inputs=[session_dd], outputs=[memory_md])
    session_dd.change(
        fn=refresh_doc_choices,
        inputs=[session_dd, doc_dd],
        outputs=[doc_dd],
    ).then(
        fn=rag_scope_hint,
        inputs=[session_dd, doc_dd],
        outputs=[rag_hint],
    )
    doc_dd.change(fn=rag_scope_hint, inputs=[session_dd, doc_dd], outputs=[rag_hint])

    discover_outputs = [
        ingest_status,
        url_choices,
        session_dd,
        advanced.trace_summary,
        advanced.trace_box,
        memory_md,
        doc_dd,
    ]

    discover_btn.click(
        fn=discover_sources,
        inputs=[topic, session_dd],
        outputs=discover_outputs,
    )

    auto_btn.click(
        fn=auto_search_ingest,
        inputs=[topic, session_dd],
        outputs=discover_outputs,
    )

    ingest_btn.click(
        fn=ingest_selected,
        inputs=[topic, urls_text, url_choices, upload_files, session_dd],
        outputs=[
            ingest_status,
            memory_md,
            advanced.trace_box,
            advanced.trace_summary,
            session_dd,
            doc_dd,
        ],
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question, session_dd, doc_dd, chatbot],
        outputs=[chatbot, advanced.trace_box, advanced.trace_summary, rag_hint, question],
    )
    question.submit(
        fn=ask_question,
        inputs=[question, session_dd, doc_dd, chatbot],
        outputs=[chatbot, advanced.trace_box, advanced.trace_summary, rag_hint, question],
    )


def researchmind_allowed_paths() -> list[str]:
    from researchmind.config import get_config

    cfg = get_config()
    root = cfg.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return [str(root)]
