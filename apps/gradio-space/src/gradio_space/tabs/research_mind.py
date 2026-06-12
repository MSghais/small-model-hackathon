from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from agent.runner import AgentRunner
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key, model_status
from gradio_space.research_helpers import (
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
from inference.factory import get_backend
from researchmind.config import get_config
from researchmind.ingest import IngestPipeline

logger = logging.getLogger(__name__)

INGEST_MODES = [
    ("Suggest URLs (confirm)", "suggest"),
    ("Auto search & ingest", "auto"),
]


def discover_sources(
    topic: str,
    ingest_mode: str,
    session_id: str,
) -> tuple[str, gr.Update, str, str, str, str, object]:
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

    auto_search = ingest_mode == "auto"
    try:
        runner = AgentRunner()
        if auto_search:
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
            return (
                format_ingest_status(result),
                gr.update(choices=[], value=[]),
                result.session_id,
                trace_summary_markdown(result.trace_path),
                trace_json,
                memory_summary(result.session_id),
                refresh_doc_choices(result.session_id, []),
            )

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
                f"Found **{len(choices)} verified URL(s)** via web search "
                f"(Google + fallbacks). Select sources and click **Ingest selected**."
            )
        trace_json = load_trace_json(discover.trace_path)
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


def ingest_selected(
    topic: str,
    urls_text: str,
    selected_urls: list[str],
    upload_files: list[str] | None,
    session_id: str,
) -> tuple[str, str, str, str, object, object]:
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
) -> tuple[list[dict], str, str, str]:
    if not question.strip():
        return chat_history or [], "Enter a question.", "", rag_scope_hint(session_id, doc_ids)

    try:
        answer, trace_json, trace_summary = run_research_question(
            question,
            session_id=session_id,
            doc_ids=doc_ids,
        )
        history = list(chat_history or [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, trace_json, trace_summary, rag_scope_hint(session_id, doc_ids)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research chat failed")
        history = list(chat_history or [])
        history.append({"role": "user", "content": question})
        err = f"Chat error: {exc}"
        history.append({"role": "assistant", "content": err})
        return history, err, err, rag_scope_hint(session_id, doc_ids)


def build_research_mind_tab() -> None:
    """ResearchMind UI — ingest, memory, trace, and corpus chat."""
    model_key = get_active_model_key()
    cfg = get_config()

    gr.Markdown(
        """
### ResearchMind

Scrape sources once, index into **MemRAG** (local SQLite + embeddings), then ask questions **offline** with citations.
"""
    )
    gr.Markdown(model_status(model_key))
    gr.Markdown(f"Memory store: `{cfg.data_dir.resolve()}`")

    with gr.Row():
        session_dd = gr.Dropdown(
            label="Session",
            choices=list_session_choices(),
            value="",
            interactive=True,
        )
        refresh_btn = gr.Button("Refresh sessions", size="sm")

    with gr.Tabs():
        with gr.Tab("Ingest"):
            gr.Markdown(
                """
- **Suggest mode:** Google web search → verified URLs → you confirm → ingest
- **Auto search:** same search, ingests top verified URLs immediately
- **Direct:** paste URLs or upload PDF/DOCX
"""
            )
            with gr.Row():
                topic = gr.Textbox(
                    label="Topic (optional)",
                    placeholder="e.g. Photosynthesis, American Revolution",
                )
                ingest_mode = gr.Dropdown(
                    label="Ingest mode",
                    choices=[m[0] for m in INGEST_MODES],
                    value=INGEST_MODES[0][0],
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

            discover_btn = gr.Button("Discover sources", variant="secondary")
            url_choices = gr.CheckboxGroup(label="Suggested URLs to ingest", choices=[])
            ingest_btn = gr.Button("Ingest selected", variant="primary")
            ingest_status = gr.Markdown()

        with gr.Tab("Memory"):
            gr.Markdown("Indexed documents and chunk counts for the selected session.")
            memory_md = gr.Markdown(value=memory_summary(""))
            refresh_memory_btn = gr.Button("Refresh memory view", size="sm")

        with gr.Tab("Trace"):
            trace_summary = gr.Markdown()
            trace_box = gr.Textbox(label="Trace JSON", lines=14, interactive=False)

    gr.Markdown("---")
    gr.Markdown("### Chat with your corpus")
    gr.Markdown(
        "Ask questions about ingested sources. Limit search to specific documents below, "
        "or leave all checked to search the whole session."
    )
    rag_hint = gr.Markdown(value=rag_scope_hint("", []))
    doc_dd = gr.CheckboxGroup(
        label="Documents in session",
        choices=[],
        value=[],
    )
    chatbot = gr.Chatbot(label="Research chat", height=360)
    question = gr.Textbox(
        label="Question",
        placeholder="What do these sources say about AI agents?",
    )
    ask_btn = gr.Button("Ask", variant="primary")

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

    discover_btn.click(
        fn=lambda topic, mode, sid: discover_sources(
            topic,
            "auto" if mode == INGEST_MODES[1][0] else "suggest",
            sid,
        ),
        inputs=[topic, ingest_mode, session_dd],
        outputs=[
            ingest_status,
            url_choices,
            session_dd,
            trace_summary,
            trace_box,
            memory_md,
            doc_dd,
        ],
    )

    ingest_btn.click(
        fn=ingest_selected,
        inputs=[topic, urls_text, url_choices, upload_files, session_dd],
        outputs=[ingest_status, memory_md, trace_box, trace_summary, session_dd, doc_dd],
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question, session_dd, doc_dd, chatbot],
        outputs=[chatbot, trace_box, trace_summary, rag_hint],
    )
    question.submit(
        fn=ask_question,
        inputs=[question, session_dd, doc_dd, chatbot],
        outputs=[chatbot, trace_box, trace_summary, rag_hint],
    )


def researchmind_allowed_paths() -> list[str]:
    cfg = get_config()
    root = cfg.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return [str(root)]
