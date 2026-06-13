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
    parse_urls_text,
    rag_scope_hint,
    refresh_doc_choices,
    refresh_sessions,
    resolve_doc_ids,
    resolve_session,
    resolve_topic,
    run_research_question,
    trace_summary_markdown,
)
from gradio_space.ui.components import build_advanced_panel, DOC_CHOICE_LIST_CLASSES, WorkspaceWidgets
from inference.factory import get_backend

logger = logging.getLogger(__name__)


def _require_topic(topic: str | None) -> str | None:
    if not (topic or "").strip():
        return "Enter a research topic first — it names your session and guides web search."
    return None


def discover_sources(
    topic: str,
    session_id: str,
    workspace_topic: str = "",
    workspace_session: str = "",
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, object, str, str, str, str, object, object]:
    topic = resolve_topic(topic, workspace_topic)
    session_id = resolve_session(session_id, workspace_session)
    progress(0, desc="Searching web…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return (
            load_error,
            gr.update(choices=[], value=[], visible=False),
            session_id,
            load_error,
            load_error,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
            gr.update(visible=False),
        )

    topic_error = _require_topic(topic)
    if topic_error:
        return (
            topic_error,
            gr.update(choices=[], value=[], visible=False),
            session_id,
            topic_error,
            topic_error,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
            gr.update(visible=False),
        )

    try:
        runner = AgentRunner()
        discover = runner.run_researchmind_discover(
            topic=topic.strip(),
            auto_search=False,
            session_id=session_id or None,
            model_key=model_key,
            backend=get_backend(model_key),
        )
        choices = discover.suggested_urls
        if not choices:
            summary = (
                "No verified URLs found. Try a more specific topic, paste URLs manually, "
                "or use **Auto-ingest from web**."
            )
        else:
            summary = (
                f"Found **{len(choices)}** verified URL(s). Review the list, then click "
                "**Ingest selected sources**."
            )
        trace_json = load_trace_json(discover.trace_path)
        progress(1.0, desc="Done")
        return (
            summary,
            gr.update(choices=choices, value=choices, visible=bool(choices)),
            refresh_sessions(discover.session_id),
            trace_summary_markdown(discover.trace_path),
            trace_json,
            memory_summary(discover.session_id),
            refresh_doc_choices(discover.session_id, []),
            gr.update(visible=bool(choices)),
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Discover error: {exc}"
        return (
            msg,
            gr.update(choices=[], value=[], visible=False),
            session_id,
            msg,
            msg,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
            gr.update(visible=False),
        )


def auto_search_ingest(
    topic: str,
    session_id: str,
    workspace_topic: str = "",
    workspace_session: str = "",
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, object, str, str, str, str, object, object]:
    topic = resolve_topic(topic, workspace_topic)
    session_id = resolve_session(session_id, workspace_session)
    progress(0, desc="Auto search & ingest…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return (
            load_error,
            gr.update(choices=[], value=[], visible=False),
            session_id,
            load_error,
            load_error,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
            gr.update(visible=False),
        )

    topic_error = _require_topic(topic)
    if topic_error:
        return (
            topic_error,
            gr.update(choices=[], value=[], visible=False),
            session_id,
            topic_error,
            topic_error,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
            gr.update(visible=False),
        )

    try:
        runner = AgentRunner()
        result = runner.run_researchmind_ingest(
            topic=topic.strip(),
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
            gr.update(choices=[], value=[], visible=False),
            refresh_sessions(result.session_id),
            trace_summary_markdown(result.trace_path),
            trace_json,
            memory_summary(result.session_id),
            refresh_doc_choices(result.session_id, []),
            gr.update(visible=False),
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Auto ingest error: {exc}"
        return (
            msg,
            gr.update(choices=[], value=[], visible=False),
            session_id,
            msg,
            msg,
            memory_summary(session_id),
            refresh_doc_choices(session_id, []),
            gr.update(visible=False),
        )


def ingest_selected(
    topic: str | None,
    urls_text: str | None,
    selected_urls: list[str] | None,
    upload_files: list[str] | None,
    session_id: str | None,
    workspace_topic: str = "",
    workspace_session: str = "",
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str, object, object]:
    topic = resolve_topic(topic, workspace_topic) or None
    session_id = resolve_session(session_id or "", workspace_session) or None
    progress(0, desc="Ingesting sources…")
    sid = session_id or ""
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return (
            load_error,
            memory_summary(sid),
            load_error,
            load_error,
            refresh_sessions(sid),
            refresh_doc_choices(sid, []),
        )

    topic_error = _require_topic(topic)
    if topic_error:
        return (
            topic_error,
            memory_summary(sid),
            topic_error,
            topic_error,
            refresh_sessions(sid),
            refresh_doc_choices(sid, []),
        )

    direct_urls = parse_urls_text(urls_text or "")
    all_urls = list(dict.fromkeys([*direct_urls, *(selected_urls or [])]))
    files = [Path(p) for p in (upload_files or [])]

    if not all_urls and not files:
        msg = "Add URLs, select suggested sources, or upload a file — then ingest."
        return (
            msg,
            memory_summary(sid),
            msg,
            msg,
            refresh_sessions(sid),
            refresh_doc_choices(sid, []),
        )

    try:
        logger.info("Ingesting %d URL(s) and %d file(s)", len(all_urls), len(files))
        runner = AgentRunner()
        result = runner.run_researchmind_ingest(
            topic=(topic or "").strip(),
            urls=all_urls,
            files=files,
            auto_search=False,
            session_id=sid or None,
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
            memory_summary(sid),
            msg,
            msg,
            refresh_sessions(sid),
            refresh_doc_choices(sid, []),
        )


def ask_question(
    question: str,
    session_id: str,
    doc_ids: list[str] | None,
    chat_history: list[dict],
    workspace_session: str = "",
    workspace_doc_ids: list[str] | None = None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[list[dict], str, str, str, str]:
    session_id = resolve_session(session_id, workspace_session)
    doc_ids = resolve_doc_ids(doc_ids, workspace_doc_ids)
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


def build_research_mind_tab(workspace: WorkspaceWidgets) -> None:
    gr.Markdown("### ResearchMind", elem_classes=["form-tab-heading"])
    gr.HTML(
        '<p class="tab-subtitle">'
        "Start with a topic, add sources to your library, then ask questions with citations."
        "</p>"
    )

    with gr.Column(elem_classes=["form-primary"]):
        topic = gr.Textbox(
            label="What are you researching?",
            placeholder="e.g. AI agents, Photosynthesis, American Revolution…",
            lines=2,
            max_lines=3,
            elem_classes=["form-topic-input"],
        )

    with gr.Row(elem_classes=["form-secondary"]):
        session_dd = gr.Dropdown(
            label="Session",
            choices=list_session_choices(),
            value="",
            scale=4,
        )
        refresh_btn = gr.Button("↻", size="sm", scale=0, min_width=40)

    with gr.Row(elem_classes=["rm-workflow-columns"]):
        with gr.Column(scale=1, elem_classes=["rm-ingest-col"]):
            gr.HTML('<p class="form-section-label">Step 1 · Add sources</p>')

            with gr.Row(elem_classes=["rm-action-row"]):
                discover_btn = gr.Button("Discover on web", variant="secondary", size="sm")
                auto_btn = gr.Button("Auto-ingest from web", variant="secondary", size="sm")

            with gr.Accordion("Suggested URLs from web search", open=True, visible=False) as urls_acc:
                url_choices = gr.CheckboxGroup(
                    label="Select sources to ingest",
                    choices=[],
                    value=[],
                    elem_classes=DOC_CHOICE_LIST_CLASSES,
                )

            with gr.Accordion(
                "Paste URLs or upload files",
                open=False,
                elem_classes=["form-optional-accordion"],
            ):
                urls_text = gr.Textbox(
                    label="URLs (one per line)",
                    lines=3,
                    placeholder="https://en.wikipedia.org/wiki/...",
                )
                upload_files = gr.File(
                    label="Upload PDF or DOCX",
                    file_count="multiple",
                    file_types=[".pdf", ".docx"],
                )

            with gr.Row(elem_classes=["form-cta-row"]):
                ingest_btn = gr.Button(
                    "Ingest selected sources",
                    variant="primary",
                    elem_classes=["primary-cta"],
                )

            ingest_status = gr.Markdown(
                value="_Enter a topic, then discover or paste sources to ingest._",
                elem_classes=["form-status"],
            )

            with gr.Accordion("Indexed documents", open=False):
                memory_md = gr.Markdown(value=memory_summary(""))
                refresh_memory_btn = gr.Button("Refresh", size="sm")

            advanced = build_advanced_panel()

        with gr.Column(scale=1, elem_classes=["rm-ask-col"]):
            gr.HTML('<p class="form-section-label">Step 2 · Ask questions</p>')

            chatbot = gr.Chatbot(
                label="Answers",
                height=320,
                placeholder="Ask a question after ingesting sources — answers include citations.",
            )

            with gr.Column(elem_classes=["form-primary"]):
                question = gr.Textbox(
                    label="Your question",
                    placeholder="What do these sources say about AI agents?",
                    lines=2,
                    max_lines=4,
                    elem_classes=["form-ask-input"],
                )

            with gr.Accordion(
                "Limit to specific documents",
                open=False,
                elem_classes=["form-optional-accordion"],
            ):
                doc_dd = gr.CheckboxGroup(
                    label="Documents (empty = all in session)",
                    choices=[],
                    value=[],
                    elem_classes=DOC_CHOICE_LIST_CLASSES,
                )

            rag_hint = gr.Markdown(
                value=rag_scope_hint("", []),
                elem_classes=["form-status"],
            )

            with gr.Row(elem_classes=["form-cta-row"]):
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
        urls_acc,
    ]

    discover_btn.click(
        fn=discover_sources,
        inputs=[topic, session_dd, workspace.topic, workspace.session_dd],
        outputs=discover_outputs,
    )

    auto_btn.click(
        fn=auto_search_ingest,
        inputs=[topic, session_dd, workspace.topic, workspace.session_dd],
        outputs=discover_outputs,
    )

    ingest_btn.click(
        fn=ingest_selected,
        inputs=[
            topic,
            urls_text,
            url_choices,
            upload_files,
            session_dd,
            workspace.topic,
            workspace.session_dd,
        ],
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
        inputs=[
            question,
            session_dd,
            doc_dd,
            chatbot,
            workspace.session_dd,
            workspace.doc_dd,
        ],
        outputs=[chatbot, advanced.trace_box, advanced.trace_summary, rag_hint, question],
    )
    question.submit(
        fn=ask_question,
        inputs=[
            question,
            session_dd,
            doc_dd,
            chatbot,
            workspace.session_dd,
            workspace.doc_dd,
        ],
        outputs=[chatbot, advanced.trace_box, advanced.trace_summary, rag_hint, question],
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


def researchmind_allowed_paths() -> list[str]:
    from researchmind.config import get_config

    cfg = get_config()
    root = cfg.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return [str(root)]
