from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from agent.runner import AgentRunner
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key, model_status
from inference.factory import get_backend
from researchmind.config import get_config
from researchmind.ingest import IngestPipeline

INGEST_MODES = [
    ("Suggest URLs (confirm)", "suggest"),
    ("Auto search & ingest", "auto"),
]


def _error_md(message: str) -> str:
    safe = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div style="padding:12px;border:1px solid #c44;border-radius:8px;'
        f'background:#fff5f5;color:#8a1f1f;">{safe}</div>'
    )


def list_session_choices() -> list[tuple[str, str]]:
    store = IngestPipeline().store
    sessions = store.list_sessions()
    choices: list[tuple[str, str]] = [("New session", "")]
    for s in sessions:
        label = f"{s.topic or 'Untitled'} ({s.id})"
        choices.append((label, s.id))
    return choices


def refresh_sessions(current: str) -> gr.Dropdown:
    choices = list_session_choices()
    values = [c[1] for c in choices]
    value = current if current in values else ""
    return gr.Dropdown(choices=choices, value=value)


def discover_sources(
    topic: str,
    ingest_mode: str,
    session_id: str,
) -> tuple[str, gr.Update, str, str, str]:
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return load_error, gr.update(choices=[], value=[]), "", load_error, load_error

    if not topic.strip():
        msg = "Enter a topic to discover sources."
        return msg, gr.update(choices=[], value=[]), session_id, msg, msg

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
            return (
                result.message,
                gr.update(choices=[], value=[]),
                result.session_id,
                f"Auto-ingest complete for session `{result.session_id}`",
                result.trace_path,
            )

        discover = runner.run_researchmind_discover(
            topic=topic,
            auto_search=False,
            session_id=session_id or None,
            model_key=model_key,
            backend=get_backend(model_key),
        )
        choices = discover.suggested_urls
        summary = (
            f"Suggested {len(choices)} URL(s). Select sources and click **Ingest selected**."
        )
        return (
            summary,
            gr.update(choices=choices, value=choices),
            discover.session_id,
            summary,
            discover.trace_path,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Discover error: {exc}"
        return msg, gr.update(choices=[], value=[]), session_id, msg, msg


def ingest_selected(
    topic: str,
    urls_text: str,
    selected_urls: list[str],
    upload_files: list[str] | None,
    session_id: str,
) -> tuple[str, str, str, gr.Dropdown]:
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        dd = refresh_sessions(session_id)
        return load_error, load_error, load_error, dd

    direct_urls = [ln.strip() for ln in urls_text.splitlines() if ln.strip()]
    all_urls = list(dict.fromkeys([*direct_urls, *(selected_urls or [])]))
    files = [Path(p) for p in (upload_files or [])]

    if not all_urls and not files:
        msg = "Provide URLs, select suggested sources, or upload a file."
        return msg, msg, msg, refresh_sessions(session_id)

    try:
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
        docs = IngestPipeline().store.list_documents(session_id=result.session_id)
        sources_table = "\n".join(
            f"- **{d.title}** (`{d.source_type}`) — {d.uri}" for d in docs
        ) or "_No documents yet._"
        return result.message, sources_table, result.trace_path, refresh_sessions(result.session_id)
    except Exception as exc:  # noqa: BLE001
        msg = f"Ingest error: {exc}"
        return msg, msg, msg, refresh_sessions(session_id)


def ask_question(
    question: str,
    session_id: str,
    chat_history: list[dict],
) -> tuple[list[dict], str, str]:
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        history = list(chat_history or [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": load_error})
        return history, load_error, load_error

    if not question.strip():
        return chat_history or [], "Enter a question.", ""

    if not session_id:
        store = IngestPipeline().store
        session_id = store.create_session().id

    try:
        runner = AgentRunner()
        result = runner.run_researchmind_chat(
            question=question,
            session_id=session_id,
            model_key=model_key,
            backend=get_backend(model_key),
        )
        history = list(chat_history or [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.answer})
        trace_json = json.dumps(
            {
                "trace_path": result.trace_path,
                "citations": [c.model_dump() for c in result.citations],
            },
            indent=2,
        )
        return history, trace_json, result.trace_path
    except Exception as exc:  # noqa: BLE001
        history = list(chat_history or [])
        history.append({"role": "user", "content": question})
        err = f"Chat error: {exc}"
        history.append({"role": "assistant", "content": err})
        return history, err, err


def build_research_mind_tab() -> None:
    model_key = get_active_model_key()
    cfg = get_config()

    gr.Markdown(
        """
### ResearchMind

Scrape sources once, index into **MemRAG** (local SQLite + embeddings), then ask questions **offline** with citations.

- **Suggest mode:** local model proposes URLs → you confirm → ingest
- **Auto search:** DuckDuckGo top URLs ingested immediately (network at ingest only)
- **Direct:** paste URLs or upload PDF/DOCX
"""
    )
    gr.Markdown(model_status(model_key))
    gr.Markdown(f"Memory store: `{cfg.data_dir.resolve()}`")

    session_dd = gr.Dropdown(
        label="Session",
        choices=list_session_choices(),
        value="",
        interactive=True,
    )
    refresh_btn = gr.Button("Refresh sessions", size="sm")

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
    sources_md = gr.Markdown(label="Ingested sources")

    gr.Markdown("---")
    gr.Markdown("#### Ask (offline after ingest)")

    chatbot = gr.Chatbot(label="Research chat", height=360)
    question = gr.Textbox(label="Question", placeholder="What does your corpus say about…?")
    ask_btn = gr.Button("Ask", variant="primary")

    with gr.Accordion("Trace & debug", open=False):
        trace_summary = gr.Markdown()
        trace_box = gr.Textbox(label="Trace JSON", lines=10, interactive=False)

    refresh_btn.click(fn=refresh_sessions, inputs=[session_dd], outputs=[session_dd])

    discover_btn.click(
        fn=lambda topic, mode, sid: discover_sources(
            topic,
            "auto" if mode == INGEST_MODES[1][0] else "suggest",
            sid,
        ),
        inputs=[topic, ingest_mode, session_dd],
        outputs=[ingest_status, url_choices, session_dd, trace_summary, trace_box],
    )

    ingest_btn.click(
        fn=ingest_selected,
        inputs=[topic, urls_text, url_choices, upload_files, session_dd],
        outputs=[ingest_status, sources_md, trace_box, session_dd],
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question, session_dd, chatbot],
        outputs=[chatbot, trace_box, trace_summary],
    )
    question.submit(
        fn=ask_question,
        inputs=[question, session_dd, chatbot],
        outputs=[chatbot, trace_box, trace_summary],
    )


def researchmind_allowed_paths() -> list[str]:
    cfg = get_config()
    root = cfg.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return [str(root)]
