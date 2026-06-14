from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from agent.models import ResearchIngestResult
from agent.runner import AgentRunner
from gradio_space.model_loading import chat, ensure_model_loaded, get_active_model_key
from gradio_space.spaces_runtime import gpu_task
from inference.factory import get_backend
from researchmind.ingest import IngestPipeline


def resolve_topic(local: str | None, workspace: str | None) -> str:
    """Tab-local topic overrides workspace default when set."""
    return (local or "").strip() or (workspace or "").strip()


def resolve_session(local: str | None, workspace: str | None) -> str:
    return (local or "").strip() or (workspace or "").strip()


def resolve_doc_ids(local: list[str] | None, workspace: list[str] | None) -> list[str]:
    if local:
        return list(local)
    return list(workspace or [])


def pick_session_for_topic(topic_hint: str = "") -> str:
    """Best-effort session id for a topic hint (substring match on session topic)."""
    hint = (topic_hint or "").lower().strip()
    store = IngestPipeline().store
    sessions = store.list_sessions()
    if hint:
        for s in sessions:
            if hint in (s.topic or "").lower():
                return s.id
    return sessions[0].id if sessions else ""


def list_session_choices() -> list[tuple[str, str]]:
    store = IngestPipeline().store
    sessions = store.list_sessions()
    choices: list[tuple[str, str]] = [("New session (chat only)", "")]
    for s in sessions:
        label = f"{s.topic or 'Untitled'} ({s.id})"
        choices.append((label, s.id))
    return choices


def refresh_sessions(current: str):
    choices = list_session_choices()
    values = [c[1] for c in choices]
    if current and current not in values:
        # New session may be selected before choices refresh (e.g. after discover).
        choices.append((f"Session ({current})", current))
        values.append(current)
    value = current if current in values else ""
    return gr.update(choices=choices, value=value)


def list_doc_choices(session_id: str | None) -> list[tuple[str, str]]:
    store = IngestPipeline().store
    docs = store.list_documents(session_id=session_id or None)
    choices: list[tuple[str, str]] = []
    for d in docs:
        label = f"{d.title} ({d.source_type})"
        if len(d.uri) > 60:
            label += f" — {d.uri[:57]}…"
        else:
            label += f" — {d.uri}"
        choices.append((label, d.id))
    return choices


def refresh_doc_choices(session_id: str, current: list[str] | None):
    choices = list_doc_choices(session_id or None)
    valid = {c[1] for c in choices}
    selected = [doc_id for doc_id in (current or []) if doc_id in valid]
    default_selected = [c[1] for c in choices] if choices and not selected else selected
    return gr.update(choices=choices, value=default_selected)


def load_trace_json(trace_path: str) -> str:
    if not trace_path:
        return ""
    if trace_path.strip().startswith("{"):
        return trace_path
    path = Path(trace_path)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return trace_path


def trace_as_dict(value: str | dict | None) -> dict:
    """Normalize trace payloads for gr.JSON (dict only, never invalid strings)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"error": text[:2000]}
        return parsed if isinstance(parsed, dict) else {"data": parsed}
    return {"message": text[:2000]}


def trace_summary_markdown(trace_path: str) -> str:
    raw = load_trace_json(trace_path)
    if not raw or not raw.strip().startswith("{"):
        return raw or "_No trace yet._"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return f"Trace file: `{trace_path}`"

    lines = [
        f"**Run** `{data.get('run_id', '?')}` · skill `{data.get('skill', '?')}`",
        "",
    ]
    for step in data.get("steps", []):
        if step.get("type") != "note":
            continue
        msg = step.get("message", "")
        extra = {k: v for k, v in step.items() if k not in ("type", "message")}
        detail = ""
        if extra:
            detail = " — " + ", ".join(f"{k}={v!r}" for k, v in extra.items())
        lines.append(f"- {msg}{detail}")
    if len(lines) <= 2:
        lines.append("_No notes in trace. See Trace JSON below._")
    return "\n".join(lines)


def format_ingest_status(result: ResearchIngestResult) -> str:
    lines = [result.message, ""]
    if result.ingested:
        lines.append("**Ingested**")
        lines.extend(f"- {url}" for url in result.ingested)
        lines.append("")
    if result.skipped:
        lines.append("**Skipped (duplicate)**")
        lines.extend(f"- {url}" for url in result.skipped)
        lines.append("")
    if result.failures:
        lines.append("**Failed**")
        for failure in result.failures:
            lines.append(f"- `{failure.url}` — _{failure.stage}_: {failure.reason}")
        lines.append("")
        lines.append("_Open the **Trace** tab for full JSON._")
    return "\n".join(lines).strip()


def memory_summary(session_id: str) -> str:
    store = IngestPipeline().store
    docs = store.list_documents(session_id=session_id or None)
    chunks = store.count_chunks()
    if not docs:
        return f"_No documents indexed yet._ Total chunks in store: **{chunks}**."
    scope = f"session `{session_id}`" if session_id else "all sessions"
    lines = [f"**{len(docs)}** document(s) in {scope} · **{chunks}** total chunks in store\n"]
    for d in docs:
        lines.append(f"- **{d.title}** (`{d.source_type}`) — {d.uri}")
    return "\n".join(lines)


def parse_urls_text(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def merge_lesson_urls(pasted: str, selected: list[str] | None) -> list[str]:
    direct = parse_urls_text(pasted)
    return list(dict.fromkeys([*direct, *(selected or [])]))


def format_citations_markdown(trace_json: str) -> str:
    """Extract citation lines from RAG trace JSON for chat display."""
    if not trace_json or not trace_json.strip().startswith("{"):
        return ""
    try:
        data = json.loads(trace_json)
    except json.JSONDecodeError:
        return ""
    citations = data.get("citations") or []
    if not citations:
        return ""
    lines = ["", "---", "**Sources:**"]
    for i, cite in enumerate(citations[:5], start=1):
        title = cite.get("title") or cite.get("uri") or "Source"
        uri = cite.get("uri") or ""
        lines.append(f"{i}. [{title}]({uri})" if uri else f"{i}. {title}")
    if len(citations) > 5:
        lines.append(f"_…and {len(citations) - 5} more (see Advanced trace)._")
    return "\n".join(lines)


def rag_scope_hint(session_id: str, doc_ids: list[str] | None) -> str:
    if doc_ids:
        return f"RAG scope: **{len(doc_ids)}** selected document(s)."
    if session_id:
        n = len(IngestPipeline().store.list_documents(session_id=session_id))
        return f"RAG scope: all **{n}** document(s) in session `{session_id}`."
    return "RAG scope: **entire** indexed corpus (all sessions)."


@gpu_task(duration=180)
def run_research_question(
    question: str,
    *,
    session_id: str,
    doc_ids: list[str] | None,
    model_key: str | None = None,
) -> tuple[str, str, str]:
    """Returns (answer_markdown, trace_json, trace_summary_md)."""
    key = model_key or get_active_model_key()
    load_error = ensure_model_loaded(key)
    if load_error:
        return load_error, load_error, load_error

    if not question.strip():
        return "Enter a question.", "", ""

    runner = AgentRunner()
    result = runner.run_researchmind_chat(
        question=question,
        session_id=session_id or "",
        doc_ids=doc_ids or None,
        model_key=key,
        backend=get_backend(key),
    )
    sid = session_id or result.session_id
    trace_json = json.dumps(
        {
            "trace_path": result.trace_path,
            "citations": [c.model_dump() for c in result.citations],
            "scope": {
                "session_id": sid,
                "doc_ids": doc_ids or [],
            },
        },
        indent=2,
    )
    return (
        result.answer,
        trace_json,
        trace_summary_markdown(result.trace_path),
    )


def rag_aware_chat(
    message: str,
    history: list,
    model_key: str,
    use_rag: bool,
    session_id: str,
    doc_ids: list[str] | None,
) -> tuple[str, str, str]:
    """Returns (reply, trace_json, trace_summary) for debug chat."""
    if not use_rag:
        return chat(message, history, model_key), "", ""

    answer, trace_json, trace_summary = run_research_question(
        message,
        session_id=session_id,
        doc_ids=doc_ids,
        model_key=model_key,
    )
    citations = format_citations_markdown(trace_json)
    if citations:
        answer = f"{answer}\n{citations}"
    return answer, trace_json, trace_summary
