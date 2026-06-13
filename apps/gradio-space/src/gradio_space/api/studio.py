from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import gradio as gr

from echocoach.config import get_echo_coach_config
from echocoach.pipeline import run_echo_coach
from echocoach.prompts import TeacherVoiceMode
from echocoach.teacher_voice import RAG_MODES, run_teacher_voice_text_turn
from gradio_space.api.serializers import err, ok, update_value
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key, model_status
from gradio_space.research_helpers import list_session_choices
from gradio_space.tabs.education_pptx import generate_lesson_slides
from gradio_space.tabs.research_mind import ingest_selected
from gradio_space.ui.studio_html import (
    render_doc_cards,
    render_echo_coach_panel,
    render_slide_canvas,
)
from inference.factory import get_backend
from researchmind.ingest import IngestPipeline

_echo_config = get_echo_coach_config()


class _NoopProgress:
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def tqdm(self, iterable: Any, **kwargs: Any) -> Any:
        return iterable


def _elapsed_seconds_from_log(processing_log: str) -> float | None:
    match = re.search(r"\*\*Elapsed:\*\* ([\d.]+)s", processing_log or "")
    if not match:
        return None
    return float(match.group(1))


def _progress_from_trace(trace_json: str) -> dict[str, Any]:
    import json

    try:
        trace = json.loads(trace_json)
    except json.JSONDecodeError:
        return {"steps": []}
    steps = []
    for step in trace.get("steps", []):
        if step.get("type") != "step":
            continue
        duration_ms = step.get("duration_ms")
        steps.append(
            {
                "name": step.get("name"),
                "label": step.get("label"),
                "detail": step.get("detail", ""),
                "duration_s": round(duration_ms / 1000, 1) if duration_ms is not None else None,
                "status": "done",
            }
        )
    return {"steps": steps}


def _doc_meta(doc: Any) -> str:
    uri = str(doc.uri or "")
    if len(uri) > 48:
        uri = uri[:45] + "…"
    return f"{doc.source_type} · {uri}"


def _documents_payload(session_id: str) -> list[dict[str, Any]]:
    store = IngestPipeline().store
    docs = store.list_documents(session_id=session_id or None)
    return [
        {
            "id": d.id,
            "title": d.title,
            "source_type": d.source_type,
            "uri": d.uri,
            "meta": _doc_meta(d),
        }
        for d in docs
    ]


def _sessions_payload() -> list[dict[str, str]]:
    sessions: list[dict[str, str]] = []
    for label, sid in list_session_choices():
        if sid == "":
            continue
        topic = label.split(" (")[0] if " (" in label else label
        sessions.append({"id": sid, "label": label, "topic": topic})
    return sessions


def _pick_session(topic_hint: str = "") -> str:
    sessions = _sessions_payload()
    hint = (topic_hint or "").lower()
    for s in sessions:
        if hint and hint in s["topic"].lower():
            return s["id"]
    return sessions[0]["id"] if sessions else ""


def api_list_sessions() -> dict[str, Any]:
    return ok(sessions=_sessions_payload())


def api_list_documents(session_id: str = "") -> dict[str, Any]:
    docs = _documents_payload(session_id)
    html_cards = render_doc_cards(docs, rag_active=bool(docs))
    return ok(session_id=session_id, documents=docs, documents_html=html_cards)


def api_ingest_url(topic: str, url: str, session_id: str = "") -> dict[str, Any]:
    if not url.strip():
        return err("Paste a URL to ingest.")
    status, _memory, _trace, _summary, sess_up, _docs = ingest_selected(
        topic,
        url.strip(),
        [],
        None,
        session_id or None,
        _NoopProgress(),
    )
    sid = update_value(sess_up, session_id)
    docs = _documents_payload(sid)
    return ok(
        status=status,
        session_id=sid,
        documents=docs,
        documents_html=render_doc_cards(docs, rag_active=bool(docs)),
    )


def api_ingest_files(
    topic: str,
    session_id: str,
    file_paths: list[str],
) -> dict[str, Any]:
    if not file_paths:
        return err("Upload at least one PDF or DOCX file.")
    status, _memory, _trace, _summary, sess_up, _docs = ingest_selected(
        topic,
        "",
        [],
        file_paths,
        session_id or None,
        _NoopProgress(),
    )
    sid = update_value(sess_up, session_id)
    docs = _documents_payload(sid)
    return ok(
        status=status,
        session_id=sid,
        documents=docs,
        documents_html=render_doc_cards(docs, rag_active=bool(docs)),
    )


def api_generate_slides(
    topic: str,
    grade: str = "6",
    slide_count: int = 5,
    session_id: str = "",
    use_rag: bool = True,
) -> dict[str, Any]:
    sid = session_id or _pick_session(topic)
    source_label = "RAG (indexed sources)" if use_rag and sid else "None (model only)"
    workflow_label = "Two-step (discover & confirm)"

    outline_md, preview_html, gallery, pptx, docx, html_export, processing_log, trace_sum, trace_json, status = (
        generate_lesson_slides(
            topic,
            grade,
            int(slide_count),
            source_label,
            workflow_label,
            "",
            [],
            None,
            sid if use_rag else "",
            None,
            _NoopProgress(),
            skip_preview_images=True,
        )
    )

    if preview_html and "form-error" in preview_html:
        return err(status or "Generation failed.", status=status, progress_log=processing_log)

    downloads = {
        "pptx": pptx,
        "docx": docx,
        "html": html_export,
    }
    return ok(
        topic=topic,
        session_id=sid,
        outline_md=outline_md,
        preview_html=preview_html,
        canvas_html=render_slide_canvas(preview_html),
        gallery=gallery or [],
        downloads=downloads,
        status=status,
        progress_log=processing_log,
        trace_summary=trace_sum,
        trace_json=trace_json,
        elapsed_seconds=_elapsed_seconds_from_log(processing_log),
        progress=_progress_from_trace(trace_json),
    )


def api_teacher_voice_turn(
    message: str,
    mode: TeacherVoiceMode = "lesson",
    topic: str = "",
    session_id: str = "",
    use_rag: bool = True,
    history: list[list[str]] | None = None,
) -> dict[str, Any]:
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return err(load_error)

    if not message.strip():
        return err("Enter a message or record audio first.")

    hist = history or []
    try:
        result = run_teacher_voice_text_turn(
            message.strip(),
            hist,
            mode=mode,
            language=_echo_config.language_choices()[0][1],
            topic=topic.strip() or None,
            backend=get_backend(model_key),
            use_rag=use_rag and mode in RAG_MODES,
            session_id=session_id or None,
            doc_ids=None,
        )
    except Exception as exc:  # noqa: BLE001
        return err(str(exc))

    return ok(
        history=result.history,
        assistant=result.assistant_text,
        status=result.rag_status or "Turn complete.",
        voiceout_path=result.voiceout_path,
    )


def api_analyze_pitch(
    audio_path: str,
    language: str = "en",
    asr_preset: str | None = None,
) -> dict[str, Any]:
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return err(load_error)

    if not audio_path or not Path(audio_path).is_file():
        return err("Record or upload audio before analyzing.")

    preset = asr_preset or _echo_config.asr_preset
    try:
        result = run_echo_coach(
            audio_path,
            language=language,
            asr_preset=preset,
            backend=get_backend(model_key),
            speak_rewrite=False,
        )
    except Exception as exc:  # noqa: BLE001
        return err(str(exc))

    panel = render_echo_coach_panel(
        pace_score=result.pace.score,
        wpm=result.pace.wpm,
        tip=result.coach.one_tip,
        report_md=result.report_markdown,
    )
    return ok(
        transcript_html=result.transcript_html,
        report_md=result.report_markdown,
        pace_score=result.pace.score,
        wpm=result.pace.wpm,
        tip=result.coach.one_tip,
        filler_chart=result.filler_chart_path,
        pace_chart=result.pace_chart_path,
        voiceout_path=result.voiceout_path,
        coach_panel_html=panel,
    )


def api_model_status() -> dict[str, Any]:
    key = get_active_model_key()
    status_md = model_status(key)
    return ok(model_key=key, status_markdown=status_md)


def api_save_upload(filename: str, content_base64: str) -> dict[str, Any]:
    """Save uploaded file bytes to a temp path for downstream ingest/analyze."""
    if not content_base64:
        return err("Empty upload.")
    try:
        raw = base64.b64decode(content_base64)
    except Exception as exc:  # noqa: BLE001
        return err(f"Invalid upload encoding: {exc}")

    suffix = Path(filename or "upload.bin").suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="studio_")
    tmp.write(raw)
    tmp.close()
    return ok(path=tmp.name, filename=filename)


def register_studio_apis(server: gr.Server) -> None:
    """Register Studio JSON APIs on a gradio.Server instance."""

    @server.api(name="list_sessions")
    def _list_sessions() -> dict[str, Any]:
        return api_list_sessions()

    @server.api(name="list_documents")
    def _list_documents(session_id: str = "") -> dict[str, Any]:
        return api_list_documents(session_id)

    @server.api(name="ingest_url")
    def _ingest_url(topic: str, url: str, session_id: str = "") -> dict[str, Any]:
        return api_ingest_url(topic, url, session_id)

    @server.api(name="ingest_files")
    def _ingest_files(
        topic: str,
        session_id: str,
        file_paths: list[str],
    ) -> dict[str, Any]:
        return api_ingest_files(topic, session_id, file_paths)

    @server.api(name="generate_slides")
    def _generate_slides(
        topic: str,
        grade: str = "6",
        slide_count: int = 5,
        session_id: str = "",
        use_rag: bool = True,
    ) -> dict[str, Any]:
        return api_generate_slides(topic, grade, slide_count, session_id, use_rag)

    @server.api(name="teacher_voice_turn")
    def _teacher_voice_turn(
        message: str,
        mode: Literal["explain", "lesson", "pitch"] = "lesson",
        topic: str = "",
        session_id: str = "",
        use_rag: bool = True,
        history: list[list[str]] | None = None,
    ) -> dict[str, Any]:
        return api_teacher_voice_turn(message, mode, topic, session_id, use_rag, history)

    @server.api(name="analyze_pitch")
    def _analyze_pitch(
        audio_path: str,
        language: str = "en",
        asr_preset: str | None = None,
    ) -> dict[str, Any]:
        return api_analyze_pitch(audio_path, language, asr_preset)

    @server.api(name="model_status")
    def _model_status() -> dict[str, Any]:
        return api_model_status()

    @server.api(name="save_upload")
    def _save_upload(filename: str, content_base64: str) -> dict[str, Any]:
        return api_save_upload(filename, content_base64)
