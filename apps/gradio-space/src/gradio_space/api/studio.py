from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import gradio as gr

from echocoach.config import get_echo_coach_config
from echocoach.pipeline import run_echo_coach
from echocoach.prompts import TeacherVoiceMode, resolve_aya_preset
from echocoach.recording import (
    ServerRecordingError,
    recording_backend_status,
    recording_elapsed_seconds,
    recording_level_warning,
    start_server_recording,
    stop_server_recording,
)
from echocoach.teacher_voice import RAG_MODES, run_teacher_voice_text_turn, run_teacher_voice_turn
from gradio_space.api.serializers import err, ok, unwrap_update, update_value
from gradio_space.model_loading import (
    ensure_model_loaded,
    get_active_model_key,
    model_status,
    reload_model,
    select_and_reload_model,
    set_runtime_model_key,
)
from gradio_space.research_helpers import (
    list_session_choices,
    memory_summary,
    pick_session_for_topic,
    rag_aware_chat,
    rag_scope_hint,
    resolve_doc_ids,
    resolve_session,
)
from gradio_space.tabs.education_pptx import SOURCE_MODES, SEARCH_WORKFLOWS, generate_lesson_slides
from gradio_space.tabs.research_mind import (
    ask_question,
    auto_search_ingest,
    discover_sources,
    ingest_selected,
)
from gradio_space.ui.studio_html import (
    render_doc_cards,
    render_echo_coach_panel,
    render_gallery_strip,
    render_slide_canvas,
    render_trace_details,
)
from gradio_space.voice_helpers import speak_last_assistant_reply
from inference.config import get_app_config, get_model_config
from inference.factory import get_backend
from researchmind.config import get_config as get_research_config
from researchmind.ingest import IngestPipeline

_echo_config = get_echo_coach_config()
_app_config = get_app_config()
_SAMPLE_PITCH_AUDIO = (
    Path(__file__).resolve().parents[5]
    / "libs"
    / "echocoach"
    / "tests"
    / "fixtures"
    / "silence_2s.wav"
)

_SOURCE_LABELS = {value: label for label, value in SOURCE_MODES}
_WORKFLOW_LABELS = {value: label for label, value in SEARCH_WORKFLOWS}


class _NoopProgress:
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def tqdm(self, iterable: Any, **kwargs: Any) -> Any:
        return iterable


def _elapsed_seconds_from_log(processing_log: str) -> float | None:
    match = re.search(r"Elapsed:\s*([\d.]+)s", processing_log or "")
    if not match:
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


def _session_has_rag_sources(session_id: str, doc_ids: list[str] | None) -> bool:
    if not session_id:
        return False
    docs = _documents_payload(session_id)
    if not docs:
        return False
    if doc_ids:
        valid = {d["id"] for d in docs}
        return any(doc_id in valid for doc_id in doc_ids)
    return True


def _sessions_payload() -> list[dict[str, str]]:
    sessions: list[dict[str, str]] = []
    for label, sid in list_session_choices():
        if sid == "":
            continue
        topic = label.split(" (")[0] if " (" in label else label
        sessions.append({"id": sid, "label": label, "topic": topic})
    return sessions


def _pick_session(topic_hint: str = "") -> str:
    return pick_session_for_topic(topic_hint)


def _voice_stack_summary() -> str:
    asr = _echo_config.get_asr()
    tts = _echo_config.get_tts()
    lines = [
        f"ASR: {asr.label} ({_echo_config.asr_preset})",
        f"TTS: {tts.label} ({_echo_config.tts_preset})",
        f"Coach model: {_echo_config.coach_model}",
        f"Coach fallbacks: {', '.join(_echo_config.coach_fallbacks) or 'none'}",
        f"Max recording: {_echo_config.max_seconds}s",
    ]
    return "\n".join(lines)


def _coach_model_key(
    coach_model: str | None = None,
    *,
    language: str = "en",
    coach_variant: str = "auto",
) -> str:
    if coach_model and coach_model.strip():
        key = coach_model.strip()
    elif coach_variant and coach_variant not in ("auto", ""):
        key = coach_variant.strip()
    else:
        key = resolve_aya_preset(language, coach_variant)
    if key in ("tiny-aya-water", "tiny-aya-fire", "tiny-aya-earth", "auto"):
        key = "tiny-aya-global"
    return key


def _coach_model_label(model_key: str) -> str:
    try:
        return get_model_config(model_key).label
    except Exception:
        return model_key


def _coach_model_candidates(
    coach_model: str | None = None,
    *,
    language: str = "en",
    coach_variant: str = "auto",
) -> list[str]:
    if coach_model and coach_model.strip():
        return [coach_model.strip()]
    primary = _coach_model_key(None, language=language, coach_variant=coach_variant)
    chain: list[str] = []
    seen: set[str] = set()
    for key in (primary, *_echo_config.coach_fallbacks):
        if key and key not in seen:
            seen.add(key)
            chain.append(key)
    return chain or [primary]


def _ensure_coach_loaded(
    coach_model: str | None = None,
    *,
    language: str = "en",
    coach_variant: str = "auto",
) -> tuple[str, str | None, str | None]:
    """Load the first coach preset that succeeds. Returns (key, error, fallback_note)."""
    candidates = _coach_model_candidates(
        coach_model,
        language=language,
        coach_variant=coach_variant,
    )
    errors: list[str] = []
    for index, key in enumerate(candidates):
        load_error = ensure_model_loaded(key)
        if not load_error:
            if index == 0:
                return key, None, None
            label = _coach_model_label(key)
            note = (
                f"Primary coach unavailable — using fallback **{label}** (`{key}`). "
                "Replies still follow your target language via prompts."
            )
            return key, None, note
        errors.append(load_error)
    return candidates[-1], errors[-1], None


def _coach_turn_status(base: str | None, fallback_note: str | None) -> str:
    status = (base or "Turn complete.").strip()
    if fallback_note:
        return f"{fallback_note} {status}".strip()
    return status


def _voice_language_codes() -> list[str]:
    return [code for _, code in _echo_config.language_choices()]


def _paths_summary() -> str:
    rm = get_research_config()
    lines = []
    if _app_config.presets_path:
        lines.append(f"Model presets: {_app_config.presets_path}")
    else:
        lines.append("Model presets: built-in defaults")
    lines.append(f"ResearchMind store: {rm.data_dir.resolve()}")
    return "\n".join(lines)


def _resolve_source_labels(
    source_mode: str,
    search_workflow: str,
    use_rag: bool,
    session_id: str,
    doc_ids: list[str] | None,
) -> tuple[str, str, str, list[str]]:
    """Return source_label, workflow_label, effective_session, effective_docs."""
    mode = (source_mode or "").strip().lower()
    if not mode:
        sid = (session_id or "").strip()
        has_sources = _session_has_rag_sources(sid, doc_ids) if use_rag else False
        if use_rag and has_sources:
            return (
                _SOURCE_LABELS["rag"],
                _WORKFLOW_LABELS["two_step"],
                sid,
                doc_ids or [],
            )
        return _SOURCE_LABELS["none"], _WORKFLOW_LABELS["two_step"], "", []

    workflow_key = (search_workflow or "two_step").strip().lower()
    if workflow_key not in _WORKFLOW_LABELS:
        workflow_key = "two_step"

    if mode not in _SOURCE_LABELS:
        mode = "none"

    sid = (session_id or "").strip()
    if mode == "rag" and not sid:
        sid = ""

    return (
        _SOURCE_LABELS[mode],
        _WORKFLOW_LABELS[workflow_key],
        sid if mode == "rag" else sid,
        doc_ids or [] if mode == "rag" else [],
    )


def api_list_sessions() -> dict[str, Any]:
    return ok(sessions=_sessions_payload())


def api_list_documents(session_id: str = "") -> dict[str, Any]:
    docs = _documents_payload(session_id)
    html_cards = render_doc_cards(docs, rag_active=bool(docs))
    return ok(
        session_id=session_id,
        documents=docs,
        documents_html=html_cards,
        memory_markdown=memory_summary(session_id),
    )


def api_session_memory(session_id: str = "") -> dict[str, Any]:
    return ok(memory_markdown=memory_summary(session_id))


def _ingest_response(
    status: str,
    session_id: str,
    trace_json: str = "",
    trace_summary: str = "",
) -> dict[str, Any]:
    sid = session_id or ""
    docs = _documents_payload(sid)
    return ok(
        status=status,
        session_id=sid,
        documents=docs,
        documents_html=render_doc_cards(docs, rag_active=bool(docs)),
        trace_json=trace_json,
        trace_summary=trace_summary,
        trace_html=render_trace_details(trace_summary=trace_summary, trace_json=trace_json),
    )


def api_discover_sources(topic: str, session_id: str = "") -> dict[str, Any]:
    if not (topic or "").strip():
        return err("Enter a workspace topic before discovering sources.")
    summary, url_up, sess_up, trace_sum, trace_json, _memory, _doc_up, _acc_up = discover_sources(
        topic,
        session_id,
        "",
        "",
        _NoopProgress(),
    )
    url_payload = unwrap_update(url_up)
    urls = list(url_payload.get("choices") or []) if isinstance(url_payload, dict) else []
    selected = list(url_payload.get("value") or urls) if isinstance(url_payload, dict) else urls
    sid = update_value(sess_up, session_id)
    trace_str = trace_json if isinstance(trace_json, str) else ""
    if summary and "error" in summary.lower() and not urls:
        return err(strip_md_summary(summary), status=summary, urls=[], session_id=sid)
    return ok(
        status=summary,
        urls=urls,
        selected_urls=selected,
        session_id=sid,
        trace_summary=trace_sum,
        trace_json=trace_str,
        trace_html=render_trace_details(trace_summary=trace_sum, trace_json=trace_str),
    )


def api_auto_search_ingest(topic: str, session_id: str = "") -> dict[str, Any]:
    if not (topic or "").strip():
        return err("Enter a workspace topic before auto-ingest.")
    status, _url_up, sess_up, trace_sum, trace_json, _memory, _doc_up, _acc_up = auto_search_ingest(
        topic,
        session_id,
        "",
        "",
        _NoopProgress(),
    )
    sid = update_value(sess_up, session_id)
    if status and "error" in status.lower() and "ingested" not in status.lower():
        return err(strip_md_summary(status), status=status, session_id=sid)
    return _ingest_response(status, sid, trace_json=str(trace_json or ""), trace_summary=trace_sum)


def api_ingest_sources(
    topic: str,
    session_id: str = "",
    urls_text: str = "",
    selected_urls: list[str] | None = None,
    file_paths: list[str] | None = None,
) -> dict[str, Any]:
    has_urls = bool((urls_text or "").strip() or (selected_urls or []))
    has_files = bool(file_paths)
    if not has_urls and not has_files:
        return err("Add URLs, select suggested sources, or upload a file — then ingest.")
    status, _memory, trace_json, trace_sum, sess_up, _doc_up = ingest_selected(
        topic,
        urls_text,
        selected_urls or [],
        file_paths,
        session_id or None,
        "",
        "",
        _NoopProgress(),
    )
    sid = update_value(sess_up, session_id)
    if status and "error" in status.lower() and "ingested" not in status.lower():
        return err(strip_md_summary(status), status=status, session_id=sid)
    return _ingest_response(status, sid, trace_json=str(trace_json or ""), trace_summary=trace_sum)


def strip_md_summary(text: str) -> str:
    return re.sub(r"\*\*", "", str(text or "")).strip()


def api_ingest_url(topic: str, url: str, session_id: str = "") -> dict[str, Any]:
    if not url.strip():
        return err("Paste a URL to ingest.")
    return api_ingest_sources(topic, session_id, urls_text=url.strip())


def api_ingest_files(
    topic: str,
    session_id: str,
    file_paths: list[str],
) -> dict[str, Any]:
    if not file_paths:
        return err("Upload at least one PDF or DOCX file.")
    return api_ingest_sources(topic, session_id, file_paths=file_paths)


def api_research_chat(
    question: str,
    session_id: str = "",
    doc_ids: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not question.strip():
        return err("Enter a question.")
    hist, trace_json, trace_sum, rag_hint, _cleared = ask_question(
        question,
        session_id,
        doc_ids or [],
        history or [],
        "",
        doc_ids or [],
        _NoopProgress(),
    )
    assistant = ""
    for msg in reversed(hist or []):
        if msg.get("role") == "assistant":
            assistant = str(msg.get("content") or "")
            break
    trace_str = trace_json if isinstance(trace_json, str) else ""
    return ok(
        history=hist,
        assistant=assistant,
        rag_hint=rag_hint,
        trace_json=trace_str,
        trace_summary=trace_sum,
        trace_html=render_trace_details(trace_summary=trace_sum, trace_json=trace_str),
    )


def api_debug_chat(
    message: str,
    history: list[list[str]] | None = None,
    use_rag: bool = False,
    session_id: str = "",
    doc_ids: list[str] | None = None,
    model_key: str = "",
    workspace_session_id: str = "",
    workspace_doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not (message or "").strip():
        return err("Enter a message.")
    key = (model_key or "").strip() or get_active_model_key()
    load_error = ensure_model_loaded(key)
    if load_error:
        return err(load_error)

    sid = resolve_session(session_id, workspace_session_id)
    docs = resolve_doc_ids(doc_ids, workspace_doc_ids)
    hist = history or []
    reply, trace_json, trace_summary = rag_aware_chat(
        message.strip(),
        hist,
        key,
        use_rag,
        sid,
        docs,
    )
    new_history = list(hist)
    new_history.append([message.strip(), reply])
    return ok(
        history=new_history,
        assistant=reply,
        rag_hint=rag_scope_hint(sid, docs),
        trace_json=trace_json,
        trace_summary=trace_summary,
        trace_html=render_trace_details(trace_summary=trace_summary, trace_json=trace_json),
    )


def api_generate_slides(
    topic: str,
    grade: str = "6",
    slide_count: int = 5,
    session_id: str = "",
    use_rag: bool = True,
    doc_ids: list[str] | None = None,
    source_mode: str = "",
    search_workflow: str = "two_step",
    urls_text: str = "",
    selected_urls: list[str] | None = None,
    file_paths: list[str] | None = None,
) -> dict[str, Any]:
    rag_docs = doc_ids or []
    sid = (session_id or "").strip()
    if not (source_mode or "").strip() and use_rag and not sid:
        sid = _pick_session(topic)

    source_label, workflow_label, effective_sid, effective_docs = _resolve_source_labels(
        source_mode,
        search_workflow,
        use_rag,
        sid,
        rag_docs,
    )

    rag_notice = ""
    if (source_mode or "").strip().lower() == "rag" or (
        not (source_mode or "").strip() and use_rag
    ):
        has_sources = _session_has_rag_sources(sid, rag_docs)
        if use_rag and not has_sources and source_label == _SOURCE_LABELS["rag"]:
            rag_notice = (
                "Cross-Reference Sources is on, but this session has no indexed documents — "
                "generated from model knowledge only. Ingest sources in Step 1 to enable RAG."
            )
            source_label = _SOURCE_LABELS["none"]
            effective_sid = ""
            effective_docs = []

    upload_files = file_paths if file_paths else None

    gen = generate_lesson_slides(
        topic,
        grade,
        int(slide_count),
        source_label,
        workflow_label,
        urls_text or "",
        selected_urls or [],
        upload_files,
        effective_sid,
        effective_docs,
        topic,
        effective_sid,
        effective_docs,
        _NoopProgress(),
        skip_preview_images=False,
    )
    last: tuple | None = None
    for item in gen:
        last = item
    if last is None:
        return err("Generation failed before producing output.")

    (
        outline_md,
        preview_html,
        gallery,
        pptx,
        docx,
        html_export,
        processing_log,
        trace_sum,
        trace_json,
        status,
    ) = last

    if preview_html and "form-error" in preview_html:
        return err(status or "Generation failed.", status=status, progress_log=processing_log)

    if rag_notice:
        status = f"{rag_notice}\n\n{status or 'Slides generated.'}".strip()

    downloads = {
        "pptx": pptx,
        "docx": docx,
        "html": html_export,
    }
    trace_str = trace_json if isinstance(trace_json, str) else ""
    return ok(
        topic=topic,
        session_id=sid,
        outline_md=outline_md,
        preview_html=preview_html,
        canvas_html=render_slide_canvas(preview_html),
        gallery=gallery or [],
        gallery_html=render_gallery_strip(gallery or []),
        downloads=downloads,
        status=status,
        rag_fallback=bool(rag_notice),
        progress_log=processing_log,
        trace_summary=trace_sum,
        trace_json=trace_str,
        trace_html=render_trace_details(
            trace_summary=trace_sum,
            trace_json=trace_str,
            progress_log=processing_log,
        ),
        elapsed_seconds=_elapsed_seconds_from_log(processing_log),
        progress=_progress_from_trace(trace_str),
    )


def api_teacher_voice_turn(
    message: str,
    mode: TeacherVoiceMode = "lesson",
    topic: str = "",
    session_id: str = "",
    use_rag: bool = True,
    history: list | None = None,
    doc_ids: list[str] | None = None,
    language: str = "en",
    asr_preset: str | None = None,
    auto_voiceout: bool = True,
    coach_model: str = "",
    coach_variant: str = "auto",
) -> dict[str, Any]:
    model_key, load_error, fallback_note = _ensure_coach_loaded(
        coach_model or None,
        language=language,
        coach_variant=coach_variant,
    )
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
            language=language,
            topic=topic.strip() or None,
            backend=get_backend(model_key),
            coach_model=model_key,
            use_rag=use_rag and mode in RAG_MODES,
            session_id=session_id or None,
            doc_ids=doc_ids or None,
            auto_voiceout=auto_voiceout,
        )
    except Exception as exc:  # noqa: BLE001
        return err(str(exc))

    return ok(
        history=result.history,
        assistant=result.assistant_text,
        status=_coach_turn_status(result.rag_status, fallback_note),
        voiceout_path=result.voiceout_path,
        voiceout_warning=result.voiceout_warning,
        rag_references=result.rag_references,
        coach_model=model_key,
        coach_fallback=bool(fallback_note),
    )

def api_teacher_voice_audio_turn(
    audio_path: str,
    mode: TeacherVoiceMode = "lesson",
    topic: str = "",
    session_id: str = "",
    use_rag: bool = True,
    history: list | None = None,
    doc_ids: list[str] | None = None,
    language: str = "en",
    asr_preset: str | None = None,
    auto_voiceout: bool = True,
    coach_model: str = "",
    coach_variant: str = "auto",
) -> dict[str, Any]:
    model_key, load_error, fallback_note = _ensure_coach_loaded(
        coach_model or None,
        language=language,
        coach_variant=coach_variant,
    )
    if load_error:
        return err(load_error)

    if not audio_path or not Path(audio_path).is_file():
        return err("Record or upload audio first.")

    hist = history or []
    preset = asr_preset or _echo_config.asr_preset
    max_turn = min(15, _echo_config.max_seconds)
    try:
        result = run_teacher_voice_turn(
            audio_path,
            hist,
            mode=mode,
            language=language,
            asr_preset=preset,
            topic=topic.strip() or None,
            backend=get_backend(model_key),
            coach_model=model_key,
            use_rag=use_rag and mode in RAG_MODES,
            session_id=session_id or None,
            doc_ids=doc_ids or None,
            max_turn_seconds=max_turn,
            auto_voiceout=auto_voiceout,
        )
    except Exception as exc:  # noqa: BLE001
        return err(str(exc))

    return ok(
        history=result.history,
        assistant=result.assistant_text,
        status=_coach_turn_status(result.rag_status, fallback_note),
        voiceout_path=result.voiceout_path,
        voiceout_warning=result.voiceout_warning,
        user_text=result.user_text,
        rag_references=result.rag_references,
        coach_model=model_key,
        coach_fallback=bool(fallback_note),
    )


def api_language_lesson_turn(
    message: str = "",
    audio_path: str = "",
    mode: TeacherVoiceMode = "lesson",
    topic: str = "",
    session_id: str = "",
    use_rag: bool = True,
    history: list | None = None,
    doc_ids: list[str] | None = None,
    language: str = "en",
    asr_preset: str | None = None,
    auto_voiceout: bool = True,
    coach_model: str = "",
    coach_variant: str = "auto",
) -> dict[str, Any]:
    """Unified Language lessons turn — routes to text or audio pipeline."""
    if audio_path and audio_path.strip():
        return api_teacher_voice_audio_turn(
            audio_path.strip(),
            mode=mode,
            topic=topic,
            session_id=session_id,
            use_rag=use_rag,
            history=history,
            doc_ids=doc_ids,
            language=language,
            asr_preset=asr_preset,
            auto_voiceout=auto_voiceout,
            coach_model=coach_model,
            coach_variant=coach_variant,
        )
    return api_teacher_voice_turn(
        message,
        mode=mode,
        topic=topic,
        session_id=session_id,
        use_rag=use_rag,
        history=history,
        doc_ids=doc_ids,
        language=language,
        asr_preset=asr_preset,
        auto_voiceout=auto_voiceout,
        coach_model=coach_model,
        coach_variant=coach_variant,
    )


def api_teacher_voice_clear() -> dict[str, Any]:
    return ok(
        history=[],
        assistant="",
        status="Conversation cleared.",
    )


def api_teacher_voice_speak(
    history: list | None = None,
    language: str = "en",
    first_sentence_only: bool = False,
) -> dict[str, Any]:
    playback, status = speak_last_assistant_reply(
        history or [],
        language,
        first_sentence_only=first_sentence_only,
    )
    if not playback:
        return err(status)
    return ok(voiceout_path=playback, status=status)


def api_load_sample_pitch() -> dict[str, Any]:
    if not _SAMPLE_PITCH_AUDIO.is_file():
        return err(
            f"Sample clip missing at `{_SAMPLE_PITCH_AUDIO}`. "
            "Run `uv run python libs/echocoach/tests/make_fixture.py`."
        )
    return ok(
        audio_path=str(_SAMPLE_PITCH_AUDIO),
        status="Sample clip loaded — click Analyze pitch when ready.",
    )


def api_analyze_pitch(
    audio_path: str,
    language: str = "en",
    asr_preset: str | None = None,
    speak_rewrite: bool = False,
) -> dict[str, Any]:
    model_key, load_error, _fallback_note = _ensure_coach_loaded(None, language=language)
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
            coach_model=model_key,
            backend=get_backend(model_key),
            speak_rewrite=speak_rewrite,
        )
    except Exception as exc:  # noqa: BLE001
        return err(str(exc))

    panel = render_echo_coach_panel(
        pace_score=result.pace.score,
        wpm=result.pace.wpm,
        tip=result.coach.one_tip,
        report_md=result.report_markdown,
        transcript_html=result.transcript_html,
        filler_chart=result.filler_chart_path,
        pace_chart=result.pace_chart_path,
        voiceout_path=result.voiceout_path,
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


def api_model_choices() -> dict[str, Any]:
    key = get_active_model_key()
    active = _app_config.get_model(key)
    allow_switch = bool(
        _app_config.allow_model_switch and len(_app_config.models) > 1
    )
    choices = []
    if allow_switch:
        choices = [{"key": k, "label": label} for label, k in _app_config.model_choices()]
    return ok(
        active_model=key,
        active_label=active.label,
        active_backend=active.backend,
        allow_model_switch=allow_switch,
        choices=choices,
        voice_stack=_voice_stack_summary(),
        paths=_paths_summary(),
    )


def api_set_active_model(model_key: str = "") -> dict[str, Any]:
    key = (model_key or "").strip() or get_active_model_key()
    try:
        status_md = select_and_reload_model(key)
    except KeyError as exc:
        return err(str(exc), model_key=key)
    if status_md.lower().startswith("error") or "failed" in status_md.lower():
        return err(status_md, status_markdown=status_md, model_key=key)
    return ok(status_markdown=status_md, model_key=key)


def api_reload_model(model_key: str = "") -> dict[str, Any]:
    key = (model_key or "").strip() or get_active_model_key()
    status_md = reload_model(key)
    if status_md.lower().startswith("error") or "failed" in status_md.lower():
        return err(status_md, status_markdown=status_md, model_key=key)
    return ok(status_markdown=status_md, model_key=key)


def api_recording_status() -> dict[str, Any]:
    status = recording_backend_status()
    return ok(
        backend=status,
        message=status,
        max_seconds=_echo_config.max_seconds,
    )


def api_recording_start(max_seconds: int | None = None) -> dict[str, Any]:
    limit = int(max_seconds or _echo_config.max_seconds)
    try:
        start_server_recording(limit)
    except ServerRecordingError as exc:
        return err(str(exc))
    return ok(
        status=f"Recording… speak now, then stop (auto-stops after {limit}s).",
        max_seconds=limit,
    )


def api_recording_stop() -> dict[str, Any]:
    try:
        elapsed = recording_elapsed_seconds()
        path = stop_server_recording()
        warning = recording_level_warning(path)
    except ServerRecordingError as exc:
        return err(str(exc))
    except Exception as exc:  # noqa: BLE001
        return err(f"Recording failed: {exc}")

    status = f"Recording saved ({elapsed:.1f}s)."
    if warning:
        status += f" Warning: {warning}"
    return ok(path=str(path), elapsed_seconds=elapsed, status=status, warning=warning or "")


def api_voice_presets() -> dict[str, Any]:
    tts = _echo_config.get_tts()
    voice_langs = _voice_language_codes()
    coach_chain = _echo_config.coach_model_chain()
    coach_chain_labels = [_coach_model_label(key) for key in coach_chain]
    fallback_label = coach_chain_labels[1] if len(coach_chain_labels) > 1 else None
    return ok(
        languages=[{"label": label, "value": value} for label, value in _echo_config.language_choices()],
        asr_presets=[{"label": label, "value": value} for label, value in _echo_config.asr_choices()],
        coach_variants=[
            {"label": "Tiny Aya Global (70+ languages)", "value": "tiny-aya-global"},
        ],
        default_language=_echo_config.language_choices()[0][1] if _echo_config.language_choices() else "en",
        default_asr=_echo_config.asr_preset,
        default_coach=_echo_config.coach_model,
        coach_fallbacks=list(_echo_config.coach_fallbacks),
        coach_chain=coach_chain,
        coach_chain_labels=coach_chain_labels,
        voice_languages=voice_langs,
        max_seconds=_echo_config.max_seconds,
        voiceout_note=(
            f"Voice in/out: {len(voice_langs)} languages via Piper · "
            f"Coach: {coach_chain_labels[0]}"
            + (f" (fallback: {fallback_label})" if fallback_label else "")
        ),
    )


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

    @server.api(name="session_memory")
    def _session_memory(session_id: str = "") -> dict[str, Any]:
        return api_session_memory(session_id)

    @server.api(name="discover_sources")
    def _discover_sources(topic: str, session_id: str = "") -> dict[str, Any]:
        return api_discover_sources(topic, session_id)

    @server.api(name="auto_search_ingest")
    def _auto_search_ingest(topic: str, session_id: str = "") -> dict[str, Any]:
        return api_auto_search_ingest(topic, session_id)

    @server.api(name="ingest_sources")
    def _ingest_sources(
        topic: str,
        session_id: str = "",
        urls_text: str = "",
        selected_urls: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return api_ingest_sources(
            topic, session_id, urls_text, selected_urls, file_paths
        )

    @server.api(name="ingest_url")
    def _ingest_url(topic: str, url: str, session_id: str = "") -> dict[str, Any]:
        return api_ingest_url(topic, url, session_id)

    @server.api(name="research_chat")
    def _research_chat(
        question: str,
        session_id: str = "",
        doc_ids: list[str] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return api_research_chat(question, session_id, doc_ids, history)

    @server.api(name="debug_chat")
    def _debug_chat(
        message: str,
        history: list[list[str]] | None = None,
        use_rag: bool = False,
        session_id: str = "",
        doc_ids: list[str] | None = None,
        model_key: str = "",
        workspace_session_id: str = "",
        workspace_doc_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return api_debug_chat(
            message,
            history,
            use_rag,
            session_id,
            doc_ids,
            model_key,
            workspace_session_id,
            workspace_doc_ids,
        )

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
        doc_ids: list[str] | None = None,
        source_mode: str = "",
        search_workflow: str = "two_step",
        urls_text: str = "",
        selected_urls: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return api_generate_slides(
            topic,
            grade,
            slide_count,
            session_id,
            use_rag,
            doc_ids,
            source_mode,
            search_workflow,
            urls_text,
            selected_urls,
            file_paths,
        )

    @server.api(name="language_lesson_turn")
    def _language_lesson_turn(
        message: str = "",
        audio_path: str = "",
        mode: Literal["explain", "lesson"] = "lesson",
        topic: str = "",
        session_id: str = "",
        use_rag: bool = True,
        history: list | None = None,
        doc_ids: list[str] | None = None,
        language: str = "en",
        asr_preset: str | None = None,
        auto_voiceout: bool = True,
        coach_model: str = "",
        coach_variant: str = "auto",
    ) -> dict[str, Any]:
        return api_language_lesson_turn(
            message,
            audio_path,
            mode,
            topic,
            session_id,
            use_rag,
            history,
            doc_ids,
            language,
            asr_preset,
            auto_voiceout,
            coach_model,
            coach_variant,
        )

    @server.api(name="teacher_voice_turn")
    def _teacher_voice_turn(
        message: str,
        mode: Literal["explain", "lesson", "pitch"] = "lesson",
        topic: str = "",
        session_id: str = "",
        use_rag: bool = True,
        history: list | None = None,
        doc_ids: list[str] | None = None,
        language: str = "en",
        asr_preset: str | None = None,
        auto_voiceout: bool = True,
        coach_model: str = "",
        coach_variant: str = "auto",
    ) -> dict[str, Any]:
        return api_teacher_voice_turn(
            message,
            mode,
            topic,
            session_id,
            use_rag,
            history,
            doc_ids,
            language,
            asr_preset,
            auto_voiceout,
            coach_model,
            coach_variant,
        )

    @server.api(name="teacher_voice_audio_turn")
    def _teacher_voice_audio_turn(
        audio_path: str,
        mode: Literal["explain", "lesson", "pitch"] = "lesson",
        topic: str = "",
        session_id: str = "",
        use_rag: bool = True,
        history: list | None = None,
        doc_ids: list[str] | None = None,
        language: str = "en",
        asr_preset: str | None = None,
        auto_voiceout: bool = True,
        coach_model: str = "",
        coach_variant: str = "auto",
    ) -> dict[str, Any]:
        return api_teacher_voice_audio_turn(
            audio_path,
            mode,
            topic,
            session_id,
            use_rag,
            history,
            doc_ids,
            language,
            asr_preset,
            auto_voiceout,
            coach_model,
            coach_variant,
        )

    @server.api(name="teacher_voice_clear")
    def _teacher_voice_clear() -> dict[str, Any]:
        return api_teacher_voice_clear()

    @server.api(name="teacher_voice_speak")
    def _teacher_voice_speak(
        history: list | None = None,
        language: str = "en",
        first_sentence_only: bool = False,
    ) -> dict[str, Any]:
        return api_teacher_voice_speak(history, language, first_sentence_only)

    @server.api(name="load_sample_pitch")
    def _load_sample_pitch() -> dict[str, Any]:
        return api_load_sample_pitch()

    @server.api(name="analyze_pitch")
    def _analyze_pitch(
        audio_path: str,
        language: str = "en",
        asr_preset: str | None = None,
        speak_rewrite: bool = False,
    ) -> dict[str, Any]:
        return api_analyze_pitch(audio_path, language, asr_preset, speak_rewrite)

    @server.api(name="model_status")
    def _model_status() -> dict[str, Any]:
        return api_model_status()

    @server.api(name="model_choices")
    def _model_choices() -> dict[str, Any]:
        return api_model_choices()

    @server.api(name="set_active_model")
    def _set_active_model(model_key: str = "") -> dict[str, Any]:
        return api_set_active_model(model_key)

    @server.api(name="reload_model")
    def _reload_model(model_key: str = "") -> dict[str, Any]:
        return api_reload_model(model_key)

    @server.api(name="recording_status")
    def _recording_status() -> dict[str, Any]:
        return api_recording_status()

    @server.api(name="recording_start")
    def _recording_start(max_seconds: int | None = None) -> dict[str, Any]:
        return api_recording_start(max_seconds)

    @server.api(name="recording_stop")
    def _recording_stop() -> dict[str, Any]:
        return api_recording_stop()

    @server.api(name="voice_presets")
    def _voice_presets() -> dict[str, Any]:
        return api_voice_presets()

    @server.api(name="save_upload")
    def _save_upload(filename: str, content_base64: str) -> dict[str, Any]:
        return api_save_upload(filename, content_base64)
