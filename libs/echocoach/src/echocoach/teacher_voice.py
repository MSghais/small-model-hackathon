"""Turn-based TeacherVoice conversation pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.runner import AgentRunner
from agent.trace import TraceRecorder
from inference.base import InferenceBackend
from inference.response_clean import (
    needs_teacher_compaction,
    reply_ends_complete_sentence,
    strip_reasoning_output,
)
from researchmind.ingest import IngestPipeline
from researchmind.scope import retrieval_query

from echocoach.asr.factory import get_asr_backend
from echocoach.audio_io import clamp_duration, load_audio_mono_16k, write_wav_temp
from echocoach.config import get_echo_coach_config, outputs_dir
from echocoach.prompts import TeacherVoiceMode, system_prompt_for_mode, topic_context_block
from echocoach.voiceout import extract_message_text, strip_references_for_tts, synthesize_voice_reply

RAG_MODES: frozenset[TeacherVoiceMode] = frozenset({"explain", "lesson"})
_VOICE_USER_SUFFIX = (
    "Reply now in 2-4 complete spoken sentences only. "
    "No planning, outlines, sentence labels, or meta commentary."
)


@dataclass
class RagContext:
    context_block: str
    references_markdown: str
    chunk_count: int
    warning: str | None = None


@dataclass
class TeacherVoiceTurnResult:
    user_text: str
    assistant_text: str
    history: list[dict[str, str]]
    voiceout_path: str | None
    voiceout_first_path: str | None
    voiceout_warning: str | None
    rag_references: str | None
    rag_status: str | None
    trace_path: str
    trace: dict[str, Any] = field(default_factory=dict)


def _assistant_content_for_chat(
    display_text: str,
    *,
    voice_path: str | None = None,
) -> str | list:
    if voice_path:
        return [display_text, {"path": voice_path}]
    return display_text


def append_chat_turn(
    history: list,
    user_text: str,
    assistant_text: str,
    *,
    assistant_display: str | None = None,
    voice_path: str | None = None,
) -> list[dict[str, Any]]:
    """Append a turn in Gradio 5 messages format."""
    updated: list[dict[str, Any]] = []
    for item in history or []:
        if isinstance(item, dict) and "role" in item and "content" in item:
            updated.append({"role": str(item["role"]), "content": item["content"]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            updated.append({"role": "user", "content": extract_message_text(user_msg)})
            if assistant_msg:
                updated.append(
                    {"role": "assistant", "content": extract_message_text(assistant_msg)}
                )
    updated.append({"role": "user", "content": user_text})
    display_text = assistant_display if assistant_display is not None else assistant_text
    updated.append(
        {
            "role": "assistant",
            "content": _assistant_content_for_chat(display_text, voice_path=voice_path),
        }
    )
    return updated


def _message_text_for_llm(role: str, content: object) -> str:
    text = extract_message_text(content)
    if role == "assistant":
        return strip_reasoning_output(text)
    return text


def history_to_messages(history: list) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            role = str(item["role"])
            messages.append(
                {"role": role, "content": _message_text_for_llm(role, item["content"])}
            )
        else:
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": extract_message_text(user_msg)})
            if assistant_msg:
                messages.append(
                    {
                        "role": "assistant",
                        "content": strip_reasoning_output(extract_message_text(assistant_msg)),
                    }
                )
    return messages


def fetch_rag_context(
    question: str,
    *,
    session_id: str,
    doc_ids: list[str] | None,
) -> RagContext | None:
    """Retrieve passages for diagnostics/tests. Production turns use AgentRunner."""
    from researchmind.config import get_config as get_researchmind_config
    from researchmind.ingest import IngestPipeline
    from researchmind.citations import format_context_block, format_references
    from researchmind.retrieve import retrieve
    from researchmind.scope import rag_scope_warning, resolve_retrieve_scope

    store = IngestPipeline().store
    cfg = get_researchmind_config()
    scope_session, scope_docs = resolve_retrieve_scope(session_id or None, doc_ids)
    chunks = retrieve(
        question,
        store,
        config=cfg,
        session_id=scope_session,
        doc_ids=scope_docs,
    )
    if not chunks:
        warning = rag_scope_warning(session_id=session_id or None, doc_ids=doc_ids)
        return RagContext(context_block="", references_markdown="", chunk_count=0, warning=warning)

    context_block, citations = format_context_block(chunks)
    refs = format_references(citations)
    return RagContext(
        context_block=context_block,
        references_markdown=refs,
        chunk_count=len(chunks),
    )


def _rag_turn_via_agent(
    user_text: str,
    *,
    mode: TeacherVoiceMode,
    topic: str | None,
    session_id: str,
    doc_ids: list[str] | None,
    model_key: str,
    backend: InferenceBackend,
    trace: TraceRecorder,
    language: str = "en",
) -> tuple[str, str | None, str | None, str]:
    """Grounded answer via ResearchMind harness. Returns text, refs, status, display."""
    query = retrieval_query(user_text, topic=topic)
    trace.log_note("rag_query", query=query, session_id=session_id or None, doc_ids=doc_ids or [])

    result = AgentRunner().run_researchmind_chat(
        question=query,
        session_id=session_id or "",
        doc_ids=doc_ids,
        model_key=model_key,
        backend=backend,
    )

    citation_count = len(result.citations)
    if citation_count:
        rag_status = (
            f"Retrieved passages from **{citation_count}** source(s) "
            f"for grounded answer."
        )
    else:
        rag_status = (
            "_No indexed passages matched this question — reply uses model guidance only._"
        )
    trace.log_note(
        "rag_retrieve",
        citations=citation_count,
        session_id=session_id or None,
        doc_ids=doc_ids or [],
        research_trace=result.trace_path,
    )

    raw_answer = strip_references_for_tts(result.answer.strip())
    assistant_text, display_reply = _finalize_voice_reply(
        raw_answer,
        mode=mode,
        backend=backend,
        trace=trace,
        language=language,
    )
    rag_refs = result.references_markdown or None
    return assistant_text, rag_refs, rag_status, display_reply


def _indexed_scope_available(session_id: str, doc_ids: list[str] | None) -> bool:
    store = IngestPipeline().store
    if doc_ids:
        return True
    if session_id:
        return bool(store.list_documents(session_id=session_id))
    return bool(store.list_documents())


def _rag_off_status(session_id: str, doc_ids: list[str] | None) -> str | None:
    if _indexed_scope_available(session_id, doc_ids):
        return (
            "_Sources are indexed but RAG is off — enable **Answer from my indexed sources** "
            "for cited, source-grounded replies._"
        )
    return (
        "_No sources used. Set a focus topic, **Discover/Auto-ingest** sources, then enable "
        "**Answer from my indexed sources** for citations._"
    )


def _compact_teacher_reply(
    raw_reply: str,
    *,
    mode: TeacherVoiceMode,
    backend: InferenceBackend,
    trace: TraceRecorder,
    language: str = "en",
) -> str:
    seed = strip_reasoning_output(raw_reply).strip() or raw_reply.strip()[:1200]
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt_for_mode(mode, language=language)}\n\n"
                "Rewrite the draft below into ONLY 2-4 spoken sentences for voice playback. "
                "Keep any [n] citations. No planning or labels."
            ),
        },
        {"role": "user", "content": seed},
    ]
    compact_raw = backend.chat(messages, max_tokens=220, temperature=0.2)
    trace.log_note("teacher_compact")
    trace.log_llm(messages[-1]["content"], compact_raw)
    compact = strip_reasoning_output(compact_raw).strip()
    return compact or seed


def _finalize_voice_reply(
    raw_reply: str,
    *,
    mode: TeacherVoiceMode,
    backend: InferenceBackend,
    trace: TraceRecorder,
    language: str = "en",
) -> tuple[str, str]:
    """Normalize model output into a complete spoken reply and chat display text."""
    assistant_text = strip_reasoning_output(raw_reply).strip()
    needs_fix = (
        not assistant_text
        or needs_teacher_compaction(raw_reply)
        or needs_teacher_compaction(assistant_text)
        or not reply_ends_complete_sentence(assistant_text)
    )
    if needs_fix:
        assistant_text = _compact_teacher_reply(
            raw_reply,
            mode=mode,
            backend=backend,
            trace=trace,
            language=language,
        )
    if not reply_ends_complete_sentence(assistant_text):
        assistant_text = _compact_teacher_reply(
            assistant_text or raw_reply,
            mode=mode,
            backend=backend,
            trace=trace,
            language=language,
        )
    return assistant_text, assistant_text


def build_teacher_messages(
    *,
    mode: TeacherVoiceMode,
    history: list,
    user_text: str,
    topic: str | None = None,
    rag: RagContext | None = None,
    language: str = "en",
) -> list[dict[str, str]]:
    system = system_prompt_for_mode(mode, language=language)
    topic_line = topic_context_block(topic, mode)
    if topic_line:
        system = f"{system}\n\n{topic_line}"

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history_to_messages(history))

    user_parts: list[str] = []
    if rag and rag.context_block:
        user_parts.append(
            "Use these source excerpts as grounding. Cite with [1], [2], etc. when relevant.\n\n"
            f"{rag.context_block}"
        )
    user_parts.append(f"{user_text.strip()}\n\n{_VOICE_USER_SUFFIX}")
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})
    return messages


def _generate_teacher_reply(
    user_text: str,
    history: list,
    *,
    trace: TraceRecorder,
    mode: TeacherVoiceMode,
    language: str,
    topic: str | None,
    model_key: str,
    backend: InferenceBackend,
    use_rag: bool,
    session_id: str,
    doc_ids: list[str] | None,
    tts_key: str,
    auto_voiceout: bool = True,
) -> TeacherVoiceTurnResult:
    rag_refs: str | None = None
    rag_status: str | None = None

    if use_rag and mode in RAG_MODES:
        assistant_text, rag_refs, rag_status, display_reply = _rag_turn_via_agent(
            user_text,
            mode=mode,
            topic=topic,
            session_id=session_id,
            doc_ids=doc_ids,
            model_key=model_key,
            backend=backend,
            trace=trace,
            language=language,
        )
    else:
        messages = build_teacher_messages(
            mode=mode,
            history=history,
            user_text=user_text,
            topic=topic,
            language=language,
        )
        raw_reply = backend.chat(messages, max_tokens=512, temperature=0.2)
        assistant_text, display_reply = _finalize_voice_reply(
            raw_reply,
            mode=mode,
            backend=backend,
            trace=trace,
            language=language,
        )
        trace.log_llm(messages[-1]["content"], raw_reply)
        if mode in RAG_MODES:
            rag_status = _rag_off_status(session_id, doc_ids)

    voiceout_path: str | None = None
    voiceout_first: str | None = None
    voiceout_warning: str | None = None
    if auto_voiceout:
        voiceout_path, voiceout_first, voiceout_warning = synthesize_voice_reply(
            strip_references_for_tts(assistant_text),
            language=language,
            tts_preset=tts_key,
            chunk_first=True,
            out_subdir="teacher_voice",
        )
        if voiceout_path:
            trace.set_artifact(voiceout_path)

    new_history = append_chat_turn(
        history,
        user_text,
        assistant_text,
        assistant_display=display_reply,
        voice_path=voiceout_path,
    )

    trace_path = trace.save()
    return TeacherVoiceTurnResult(
        user_text=user_text,
        assistant_text=assistant_text,
        history=new_history,
        voiceout_path=voiceout_path,
        voiceout_first_path=voiceout_first,
        voiceout_warning=voiceout_warning,
        rag_references=rag_refs,
        rag_status=rag_status,
        trace_path=str(trace_path),
        trace=trace.to_dict(),
    )


def run_teacher_voice_text_turn(
    user_text: str,
    history: list,
    *,
    mode: TeacherVoiceMode = "explain",
    language: str = "en",
    topic: str | None = None,
    tts_preset: str | None = None,
    coach_model: str | None = None,
    backend: InferenceBackend,
    use_rag: bool = False,
    session_id: str = "",
    doc_ids: list[str] | None = None,
    auto_voiceout: bool = True,
) -> TeacherVoiceTurnResult:
    """Process a typed user message (skips ASR)."""
    user_text = user_text.strip()
    if not user_text:
        raise ValueError("Type a message to send.")

    config = get_echo_coach_config()
    tts_key = tts_preset or config.realtime_tts_preset or config.tts_preset
    model_key = coach_model or config.coach_model
    run_id = uuid.uuid4().hex[:12]

    trace = TraceRecorder(
        skill="teacher-voice",
        model=model_key,
        user_input={
            "mode": mode,
            "language": language,
            "topic": topic,
            "input_type": "text",
            "user_text": user_text,
            "tts_preset": tts_key,
            "use_rag": use_rag,
            "session_id": session_id,
            "doc_ids": doc_ids or [],
        },
        run_id=run_id,
    )
    trace.log_note("text_input", chars=len(user_text))

    return _generate_teacher_reply(
        user_text,
        history,
        trace=trace,
        mode=mode,
        language=language,
        topic=topic,
        model_key=model_key,
        backend=backend,
        use_rag=use_rag,
        session_id=session_id,
        doc_ids=doc_ids,
        tts_key=tts_key,
        auto_voiceout=auto_voiceout,
    )


def run_teacher_voice_turn(
    audio_path: str,
    history: list,
    *,
    mode: TeacherVoiceMode = "explain",
    language: str = "en",
    topic: str | None = None,
    asr_preset: str | None = None,
    tts_preset: str | None = None,
    coach_model: str | None = None,
    backend: InferenceBackend,
    use_rag: bool = False,
    session_id: str = "",
    doc_ids: list[str] | None = None,
    max_turn_seconds: int | None = None,
    auto_voiceout: bool = True,
) -> TeacherVoiceTurnResult:
    if not audio_path:
        raise ValueError("No audio recording provided.")

    config = get_echo_coach_config()
    asr_key = asr_preset or config.asr_preset
    tts_key = tts_preset or config.realtime_tts_preset or config.tts_preset
    model_key = coach_model or config.coach_model
    turn_cap = max_turn_seconds or min(15, config.max_seconds)
    run_id = uuid.uuid4().hex[:12]
    out_base = outputs_dir()

    trace = TraceRecorder(
        skill="teacher-voice",
        model=model_key,
        user_input={
            "mode": mode,
            "language": language,
            "topic": topic,
            "asr_preset": asr_key,
            "tts_preset": tts_key,
            "use_rag": use_rag,
            "session_id": session_id,
            "doc_ids": doc_ids or [],
            "audio_path": audio_path,
        },
        run_id=run_id,
    )

    audio, duration = load_audio_mono_16k(audio_path)
    audio = clamp_duration(audio, turn_cap)
    clipped_path = write_wav_temp(audio, out_base / "clips", stem=f"tv_{run_id}")
    trace.log_note("audio_loaded", duration_seconds=duration, path=str(clipped_path))

    asr = get_asr_backend(asr_key)
    user_text = asr.transcribe(str(clipped_path), language=language).strip()
    if not user_text:
        raise ValueError("Could not transcribe speech — try speaking louder or uploading clearer audio.")
    trace.log_note("asr_complete", preset=asr_key, chars=len(user_text))

    from echocoach.omni import is_omni_profile, try_omni_turn

    if is_omni_profile():
        system = system_prompt_for_mode(mode, language=language)
        topic_line = topic_context_block(topic, mode)
        if topic_line:
            system = f"{system}\n\n{topic_line}"
        omni_user, omni_reply, omni_wav_or_note = try_omni_turn(
            str(clipped_path),
            language=language,
            history=history,
            system_prompt=system,
        )
        if omni_wav_or_note and omni_user and omni_reply and Path(omni_wav_or_note).is_file():
            trace.log_note("omni_turn", path=omni_wav_or_note)
            new_history = append_chat_turn(
                history,
                omni_user,
                omni_reply,
                voice_path=omni_wav_or_note,
            )
            trace_path = trace.save()
            return TeacherVoiceTurnResult(
                user_text=omni_user,
                assistant_text=omni_reply,
                history=new_history,
                voiceout_path=omni_wav_or_note,
                voiceout_first_path=omni_wav_or_note,
                voiceout_warning=None,
                rag_references=None,
                rag_status=None,
                trace_path=str(trace_path),
                trace=trace.to_dict(),
            )
        if omni_wav_or_note:
            trace.log_note("omni_fallback", message=omni_wav_or_note)

    return _generate_teacher_reply(
        user_text,
        history,
        trace=trace,
        mode=mode,
        language=language,
        topic=topic,
        model_key=model_key,
        backend=backend,
        use_rag=use_rag,
        session_id=session_id,
        doc_ids=doc_ids,
        tts_key=tts_key,
        auto_voiceout=auto_voiceout,
    )
