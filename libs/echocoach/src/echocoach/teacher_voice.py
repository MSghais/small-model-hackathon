"""Turn-based TeacherVoice conversation pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.trace import TraceRecorder
from inference.base import InferenceBackend
from inference.response_clean import strip_reasoning_output
from researchmind.citations import format_context_block, format_references
from researchmind.config import get_config as get_researchmind_config
from researchmind.ingest import IngestPipeline
from researchmind.retrieve import retrieve

from echocoach.asr.factory import get_asr_backend
from echocoach.audio_io import clamp_duration, load_audio_mono_16k, write_wav_temp
from echocoach.config import get_echo_coach_config, outputs_dir
from echocoach.prompts import TeacherVoiceMode, system_prompt_for_mode, topic_context_block
from echocoach.voiceout import strip_references_for_tts, synthesize_voice_reply

RAG_MODES: frozenset[TeacherVoiceMode] = frozenset({"explain", "lesson"})


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
    trace_path: str
    trace: dict[str, Any] = field(default_factory=dict)


def append_chat_turn(
    history: list,
    user_text: str,
    assistant_text: str,
) -> list[dict[str, str]]:
    """Append a turn in Gradio 5 messages format."""
    updated: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict) and "role" in item and "content" in item:
            updated.append({"role": str(item["role"]), "content": str(item["content"])})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            updated.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                updated.append({"role": "assistant", "content": str(assistant_msg)})
    updated.append({"role": "user", "content": user_text})
    updated.append({"role": "assistant", "content": assistant_text})
    return updated


def history_to_messages(history: list) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                messages.append({"role": "assistant", "content": str(assistant_msg)})
    return messages


def fetch_rag_context(
    question: str,
    *,
    session_id: str,
    doc_ids: list[str] | None,
) -> RagContext | None:
    store = IngestPipeline().store
    cfg = get_researchmind_config()
    scope_session = session_id if session_id and not doc_ids else None
    scope_docs = doc_ids if doc_ids else None
    chunks = retrieve(
        question,
        store,
        config=cfg,
        session_id=scope_session,
        doc_ids=scope_docs,
    )
    if not chunks:
        if doc_ids:
            warning = "No passages in selected documents for this question."
        elif session_id:
            warning = "No indexed sources in this session yet."
        else:
            warning = "No indexed sources in the corpus yet."
        return RagContext(context_block="", references_markdown="", chunk_count=0, warning=warning)

    context_block, citations = format_context_block(chunks)
    refs = format_references(citations)
    return RagContext(
        context_block=context_block,
        references_markdown=refs,
        chunk_count=len(chunks),
    )


def build_teacher_messages(
    *,
    mode: TeacherVoiceMode,
    history: list,
    user_text: str,
    topic: str | None = None,
    rag: RagContext | None = None,
) -> list[dict[str, str]]:
    system = system_prompt_for_mode(mode)
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
    user_parts.append(user_text.strip())
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})
    return messages


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
) -> TeacherVoiceTurnResult:
    if not audio_path:
        raise ValueError("No audio recording provided.")

    config = get_echo_coach_config()
    asr_key = asr_preset or config.asr_preset
    tts_key = tts_preset or config.tts_preset
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
        system = system_prompt_for_mode(mode)
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
            new_history = append_chat_turn(history, omni_user, omni_reply)
            trace_path = trace.save()
            return TeacherVoiceTurnResult(
                user_text=omni_user,
                assistant_text=omni_reply,
                history=new_history,
                voiceout_path=omni_wav_or_note,
                voiceout_first_path=omni_wav_or_note,
                voiceout_warning=None,
                rag_references=None,
                trace_path=str(trace_path),
                trace=trace.to_dict(),
            )
        if omni_wav_or_note:
            trace.log_note("omni_fallback", message=omni_wav_or_note)

    rag: RagContext | None = None
    rag_refs: str | None = None
    if use_rag and mode in RAG_MODES:
        sid = session_id
        if not sid:
            sid = IngestPipeline().store.create_session().id
        rag = fetch_rag_context(user_text, session_id=sid, doc_ids=doc_ids)
        if rag:
            trace.log_note(
                "rag_retrieve",
                chunks=rag.chunk_count,
                warning=rag.warning,
            )
            if rag.references_markdown:
                rag_refs = rag.references_markdown

    messages = build_teacher_messages(
        mode=mode,
        history=history,
        user_text=user_text,
        topic=topic,
        rag=rag if rag and rag.context_block else None,
    )
    raw_reply = backend.chat(messages, max_tokens=512, temperature=0.5)
    assistant_text = strip_reasoning_output(raw_reply).strip()
    trace.log_llm(messages[-1]["content"], raw_reply)

    if rag_refs:
        assistant_text = f"{assistant_text}\n\n{rag_refs}"

    voiceout_path, voiceout_first, voiceout_warning = synthesize_voice_reply(
        strip_references_for_tts(assistant_text),
        language=language,
        tts_preset=tts_key,
        chunk_first=True,
        out_subdir="teacher_voice",
    )
    if voiceout_path:
        trace.set_artifact(voiceout_path)

    new_history = append_chat_turn(history, user_text, assistant_text)

    trace_path = trace.save()
    return TeacherVoiceTurnResult(
        user_text=user_text,
        assistant_text=assistant_text,
        history=new_history,
        voiceout_path=voiceout_path,
        voiceout_first_path=voiceout_first,
        voiceout_warning=voiceout_warning,
        rag_references=rag_refs,
        trace_path=str(trace_path),
        trace=trace.to_dict(),
    )
