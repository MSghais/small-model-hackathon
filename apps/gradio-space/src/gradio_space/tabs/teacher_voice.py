from __future__ import annotations

import gradio as gr

from echocoach.config import get_echo_coach_config
from echocoach.omni import omni_status_message
from echocoach.prompts import MODE_LABELS, TeacherVoiceMode
from echocoach.teacher_voice import RAG_MODES, run_teacher_voice_turn
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key
from gradio_space.research_helpers import (
    list_session_choices,
    rag_scope_hint,
    refresh_doc_choices,
    refresh_sessions,
)
from gradio_space.ui.components import (
    build_recording_block,
    tab_hero,
    wire_recording_handlers,
)
from gradio_space.voice_helpers import speak_last_assistant_reply
from inference.factory import get_backend

_config = get_echo_coach_config()
_TURN_MAX = min(15, _config.max_seconds)
_MODE_CHOICES = [(label, key) for key, label in MODE_LABELS.items()]


def _empty_turn() -> tuple:
    return (
        [],
        None,
        "Record your question, then click **Send turn**.",
        "",
        {},
    )


def send_turn(
    audio_path: str | None,
    history: list,
    mode: TeacherVoiceMode,
    language: str,
    asr_preset: str,
    topic: str,
    use_rag: bool,
    session_id: str,
    doc_ids: list[str] | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    progress(0, desc="Loading model…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return [], None, load_error, "", {}

    if not audio_path:
        return _empty_turn()

    try:
        progress(0.15, desc="Listening…")
        result = run_teacher_voice_turn(
            audio_path,
            history,
            mode=mode,
            language=language,
            asr_preset=asr_preset,
            topic=topic.strip() or None,
            backend=get_backend(model_key),
            use_rag=use_rag and mode in RAG_MODES,
            session_id=session_id or None,
            doc_ids=doc_ids or None,
            max_turn_seconds=_TURN_MAX,
        )
    except Exception as exc:  # noqa: BLE001
        return [], None, f"TeacherVoice failed: {exc}", "", {}

    progress(1.0, desc="Done")
    status = (
        f"Turn complete — user {result.user_chars} chars, "
        f"teacher {result.assistant_chars} chars."
    )
    if result.voiceout_warning:
        status += f" VoiceOut: {result.voiceout_warning}"

    playback = result.voiceout_path
    if playback:
        playback = str(playback)

    return (
        result.history,
        playback,
        status,
        f"Trace saved: `{result.trace_path}`",
        result.trace,
    )


def clear_conversation() -> tuple:
    return _empty_turn()


def _format_speak_status(status: str) -> str:
    if status.startswith("VoiceOut ready"):
        return f"**{status}**"
    return f"**VoiceOut:** {status}"


def speak_full_reply(history: list, language: str) -> tuple[str | None, str, str]:
    playback, status = speak_last_assistant_reply(history, language, first_sentence_only=False)
    return playback, status, _format_speak_status(status)


def speak_quick_reply(history: list, language: str) -> tuple[str | None, str, str]:
    playback, status = speak_last_assistant_reply(history, language, first_sentence_only=True)
    return playback, status, _format_speak_status(status)


def _topic_visible(mode: str) -> dict:
    return gr.update(visible=mode in ("explain", "lesson"))


def _rag_visible(mode: str) -> dict:
    return gr.update(visible=mode in RAG_MODES)


def build_teacher_voice_tab() -> None:
    lang_choices = _config.language_choices()
    asr_choices = _config.asr_choices()
    default_lang = lang_choices[0][1] if lang_choices else "en"
    default_asr = _config.asr_preset
    omni_note = omni_status_message()

    tab_hero(
        "Turn-based voice conversation with a local teacher — record, send, hear the reply.",
        steps=["Mode", "Record", "Send", "Listen"],
        active_step=0,
    )
    if omni_note:
        gr.Markdown(omni_note)

    with gr.Row():
        with gr.Column(scale=1):
            mode_dd = gr.Radio(
                label="Mode",
                choices=_MODE_CHOICES,
                value="explain",
                elem_classes=["mode-cards"],
            )
            topic_tb = gr.Textbox(
                label="Topic (Explain / Lesson modes)",
                placeholder="e.g. Photosynthesis for grade 6",
            )

            use_rag = gr.Checkbox(
                label="Use my ResearchMind sources",
                value=False,
            )
            with gr.Row():
                session_dd = gr.Dropdown(
                    label="Session",
                    choices=list_session_choices(),
                    value="",
                    scale=4,
                )
                refresh_sessions_btn = gr.Button("↻", size="sm", scale=0, min_width=40)
            doc_dd = gr.CheckboxGroup(
                label="Documents (empty = all in session)",
                choices=[],
                value=[],
            )
            rag_hint = gr.Markdown(value="_RAG off — model knowledge only._")

            rec = build_recording_block(
                max_seconds=_TURN_MAX,
                default_seconds=_TURN_MAX,
                lang_choices=lang_choices,
                asr_choices=asr_choices,
                default_lang=default_lang,
                default_asr=default_asr,
                audio_label="Your turn (mic or upload)",
                advanced_open=True,
            )
            status = gr.Textbox(label="Status", interactive=False, lines=3)
            rec.status = status

            with gr.Row():
                send_btn = gr.Button("Send turn", variant="primary", elem_classes=["primary-cta"])
                clear_btn = gr.Button("Clear", variant="secondary")

            wire_recording_handlers(
                rec,
                stop_next_action="Click **Send turn**.",
                status_output=status,
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=400)
            voiceout = gr.Audio(
                label="Teacher reply",
                type="filepath",
                autoplay=True,
            )

    with gr.Accordion("Advanced & debug", open=False):
        with gr.Row():
            speak_full_btn = gr.Button("Speak last reply", variant="secondary")
            speak_quick_btn = gr.Button("Speak first sentence", variant="secondary")
        speak_status = gr.Markdown(
            value="_VoiceOut auto-plays after each turn. Use Speak buttons to replay._"
        )
        trace_note = gr.Markdown()
        trace_json = gr.JSON(label="Trace")

    mode_dd.change(fn=_topic_visible, inputs=[mode_dd], outputs=[topic_tb])
    mode_dd.change(fn=_rag_visible, inputs=[mode_dd], outputs=[use_rag])

    def _update_rag_hint(rag_on: bool, sid: str, docs: list[str] | None) -> str:
        if not rag_on:
            return "_RAG off — model knowledge only._"
        return rag_scope_hint(sid, docs)

    refresh_sessions_btn.click(fn=refresh_sessions, inputs=[session_dd], outputs=[session_dd])
    session_dd.change(fn=refresh_doc_choices, inputs=[session_dd, doc_dd], outputs=[doc_dd])
    for trigger in (use_rag, session_dd, doc_dd):
        trigger.change(
            fn=_update_rag_hint,
            inputs=[use_rag, session_dd, doc_dd],
            outputs=[rag_hint],
        )

    send_btn.click(
        send_turn,
        inputs=[
            rec.audio_in,
            chatbot,
            mode_dd,
            rec.language,
            rec.asr_preset,
            topic_tb,
            use_rag,
            session_dd,
            doc_dd,
        ],
        outputs=[chatbot, voiceout, status, trace_note, trace_json],
    )

    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, voiceout, status, trace_note, trace_json],
    )

    speak_full_btn.click(
        speak_full_reply,
        inputs=[chatbot, rec.language],
        outputs=[voiceout, status, speak_status],
    )
    speak_quick_btn.click(
        speak_quick_reply,
        inputs=[chatbot, rec.language],
        outputs=[voiceout, status, speak_status],
    )


def teacher_voice_allowed_paths() -> list[str]:
    paths: list[str] = []
    if _config.presets_path:
        paths.append(str(_config.presets_path.parent))
    from echocoach.config import outputs_dir

    paths.append(str(outputs_dir()))
    paths.append(str(outputs_dir() / "recordings"))
    paths.append(str(outputs_dir() / "teacher_voice"))
    return paths
