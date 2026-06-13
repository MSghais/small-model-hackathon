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
    build_advanced_panel,
    build_recording_block,
    empty_state,
    wire_recording_handlers,
)
from gradio_space.voice_helpers import speak_last_assistant_reply
from inference.factory import get_backend

_config = get_echo_coach_config()
_TURN_MAX = min(15, _config.max_seconds)
_MODE_CHOICES = [(label, key) for key, label in MODE_LABELS.items()]


def _conversation_visibility(history: list | None) -> tuple[dict, dict]:
    has_messages = bool(history)
    return (
        gr.update(visible=not has_messages),
        gr.update(visible=has_messages),
    )


def _voiceout_update(path: str | None) -> dict:
    return gr.update(value=path, visible=bool(path))


def _empty_turn() -> tuple:
    return (
        [],
        _voiceout_update(None),
        "_Record a question, then click **Send turn**._",
        "",
        {},
        *_conversation_visibility([]),
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
        return (
            history or [],
            _voiceout_update(None),
            load_error,
            "",
            {},
            *_conversation_visibility(history),
        )

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
        return (
            history or [],
            _voiceout_update(None),
            f"**TeacherVoice failed:** {exc}",
            "",
            {},
            *_conversation_visibility(history),
        )

    progress(1.0, desc="Done")
    status = (
        f"**Turn complete** — you spoke {len(result.user_text)} chars, "
        f"teacher replied with {len(result.assistant_text)} chars."
    )
    if result.voiceout_warning:
        status += f" VoiceOut note: {result.voiceout_warning}"

    playback = str(result.voiceout_path) if result.voiceout_path else None

    return (
        result.history,
        _voiceout_update(playback),
        status,
        f"Trace saved: `{result.trace_path}`",
        result.trace,
        *_conversation_visibility(result.history),
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


def _update_rag_hint(rag_on: bool, sid: str, docs: list[str] | None) -> str:
    if not rag_on:
        return "_Using model knowledge only — enable ResearchMind sources below to cite your library._"
    return rag_scope_hint(sid, docs)


def _on_mode_change(mode: str) -> tuple:
    topic_mode = mode in ("explain", "lesson")
    rag_mode = mode in RAG_MODES
    if mode == "lesson":
        topic_up = gr.update(
            visible=topic_mode,
            label="What lesson are we planning?",
            placeholder="e.g. Photosynthesis for grade 6",
        )
    elif mode == "explain":
        topic_up = gr.update(
            visible=topic_mode,
            label="What should the teacher explain?",
            placeholder="e.g. How photosynthesis works",
        )
    else:
        topic_up = gr.update(visible=False, value="")
    rag_acc = gr.update(visible=rag_mode)
    use_rag = gr.update(value=False) if not rag_mode else gr.update()
    return topic_up, rag_acc, use_rag


def build_teacher_voice_tab() -> None:
    lang_choices = _config.language_choices()
    asr_choices = _config.asr_choices()
    default_lang = lang_choices[0][1] if lang_choices else "en"
    default_asr = _config.asr_preset
    omni_note = omni_status_message()

    gr.Markdown("### TeacherVoice", elem_classes=["form-tab-heading"])
    gr.HTML(
        '<p class="tab-subtitle">'
        "Pick a mode, record your question, and hear a spoken reply from your local teacher."
        "</p>"
    )
    if omni_note:
        gr.Markdown(omni_note, elem_classes=["form-status"])
    gr.HTML(
        '<p class="cross-link">Want charts and filler analysis? Use '
        "<strong>EchoCoach</strong> for pitch feedback.</p>"
    )

    with gr.Row(elem_classes=["tv-workflow-columns"]):
        with gr.Column(scale=1, elem_classes=["tv-input-col"]):
            gr.HTML('<p class="form-section-label">Step 1 · Choose mode & speak</p>')

            mode_dd = gr.Radio(
                label="How do you want to practice?",
                choices=_MODE_CHOICES,
                value="explain",
                elem_classes=["mode-cards"],
            )

            topic_tb = gr.Textbox(
                label="What should the teacher explain?",
                placeholder="e.g. How photosynthesis works",
                lines=2,
                max_lines=3,
                elem_classes=["form-topic-input"],
            )

            with gr.Accordion(
                "ResearchMind sources (optional)",
                open=False,
                visible=True,
                elem_classes=["form-optional-accordion"],
            ) as rag_acc:
                use_rag = gr.Checkbox(
                    label="Answer from my indexed sources (with citations)",
                    value=False,
                )
                with gr.Row(elem_classes=["form-secondary"]):
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
                rag_hint = gr.Markdown(
                    value="_Using model knowledge only — enable ResearchMind sources above to cite your library._",
                    elem_classes=["form-status"],
                )

            with gr.Column(elem_classes=["form-primary"]):
                rec = build_recording_block(
                    max_seconds=_TURN_MAX,
                    default_seconds=_TURN_MAX,
                    lang_choices=lang_choices,
                    asr_choices=asr_choices,
                    default_lang=default_lang,
                    default_asr=default_asr,
                    audio_label="Your turn (mic or upload, up to 15s)",
                    compact=True,
                )

            status = gr.Markdown(
                value="_Record or upload audio, then send._",
                elem_classes=["form-status"],
            )
            rec.status = status

            with gr.Row(elem_classes=["form-cta-row"]):
                send_btn = gr.Button(
                    "Send turn",
                    variant="primary",
                    elem_classes=["primary-cta"],
                )
            clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

            wire_recording_handlers(
                rec,
                stop_next_action="Click **Send turn**.",
                status_output=status,
            )

            with gr.Accordion(
                "Replay teacher audio",
                open=False,
                elem_classes=["form-optional-accordion"],
            ):
                with gr.Row(elem_classes=["tv-replay-row"]):
                    speak_full_btn = gr.Button("Speak full reply", variant="secondary", size="sm")
                    speak_quick_btn = gr.Button("Speak first sentence", variant="secondary", size="sm")
                speak_status = gr.Markdown(
                    value="_VoiceOut auto-plays after each turn. Use replay if you missed it._",
                    elem_classes=["form-status"],
                )

            advanced = build_advanced_panel(use_json=True)

        with gr.Column(scale=2, elem_classes=["tv-results-col"]):
            gr.HTML('<p class="form-section-label">Step 2 · Conversation</p>')

            chat_empty = gr.HTML(
                value=empty_state(
                    "Your back-and-forth with the teacher will show here. "
                    "Choose a mode on the left, record a question, and click **Send turn**."
                )
            )

            with gr.Column(visible=False) as chat_panel:
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=360,
                    placeholder="Ask anything — the teacher replies in text and spoken audio.",
                )
                voiceout = gr.Audio(
                    label="Teacher reply (auto-plays)",
                    type="filepath",
                    autoplay=True,
                    visible=False,
                )

    mode_dd.change(
        fn=_on_mode_change,
        inputs=[mode_dd],
        outputs=[topic_tb, rag_acc, use_rag],
    ).then(
        fn=_update_rag_hint,
        inputs=[use_rag, session_dd, doc_dd],
        outputs=[rag_hint],
    )

    refresh_sessions_btn.click(fn=refresh_sessions, inputs=[session_dd], outputs=[session_dd])
    session_dd.change(fn=refresh_doc_choices, inputs=[session_dd, doc_dd], outputs=[doc_dd])
    for trigger in (use_rag, session_dd, doc_dd):
        trigger.change(
            fn=_update_rag_hint,
            inputs=[use_rag, session_dd, doc_dd],
            outputs=[rag_hint],
        )

    turn_outputs = [
        chatbot,
        voiceout,
        status,
        advanced.trace_summary,
        advanced.trace_box,
        chat_empty,
        chat_panel,
    ]

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
        outputs=turn_outputs,
    )

    clear_btn.click(clear_conversation, outputs=turn_outputs)

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
