from __future__ import annotations

import gradio as gr

from echocoach.config import get_echo_coach_config
from echocoach.prompts import MODE_LABELS, TeacherVoiceMode
from echocoach.recording import (
    ServerRecordingError,
    recording_backend_status,
    recording_elapsed_seconds,
    recording_level_warning,
    start_server_recording,
    stop_server_recording,
)
from echocoach.teacher_voice import RAG_MODES, run_teacher_voice_turn
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key, model_status
from gradio_space.research_helpers import (
    list_doc_choices,
    list_session_choices,
    rag_scope_hint,
    refresh_doc_choices,
    refresh_sessions,
)
from echocoach.omni import omni_status_message
from gradio_space.voice_helpers import speak_last_assistant_reply
from inference.factory import get_backend

_config = get_echo_coach_config()
_TURN_MAX = min(15, _config.max_seconds)
_MODE_CHOICES = [(label, key) for key, label in MODE_LABELS.items()]


def _empty_turn() -> tuple:
    return (
        [],
        None,
        "Start recording, speak your question, stop, then click **Send turn**.",
        "",
        {},
    )


def ui_start_recording(max_seconds: int) -> tuple[str, dict, dict]:
    try:
        start_server_recording(int(max_seconds))
    except ServerRecordingError as exc:
        return (
            str(exc),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    return (
        (
            f"Recording… speak now, then click **Stop recording** "
            f"(auto-stops after {int(max_seconds)}s)."
        ),
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


def ui_stop_recording() -> tuple[str | None, str, dict, dict]:
    try:
        elapsed = recording_elapsed_seconds()
        path = stop_server_recording()
        warning = recording_level_warning(path)
    except ServerRecordingError as exc:
        return (
            None,
            str(exc),
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            None,
            f"Recording failed: {exc}",
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    status = f"Recording saved ({elapsed:.1f}s). Click **Send turn** to talk to TeacherVoice."
    if warning:
        status += f" Warning: {warning}"
    return (
        gr.update(value=str(path)),
        status,
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def clear_conversation() -> tuple:
    return _empty_turn()


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
) -> tuple:
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return (
            history,
            None,
            load_error,
            "",
            {},
        )

    if not audio_path:
        return (
            history,
            None,
            "Record or upload audio, then click **Send turn**.",
            "",
            {},
        )

    try:
        result = run_teacher_voice_turn(
            audio_path,
            history,
            mode=mode,
            language=language,
            topic=topic or None,
            asr_preset=asr_preset,
            backend=get_backend(model_key),
            use_rag=use_rag and mode in RAG_MODES,
            session_id=session_id,
            doc_ids=doc_ids,
            max_turn_seconds=_TURN_MAX,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            history,
            None,
            f"TeacherVoice failed: {exc}",
            "",
            {},
        )

    status = f"Turn complete — transcribed {len(result.user_text)} chars, replied in voice."
    if result.voiceout_warning:
        status += f" VoiceOut: {result.voiceout_warning}"

    playback = result.voiceout_first_path or result.voiceout_path
    return (
        result.history,
        playback,
        status,
        f"Trace saved: `{result.trace_path}`",
        result.trace,
    )


def speak_full_reply(history: list, language: str) -> tuple[str | None, str]:
    playback, status = speak_last_assistant_reply(history, language, first_sentence_only=False)
    return playback, status


def speak_quick_reply(history: list, language: str) -> tuple[str | None, str]:
    playback, status = speak_last_assistant_reply(history, language, first_sentence_only=True)
    return playback, status


def build_teacher_voice_tab() -> None:
    lang_choices = _config.language_choices()
    asr_choices = _config.asr_choices()
    default_lang = lang_choices[0][1] if lang_choices else "en"
    default_asr = _config.asr_preset
    mic_status = recording_backend_status()

    omni_note = omni_status_message()
    gr.Markdown(
        f"""
**TeacherVoice** — turn-based voice conversation with a local teacher (not full duplex).

1. Choose a mode → record a short turn (max **{_TURN_MAX}s**) → **Send turn** → hear the reply.
2. **Explain** — tutor any topic. **Lesson coach** — outline and discuss lessons. **Pitch practice** — live speaking tips.
3. For deep pitch analysis (pace charts, filler counts), use the **EchoCoach** tab.

Latency is typically a few seconds per turn on GPU; CPU may take longer.
{omni_note or ""}
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            mode_dd = gr.Dropdown(
                label="Mode",
                choices=_MODE_CHOICES,
                value="explain",
            )
            topic_tb = gr.Textbox(
                label="Topic (Explain / Lesson modes)",
                placeholder="e.g. Photosynthesis for grade 6",
            )
            record_status_md = gr.Markdown(mic_status)
            with gr.Accordion("Record from this computer", open=True):
                record_seconds = gr.Slider(
                    label="Max turn length (seconds)",
                    minimum=3,
                    maximum=_TURN_MAX,
                    value=_TURN_MAX,
                    step=1,
                )
                with gr.Row():
                    record_start_btn = gr.Button("Start recording", variant="secondary")
                    record_stop_btn = gr.Button("Stop recording", variant="stop", interactive=False)
            audio_in = gr.Audio(
                label="Your turn (browser mic or upload)",
                sources=["upload", "microphone"],
                type="filepath",
                format="wav",
            )
            language = gr.Dropdown(label="Language", choices=lang_choices, value=default_lang)
            asr_preset = gr.Dropdown(label="ASR preset", choices=asr_choices, value=default_asr)
            with gr.Accordion("ResearchMind RAG (Explain / Lesson)", open=False):
                use_rag = gr.Checkbox(label="Ground answers in ingested sources", value=False)
                session_dd = gr.Dropdown(
                    label="Session",
                    choices=list_session_choices(),
                    value="",
                )
                refresh_sessions_btn = gr.Button("Refresh sessions", size="sm")
                doc_dd = gr.CheckboxGroup(label="Documents (empty = all in session)", choices=[], value=[])
                rag_hint = gr.Markdown(value=rag_scope_hint("", []))
            with gr.Row():
                send_btn = gr.Button("Send turn", variant="primary")
                clear_btn = gr.Button("Clear conversation", variant="secondary")
            status = gr.Textbox(label="Status", interactive=False, lines=3)
            coach_status = gr.Markdown(model_status(get_active_model_key()))

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=360)
            with gr.Row():
                speak_full_btn = gr.Button("Speak last reply", variant="secondary")
                speak_quick_btn = gr.Button("Speak first sentence", variant="secondary")
            voiceout = gr.Audio(
                label="Teacher reply (auto after Send turn, or use Speak buttons)",
                type="filepath",
                autoplay=True,
            )
            trace_note = gr.Markdown()
            trace_json = gr.JSON(label="Trace")

    record_start_btn.click(
        ui_start_recording,
        inputs=[record_seconds],
        outputs=[status, record_start_btn, record_stop_btn],
    )
    record_stop_btn.click(
        ui_stop_recording,
        outputs=[audio_in, status, record_start_btn, record_stop_btn],
    ).then(
        lambda: recording_backend_status(),
        outputs=[record_status_md],
    )

    refresh_sessions_btn.click(fn=refresh_sessions, inputs=[session_dd], outputs=[session_dd])
    session_dd.change(
        fn=refresh_doc_choices,
        inputs=[session_dd, doc_dd],
        outputs=[doc_dd],
    )
    for trigger in (use_rag, session_dd, doc_dd):
        trigger.change(
            fn=lambda rag_on, sid, docs: (
                rag_scope_hint(sid, docs) if rag_on else "_RAG off — model knowledge only._"
            ),
            inputs=[use_rag, session_dd, doc_dd],
            outputs=[rag_hint],
        )

    send_btn.click(
        send_turn,
        inputs=[
            audio_in,
            chatbot,
            mode_dd,
            language,
            asr_preset,
            topic_tb,
            use_rag,
            session_dd,
            doc_dd,
        ],
        outputs=[chatbot, voiceout, status, trace_note, trace_json],
    )

    clear_btn.click(clear_conversation, outputs=[chatbot, voiceout, status, trace_note, trace_json])

    speak_full_btn.click(
        speak_full_reply,
        inputs=[chatbot, language],
        outputs=[voiceout, status],
    )
    speak_quick_btn.click(
        speak_quick_reply,
        inputs=[chatbot, language],
        outputs=[voiceout, status],
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
