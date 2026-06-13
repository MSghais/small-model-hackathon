from __future__ import annotations

import gradio as gr

from echocoach.config import get_echo_coach_config
from echocoach.omni import omni_status_message
from echocoach.prompts import MODE_LABELS, TeacherVoiceMode
from echocoach.teacher_voice import RAG_MODES, run_teacher_voice_text_turn, run_teacher_voice_turn
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key
from gradio_space.research_helpers import (
    list_session_choices,
    memory_summary,
    rag_scope_hint,
    refresh_doc_choices,
    refresh_sessions,
    resolve_doc_ids,
    resolve_session,
    resolve_topic,
    trace_as_dict,
)
from gradio_space.tabs.research_mind import (
    auto_search_ingest,
    discover_sources,
    ingest_selected,
)
from gradio_space.ui.components import (
    build_advanced_panel,
    build_recording_block,
    DOC_CHOICE_LIST_CLASSES,
    wire_recording_handlers,
    WorkspaceWidgets,
)
from gradio_space.voice_helpers import speak_last_assistant_reply
from inference.factory import get_backend

_config = get_echo_coach_config()
_TURN_MAX = min(15, _config.max_seconds)
_MODE_CHOICES = [(label, key) for key, label in MODE_LABELS.items()]
_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"
_REASONING_TAGS = [
    (_THINK_OPEN, _THINK_CLOSE),
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
]


def _empty_turn() -> tuple:
    return (
        [],
        "_Type a message or record audio, then send._",
        "",
        {},
        "",
    )


def _turn_result(result) -> tuple:
    status = (
        f"**Turn complete** — you sent {len(result.user_text)} chars, "
        f"teacher replied with {len(result.assistant_text)} chars."
    )
    if result.rag_status:
        status += f"\n\n{result.rag_status}"
    if result.voiceout_warning:
        first_line = result.voiceout_warning.split("\n", 1)[0].strip()
        if len(first_line) > 120:
            first_line = first_line[:117] + "…"
        status += f" VoiceOut note: {first_line} _(details in Advanced)_"

    return (
        result.history,
        status,
        f"Trace saved: `{result.trace_path}`",
        trace_as_dict(result.trace),
        "",
    )


def _turn_error(history: list | None, message: str) -> tuple:
    return (
        history or [],
        f"**TeacherVoice failed:** {message}",
        "",
        {},
        gr.update(),
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
    workspace_topic: str,
    workspace_session: str,
    workspace_doc_ids: list[str] | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    topic = resolve_topic(topic, workspace_topic)
    session_id = resolve_session(session_id, workspace_session)
    doc_ids = resolve_doc_ids(doc_ids, workspace_doc_ids)
    progress(0, desc="Loading model…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return _turn_error(history, load_error)

    if not audio_path:
        return (
            history or [],
            "_Record or upload audio, then click **Send voice turn**._",
            "",
            {},
            gr.update(),
        )

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
        return _turn_error(history, str(exc))

    progress(1.0, desc="Done")
    return _turn_result(result)


def send_text_turn(
    message: str,
    history: list,
    mode: TeacherVoiceMode,
    language: str,
    topic: str,
    use_rag: bool,
    session_id: str,
    doc_ids: list[str] | None,
    workspace_topic: str,
    workspace_session: str,
    workspace_doc_ids: list[str] | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    topic = resolve_topic(topic, workspace_topic)
    session_id = resolve_session(session_id, workspace_session)
    doc_ids = resolve_doc_ids(doc_ids, workspace_doc_ids)
    progress(0, desc="Loading model…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return _turn_error(history, load_error)

    if not message.strip():
        return (
            history or [],
            "_Type your question above, then click **Send text turn**._",
            "",
            {},
            gr.update(),
        )

    try:
        progress(0.2, desc="Thinking…")
        result = run_teacher_voice_text_turn(
            message,
            history,
            mode=mode,
            language=language,
            topic=topic.strip() or None,
            backend=get_backend(model_key),
            use_rag=use_rag and mode in RAG_MODES,
            session_id=session_id or None,
            doc_ids=doc_ids or None,
        )
    except Exception as exc:  # noqa: BLE001
        return _turn_error(history, str(exc))

    progress(1.0, desc="Done")
    return _turn_result(result)


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
        return (
            "_Using model knowledge only. Use **Discover** or **Auto-ingest** below, "
            "then check **Answer from my indexed sources**._"
        )
    return rag_scope_hint(sid, docs)


def _ingest_succeeded(status: str) -> bool:
    text = (status or "").lower()
    return not any(
        marker in text
        for marker in (
            "error",
            "enter a research topic",
            "add urls",
            "no verified urls found",
        )
    )


def _enable_rag_after_ingest(
    status: str,
    session_id: str,
    doc_ids: list[str] | None,
) -> tuple[dict, str]:
    if _ingest_succeeded(status):
        return gr.update(value=True), _update_rag_hint(True, session_id, doc_ids)
    return gr.update(), _update_rag_hint(False, session_id, doc_ids)


def _discover_for_json(
    topic: str,
    session_id: str,
    workspace_topic: str,
    workspace_session: str,
    progress: gr.Progress = gr.Progress(),
):
    results = list(
        discover_sources(topic, session_id, workspace_topic, workspace_session, progress)
    )
    results[4] = trace_as_dict(results[4])
    return tuple(results)


def _auto_ingest_for_json(
    topic: str,
    session_id: str,
    workspace_topic: str,
    workspace_session: str,
    progress: gr.Progress = gr.Progress(),
):
    results = list(
        auto_search_ingest(topic, session_id, workspace_topic, workspace_session, progress)
    )
    results[4] = trace_as_dict(results[4])
    return tuple(results)


def _ingest_for_json(
    topic: str,
    urls_text: str,
    selected_urls: list[str],
    upload_files: list[str] | None,
    session_id: str,
    workspace_topic: str,
    workspace_session: str,
    progress: gr.Progress = gr.Progress(),
):
    results = list(
        ingest_selected(
            topic,
            urls_text,
            selected_urls,
            upload_files,
            session_id,
            workspace_topic,
            workspace_session,
            progress,
        )
    )
    results[2] = trace_as_dict(results[2])
    return tuple(results)


def _on_mode_change(mode: str) -> tuple:
    topic_mode = mode in ("explain", "lesson")
    rag_mode = mode in RAG_MODES
    if mode == "lesson":
        topic_up = gr.update(
            visible=topic_mode,
            label="Focus topic",
            placeholder="e.g. Photosynthesis for grade 6 — for web search and lesson context",
        )
        message_up = gr.update(
            label="Your message",
            placeholder="e.g. What are the main steps of photosynthesis?",
        )
    elif mode == "explain":
        topic_up = gr.update(
            visible=topic_mode,
            label="Focus topic",
            placeholder="e.g. Photosynthesis — for web search and lesson context",
        )
        message_up = gr.update(
            label="Your message",
            placeholder="e.g. How does photosynthesis work?",
        )
    else:
        topic_up = gr.update(visible=False, value="")
        message_up = gr.update(
            label="Your message",
            placeholder="e.g. Here is my opening line — how can I improve it?",
        )
    rag_acc = gr.update(visible=rag_mode)
    use_rag = gr.update(value=False) if not rag_mode else gr.update()
    return topic_up, message_up, rag_acc, use_rag


def build_teacher_voice_tab(workspace: WorkspaceWidgets) -> None:
    lang_choices = _config.language_choices()
    asr_choices = _config.asr_choices()
    default_lang = lang_choices[0][1] if lang_choices else "en"
    default_asr = _config.asr_preset
    omni_note = omni_status_message()

    gr.Markdown("### TeacherVoice", elem_classes=["form-tab-heading"])
    gr.HTML(
        '<p class="tab-subtitle">'
        "Pick a mode, type a question or record audio, and hear a spoken reply from your local teacher."
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
                label="Focus topic",
                placeholder="e.g. Photosynthesis — used for web search and lesson context",
                lines=1,
                max_lines=2,
                elem_classes=["form-secondary"],
            )

            with gr.Accordion(
                "ResearchMind sources (optional)",
                open=False,
                visible=True,
                elem_classes=["form-optional-accordion"],
            ) as rag_acc:
                gr.Markdown(
                    "Set **Focus topic** above, then discover or ingest sources. "
                    "Enable RAG to ground answers in your library.",
                    elem_classes=["form-status"],
                )

                with gr.Row(elem_classes=["rm-action-row"]):
                    discover_btn = gr.Button("Discover on web", variant="secondary", size="sm")
                    auto_btn = gr.Button("Auto-ingest from web", variant="secondary", size="sm")

                with gr.Accordion(
                    "Suggested URLs from web search",
                    open=True,
                    visible=False,
                ) as urls_acc:
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

                ingest_btn = gr.Button(
                    "Ingest selected sources",
                    variant="secondary",
                    size="sm",
                )

                ingest_status = gr.Markdown(
                    value="_Set focus topic, then discover or auto-ingest sources._",
                    elem_classes=["form-status"],
                )

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
                    elem_classes=DOC_CHOICE_LIST_CLASSES,
                )
                rag_hint = gr.Markdown(
                    value=_update_rag_hint(False, "", []),
                    elem_classes=["form-status"],
                )

                with gr.Accordion("Indexed in this session", open=False):
                    indexed_md = gr.Markdown(value=memory_summary(""))
                    refresh_indexed_btn = gr.Button("Refresh", size="sm")

            message_tb = gr.Textbox(
                label="Your message",
                placeholder="e.g. How does photosynthesis work?",
                lines=3,
                max_lines=6,
                elem_classes=["form-ask-input"],
            )

            with gr.Row(elem_classes=["form-cta-row"]):
                send_text_btn = gr.Button(
                    "Send text turn",
                    variant="primary",
                    elem_classes=["primary-cta"],
                )

            gr.HTML('<p class="tv-or-divider">— or record your voice —</p>')

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
                value="_Type a message or record audio, then send._",
                elem_classes=["form-status"],
            )
            rec.status = status

            with gr.Row(elem_classes=["form-cta-row"]):
                send_voice_btn = gr.Button(
                    "Send voice turn",
                    variant="secondary",
                )
            clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

            wire_recording_handlers(
                rec,
                stop_next_action="Click **Send voice turn**.",
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
                voiceout = gr.Audio(
                    label="Replay audio",
                    type="filepath",
                    visible=False,
                )
                speak_status = gr.Markdown(
                    value="_Each reply includes an audio player in the chat. Use replay to regenerate speech._",
                    elem_classes=["form-status"],
                )

            advanced = build_advanced_panel(use_json=True)

        with gr.Column(scale=2, elem_classes=["tv-results-col"]):
            gr.HTML('<p class="form-section-label">Step 2 · Conversation</p>')

            chatbot = gr.Chatbot(
                label="Conversation",
                height=360,
                reasoning_tags=_REASONING_TAGS,
                placeholder=(
                    "Your back-and-forth with the teacher will show here. "
                    "Type a message or record audio on the left, then send a turn."
                ),
            )

    mode_dd.change(
        fn=_on_mode_change,
        inputs=[mode_dd],
        outputs=[topic_tb, message_tb, rag_acc, use_rag],
    ).then(
        fn=_update_rag_hint,
        inputs=[use_rag, session_dd, doc_dd],
        outputs=[rag_hint],
    )

    refresh_sessions_btn.click(fn=refresh_sessions, inputs=[session_dd], outputs=[session_dd])
    refresh_indexed_btn.click(fn=memory_summary, inputs=[session_dd], outputs=[indexed_md])
    session_dd.change(fn=memory_summary, inputs=[session_dd], outputs=[indexed_md])
    session_dd.change(fn=refresh_doc_choices, inputs=[session_dd, doc_dd], outputs=[doc_dd])
    for trigger in (use_rag, session_dd, doc_dd):
        trigger.change(
            fn=_update_rag_hint,
            inputs=[use_rag, session_dd, doc_dd],
            outputs=[rag_hint],
        )

    discover_outputs = [
        ingest_status,
        url_choices,
        session_dd,
        advanced.trace_summary,
        advanced.trace_box,
        indexed_md,
        doc_dd,
        urls_acc,
    ]

    discover_btn.click(
        fn=_discover_for_json,
        inputs=[topic_tb, session_dd, workspace.topic, workspace.session_dd],
        outputs=discover_outputs,
    ).then(
        fn=_update_rag_hint,
        inputs=[use_rag, session_dd, doc_dd],
        outputs=[rag_hint],
    )

    auto_btn.click(
        fn=_auto_ingest_for_json,
        inputs=[topic_tb, session_dd, workspace.topic, workspace.session_dd],
        outputs=discover_outputs,
    ).then(
        fn=_enable_rag_after_ingest,
        inputs=[ingest_status, session_dd, doc_dd],
        outputs=[use_rag, rag_hint],
    )

    ingest_btn.click(
        fn=_ingest_for_json,
        inputs=[
            topic_tb,
            urls_text,
            url_choices,
            upload_files,
            session_dd,
            workspace.topic,
            workspace.session_dd,
        ],
        outputs=[
            ingest_status,
            indexed_md,
            advanced.trace_box,
            advanced.trace_summary,
            session_dd,
            doc_dd,
        ],
    ).then(
        fn=_enable_rag_after_ingest,
        inputs=[ingest_status, session_dd, doc_dd],
        outputs=[use_rag, rag_hint],
    )

    turn_outputs = [
        chatbot,
        status,
        advanced.trace_summary,
        advanced.trace_box,
        message_tb,
    ]

    text_turn_inputs = [
        message_tb,
        chatbot,
        mode_dd,
        rec.language,
        topic_tb,
        use_rag,
        session_dd,
        doc_dd,
        workspace.topic,
        workspace.session_dd,
        workspace.doc_dd,
    ]

    voice_turn_inputs = [
        rec.audio_in,
        chatbot,
        mode_dd,
        rec.language,
        rec.asr_preset,
        topic_tb,
        use_rag,
        session_dd,
        doc_dd,
        workspace.topic,
        workspace.session_dd,
        workspace.doc_dd,
    ]

    send_text_btn.click(send_text_turn, inputs=text_turn_inputs, outputs=turn_outputs)
    message_tb.submit(send_text_turn, inputs=text_turn_inputs, outputs=turn_outputs)

    send_voice_btn.click(send_turn, inputs=voice_turn_inputs, outputs=turn_outputs)

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
        inputs=[workspace.topic, topic_tb],
        outputs=[topic_tb],
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


def teacher_voice_allowed_paths() -> list[str]:
    paths: list[str] = []
    if _config.presets_path:
        paths.append(str(_config.presets_path.parent))
    from echocoach.config import outputs_dir

    paths.append(str(outputs_dir()))
    paths.append(str(outputs_dir() / "recordings"))
    paths.append(str(outputs_dir() / "teacher_voice"))
    return paths
