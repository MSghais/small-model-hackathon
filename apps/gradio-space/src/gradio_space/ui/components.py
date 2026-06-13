from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gradio as gr

from echocoach.recording import (
    ServerRecordingError,
    recording_backend_status,
    recording_elapsed_seconds,
    recording_level_warning,
    start_server_recording,
    stop_server_recording,
)
from gradio_space.research_helpers import (
    list_session_choices,
    rag_scope_hint,
    refresh_doc_choices,
    refresh_sessions,
)

# Shared elem_classes for document / URL CheckboxGroup rows (see styles.css).
DOC_CHOICE_LIST_CLASSES = ["doc-choice-list"]


def build_step_indicator(steps: list[str], active_index: int = 0) -> str:
    """Render a horizontal step strip as HTML."""
    parts: list[str] = ['<div class="step-strip">']
    for i, label in enumerate(steps):
        if i > 0:
            parts.append('<span class="step-arrow">→</span>')
        if i < active_index:
            state = "done"
        elif i == active_index:
            state = "active"
        else:
            state = ""
        cls = f"step-pill {state}".strip()
        parts.append(
            f'<span class="{cls}"><span class="num">{i + 1}</span>{label}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


def tab_hero(subtitle: str, steps: list[str] | None = None, active_step: int = 0) -> gr.HTML:
    html = f'<p class="tab-subtitle">{subtitle}</p>'
    if steps:
        html += build_step_indicator(steps, active_step)
    return gr.HTML(html)


@dataclass
class SessionPickerWidgets:
    session_dd: gr.Dropdown
    refresh_btn: gr.Button
    doc_dd: gr.CheckboxGroup | None = None
    rag_hint: gr.Markdown | None = None

    def wire(
        self,
        *,
        on_session_change: Callable | None = None,
        extra_session_outputs: list | None = None,
    ) -> None:
        session_outputs = list(extra_session_outputs or [])
        if self.doc_dd is not None:
            self.session_dd.change(
                fn=refresh_doc_choices,
                inputs=[self.session_dd, self.doc_dd],
                outputs=[self.doc_dd],
            )
        if on_session_change is not None:
            self.session_dd.change(
                fn=on_session_change,
                inputs=[self.session_dd],
                outputs=session_outputs,
            )
        self.refresh_btn.click(
            fn=refresh_sessions,
            inputs=[self.session_dd],
            outputs=[self.session_dd],
        )


def build_session_picker(
    *,
    include_docs: bool = False,
    doc_label: str = "Documents (empty = all in session)",
    session_label: str = "Session",
) -> SessionPickerWidgets:
    with gr.Row():
        session_dd = gr.Dropdown(
            label=session_label,
            choices=list_session_choices(),
            value="",
            interactive=True,
            scale=4,
        )
        refresh_btn = gr.Button("↻", size="sm", scale=0, min_width=40)

    doc_dd = None
    rag_hint = None
    if include_docs:
        with gr.Accordion("Limit to documents", open=False):
            doc_dd = gr.CheckboxGroup(
                label=doc_label,
                choices=[],
                value=[],
                elem_classes=DOC_CHOICE_LIST_CLASSES,
            )
            rag_hint = gr.Markdown(value=rag_scope_hint("", []))
            doc_dd.change(
                fn=rag_scope_hint,
                inputs=[session_dd, doc_dd],
                outputs=[rag_hint],
            )
            session_dd.change(
                fn=rag_scope_hint,
                inputs=[session_dd, doc_dd],
                outputs=[rag_hint],
            )

    return SessionPickerWidgets(
        session_dd=session_dd,
        refresh_btn=refresh_btn,
        doc_dd=doc_dd,
        rag_hint=rag_hint,
    )


@dataclass
class RecordingWidgets:
    record_status_md: gr.Markdown
    audio_in: gr.Audio
    record_start_btn: gr.Button
    record_stop_btn: gr.Button
    record_seconds: gr.Slider
    sample_btn: gr.Button | None = None
    language: gr.Dropdown | None = None
    asr_preset: gr.Dropdown | None = None

    status: gr.Textbox | gr.Markdown | None = None


def build_recording_block(
    *,
    max_seconds: int,
    default_seconds: int | None = None,
    lang_choices: list[tuple[str, str]],
    asr_choices: list[tuple[str, str]],
    default_lang: str,
    default_asr: str,
    audio_label: str = "Record or upload",
    include_sample: bool = False,
    server_mic_open: bool = False,
    advanced_open: bool = False,
    compact: bool = False,
    audio_elem_classes: list[str] | None = None,
) -> RecordingWidgets:
    mic_status = recording_backend_status()
    slider_value = default_seconds or min(30, max_seconds)
    sample_btn: gr.Button | None = None

    if compact:
        record_status_md = gr.Markdown(mic_status, elem_classes=["form-status", "ec-mic-hint"])
        audio_classes = ["ec-audio-primary", *(audio_elem_classes or [])]
        audio_in = gr.Audio(
            label=audio_label,
            sources=["upload", "microphone"],
            type="filepath",
            format="wav",
            elem_classes=audio_classes,
        )
        with gr.Row(elem_classes=["ec-record-row"]):
            record_start_btn = gr.Button("Start recording", variant="secondary", size="sm")
            record_stop_btn = gr.Button("Stop recording", variant="stop", size="sm", interactive=False)
            if include_sample:
                sample_btn = gr.Button("Try sample clip", variant="secondary", size="sm")
        with gr.Accordion(
            "Recording options",
            open=False,
            elem_classes=["form-optional-accordion"],
        ):
            gr.Markdown(
                "Open **http://localhost:7860** in Chrome or Firefox (not Cursor's preview) "
                "and allow microphone access. On Linux you can also use **Start recording** "
                "for server-side capture. Use **Upload** if the browser mic fails."
            )
            record_seconds = gr.Slider(
                label="Max length (seconds)",
                minimum=3,
                maximum=max_seconds,
                value=slider_value,
                step=1,
            )
            language = gr.Dropdown(label="Language", choices=lang_choices, value=default_lang)
            asr_preset = gr.Dropdown(label="ASR preset", choices=asr_choices, value=default_asr)
    else:
        record_status_md = gr.Markdown(mic_status)
        with gr.Accordion("Recording help", open=False):
            gr.Markdown(
                "Open **http://localhost:7860** in Chrome or Firefox (not Cursor's preview) "
                "and allow microphone access. Use **Upload** if the browser mic fails."
            )
        audio_in = gr.Audio(
            label=audio_label,
            sources=["upload", "microphone"],
            type="filepath",
            format="wav",
            elem_classes=audio_elem_classes or None,
        )
        with gr.Accordion("Server microphone (Linux)", open=server_mic_open):
            record_seconds = gr.Slider(
                label="Max length (seconds)",
                minimum=3,
                maximum=max_seconds,
                value=slider_value,
                step=1,
            )
            with gr.Row():
                record_start_btn = gr.Button("Start recording", variant="secondary")
                record_stop_btn = gr.Button("Stop recording", variant="stop", interactive=False)
        if include_sample:
            sample_btn = gr.Button("Load sample clip", variant="secondary")
        language = None
        asr_preset = None
        with gr.Accordion("Voice settings", open=advanced_open):
            language = gr.Dropdown(label="Language", choices=lang_choices, value=default_lang)
            asr_preset = gr.Dropdown(label="ASR preset", choices=asr_choices, value=default_asr)

    return RecordingWidgets(
        record_status_md=record_status_md,
        audio_in=audio_in,
        record_start_btn=record_start_btn,
        record_stop_btn=record_stop_btn,
        record_seconds=record_seconds,
        sample_btn=sample_btn,
        language=language,
        asr_preset=asr_preset,
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


def ui_stop_recording(*, next_action: str) -> tuple[str | None, str, dict, dict]:
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

    status = f"Recording saved ({elapsed:.1f}s). {next_action}"
    if warning:
        status += f" Warning: {warning}"
    return (
        gr.update(value=str(path)),
        status,
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def wire_recording_handlers(
    rec: RecordingWidgets,
    *,
    stop_next_action: str,
    status_output: gr.Textbox | gr.Markdown | None = None,
    sample_loader: Callable[[], tuple] | None = None,
) -> None:
    status_out = status_output or rec.status
    if status_out is None:
        raise ValueError("wire_recording_handlers requires status_output or rec.status")

    rec.record_start_btn.click(
        ui_start_recording,
        inputs=[rec.record_seconds],
        outputs=[status_out, rec.record_start_btn, rec.record_stop_btn],
    )
    rec.record_stop_btn.click(
        lambda: ui_stop_recording(next_action=stop_next_action),
        outputs=[rec.audio_in, status_out, rec.record_start_btn, rec.record_stop_btn],
    ).then(
        lambda: recording_backend_status(),
        outputs=[rec.record_status_md],
    )

    if rec.sample_btn is not None and sample_loader is not None:
        rec.sample_btn.click(sample_loader, outputs=[rec.audio_in, status_out])


@dataclass
class AdvancedPanelWidgets:
    trace_summary: gr.Markdown
    trace_box: gr.Textbox | gr.JSON


def build_advanced_panel(
    *,
    use_json: bool = False,
    trace_lines: int = 12,
) -> AdvancedPanelWidgets:
    with gr.Accordion("Advanced & debug", open=False):
        trace_summary = gr.Markdown()
        if use_json:
            trace_box = gr.JSON(label="Trace")
        else:
            trace_box = gr.Textbox(
                label="Agent trace (JSON)",
                lines=trace_lines,
                max_lines=20,
                interactive=False,
            )
    return AdvancedPanelWidgets(trace_summary=trace_summary, trace_box=trace_box)


def empty_state(message: str) -> str:
    return f'<div class="empty-state">{message}</div>'
