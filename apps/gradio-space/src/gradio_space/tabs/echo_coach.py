from __future__ import annotations

from pathlib import Path

import gradio as gr

from echocoach.config import get_echo_coach_config
from echocoach.pipeline import run_echo_coach
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key
from gradio_space.spaces_runtime import gpu_task
from gradio_space.ui.components import (
    build_advanced_panel,
    build_recording_block,
    empty_state,
    wire_recording_handlers,
)
from inference.factory import get_backend

_config = get_echo_coach_config()
_SAMPLE_AUDIO = (
    Path(__file__).resolve().parents[5]
    / "libs"
    / "echocoach"
    / "tests"
    / "fixtures"
    / "silence_2s.wav"
)


def _error_html(message: str) -> str:
    safe = (
        message.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<div class="form-error">{safe}</div>'
    )


def _error_outputs(message: str) -> tuple:
    return (
        message,
        _error_html(message),
        "",
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        f"Trace: {message}",
        {},
        gr.update(visible=False),
        gr.update(visible=True),
    )


def load_sample_pitch() -> tuple[str | None, str]:
    if not _SAMPLE_AUDIO.is_file():
        return (
            None,
            f"Sample clip missing at `{_SAMPLE_AUDIO}`. Run `uv run python libs/echocoach/tests/make_fixture.py`.",
        )
    return (
        gr.update(value=str(_SAMPLE_AUDIO)),
        "Sample clip loaded — click **Analyze pitch** when ready.",
    )


def analyze_pitch(
    audio_path: str | None,
    language: str,
    asr_preset: str,
    speak_rewrite: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    progress(0, desc="Loading model…")
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return _error_outputs(load_error)

    if not audio_path:
        return _error_outputs("Record or upload a pitch (up to 30 seconds), then click **Analyze pitch**.")

    try:
        progress(0.2, desc="Transcribing & analyzing…")
        result = run_echo_coach(
            audio_path,
            language=language,
            asr_preset=asr_preset,
            backend=get_backend(model_key),
            speak_rewrite=speak_rewrite,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_outputs(f"EchoCoach failed: {exc}")

    progress(1.0, desc="Done")
    status = "**Analysis complete.** Review transcript, charts, and VoiceOut on the right."
    if result.voiceout_warning:
        status += f" VoiceOut note: {result.voiceout_warning}"

    has_filler = bool(result.filler_chart_path)
    has_pace = bool(result.pace_chart_path)
    has_voiceout = bool(result.voiceout_path)

    return (
        status,
        result.transcript_html,
        result.report_markdown,
        gr.update(value=result.filler_chart_path, visible=has_filler),
        gr.update(value=result.pace_chart_path, visible=has_pace),
        gr.update(value=result.voiceout_path, visible=has_voiceout),
        f"Trace saved: `{result.trace_path}`",
        result.trace,
        gr.update(visible=False),
        gr.update(visible=True),
    )


def build_echo_coach_tab() -> None:
    lang_choices = _config.language_choices()
    asr_choices = _config.asr_choices()
    default_lang = lang_choices[0][1] if lang_choices else "en"
    default_asr = _config.asr_preset

    gr.Markdown("### EchoCoach", elem_classes=["form-tab-heading"])
    gr.HTML(
        '<p class="tab-subtitle">'
        "Record a short pitch and get transcript, pace analysis, filler highlights, and spoken feedback."
        "</p>"
    )
    gr.HTML(
        '<p class="cross-link">Want live coaching? Try '
        "<strong>TeacherVoice → Pitch practice</strong>.</p>"
    )

    with gr.Row(elem_classes=["ec-workflow-columns"]):
        with gr.Column(scale=1, elem_classes=["ec-input-col"]):
            gr.HTML('<p class="form-section-label">Step 1 · Record your pitch</p>')

            with gr.Column(elem_classes=["form-primary"]):
                rec = build_recording_block(
                    max_seconds=_config.max_seconds,
                    default_seconds=min(30, _config.max_seconds),
                    lang_choices=lang_choices,
                    asr_choices=asr_choices,
                    default_lang=default_lang,
                    default_asr=default_asr,
                    audio_label="Your pitch (mic or upload, up to 30s)",
                    include_sample=True,
                    compact=True,
                )

            status = gr.Markdown(
                value="_Record or upload audio, then analyze._",
                elem_classes=["form-status"],
            )
            rec.status = status

            with gr.Accordion(
                "VoiceOut options",
                open=False,
                elem_classes=["form-optional-accordion"],
            ):
                speak_rewrite = gr.Checkbox(
                    label="Speak full rewrite (otherwise summary + tip)",
                    value=False,
                )

            with gr.Row(elem_classes=["form-cta-row"]):
                analyze_btn = gr.Button(
                    "Analyze pitch",
                    variant="primary",
                    elem_classes=["primary-cta"],
                )

            wire_recording_handlers(
                rec,
                stop_next_action="Click **Analyze pitch**.",
                status_output=status,
                sample_loader=load_sample_pitch,
            )

            advanced = build_advanced_panel(use_json=True)

        with gr.Column(scale=2, elem_classes=["ec-results-col"]):
            gr.HTML('<p class="form-section-label">Step 2 · Review feedback</p>')

            results_empty = gr.HTML(
                value=empty_state(
                    "Your transcript, pace charts, filler highlights, and VoiceOut audio "
                    "will appear here after you analyze a recording."
                )
            )

            with gr.Column(visible=False) as results_panel:
                report_md = gr.Markdown(
                    label="Coach summary",
                    elem_classes=["ec-coach-report"],
                )
                transcript_html = gr.HTML(
                    label="Transcript",
                    elem_classes=["ec-transcript"],
                )
                with gr.Row(elem_classes=["ec-charts-row"]):
                    filler_chart = gr.Image(
                        label="Filler words",
                        type="filepath",
                        visible=False,
                    )
                    pace_chart = gr.Image(
                        label="Pace timeline",
                        type="filepath",
                        visible=False,
                    )
                voiceout = gr.Audio(label="VoiceOut feedback", type="filepath", visible=False)

    analyze_btn.click(
        analyze_pitch,
        inputs=[
            rec.audio_in,
            rec.language,
            rec.asr_preset,
            speak_rewrite,
        ],
        outputs=[
            status,
            transcript_html,
            report_md,
            filler_chart,
            pace_chart,
            voiceout,
            advanced.trace_summary,
            advanced.trace_box,
            results_empty,
            results_panel,
        ],
    )


def echo_coach_allowed_paths() -> list[str]:
    base = get_echo_coach_config()
    paths: list[str] = []
    if base.presets_path:
        paths.append(str(base.presets_path.parent))
    from echocoach.config import outputs_dir

    paths.append(str(outputs_dir()))
    paths.append(str(outputs_dir() / "recordings"))
    if _SAMPLE_AUDIO.is_file():
        paths.append(str(_SAMPLE_AUDIO.parent))
    return paths
