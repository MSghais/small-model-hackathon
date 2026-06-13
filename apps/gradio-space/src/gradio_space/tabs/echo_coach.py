from __future__ import annotations

from pathlib import Path

import gradio as gr

from echocoach.config import get_echo_coach_config
from echocoach.pipeline import run_echo_coach
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key
from gradio_space.ui.components import (
    build_advanced_panel,
    build_recording_block,
    tab_hero,
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


def _error_outputs(message: str) -> tuple:
    return (
        message,
        f'<p style="color:#8a1f1f;">{message}</p>',
        "",
        None,
        None,
        None,
        message,
        {},
    )


def load_sample_pitch() -> tuple[str | None, str]:
    if not _SAMPLE_AUDIO.is_file():
        return (
            None,
            f"Sample clip missing at `{_SAMPLE_AUDIO}`. Run `uv run python libs/echocoach/tests/make_fixture.py`.",
        )
    return (
        gr.update(value=str(_SAMPLE_AUDIO)),
        "Loaded 2s sample clip. Click **Analyze pitch** to test the pipeline.",
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
        return _error_outputs("Record or upload a pitch, then click **Analyze pitch**.")

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
    status = "Analysis complete."
    if result.voiceout_warning:
        status += f" VoiceOut: {result.voiceout_warning}"

    return (
        status,
        result.transcript_html,
        result.report_markdown,
        result.filler_chart_path,
        result.pace_chart_path,
        result.voiceout_path,
        f"Trace saved: `{result.trace_path}`",
        result.trace,
    )


def build_echo_coach_tab() -> None:
    lang_choices = _config.language_choices()
    asr_choices = _config.asr_choices()
    default_lang = lang_choices[0][1] if lang_choices else "en"
    default_asr = _config.asr_preset

    tab_hero(
        "Record up to 30 seconds and get transcript, pace charts, filler highlights, and VoiceOut feedback.",
        steps=["Record", "Analyze", "Results"],
        active_step=0,
    )
    gr.HTML(
        '<p class="cross-link">Want live speaking tips? Try <strong>TeacherVoice → Pitch practice</strong>.</p>'
    )

    with gr.Row():
        with gr.Column(scale=1):
            rec = build_recording_block(
                max_seconds=_config.max_seconds,
                default_seconds=min(30, _config.max_seconds),
                lang_choices=lang_choices,
                asr_choices=asr_choices,
                default_lang=default_lang,
                default_asr=default_asr,
                audio_label="Your pitch (mic or upload)",
                include_sample=True,
            )
            status = gr.Textbox(label="Status", interactive=False, lines=3)
            rec.status = status

            with gr.Accordion("VoiceOut options", open=False):
                speak_rewrite = gr.Checkbox(
                    label="VoiceOut speaks full rewrite (otherwise summary + tip)",
                    value=False,
                )

            analyze_btn = gr.Button("Analyze pitch", variant="primary", elem_classes=["primary-cta"])

            wire_recording_handlers(
                rec,
                stop_next_action="Click **Analyze pitch**.",
                status_output=status,
                sample_loader=load_sample_pitch,
            )

        with gr.Column(scale=2):
            transcript_html = gr.HTML(label="Transcript")
            report_md = gr.Markdown(label="Coach report")
            with gr.Row():
                filler_chart = gr.Image(label="Filler words", type="filepath")
                pace_chart = gr.Image(label="Pace timeline", type="filepath")
            voiceout = gr.Audio(label="VoiceOut", type="filepath")

    advanced = build_advanced_panel(use_json=True)
    trace_note = gr.Markdown()

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
            trace_note,
            advanced.trace_box,
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
