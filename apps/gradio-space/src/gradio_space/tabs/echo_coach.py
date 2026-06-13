from __future__ import annotations

import gradio as gr

from echocoach.config import get_echo_coach_config
from echocoach.pipeline import run_echo_coach
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key, model_status
from inference.factory import get_backend

_config = get_echo_coach_config()


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


def analyze_pitch(
    audio_path: str | None,
    language: str,
    asr_preset: str,
    speak_rewrite: bool,
) -> tuple:
    model_key = get_active_model_key()
    load_error = ensure_model_loaded(model_key)
    if load_error:
        return _error_outputs(load_error)

    if not audio_path:
        return _error_outputs("Record or upload a pitch (up to 30 seconds), then click **Analyze pitch**.")

    try:
        result = run_echo_coach(
            audio_path,
            language=language,
            asr_preset=asr_preset,
            backend=get_backend(model_key),
            speak_rewrite=speak_rewrite,
        )
    except Exception as exc:  # noqa: BLE001 — surface pipeline errors in UI
        return _error_outputs(f"EchoCoach failed: {exc}")

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

    gr.Markdown(
        """
Record up to **30 seconds**, then get local feedback: transcript with **filler highlights**,
**pace score**, coach **rewrite**, and **VoiceOut** audio — all on-device.

- **ASR:** configurable (`voice_models.yaml`) — Cohere Transcribe 2B or Whisper.cpp
- **Coach:** text LLM preset (`ACTIVE_MODEL` / `ECHOCOACH_COACH_MODEL`)
- **TTS:** Piper VoiceOut (optional; install `echocoach[piper]`)
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                label="Your pitch (mic or upload)",
                sources=["microphone", "upload"],
                type="filepath",
            )
            language = gr.Dropdown(
                label="Language",
                choices=lang_choices,
                value=default_lang,
            )
            asr_preset = gr.Dropdown(
                label="ASR preset",
                choices=asr_choices,
                value=default_asr,
            )
            speak_rewrite = gr.Checkbox(
                label="VoiceOut speaks full rewrite (otherwise summary + tip)",
                value=False,
            )
            analyze_btn = gr.Button("Analyze pitch", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            coach_status = gr.Markdown(model_status(get_active_model_key()))

        with gr.Column(scale=2):
            transcript_html = gr.HTML(label="Transcript")
            report_md = gr.Markdown(label="Coach report")
            with gr.Row():
                filler_chart = gr.Image(label="Filler words", type="filepath")
                pace_chart = gr.Image(label="Pace timeline", type="filepath")
            voiceout = gr.Audio(label="VoiceOut", type="filepath")
            trace_note = gr.Markdown()
            trace_json = gr.JSON(label="Trace")

    analyze_btn.click(
        analyze_pitch,
        inputs=[audio_in, language, asr_preset, speak_rewrite],
        outputs=[
            status,
            transcript_html,
            report_md,
            filler_chart,
            pace_chart,
            voiceout,
            trace_note,
            trace_json,
        ],
    )


def echo_coach_allowed_paths() -> list[str]:
    base = get_echo_coach_config()
    paths: list[str] = []
    if base.presets_path:
        paths.append(str(base.presets_path.parent))
    from echocoach.config import outputs_dir

    paths.append(str(outputs_dir()))
    return paths
