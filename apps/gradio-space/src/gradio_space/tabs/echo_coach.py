from __future__ import annotations

from pathlib import Path

import gradio as gr

from echocoach.config import get_echo_coach_config
from echocoach.pipeline import run_echo_coach
from echocoach.recording import (
    ServerRecordingError,
    record_server_wav,
    recording_backend_status,
)
from gradio_space.model_loading import ensure_model_loaded, get_active_model_key, model_status
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


def record_server_pitch_start(seconds: int) -> str:
    return f"Recording {seconds}s — speak now (using this computer's microphone)..."


def record_server_pitch(seconds: int) -> tuple[str | None, str]:
    """Capture from this machine's mic and feed the same audio input as upload."""
    try:
        path = record_server_wav(max_seconds=int(seconds))
    except ServerRecordingError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — surface unexpected recorder errors
        return None, f"Recording failed: {exc}"

    return (
        str(path),
        f"Recorded {int(seconds)}s from server microphone. Click **Analyze pitch**.",
    )


def load_sample_pitch() -> tuple[str | None, str]:
    if not _SAMPLE_AUDIO.is_file():
        return None, f"Sample clip missing at `{_SAMPLE_AUDIO}`. Run `uv run python libs/echocoach/tests/make_fixture.py`."
    return str(_SAMPLE_AUDIO), "Loaded 2s sample clip. Click **Analyze pitch** to test the pipeline."


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
    mic_status = recording_backend_status()

    gr.Markdown(
        f"""
Record up to **{_config.max_seconds} seconds**, then get local feedback: transcript with **filler highlights**,
**pace score**, coach **rewrite**, and **VoiceOut** audio — all on-device.

- **ASR:** configurable (`voice_models.yaml`) — Cohere Transcribe 2B or Whisper.cpp
- **Coach:** text LLM preset (`ACTIVE_MODEL` / `ECHOCOACH_COACH_MODEL`)
- **TTS:** Piper VoiceOut (optional; install `echocoach[piper]`)

**Browser mic:** open **http://localhost:7860** in Chrome or Firefox (not Cursor's preview, not `0.0.0.0`) and allow microphone access.
If the mic icon still fails, use **Record from this computer** or **Upload** a `.wav` / `.mp3`.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            record_status_md = gr.Markdown(mic_status)
            with gr.Accordion("Record from this computer (recommended)", open=True):
                gr.Markdown(
                    "Captures on the machine running Gradio. Status updates immediately; recording takes a few seconds."
                )
                record_seconds = gr.Slider(
                    label="Recording length (seconds)",
                    minimum=3,
                    maximum=_config.max_seconds,
                    value=min(10, _config.max_seconds),
                    step=1,
                )
                with gr.Row():
                    record_server_btn = gr.Button(
                        "Record from this computer",
                        variant="secondary",
                    )
                    sample_btn = gr.Button("Load sample clip", variant="secondary")
            audio_in = gr.Audio(
                label="Your pitch (browser mic or upload)",
                sources=["upload", "microphone"],
                type="filepath",
                format="wav",
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
            status = gr.Textbox(label="Status", interactive=False, lines=2)
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

    record_event = record_server_btn.click(
        record_server_pitch_start,
        inputs=[record_seconds],
        outputs=[status],
    ).then(
        record_server_pitch,
        inputs=[record_seconds],
        outputs=[audio_in, status],
    )
    record_event.then(
        lambda: recording_backend_status(),
        outputs=[record_status_md],
    )

    sample_btn.click(
        load_sample_pitch,
        outputs=[audio_in, status],
    )

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
    if _SAMPLE_AUDIO.is_file():
        paths.append(str(_SAMPLE_AUDIO.parent))
    return paths
