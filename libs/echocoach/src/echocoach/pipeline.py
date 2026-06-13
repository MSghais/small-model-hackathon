"""End-to-end EchoCoach pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from agent.trace import TraceRecorder
from inference.base import InferenceBackend

from echocoach.analysis.charts import build_charts
from echocoach.analysis.fillers import analyze_fillers, highlight_fillers_html
from echocoach.analysis.pace import analyze_pace
from echocoach.asr.factory import get_asr_backend
from echocoach.audio_io import clamp_duration, load_audio_mono_16k, write_wav_temp
from echocoach.coach import format_report_markdown, run_coach
from echocoach.config import get_echo_coach_config, outputs_dir
from echocoach.models import EchoCoachResult
from echocoach.tts.piper import get_tts_backend


def run_echo_coach(
    audio_path: str,
    *,
    language: str = "en",
    asr_preset: str | None = None,
    tts_preset: str | None = None,
    coach_model: str | None = None,
    backend: InferenceBackend,
    speak_rewrite: bool = False,
) -> EchoCoachResult:
    if not audio_path:
        raise ValueError("No audio recording provided.")

    config = get_echo_coach_config()
    asr_key = asr_preset or config.asr_preset
    tts_key = tts_preset or config.tts_preset
    model_key = coach_model or config.coach_model
    run_id = uuid.uuid4().hex[:12]
    out_base = outputs_dir()

    trace = TraceRecorder(
        skill="echo-coach",
        model=model_key,
        user_input={
            "language": language,
            "asr_preset": asr_key,
            "tts_preset": tts_key,
            "audio_path": audio_path,
            "speak_rewrite": speak_rewrite,
        },
        run_id=run_id,
    )

    audio, duration = load_audio_mono_16k(audio_path)
    audio = clamp_duration(audio, config.max_seconds)
    duration = len(audio) / 16_000
    clipped_path = write_wav_temp(audio, out_base / "clips", stem=run_id)

    trace.log_note("audio_loaded", duration_seconds=duration, path=str(clipped_path))

    asr = get_asr_backend(asr_key)
    transcript = asr.transcribe(str(clipped_path), language=language)
    trace.log_note("asr_complete", preset=asr_key, chars=len(transcript))

    fillers = analyze_fillers(transcript)
    pace = analyze_pace(transcript, duration)
    transcript_html = highlight_fillers_html(transcript, fillers)

    filler_chart, pace_chart = build_charts(
        transcript,
        duration,
        fillers,
        pace,
        out_base / "charts",
        run_id,
    )

    coach_feedback, system_prompt, coach_raw = run_coach(
        backend,
        transcript,
        fillers,
        pace,
        language,
    )
    trace.log_llm(system_prompt + "\n\n" + coach_user_for_trace(transcript, fillers, pace, language), coach_raw)

    report = format_report_markdown(coach_feedback, fillers, pace)

    voice_text = coach_feedback.rewrite if speak_rewrite else (
        f"{coach_feedback.summary} {coach_feedback.one_tip}".strip()
    )
    tts = get_tts_backend(tts_key)
    voiceout_path, voiceout_warning = tts.synthesize(
        voice_text,
        language=language,
        out_dir=out_base / "voiceout",
    )
    if voiceout_path:
        trace.set_artifact(voiceout_path)

    trace_path = trace.save()
    trace_dict: dict[str, Any] = trace.to_dict()

    return EchoCoachResult(
        transcript=transcript,
        transcript_html=transcript_html,
        language=language,
        duration_seconds=duration,
        fillers=fillers,
        pace=pace,
        coach=coach_feedback,
        report_markdown=report,
        filler_chart_path=str(filler_chart),
        pace_chart_path=str(pace_chart),
        voiceout_path=voiceout_path,
        voiceout_warning=voiceout_warning,
        trace_path=str(trace_path),
        trace=trace_dict,
    )


def coach_user_for_trace(transcript: str, fillers, pace, language: str) -> str:
    from echocoach.coach import coach_user_prompt

    return coach_user_prompt(transcript, fillers, pace, language)
