"""Gradio helpers for on-demand VoiceOut (shared across tabs)."""

from __future__ import annotations

from echocoach.voiceout import last_assistant_message, speak_assistant_text


def speak_last_assistant_reply(
    history: list,
    language: str,
    *,
    first_sentence_only: bool = False,
) -> tuple[str | None, str]:
    """TTS the latest assistant message. Returns (wav_path, status_message)."""
    text = last_assistant_message(history)
    if not text:
        return None, "No assistant reply to speak yet — send a turn first."

    playback, _, warning = speak_assistant_text(
        text,
        language=language or "en",
        first_sentence_only=first_sentence_only,
    )
    if not playback:
        hint = warning or "VoiceOut failed."
        if "piper" in hint.lower() or "not installed" in hint.lower():
            hint += " Install with: `uv sync --package echocoach --extra piper`"
        return None, hint

    label = "first sentence" if first_sentence_only else "full reply"
    status = f"VoiceOut ready ({label})."
    if warning:
        status += f" {warning}"
    return playback, status
