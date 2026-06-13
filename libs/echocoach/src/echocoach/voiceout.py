"""Reusable VoiceOut helpers for TeacherVoice, Chat, and other tabs."""

from __future__ import annotations

import re

from echocoach.config import get_echo_coach_config, outputs_dir
from echocoach.tts.piper import get_tts_backend


def strip_references_for_tts(text: str) -> str:
    """Remove citation/reference blocks before TTS."""
    cleaned = text
    for marker in ("\n\n## References", "\n\n**References**"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


def extract_message_text(content: object) -> str:
    """Normalize Gradio chat content (plain string or message blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text = block.strip()
            elif isinstance(block, dict):
                text = str(block.get("text") or block.get("content") or "").strip()
            else:
                text = str(block).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(content).strip()


def last_assistant_message(history: list) -> str | None:
    """Return the most recent assistant message from Gradio chat history."""
    for item in reversed(history or []):
        if isinstance(item, dict) and item.get("role") == "assistant":
            content = extract_message_text(item.get("content"))
            return content or None
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[1]:
            return extract_message_text(item[1]) or None
    return None


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def synthesize_voice_reply(
    text: str,
    *,
    language: str,
    tts_preset: str | None = None,
    chunk_first: bool = True,
    out_subdir: str = "voiceout",
) -> tuple[str | None, str | None, str | None]:
    """Return (full_wav, first_sentence_wav, warning)."""
    if not text.strip():
        return None, None, "No text to synthesize."

    config = get_echo_coach_config()
    preset = tts_preset or config.tts_preset
    tts = get_tts_backend(preset)
    out_dir = outputs_dir() / out_subdir
    full_path, warning = tts.synthesize(text, language=language, out_dir=out_dir)

    first_path = None
    if chunk_first:
        sentences = split_sentences(text)
        if len(sentences) > 1:
            first_path, first_warning = tts.synthesize(
                sentences[0],
                language=language,
                out_dir=out_dir,
            )
            if first_warning and not warning:
                warning = first_warning
        elif full_path:
            first_path = full_path

    return full_path, first_path, warning


def speak_assistant_text(
    text: str,
    *,
    language: str = "en",
    tts_preset: str | None = None,
    first_sentence_only: bool = False,
) -> tuple[str | None, str | None, str | None]:
    """Synthesize assistant reply audio. Returns (playback_path, alt_path, warning)."""
    clean = strip_references_for_tts(text)
    full_path, first_path, warning = synthesize_voice_reply(
        clean,
        language=language,
        tts_preset=tts_preset,
        chunk_first=True,
    )
    if first_sentence_only:
        return first_path or full_path, full_path, warning
    return full_path or first_path, first_path, warning
