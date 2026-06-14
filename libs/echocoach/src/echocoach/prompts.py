"""TeacherVoice mode system prompts."""

from __future__ import annotations

from typing import Literal

TeacherVoiceMode = Literal["explain", "lesson", "pitch"]

MODE_LABELS: dict[TeacherVoiceMode, str] = {
    "explain": "Explain",
    "lesson": "Lesson coach",
    "pitch": "Pitch practice",
}

LANGUAGE_LESSON_MODES: frozenset[TeacherVoiceMode] = frozenset({"explain", "lesson"})

# ISO 639-1 codes mapped to Tiny Aya regional presets (see Cohere Labs field guide).
_AYA_FIRE_LANGS = frozenset({"hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur", "ne", "si"})
_AYA_EARTH_LANGS = frozenset({"ar", "sw", "am", "ha", "fa", "he", "so", "yo", "ig", "zu", "af"})
_AYA_WATER_LANGS = frozenset(
    {"fr", "de", "es", "it", "pt", "nl", "pl", "el", "ja", "zh", "ko", "vi", "ru", "uk", "cs", "sv", "da", "fi", "no"}
)

_LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "el": "Greek",
    "ar": "Arabic",
    "ja": "Japanese",
    "zh": "Chinese",
    "vi": "Vietnamese",
    "ko": "Korean",
}

EXPLAIN_SYSTEM = """You are TeacherVoice, a friendly tutor who explains ideas in plain language.
Reply with ONLY the spoken answer (2-5 short sentences). Do not include planning, drafting,
numbered outlines, or phrases like "let me think" or "first I need to".
Use simple examples when helpful.
When source excerpts are provided, ground your answer in them and cite with [1], [2], etc."""

LESSON_SYSTEM = """You are TeacherVoice, a lesson-planning coach for teachers and students.
Reply with ONLY the spoken answer (2-5 short sentences). Do not include planning, drafting,
or meta commentary about how you will answer.
Help outline and explain lesson content verbally: learning goals, key points, and a simple flow.
If a lesson topic is set, stay focused on it.
When source excerpts are provided, use them and cite [1], [2], etc."""

PITCH_SYSTEM = """You are TeacherVoice, a supportive public-speaking coach in a live conversation.
Give brief, actionable feedback on what the student just said (opening, clarity, energy, structure).
Do not produce JSON or long reports — speak naturally in 2-4 sentences.
Suggest one concrete improvement for their next attempt. For charts and pace analysis, use Classic EchoCoach."""

_MODE_SYSTEM: dict[TeacherVoiceMode, str] = {
    "explain": EXPLAIN_SYSTEM,
    "lesson": LESSON_SYSTEM,
    "pitch": PITCH_SYSTEM,
}


def language_label(language: str) -> str:
    code = (language or "en").strip().lower().split("-")[0]
    return _LANGUAGE_LABELS.get(code, code or "English")


def language_instruction(language: str) -> str:
    label = language_label(language)
    return (
        f"Target language: {label} ({language}). "
        f"Reply ONLY in {label}. "
        "If the student writes or speaks in another language, match their language instead."
    )


def resolve_aya_preset(language: str, variant: str = "auto") -> str:
    """Return a models.yaml preset key for the Tiny Aya coach."""
    if variant and variant not in ("auto", ""):
        return variant
    code = (language or "en").strip().lower().split("-")[0]
    if code in _AYA_FIRE_LANGS:
        return "tiny-aya-fire"
    if code in _AYA_EARTH_LANGS:
        return "tiny-aya-earth"
    if code in _AYA_WATER_LANGS:
        return "tiny-aya-water"
    return "tiny-aya-global"


def system_prompt_for_mode(mode: TeacherVoiceMode, *, language: str | None = None) -> str:
    base = _MODE_SYSTEM[mode]
    if language:
        return f"{base}\n\n{language_instruction(language)}"
    return base


def topic_context_block(topic: str | None, mode: TeacherVoiceMode) -> str | None:
    topic = (topic or "").strip()
    if not topic or mode == "pitch":
        return None
    label = "Lesson topic" if mode == "lesson" else "Focus topic"
    return f"{label}: {topic}"
