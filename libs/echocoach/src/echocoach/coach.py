"""LLM coaching prompts and parsing."""

from __future__ import annotations

from inference.base import InferenceBackend
from inference.response_clean import strip_reasoning_output

from echocoach.models import CoachFeedback, FillerAnalysis, PaceAnalysis
from echocoach.utils import extract_json

COACH_SYSTEM = """You are EchoCoach, a concise public-speaking coach for students and teachers.
Respond with a single JSON object only — no markdown fences, no extra text.

Required keys:
- summary: 1-2 sentence overall assessment
- filler_feedback: specific advice about filler words
- pace_feedback: specific advice about speaking pace
- rewrite: improved 2-4 sentence version of the pitch (same language as the transcript)
- one_tip: one actionable tip for the next attempt
"""


def coach_user_prompt(
    transcript: str,
    fillers: FillerAnalysis,
    pace: PaceAnalysis,
    language: str,
) -> str:
    filler_list = ", ".join(f"{k} ({v})" for k, v in fillers.counts.items()) or "none"
    return f"""Language: {language}
Duration: {pace.duration_seconds:.1f}s
Word count: {pace.word_count}
Pace: {pace.wpm} WPM (target {pace.target_low}-{pace.target_high}) — {pace.label} (score {pace.score}/100)
Filler words: {fillers.total} total — {filler_list}

Transcript:
{transcript}
"""


def run_coach(
    backend: InferenceBackend,
    transcript: str,
    fillers: FillerAnalysis,
    pace: PaceAnalysis,
    language: str,
) -> tuple[CoachFeedback, str, str]:
    user_content = coach_user_prompt(transcript, fillers, pace, language)
    messages = [
        {"role": "system", "content": COACH_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    raw = backend.chat(messages, max_tokens=768, temperature=0.4)
    raw = strip_reasoning_output(raw)
    feedback = parse_coach_response(raw)
    return feedback, COACH_SYSTEM, raw


def parse_coach_response(raw: str) -> CoachFeedback:
    data = extract_json(raw)
    return CoachFeedback(
        summary=str(data.get("summary", "")).strip(),
        filler_feedback=str(data.get("filler_feedback", "")).strip(),
        pace_feedback=str(data.get("pace_feedback", "")).strip(),
        rewrite=str(data.get("rewrite", "")).strip(),
        one_tip=str(data.get("one_tip", "")).strip(),
    )


def format_report_markdown(
    coach: CoachFeedback,
    fillers: FillerAnalysis,
    pace: PaceAnalysis,
) -> str:
    filler_lines = (
        "\n".join(f"- **{word}**: {count}" for word, count in fillers.counts.items())
        or "- None detected"
    )
    return f"""## Pace

- **Score:** {pace.score}/100 — {pace.label}
- **WPM:** {pace.wpm} (target {pace.target_low}–{pace.target_high})
- **Duration:** {pace.duration_seconds:.1f}s · **Words:** {pace.word_count}

## Fillers ({fillers.total})

{filler_lines}

## Coach summary

{coach.summary}

### Filler feedback

{coach.filler_feedback}

### Pace feedback

{coach.pace_feedback}

### Suggested rewrite

{coach.rewrite}

### One tip

{coach.one_tip}
"""
