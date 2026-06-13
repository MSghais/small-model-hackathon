"""Speaking pace scoring."""

from __future__ import annotations

from echocoach.audio_io import count_words
from echocoach.models import PaceAnalysis

TARGET_LOW = 120
TARGET_HIGH = 160


def analyze_pace(
    transcript: str,
    duration_seconds: float,
    *,
    target_low: int = TARGET_LOW,
    target_high: int = TARGET_HIGH,
) -> PaceAnalysis:
    word_count = count_words(transcript)
    if duration_seconds <= 0:
        wpm = 0.0
    else:
        wpm = word_count / (duration_seconds / 60.0)

    score, label = _score_wpm(wpm, target_low, target_high)
    return PaceAnalysis(
        word_count=word_count,
        duration_seconds=duration_seconds,
        wpm=round(wpm, 1),
        score=score,
        label=label,
        target_low=target_low,
        target_high=target_high,
    )


def _score_wpm(wpm: float, low: int, high: int) -> tuple[int, str]:
    if wpm <= 0:
        return 0, "No speech detected"
    if low <= wpm <= high:
        return 100, "Ideal pace"
    if wpm < low:
        ratio = wpm / low
        score = max(20, int(100 * ratio))
        return score, "Too slow — pick up the energy"
    # too fast
    overshoot = (wpm - high) / high
    score = max(20, int(100 * (1.0 - min(overshoot, 0.8))))
    return score, "Too fast — pause and breathe"
