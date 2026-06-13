from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FillerSpan:
    start: int
    end: int
    word: str


@dataclass(frozen=True)
class FillerAnalysis:
    counts: dict[str, int]
    spans: list[FillerSpan]
    total: int


@dataclass(frozen=True)
class PaceAnalysis:
    word_count: int
    duration_seconds: float
    wpm: float
    score: int
    label: str
    target_low: int = 120
    target_high: int = 160


@dataclass(frozen=True)
class CoachFeedback:
    summary: str
    filler_feedback: str
    pace_feedback: str
    rewrite: str
    one_tip: str


@dataclass
class EchoCoachResult:
    transcript: str
    transcript_html: str
    language: str
    duration_seconds: float
    fillers: FillerAnalysis
    pace: PaceAnalysis
    coach: CoachFeedback
    report_markdown: str
    filler_chart_path: str | None
    pace_chart_path: str | None
    voiceout_path: str | None
    voiceout_warning: str | None
    trace_path: str
    trace: dict[str, Any] = field(default_factory=dict)
