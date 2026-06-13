"""Filler word detection and HTML highlighting."""

from __future__ import annotations

import re

from echocoach.models import FillerAnalysis, FillerSpan

DEFAULT_FILLERS = [
    "um",
    "uh",
    "uhm",
    "erm",
    "like",
    "you know",
    "basically",
    "actually",
    "literally",
    "sort of",
    "kind of",
    "i mean",
    "right",
    "okay so",
    "so yeah",
]

# Longer phrases first so "you know" wins over "you"
_SORTED_FILLERS = sorted(DEFAULT_FILLERS, key=len, reverse=True)
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in _SORTED_FILLERS) + r")\b",
    re.IGNORECASE,
)


def analyze_fillers(transcript: str, fillers: list[str] | None = None) -> FillerAnalysis:
    if fillers is None:
        pattern = _PATTERN
    else:
        ordered = sorted(fillers, key=len, reverse=True)
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(f) for f in ordered) + r")\b",
            re.IGNORECASE,
        )

    counts: dict[str, int] = {}
    spans: list[FillerSpan] = []
    for match in pattern.finditer(transcript):
        word = match.group(1).lower()
        counts[word] = counts.get(word, 0) + 1
        spans.append(FillerSpan(start=match.start(), end=match.end(), word=word))

    return FillerAnalysis(counts=counts, spans=spans, total=sum(counts.values()))


def highlight_fillers_html(transcript: str, analysis: FillerAnalysis) -> str:
    if not analysis.spans:
        safe = _escape_html(transcript)
        return f'<p style="line-height:1.6;">{safe}</p>'

    parts: list[str] = []
    cursor = 0
    for span in sorted(analysis.spans, key=lambda s: s.start):
        if span.start < cursor:
            continue
        parts.append(_escape_html(transcript[cursor : span.start]))
        parts.append(
            f'<mark style="background:#ffe08a;padding:0 2px;border-radius:3px;">'
            f"{_escape_html(transcript[span.start : span.end])}</mark>"
        )
        cursor = span.end
    parts.append(_escape_html(transcript[cursor:]))
    body = "".join(parts)
    return f'<p style="line-height:1.6;">{body}</p>'


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
