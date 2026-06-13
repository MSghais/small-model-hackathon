from __future__ import annotations

import re

_RT_OPEN = "<" + "redacted_thinking" + ">"
_RT_CLOSE = "</" + "redacted_thinking" + ">"
_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"

_THINK_BLOCKS = re.compile(
    "|".join(
        (
            re.escape(_RT_OPEN) + r".*?" + re.escape(_RT_CLOSE),
            re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE),
            r"<thinking>.*?</thinking>",
        )
    ),
    re.DOTALL | re.IGNORECASE,
)
_MALFORMED_THINK_OPEN = re.compile(r"^think>\s*", re.IGNORECASE)
_ANSWER_SPLITS = [
    re.compile(r"(?:Let's draft:|Draft:)\s*", re.IGNORECASE),
    re.compile(r"\nSummary:\s*", re.IGNORECASE),
    re.compile(r"\nAnswer:\s*", re.IGNORECASE),
    re.compile(r"\n\n(?:In summary|To summarize)[,:]\s*", re.IGNORECASE),
]
_META_TAIL = re.compile(
    r"\n\n(?:Now,|We need|Also,|But we|However,|The instruction|So we|"
    r"That means|We must|We should|We have|We can)\b",
    re.IGNORECASE,
)
_REASONING_OPENERS = (
    "we need to",
    "first,",
    "the user",
    "let me",
    "okay,",
    "now, let",
    "i need to",
)


def _normalize_extracted(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^Summary:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Answer:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_answer_from_reasoning(text: str) -> str | None:
    for pattern in _ANSWER_SPLITS:
        match = pattern.search(text)
        if not match:
            continue
        rest = _normalize_extracted(text[match.end() :])
        rest = _META_TAIL.split(rest, maxsplit=1)[0].strip()
        if len(rest) >= 40:
            return rest
    return None


def looks_like_reasoning_only(text: str) -> bool:
    sample = text[:240].lower()
    return any(sample.startswith(opener) for opener in _REASONING_OPENERS)


def prepare_display_reply(text: str) -> str:
    """Normalize model output for chat UI while preserving thinking blocks."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    if _MALFORMED_THINK_OPEN.match(cleaned):
        body = _MALFORMED_THINK_OPEN.sub("", cleaned, count=1).strip()
        return f"{_THINK_OPEN}\n{body}\n{_THINK_CLOSE}"
    return cleaned


def strip_reasoning_output(text: str) -> str:
    """Remove model chain-of-thought / thinking traces from user-visible replies."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = _THINK_BLOCKS.sub("", cleaned).strip()

    if _MALFORMED_THINK_OPEN.match(cleaned):
        body = _MALFORMED_THINK_OPEN.sub("", cleaned, count=1).strip()
        extracted = _extract_answer_from_reasoning(body)
        if extracted:
            return extracted
        cleaned = body

    if looks_like_reasoning_only(cleaned):
        extracted = _extract_answer_from_reasoning(cleaned)
        if extracted:
            return extracted

    return cleaned
