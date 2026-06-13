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
    re.compile(r"(?:Let's draft:|Let me draft:|Draft:)\s*", re.IGNORECASE),
    re.compile(r"\nSummary:\s*", re.IGNORECASE),
    re.compile(r"\nAnswer:\s*", re.IGNORECASE),
    re.compile(r"\nFinal answer:\s*", re.IGNORECASE),
    re.compile(r"\nLet me write:\s*", re.IGNORECASE),
    re.compile(r"\n\n(?:In summary|To summarize)[,:]\s*", re.IGNORECASE),
]
_ANSWER_MARKER = re.compile(
    r"(?:^|\n)(?:Final answer|Let me write|Let's draft|Let me draft|Answer|Summary):\s*",
    re.IGNORECASE | re.MULTILINE,
)
_META_TAIL = re.compile(
    r"\n\n(?:Now,|We need|Also,|But we|However,|The instruction|So we|"
    r"That means|We must|We should|We have|We can|Next,)\b",
    re.IGNORECASE,
)
_META_AFTER_ANSWER = re.compile(
    r"\n\n(?:That's about|I think it covers|I'll add|To be more precise|"
    r"Let me write|Let me count|Let me draft|I need to make sure).*",
    re.DOTALL | re.IGNORECASE,
)
_COMPLETE_SENTENCE = re.compile(r"[.!?][\"')\]]*\s*$")
_LIST_OUTLINE = re.compile(r"^\d+\.\s", re.MULTILINE)
_REASONING_OPENERS = (
    "we need to",
    "first,",
    "next,",
    "the user",
    "let me",
    "okay,",
    "now, let",
    "i need to",
    "i should",
)


def _normalize_extracted(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^Summary:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Answer:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Final answer:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _clean_answer_candidate(text: str) -> str:
    rest = _normalize_extracted(text)
    rest = _META_TAIL.split(rest, maxsplit=1)[0].strip()
    rest = _META_AFTER_ANSWER.split(rest, maxsplit=1)[0].strip()
    return rest


def _slice_until_next_marker(text: str, start: int) -> str:
    rest = text[start:]
    next_match = _ANSWER_MARKER.search(rest)
    if next_match and next_match.start() > 0:
        rest = rest[: next_match.start()]
    return rest


def _is_list_outline(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    numbered = sum(1 for line in lines if _LIST_OUTLINE.match(line))
    return numbered >= max(2, len(lines) // 2)


def _extract_answer_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _ANSWER_MARKER.finditer(text):
        rest = _clean_answer_candidate(_slice_until_next_marker(text, match.end()))
        if len(rest) >= 20 and not _is_list_outline(rest):
            candidates.append(rest)
    for pattern in _ANSWER_SPLITS:
        match = pattern.search(text)
        if not match:
            continue
        rest = _clean_answer_candidate(_slice_until_next_marker(text, match.end()))
        if len(rest) >= 20 and not _is_list_outline(rest):
            candidates.append(rest)
    return candidates


def _extract_best_answer(text: str) -> str | None:
    candidates = _extract_answer_candidates(text)
    if not candidates:
        return None
    complete = [c for c in candidates if _COMPLETE_SENTENCE.search(c)]
    pool = complete or candidates
    return max(pool, key=len)


def _extract_answer_from_reasoning(text: str) -> str | None:
    return _extract_best_answer(text)


def _split_reasoning_and_answer(text: str) -> tuple[str | None, str]:
    cleaned = text.strip()
    if not cleaned:
        return None, ""

    final = _extract_best_answer(cleaned)
    if final and final != cleaned:
        idx = cleaned.find(final)
        if idx > 0:
            return cleaned[:idx].strip(), final
        return None, final

    if looks_like_reasoning_only(cleaned):
        return cleaned, ""

    return None, cleaned


def looks_like_reasoning_only(text: str) -> bool:
    sample = text[:240].lower()
    return any(sample.startswith(opener) for opener in _REASONING_OPENERS)


def prepare_display_reply(text: str) -> str:
    """Normalize model output for chat UI while preserving thinking blocks."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    if _THINK_BLOCKS.search(cleaned):
        answer = _THINK_BLOCKS.sub("", cleaned).strip()
        return answer or cleaned

    if _MALFORMED_THINK_OPEN.match(cleaned):
        body = _MALFORMED_THINK_OPEN.sub("", cleaned, count=1).strip()
        reasoning, answer = _split_reasoning_and_answer(body)
        if answer:
            think_body = reasoning or body
            return f"{_THINK_OPEN}\n{think_body}\n{_THINK_CLOSE}\n\n{answer}"
        return f"{_THINK_OPEN}\n{body}\n{_THINK_CLOSE}"

    reasoning, answer = _split_reasoning_and_answer(cleaned)
    if reasoning and answer:
        return f"{_THINK_OPEN}\n{reasoning}\n{_THINK_CLOSE}\n\n{answer}"

    return cleaned


def strip_reasoning_output(text: str) -> str:
    """Remove model chain-of-thought / thinking traces from user-visible replies."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = _THINK_BLOCKS.sub("", cleaned).strip()
    if cleaned and not _THINK_BLOCKS.search(text):
        extracted = _extract_best_answer(cleaned)
        if extracted:
            return extracted

    if _MALFORMED_THINK_OPEN.match(cleaned):
        body = _MALFORMED_THINK_OPEN.sub("", cleaned, count=1).strip()
        extracted = _extract_best_answer(body)
        if extracted:
            return extracted
        cleaned = body

    if looks_like_reasoning_only(cleaned) or _ANSWER_MARKER.search(cleaned):
        extracted = _extract_best_answer(cleaned)
        if extracted:
            return extracted

    return cleaned
