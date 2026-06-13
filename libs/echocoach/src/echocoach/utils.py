from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from an LLM response (fenced blocks or trailing prose)."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("{")
    if start < 0:
        return json.loads(cleaned)

    end = _matching_brace_end(cleaned, start)
    if end is not None:
        return json.loads(cleaned[start : end + 1])

    fallback_end = cleaned.rfind("}")
    if fallback_end > start:
        return json.loads(cleaned[start : fallback_end + 1])
    return json.loads(cleaned)


def _matching_brace_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None
