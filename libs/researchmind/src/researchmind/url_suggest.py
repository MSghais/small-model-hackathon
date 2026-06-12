from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    pass


class ChatBackend(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str: ...


SUGGEST_SYSTEM = """You suggest reputable web URLs for research on a topic.
Return ONLY a JSON array of 3-5 full https URLs as strings.
No markdown, no explanation. Example: ["https://example.com/a", "https://example.com/b"]
"""


def suggest_urls(topic: str, backend: ChatBackend, *, max_urls: int = 5) -> list[str]:
    messages = [
        {"role": "system", "content": SUGGEST_SYSTEM},
        {"role": "user", "content": f"Topic: {topic.strip()}"},
    ]
    raw = backend.chat(messages, max_tokens=512, temperature=0.2)
    return _parse_url_list(raw, max_urls=max_urls)


def _parse_url_list(raw: str, *, max_urls: int) -> list[str]:
    cleaned = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        urls = re.findall(r"https?://[^\s\"'<>]+", raw)
        return _dedupe_urls(urls, max_urls)

    if not isinstance(data, list):
        return []
    urls = [str(u).strip() for u in data if str(u).strip().startswith("http")]
    return _dedupe_urls(urls, max_urls)


def _dedupe_urls(urls: list[str], max_urls: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= max_urls:
            break
    return out
