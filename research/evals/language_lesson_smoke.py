#!/usr/bin/env python3
"""Smoke-check language-lesson eval JSONL for TeacherVoice format."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "research" / "data"

_JSON_LEAK = re.compile(r"^\s*[\{\[]|```")
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_FRENCH_MARKERS = re.compile(
    r"\b(le|la|les|un|une|des|est|sont|pour|dans|avec|que|qui|comment|pourquoi)\b",
    re.IGNORECASE,
)
_VOICE_SUFFIX = "Reply now in 2-4 complete spoken sentences only"


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _score_row(row: dict, *, language: str) -> list[str]:
    issues: list[str] = []
    messages = row.get("messages") or []
    if len(messages) < 3:
        issues.append("missing messages")
        return issues
    system = messages[0].get("content", "")
    user = messages[-2].get("content", "")
    assistant = messages[-1].get("content", "")

    if "TeacherVoice" not in system:
        issues.append("system missing TeacherVoice")
    label = "French" if language == "fr" else "Arabic"
    if f"Target language: {label}" not in system:
        issues.append(f"system missing target language {label}")
    if _VOICE_SUFFIX not in user:
        issues.append("user missing voice suffix")
    if not (40 <= len(assistant) <= 600):
        issues.append(f"assistant length {len(assistant)} out of range")
    if _JSON_LEAK.search(assistant):
        issues.append("assistant looks like JSON/code")
    if language == "ar" and not _ARABIC.search(assistant):
        issues.append("assistant missing Arabic script")
    if language == "fr" and not _FRENCH_MARKERS.search(assistant):
        issues.append("assistant missing French markers")
    return issues


def run_eval(*, language: str) -> int:
    path = _DATA / f"language-lesson-eval-{language}.jsonl"
    if not path.is_file():
        print(f"skip {path.name} (not found)")
        return 0
    rows = _load_rows(path)
    if not rows:
        print(f"skip {path.name} (empty)")
        return 0
    bad = 0
    for index, row in enumerate(rows):
        issues = _score_row(row, language=language)
        if issues:
            bad += 1
            print(f"  row {index}: {', '.join(issues)}")
    ok = len(rows) - bad
    print(f"{language.upper()} eval: {ok}/{len(rows)} passed")
    return 0 if bad == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("fr", "ar", "both"), default="both")
    args = parser.parse_args()
    codes = ("fr", "ar") if args.language == "both" else (args.language,)
    return max(run_eval(language=code) for code in codes)


if __name__ == "__main__":
    sys.exit(main())
