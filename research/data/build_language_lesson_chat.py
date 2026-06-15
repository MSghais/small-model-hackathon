#!/usr/bin/env python3
"""Build TeacherVoice-shaped FR/AR chat JSONL from Hugging Face sources + seeds.

Exports:
  research/data/language-lesson-fr.jsonl
  research/data/language-lesson-ar.jsonl
  research/data/language-lesson-eval-fr.jsonl  (5% holdout)
  research/data/language-lesson-eval-ar.jsonl

Usage:
  uv run python research/data/build_language_lesson_chat.py
  uv run python research/data/build_language_lesson_chat.py --max-per-source 500 --skip-hub
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from echocoach.prompts import (  # noqa: E402
    system_prompt_for_mode,
    topic_context_block,
)
from echocoach.teacher_voice import _VOICE_USER_SUFFIX  # noqa: E402

VoiceMode = Literal["explain", "lesson"]

MIN_ASSISTANT_CHARS = 40
MAX_ASSISTANT_CHARS = 600
EVAL_HOLDOUT_RATIO = 0.05

DEFAULT_FR_SOURCES = (
    "FrancophonIA/english_french",
    "angeluriot/french_instruct",
    "CohereLabs/aya_dataset",
    "pinzhenchen/alpaca-cleaned-fr",
    "jpacifico/French-Alpaca-dataset-Instruct-110K",
)
DEFAULT_AR_SOURCES = (
    "arbml/CIDAR",
    "ClusterlabAi/InstAr-500k",
    "CohereLabs/aya_dataset",
)

SOURCE_CAPS: dict[str, dict[str, int]] = {
    "FrancophonIA/english_french": {"fr": 4000},
    "angeluriot/french_instruct": {"fr": 8000},
    "CohereLabs/aya_dataset": {"fr": 3000, "ar": 3000},
    "pinzhenchen/alpaca-cleaned-fr": {"fr": 2000},
    "jpacifico/French-Alpaca-dataset-Instruct-110K": {"fr": 4000},
    "arbml/CIDAR": {"ar": 8000},
    "ClusterlabAi/InstAr-500k": {"ar": 5000},
}

_INSTAR_GOOD_TASKS = frozenset(
    {
        "Open QA",
        "Extraction and Explanation",
        "Summarization",
        "Classification",
    }
)

_CODE_MARKERS = re.compile(r"```|^\s*def |^\s*class |^\s*import ", re.MULTILINE)
_JSON_START = re.compile(r"^\s*[\{\[]")


def _assistant_ok(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < MIN_ASSISTANT_CHARS or len(text) > MAX_ASSISTANT_CHARS:
        return False
    if _JSON_START.match(text):
        return False
    if _CODE_MARKERS.search(text):
        return False
    if text.count("\n") > 8:
        return False
    return True


def _pick_mode(rng: random.Random, *, topic: str | None) -> VoiceMode:
    if topic and rng.random() < 0.4:
        return "lesson"
    return "explain" if rng.random() < 0.6 else "lesson"


def _wrap_row(
    *,
    language: str,
    mode: VoiceMode,
    user_text: str,
    assistant_text: str,
    topic: str | None = None,
) -> dict[str, Any]:
    system = system_prompt_for_mode(mode, language=language)
    topic_line = topic_context_block(topic, mode)
    if topic_line:
        system = f"{system}\n\n{topic_line}"
    user_body = f"{user_text.strip()}\n\n{_VOICE_USER_SUFFIX}"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_body},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }


def _load_seeds(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file():
        return [], []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fr_rows: list[dict[str, Any]] = []
    ar_rows: list[dict[str, Any]] = []
    for lang, key in (("fr", "fr"), ("ar", "ar")):
        for item in raw.get(key, []):
            mode = item.get("mode", "explain")
            topic = item.get("topic")
            if topic in (None, "null", ""):
                topic = None
            row = _wrap_row(
                language=lang,
                mode=mode,  # type: ignore[arg-type]
                user_text=str(item["user"]),
                assistant_text=str(item["assistant"]),
                topic=str(topic) if topic else None,
            )
            (fr_rows if key == "fr" else ar_rows).append(row)
    return fr_rows, ar_rows


def _iter_english_french(max_rows: int) -> Iterator[tuple[str, str, str | None]]:
    """EN→FR parallel sentences — user asks in English, coach replies in French."""
    from datasets import load_dataset

    ds = load_dataset("FrancophonIA/english_french", split="train", streaming=True)
    count = 0
    for row in ds:
        english = (row.get("english") or "").strip()
        french = (row.get("french") or "").strip()
        if english and _assistant_ok(french):
            user = f"Translate the following to French:\n{english}"
            yield user, french, None
            count += 1
            if count >= max_rows:
                break


def _iter_french_instruct(max_rows: int) -> Iterator[tuple[str, str, str | None]]:
    from datasets import load_dataset

    ds = load_dataset("angeluriot/french_instruct", split="train", streaming=True)
    count = 0
    for row in ds:
        messages = row.get("messages") or row.get("conversation")
        if not messages:
            continue
        user_text = ""
        assistant_text = ""
        for msg in messages:
            role = (msg.get("role") or msg.get("from") or "").lower()
            content = (msg.get("content") or msg.get("value") or "").strip()
            if role in ("user", "human"):
                user_text = content
            elif role in ("assistant", "gpt", "bot") and content:
                assistant_text = content
        if user_text and _assistant_ok(assistant_text):
            yield user_text, assistant_text, None
            count += 1
            if count >= max_rows:
                break


def _iter_aya(language_code: str, max_rows: int) -> Iterator[tuple[str, str, str | None]]:
    from datasets import load_dataset

    ds = load_dataset("CohereLabs/aya_dataset", split="train")
    count = 0
    for row in ds:
        if row.get("language") != language_code:
            continue
        user_text = (row.get("inputs") or "").strip()
        assistant_text = (row.get("targets") or "").strip()
        if user_text and _assistant_ok(assistant_text):
            yield user_text, assistant_text, None
            count += 1
            if count >= max_rows:
                break


def _iter_alpaca_fr(max_rows: int) -> Iterator[tuple[str, str, str | None]]:
    from datasets import load_dataset

    ds = load_dataset("pinzhenchen/alpaca-cleaned-fr", split="train")
    count = 0
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        user_text = f"{instruction}\n{inp}".strip() if inp else instruction
        if user_text and _assistant_ok(output):
            yield user_text, output, None
            count += 1
            if count >= max_rows:
                break


def _iter_french_alpaca_110k(max_rows: int) -> Iterator[tuple[str, str, str | None]]:
    from datasets import load_dataset

    ds = load_dataset(
        "jpacifico/French-Alpaca-dataset-Instruct-110K", split="train", streaming=True
    )
    count = 0
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        user_text = f"{instruction}\n{inp}".strip() if inp else instruction
        if user_text and _assistant_ok(output):
            yield user_text, output, None
            count += 1
            if count >= max_rows:
                break


def _iter_cidar(max_rows: int) -> Iterator[tuple[str, str, str | None]]:
    from datasets import load_dataset

    ds = load_dataset("arbml/CIDAR", split="train")
    count = 0
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        user_text = f"{instruction}\n{inp}".strip() if inp else instruction
        topic = instruction[:80] if instruction else None
        if user_text and _assistant_ok(output):
            yield user_text, output, topic
            count += 1
            if count >= max_rows:
                break


def _iter_instar(max_rows: int) -> Iterator[tuple[str, str, str | None]]:
    from datasets import load_dataset

    ds = load_dataset("ClusterlabAi/InstAr-500k", split="train", streaming=True)
    count = 0
    for row in ds:
        task = row.get("task") or ""
        if task not in _INSTAR_GOOD_TASKS:
            continue
        instruction = (row.get("instruction") or "").strip()
        output = (row.get("output") or "").strip()
        topic = (row.get("topic") or "").strip() or None
        if instruction and _assistant_ok(output):
            yield instruction, output, topic
            count += 1
            if count >= max_rows:
                break


_SOURCE_LOADERS: dict[str, dict[str, Any]] = {
    "FrancophonIA/english_french": {"fr": _iter_english_french},
    "angeluriot/french_instruct": {"fr": _iter_french_instruct},
    "CohereLabs/aya_dataset": {
        "fr": lambda n: _iter_aya("fra", n),
        "ar": lambda n: _iter_aya("arb", n),
    },
    "pinzhenchen/alpaca-cleaned-fr": {"fr": _iter_alpaca_fr},
    "jpacifico/French-Alpaca-dataset-Instruct-110K": {"fr": _iter_french_alpaca_110k},
    "arbml/CIDAR": {"ar": _iter_cidar},
    "ClusterlabAi/InstAr-500k": {"ar": _iter_instar},
}


def _collect_from_source(
    source: str,
    language: str,
    max_rows: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    loaders = _SOURCE_LOADERS.get(source, {})
    loader = loaders.get(language)
    if loader is None:
        print(f"  skip {source} (no loader for {language})")
        return []
    rows: list[dict[str, Any]] = []
    try:
        for user_text, assistant_text, topic in loader(max_rows):
            mode = _pick_mode(rng, topic=topic)
            rows.append(
                _wrap_row(
                    language=language,
                    mode=mode,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    topic=topic,
                )
            )
    except Exception as exc:
        print(f"  warning: {source} failed for {language}: {exc}")
    return rows


def _split_eval(
    rows: list[dict[str, Any]], rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 20:
        return rows, []
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * EVAL_HOLDOUT_RATIO))
    return shuffled[n_eval:], shuffled[:n_eval]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_language_datasets(
    *,
    french_sources: tuple[str, ...],
    arabic_sources: tuple[str, ...],
    max_per_source: int,
    seeds_path: Path,
    skip_hub: bool,
    seed: int,
) -> None:
    rng = random.Random(seed)
    fr_rows, ar_rows = _load_seeds(seeds_path)
    print(f"Loaded {len(fr_rows)} FR + {len(ar_rows)} AR seed rows from {seeds_path.name}")

    if not skip_hub:
        for source in french_sources:
            cap = min(max_per_source, SOURCE_CAPS.get(source, {}).get("fr", max_per_source))
            print(f"Fetching FR from {source} (cap={cap})...")
            fr_rows.extend(_collect_from_source(source, "fr", cap, rng))
        for source in arabic_sources:
            cap = min(max_per_source, SOURCE_CAPS.get(source, {}).get("ar", max_per_source))
            print(f"Fetching AR from {source} (cap={cap})...")
            ar_rows.extend(_collect_from_source(source, "ar", cap, rng))

    fr_train, fr_eval = _split_eval(fr_rows, rng)
    ar_train, ar_eval = _split_eval(ar_rows, rng)

    out_fr = _DATA_DIR / "language-lesson-fr.jsonl"
    out_ar = _DATA_DIR / "language-lesson-ar.jsonl"
    eval_fr = _DATA_DIR / "language-lesson-eval-fr.jsonl"
    eval_ar = _DATA_DIR / "language-lesson-eval-ar.jsonl"

    _write_jsonl(out_fr, fr_train)
    _write_jsonl(out_ar, ar_train)
    _write_jsonl(eval_fr, fr_eval)
    _write_jsonl(eval_ar, ar_eval)

    print(
        f"Wrote FR train={len(fr_train)} eval={len(fr_eval)} -> {out_fr.name}, {eval_fr.name}\n"
        f"Wrote AR train={len(ar_train)} eval={len(ar_eval)} -> {out_ar.name}, {eval_ar.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--french-sources",
        default=",".join(DEFAULT_FR_SOURCES),
        help="Comma-separated Hugging Face dataset ids for French",
    )
    parser.add_argument(
        "--arabic-sources",
        default=",".join(DEFAULT_AR_SOURCES),
        help="Comma-separated Hugging Face dataset ids for Arabic",
    )
    parser.add_argument("--max-per-source", type=int, default=5000)
    parser.add_argument(
        "--custom-seeds",
        type=Path,
        default=_DATA_DIR / "language-lesson-seeds.yaml",
    )
    parser.add_argument(
        "--skip-hub",
        action="store_true",
        help="Only write seed rows (offline / smoke)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    french_sources = tuple(s.strip() for s in args.french_sources.split(",") if s.strip())
    arabic_sources = tuple(s.strip() for s in args.arabic_sources.split(",") if s.strip())

    build_language_datasets(
        french_sources=french_sources,
        arabic_sources=arabic_sources,
        max_per_source=args.max_per_source,
        seeds_path=args.custom_seeds,
        skip_hub=args.skip_hub,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
