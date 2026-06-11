"""
benchmarks/gaia.py
───────────────────
GAIA — General AI Assistants benchmark.

What it tests: Multi-step reasoning with tool use (web search,
code execution, file reading) toward a single final answer.

Dataset: gaia-benchmark/GAIA on HF Hub (requires auth for test split).
         We use the validation split by default, which is public.

Scoring: Exact-match on final answer string (case-insensitive, stripped).
         GAIA also supports a normalised-string partial match — implemented here.

Levels:
    Level 1 – single-step, no tools needed
    Level 2 – multi-step, 1-2 tools
    Level 3 – complex, multi-tool chains  (hardest)
"""

from __future__ import annotations
import re
import unicodedata
from typing import Any

from benchmarks.base import BaseBenchmark


SYSTEM_PROMPT = """\
You are a highly capable AI assistant.
Answer the user's question as precisely as possible.
If you need to reason step-by-step, do so in <think>...</think> tags.
Your final answer must appear on its own line starting with: ANSWER:
"""

# Tools the agent could use in a real setup — logged in prompt as context.
AVAILABLE_TOOLS_DESCRIPTION = """
Available tools (describe their use in your reasoning, then give ANSWER):
- web_search(query: str) → search results
- code_exec(code: str) → run Python code, returns stdout
- read_file(path: str) → read a local file
"""


class GAIABenchmark(BaseBenchmark):
    """
    GAIA benchmark adapter.

    Config keys (benchmark_overrides.gaia):
        data_path   – local JSONL (optional)
        split       – "validation" | "test" (default: "validation")
        levels      – list[int], e.g. [1, 2] to run only levels 1 & 2
        tool_mode   – "describe" | "none"  (default: "describe")
                      In "describe" mode tools are mentioned in prompt but
                      not actually executed (offline eval). Set up real tools
                      in your own subclass for live eval.
    """

    name = "gaia"

    def load_dataset(self) -> list[dict]:
        data_path = self.cfg.get("data_path")
        if data_path:
            return self._load_local(data_path)
        return self._load_from_hub()

    def _load_local(self, path: str) -> list[dict]:
        import json
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return self._filter_levels(samples)

    def _load_from_hub(self) -> list[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        split = self.cfg.get("split", "validation")
        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split, trust_remote_code=True)
        return self._filter_levels(list(ds))

    def _filter_levels(self, samples: list[dict]) -> list[dict]:
        levels = self.cfg.get("levels")
        if not levels:
            return samples
        return [s for s in samples if s.get("Level") in levels or s.get("level") in levels]

    # ── Prompt ────────────────────────────────────────────────────────────────

    def build_prompt(self, sample: dict) -> str:
        question = sample.get("Question", sample.get("question", ""))
        tool_block = (
            AVAILABLE_TOOLS_DESCRIPTION
            if self.cfg.get("tool_mode", "describe") == "describe"
            else ""
        )
        return (
            f"{SYSTEM_PROMPT}\n"
            f"{tool_block}\n"
            f"Question: {question}\n"
            f"Assistant:"
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_sample(self, sample: dict, prediction: str) -> dict:
        ground_truth = (
            sample.get("Final answer")
            or sample.get("final_answer")
            or sample.get("answer")
            or ""
        )
        if not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)

        extracted = self._extract_answer(prediction)
        exact_ok  = self._normalize(extracted) == self._normalize(ground_truth)

        if exact_ok:
            return {"passed": True, "score": 1.0, "note": "exact match"}

        # Partial: is ground truth contained in extracted (or vice versa)?
        n_gt   = self._normalize(ground_truth)
        n_pred = self._normalize(extracted)
        partial = (n_gt in n_pred) or (n_pred and n_pred in n_gt)
        score   = 0.5 if partial else 0.0

        return {
            "passed": False,
            "score":  score,
            "note":   f"pred='{extracted[:60]}' gt='{ground_truth[:60]}'",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Pull the text after 'ANSWER:' on its own line."""
        match = re.search(r"(?i)^ANSWER:\s*(.+)", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        # Fallback: last non-empty line
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[-1] if lines else text.strip()

    @staticmethod
    def _normalize(s: str) -> str:
        """Lower-case, strip punctuation and extra whitespace."""
        s = unicodedata.normalize("NFKC", s).lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        return re.sub(r"\s+", " ", s)
