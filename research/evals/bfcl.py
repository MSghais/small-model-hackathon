"""
benchmarks/bfcl.py
───────────────────
Berkeley Function-Calling Leaderboard (BFCL) — local eval adapter.

What it tests: Can the model call functions with correct names,
argument names, and argument values given a user request + schema?

Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard on HF Hub
         OR local JSONL at cfg["data_path"]

Scoring: Exact-match on function name + fuzzy-match on arguments.
"""

from __future__ import annotations
import json
import re
from typing import Any

from benchmarks.base import BaseBenchmark


SYSTEM_PROMPT = """\
You are a helpful assistant with access to the following functions.
To call a function, respond ONLY with a JSON object in this exact format:
{"name": "<function_name>", "arguments": {<key>: <value>, ...}}
Do not add any explanation or text outside the JSON.
"""


class BFCLBenchmark(BaseBenchmark):
    """
    BFCL function-calling benchmark.

    Config keys (in benchmark_overrides.bfcl):
        data_path   – local JSONL file (optional; falls back to HF Hub)
        categories  – list of categories to filter on (optional)
        strict      – bool, require perfect arg match (default: False)
    """

    name = "bfcl"

    # ── Dataset ───────────────────────────────────────────────────────────────

    def load_dataset(self) -> list[dict]:
        data_path = self.cfg.get("data_path")

        if data_path:
            return self._load_local(data_path)
        else:
            return self._load_from_hub()

    def _load_local(self, path: str) -> list[dict]:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _load_from_hub(self) -> list[dict]:
        """
        Pulls the BFCL dataset from HF Hub.
        Requires: pip install datasets
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets  (required to fetch BFCL from Hub)")

        ds = load_dataset(
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            split="train",
            trust_remote_code=True,
        )
        return list(ds)

    # ── Prompt ────────────────────────────────────────────────────────────────

    def build_prompt(self, sample: dict) -> str:
        functions_block = json.dumps(sample.get("function", []), indent=2)
        user_query = sample.get("question", sample.get("input", ""))

        return (
            f"{SYSTEM_PROMPT}\n"
            f"Available functions:\n{functions_block}\n\n"
            f"User: {user_query}\n"
            f"Assistant:"
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_sample(self, sample: dict, prediction: str) -> dict:
        """
        1. Parse model output as JSON.
        2. Check function name matches ground truth.
        3. Check arguments (strict = exact keys+values, else fuzzy).
        """
        ground_truth = sample.get("ground_truth", {})
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError:
                ground_truth = {}

        # ── Parse prediction ──────────────────────────────────────────────────
        parsed = self._extract_json(prediction)
        if parsed is None:
            return {
                "passed": False,
                "score":  0.0,
                "note":   "Could not parse JSON from model output",
            }

        # ── Check function name ───────────────────────────────────────────────
        expected_name = (
            ground_truth.get("name")
            or ground_truth.get("function_name")
            or ""
        )
        predicted_name = parsed.get("name", "")
        name_ok = predicted_name.strip() == expected_name.strip()

        if not name_ok:
            return {
                "passed": False,
                "score":  0.2,  # partial credit: valid JSON produced
                "note":   f"Wrong fn name: got '{predicted_name}', expected '{expected_name}'",
            }

        # ── Check arguments ───────────────────────────────────────────────────
        expected_args = ground_truth.get("arguments", ground_truth.get("args", {}))
        predicted_args = parsed.get("arguments", parsed.get("args", {}))
        strict = self.cfg.get("strict", False)

        arg_score = self._score_args(expected_args, predicted_args, strict=strict)
        passed = arg_score >= (1.0 if strict else 0.8)

        return {
            "passed": passed,
            "score":  round((0.5 + 0.5 * arg_score), 3),  # name correct = 0.5, args = 0.5
            "note":   f"arg_match={arg_score:.2f}",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract the first JSON object from free-form model output."""
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # Regex fallback: find first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _score_args(
        expected: dict,
        predicted: dict,
        strict: bool = False,
    ) -> float:
        """Return 0–1 argument match score."""
        if not expected:
            return 1.0 if not predicted else 0.8

        if strict:
            return 1.0 if expected == predicted else 0.0

        # Fuzzy: what fraction of expected keys are correctly predicted?
        hits = 0
        for key, exp_val in expected.items():
            pred_val = predicted.get(key)
            if pred_val is None:
                continue
            if str(pred_val).strip().lower() == str(exp_val).strip().lower():
                hits += 1
            elif str(exp_val).strip().lower() in str(pred_val).strip().lower():
                hits += 0.5  # partial credit

        return hits / len(expected)
