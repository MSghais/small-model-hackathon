"""
benchmarks/base.py
───────────────────
Abstract base class every benchmark extends.
"""

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from typing import Any


class BaseBenchmark(ABC):
    """
    Subclass this for each benchmark.

    Concrete subclasses must implement:
        load_dataset()  → list of sample dicts
        evaluate_sample(sample, prediction) → dict with keys:
            passed (bool), score (float 0-1), note (str)
        build_prompt(sample) → str
    """

    name: str = "base"

    def __init__(
        self,
        model_bundle: dict[str, Any],
        max_samples: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        benchmark_cfg: dict | None = None,
    ):
        self.model_bundle   = model_bundle
        self.generate       = model_bundle["generate_fn"]
        self.max_samples    = max_samples
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.cfg            = benchmark_cfg or {}

    # ── Must implement ────────────────────────────────────────────────────────

    @abstractmethod
    def load_dataset(self) -> list[dict]:
        """Return a list of sample dicts."""

    @abstractmethod
    def build_prompt(self, sample: dict) -> str:
        """Convert a sample dict into the prompt string sent to the model."""

    @abstractmethod
    def evaluate_sample(self, sample: dict, prediction: str) -> dict:
        """
        Score one prediction.

        Returns dict:
            passed  (bool)
            score   (float, 0–1)
            note    (str, optional explanation)
        """

    # ── Orchestration — override if needed ───────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Run all samples and aggregate results."""
        dataset = self.load_dataset()
        if self.max_samples:
            dataset = dataset[: self.max_samples]

        samples_out = []
        total_latency = 0.0
        errors = 0

        for sample in dataset:
            prompt = self.build_prompt(sample)
            t0 = time.perf_counter()
            try:
                prediction = self.generate(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                latency = time.perf_counter() - t0
                eval_result = self.evaluate_sample(sample, prediction)
            except Exception as exc:
                latency = time.perf_counter() - t0
                errors += 1
                eval_result = {
                    "passed": False,
                    "score":  0.0,
                    "note":   f"ERROR: {exc}",
                }
                prediction = ""

            samples_out.append(
                {
                    "id":         sample.get("id", ""),
                    "prediction": prediction,
                    "latency_s":  round(latency, 3),
                    **eval_result,
                }
            )
            total_latency += latency

        passed = sum(1 for s in samples_out if s["passed"])
        total  = len(samples_out)
        score  = (passed / total) if total else 0.0
        avg_lat = round(total_latency / total, 3) if total else 0.0

        return {
            "benchmark":    self.name,
            "passed":       passed,
            "total":        total,
            "score":        score,
            "error_count":  errors,
            "avg_latency_s": avg_lat,
            "samples":      samples_out,
        }
