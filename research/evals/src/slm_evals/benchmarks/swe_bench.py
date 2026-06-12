"""
benchmarks/swe_bench.py
────────────────────────
SWE-bench Verified — agentic coding benchmark.

What it tests: Given a GitHub issue description + repository context,
can the model produce a patch that fixes the bug (passes test suite)?

Dataset: princeton-nlp/SWE-bench_Verified on HF Hub (500 human-verified tasks).

Scoring:
    Offline mode (default): Checks patch structural validity + keyword heuristics.
    Full mode (cfg["full_eval"]=True): Runs the patch in a Docker sandbox and
    executes the test suite. Requires Docker + swebench[eval] installed.

Note: Full end-to-end SWE-bench eval requires the official harness
      (https://github.com/princeton-nlp/SWE-bench). This adapter wraps
      the offline/structural scoring path for lightweight local use,
      and delegates to the harness when full_eval is requested.
"""

from __future__ import annotations
import re
from typing import Any

from slm_evals.benchmarks.base import BaseBenchmark


SYSTEM_PROMPT = """\
You are an expert software engineer.
You will be given a GitHub issue and the relevant source code.
Produce a unified diff patch that fixes the issue.
Output ONLY the patch, starting with --- and ending with the last +++ hunk.
Do not include any explanation.
"""


class SWEBenchmark(BaseBenchmark):
    """
    SWE-bench Verified adapter.

    Config keys (benchmark_overrides.swe_bench):
        data_path       – local JSONL
        full_eval       – bool (default False); run actual test harness
        context_lines   – int, how many lines of file context to include (default 80)
        difficulty      – list[str] filter by difficulty label (optional)
    """

    name = "swe_bench"

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
        return samples

    def _load_from_hub(self) -> list[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        ds = load_dataset(
            "princeton-nlp/SWE-bench_Verified",
            split="test",
            trust_remote_code=True,
        )
        return list(ds)

    # ── Prompt ────────────────────────────────────────────────────────────────

    def build_prompt(self, sample: dict) -> str:
        issue_text   = sample.get("problem_statement", sample.get("issue", ""))
        repo         = sample.get("repo", "unknown/repo")
        hints        = sample.get("hints_text", "")
        context_snip = self._build_context_snippet(sample)

        return (
            f"{SYSTEM_PROMPT}\n"
            f"Repository: {repo}\n\n"
            f"Issue:\n{issue_text}\n\n"
            f"{'Hints: ' + hints + chr(10) if hints else ''}"
            f"Relevant code:\n{context_snip}\n\n"
            f"Patch:"
        )

    def _build_context_snippet(self, sample: dict) -> str:
        """Pull relevant file snippets from the sample if available."""
        n = self.cfg.get("context_lines", 80)

        # SWE-bench Verified includes patch/test files fields
        base_commit  = sample.get("base_commit", "")
        patch        = sample.get("patch", "")          # ground truth patch (don't expose to model)
        test_patch   = sample.get("test_patch", "")

        # We expose only the files mentioned in the issue, not the patch itself
        file_names = re.findall(r"[\w/]+\.py", sample.get("problem_statement", ""))
        if file_names:
            return f"[Files likely relevant: {', '.join(set(file_names[:5]))}]\n(Fetch via repo checkout at {base_commit})"
        return "(No inline context available — use repo checkout for full context)"

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_sample(self, sample: dict, prediction: str) -> dict:
        if self.cfg.get("full_eval", False):
            return self._full_harness_eval(sample, prediction)
        return self._structural_eval(sample, prediction)

    def _structural_eval(self, sample: dict, prediction: str) -> dict:
        """
        Lightweight offline scoring:
        - Is the output a valid unified diff?
        - Does it touch any of the expected files?
        - Does it contain meaningful change lines (+/-)?
        """
        is_diff    = self._looks_like_diff(prediction)
        expected_f = self._expected_files(sample)
        touches_f  = self._patch_touches_files(prediction, expected_f)
        has_changes = bool(re.search(r"^[+-][^+-]", prediction, re.MULTILINE))

        score = sum([is_diff * 0.4, touches_f * 0.4, has_changes * 0.2])
        passed = score >= 0.6

        return {
            "passed": passed,
            "score":  round(score, 3),
            "note": (
                f"valid_diff={is_diff}  "
                f"touches_expected_files={touches_f}  "
                f"has_changes={has_changes}"
            ),
        }

    def _full_harness_eval(self, sample: dict, prediction: str) -> dict:
        """
        Delegate to the official SWE-bench evaluation harness.
        Requires: pip install swebench  AND  Docker running.

        Returns pass/fail based on whether tests pass after applying the patch.
        """
        try:
            from swebench.harness.run_evaluation import run_instances
        except ImportError:
            raise ImportError(
                "pip install swebench  (and ensure Docker is running)"
            )

        instance_id = sample.get("instance_id", sample.get("id", "unknown"))
        result = run_instances(
            predictions={instance_id: {"model_patch": prediction}},
            instances=[sample],
            run_id="slm_bench_eval",
        )
        resolved = result.get(instance_id, {}).get("resolved", False)
        return {
            "passed": resolved,
            "score":  1.0 if resolved else 0.0,
            "note":   "full harness eval",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _looks_like_diff(text: str) -> bool:
        return bool(re.search(r"^(---|\+\+\+|@@)", text, re.MULTILINE))

    @staticmethod
    def _expected_files(sample: dict) -> list[str]:
        patch = sample.get("patch", "")
        return re.findall(r"(?:---|\+\+\+) [ab]/(.+\.py)", patch)

    @staticmethod
    def _patch_touches_files(prediction: str, expected_files: list[str]) -> float:
        if not expected_files:
            return 0.5  # can't verify, give benefit of doubt
        pred_files = re.findall(r"(?:---|\+\+\+) [ab]/(.+\.py)", prediction)
        hits = set(pred_files) & set(expected_files)
        return len(hits) / len(expected_files)
