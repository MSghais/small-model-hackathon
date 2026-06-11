"""
utils/reporter.py
──────────────────
Persist benchmark results to JSON, Markdown, and CSV.
"""

from __future__ import annotations
import csv
import json
import datetime
from pathlib import Path
from typing import Any


class Reporter:
    def __init__(
        self,
        output_dir: str,
        experiment_name: str,
        model_path: str,
    ):
        self.output_dir     = Path(output_dir)
        self.experiment_name = experiment_name
        self.model_path     = model_path
        self.timestamp      = datetime.datetime.now().isoformat(timespec="seconds")

        self.run_dir = self.output_dir / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def save(self, all_results: dict[str, Any]) -> dict[str, str]:
        """Write all three output formats. Returns dict of format → filepath."""
        payload = self._build_payload(all_results)
        paths = {
            "json":     self._write_json(payload),
            "csv":      self._write_csv(payload),
            "markdown": self._write_markdown(payload),
        }
        return paths

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_payload(self, all_results: dict[str, Any]) -> dict[str, Any]:
        aggregate_score = (
            sum(r["score"] for r in all_results.values()) / len(all_results)
            if all_results else 0.0
        )
        return {
            "experiment_name":  self.experiment_name,
            "timestamp":        self.timestamp,
            "model_path":       self.model_path,
            "aggregate_score":  round(aggregate_score, 4),
            "benchmarks":       all_results,
        }

    def _write_json(self, payload: dict) -> str:
        path = self.run_dir / "results.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return str(path)

    def _write_csv(self, payload: dict) -> str:
        path = self.run_dir / "results.csv"
        rows = []
        for bench_name, result in payload["benchmarks"].items():
            base_row = {
                "experiment":      payload["experiment_name"],
                "timestamp":       payload["timestamp"],
                "model_path":      payload["model_path"],
                "benchmark":       bench_name,
                "score":           round(result["score"], 4),
                "passed":          result["passed"],
                "total":           result["total"],
                "error_count":     result.get("error_count", 0),
                "avg_latency_s":   result.get("avg_latency_s", ""),
            }
            rows.append(base_row)

        if rows:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        return str(path)

    def _write_markdown(self, payload: dict) -> str:
        path = self.run_dir / "report.md"
        lines = [
            f"# Benchmark Report — {payload['experiment_name']}",
            "",
            f"| Field | Value |",
            f"|---|---|",
            f"| **Timestamp** | {payload['timestamp']} |",
            f"| **Model** | `{payload['model_path']}` |",
            f"| **Aggregate Score** | **{payload['aggregate_score']:.2%}** |",
            "",
            "## Results by Benchmark",
            "",
            "| Benchmark | Score | Passed | Total | Errors | Avg Latency |",
            "|---|---|---|---|---|---|",
        ]

        for bench_name, result in payload["benchmarks"].items():
            lines.append(
                f"| {bench_name} "
                f"| {result['score']:.2%} "
                f"| {result['passed']} "
                f"| {result['total']} "
                f"| {result.get('error_count', 0)} "
                f"| {result.get('avg_latency_s', 'n/a')}s |"
            )

        lines += ["", "## Per-Sample Details", ""]

        for bench_name, result in payload["benchmarks"].items():
            lines += [
                f"### {bench_name.upper()}",
                "",
                "| # | Sample ID | Status | Score | Note |",
                "|---|---|---|---|---|",
            ]
            for i, sample in enumerate(result.get("samples", []), 1):
                status = "✅" if sample.get("passed") else "❌"
                lines.append(
                    f"| {i} "
                    f"| {sample.get('id', '—')} "
                    f"| {status} "
                    f"| {sample.get('score', ''):.2f} "
                    f"| {sample.get('note', '')} |"
                )
            lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return str(path)
