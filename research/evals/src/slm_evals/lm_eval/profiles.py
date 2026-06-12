"""Load and display eval profile catalog from configs/eval_profiles.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[5]
_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
_PROFILES_FILE = _CONFIGS_DIR / "eval_profiles.yaml"


def load_profiles_catalog() -> dict[str, Any]:
    if not _PROFILES_FILE.is_file():
        raise FileNotFoundError(f"Profile catalog not found: {_PROFILES_FILE}")
    with open(_PROFILES_FILE) as f:
        return yaml.safe_load(f) or {}


def config_path_for_profile(name: str) -> Path:
    catalog = load_profiles_catalog()
    profiles = catalog.get("profiles", {})
    if name not in profiles:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown lm-eval profile {name!r}. Known: {known}")
    rel = profiles[name].get("config")
    if not rel:
        raise ValueError(f"Profile {name!r} has no config file in eval_profiles.yaml")
    path = _CONFIGS_DIR / rel
    if not path.is_file():
        raise FileNotFoundError(f"Profile config missing: {path}")
    return path


def _format_tasks(tasks: list[str] | None) -> str:
    if not tasks:
        return "—"
    return ", ".join(tasks)


def format_profiles_table(*, include_suites: bool = False, include_external: bool = False) -> str:
    catalog = load_profiles_catalog()
    lines = [
        "Eval profiles (slm-lm-eval — use --profile NAME or --config PATH)",
        "",
        f"{'Profile':<16} {'Claim':<28} {'Tasks'}",
        f"{'-'*16} {'-'*28} {'-'*40}",
    ]
    for name, meta in sorted(catalog.get("profiles", {}).items()):
        claim = meta.get("claim", "")
        tasks = _format_tasks(meta.get("tasks"))
        config = meta.get("config", "")
        lines.append(f"{name:<16} {claim:<28} {tasks}")
        if meta.get("description"):
            lines.append(f"{'':16} config: {config}")
            lines.append(f"{'':16} {meta['description']}")
            lines.append("")

    if include_suites:
        lines.extend(
            [
                "",
                "Other suites (separate CLI — see docs/eval_profiles.md)",
                "",
                f"{'Suite':<16} {'Tool':<14} {'Claim':<28}",
                f"{'-'*16} {'-'*14} {'-'*28}",
            ]
        )
        for name, meta in sorted(catalog.get("suites", {}).items()):
            tool = meta.get("tool", "")
            claim = meta.get("claim", "")
            lines.append(f"{name:<16} {tool:<14} {claim:<28}")
            benchmarks = meta.get("benchmarks")
            if benchmarks:
                lines.append(f"{'':16} benchmarks: {', '.join(benchmarks)}")
            if meta.get("command"):
                lines.append(f"{'':16} {meta['command']}")
            lines.append("")

    if include_external:
        lines.extend(
            [
                "",
                "External (not integrated in this repo)",
                "",
            ]
        )
        for name, meta in sorted(catalog.get("external", {}).items()):
            lines.append(f"- {name}: {meta.get('claim', '')} — {meta.get('description', '')}")

    lines.extend(
        [
            "",
            "Examples:",
            "  slm-lm-eval --list-profiles",
            "  slm-lm-eval --profile reasoning --preset minicpm5-1b --experiment-name baseline",
            "  slm-benchmark --list-benchmarks",
        ]
    )
    return "\n".join(lines)


def list_lm_eval_task_names(limit: int = 80) -> list[str]:
    try:
        from lm_eval.tasks import TaskManager

        tm = TaskManager()
        names = sorted(tm.all_tasks)
        return names[:limit] if limit else names
    except ImportError:
        return []


def format_lm_eval_tasks(*, limit: int = 80) -> str:
    tasks = list_lm_eval_task_names(limit=limit)
    if not tasks:
        catalog = load_profiles_catalog()
        seen: set[str] = set()
        for section in ("profiles",):
            for meta in catalog.get(section, {}).values():
                for task in meta.get("tasks") or []:
                    seen.add(task)
        tasks = sorted(seen)
        header = (
            "lm-eval not installed — showing tasks referenced in eval_profiles.yaml only.\n"
            "Install: uv sync --group lm-eval\n"
        )
    else:
        header = f"lm-eval tasks (first {len(tasks)}; run with --list-tasks --all for full list):\n"

    lines = [header.rstrip(), ""]
    col = 4
    for i in range(0, len(tasks), col):
        chunk = tasks[i : i + col]
        lines.append("  ".join(f"{t:<22}" for t in chunk))
    lines.append("")
    lines.append("Use in a run: slm-lm-eval --profile reasoning --preset minicpm5-1b")
    lines.append("Or override:  slm-lm-eval --config ... --tasks gsm8k arc_easy")
    return "\n".join(lines)


def format_agentic_benchmarks() -> str:
    from slm_evals.run_benchmark import BENCHMARK_REGISTRY

    catalog = load_profiles_catalog()
    lines = [
        "Agentic benchmarks (slm-benchmark — use --benchmarks NAME)",
        "",
        f"{'Key':<12} {'Measures'}",
        f"{'-'*12} {'-'*50}",
        "bfcl         Single-turn function calling (Berkeley FC Leaderboard)",
        "tau_bench    Multi-turn tool + user simulation",
        "gaia         End-to-end assistant tasks (reasoning + tools)",
        "swe_bench    Code patch generation (SWE-bench Verified)",
        "all          Run all four benchmarks",
        "",
        "Registered in code:",
    ]
    for key in sorted(BENCHMARK_REGISTRY):
        lines.append(f"  - {key}")

    lines.append("")
    lines.append("Preset suites from eval_profiles.yaml:")
    for name, meta in sorted(catalog.get("suites", {}).items()):
        if meta.get("tool") != "slm-benchmark":
            continue
        bms = meta.get("benchmarks") or []
        lines.append(f"  {name}: {', '.join(bms)}")
        lines.append(f"    {meta.get('description', '')}")

    lines.extend(
        [
            "",
            "Example:",
            "  slm-benchmark --model openbmb/MiniCPM5-1B --benchmarks bfcl --max-samples 20",
        ]
    )
    return "\n".join(lines)
