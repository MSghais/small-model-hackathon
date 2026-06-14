"""
slm-lm-eval — Academic benchmarks via lm-evaluation-harness
============================================================
Run GSM8K, ARC, HellaSwag, and related tasks against presets and finetuned
checkpoints.

Usage:
    uv run --package slm-evals slm-lm-eval \\
      --config research/evals/configs/lm_eval_minicpm5.yaml \\
      --preset minicpm5-1b \\
      --experiment-name minicpm5-1b__baseline
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from slm_evals.lm_eval.preset_resolver import resolve_model_spec
from slm_evals.lm_eval.profiles import (
    config_path_for_profile,
    format_lm_eval_tasks,
    format_profiles_table,
)


def _ensure_lm_eval_models_registered() -> None:
    """Import lm-eval model backends so registry includes hf."""
    import lm_eval.models  # noqa: F401 — registers bundled backends when available

    try:
        import lm_eval.models.huggingface  # noqa: F401
    except ImportError:
        pass

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT = _REPO_ROOT / "results" / "lm_eval"

_METRIC_PRIORITY = (
    "acc,none",
    "acc_norm,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "f1,none",
    "bleu,none",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness benchmarks via slm-evals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Profiles: slm-lm-eval --list-profiles\n"
            "          slm-lm-eval --profile reasoning --preset minicpm5-1b\n"
            "All tasks: slm-lm-eval --list-tasks (requires uv sync --group lm-eval)"
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Show claim-matched lm-eval profiles and other eval suites",
    )
    parser.add_argument(
        "--list-profiles-all",
        action="store_true",
        help="Like --list-profiles but include agentic suites and external notes",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List lm-eval task names (from harness, or catalog fallback)",
    )
    parser.add_argument(
        "--list-tasks-all",
        action="store_true",
        help="List all lm-eval task names (can be long)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="NAME",
        help="Shorthand for --config (e.g. reasoning, understanding, code, smoke)",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--preset", type=str, default=None, help="models.yaml preset key")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF Hub id or merged checkpoint dir",
    )
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task names (overrides config)",
    )
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max samples per task")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help="Root directory for lm-eval results",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--compare-to",
        type=str,
        default=None,
        help="Path to baseline results.json for delta table",
    )
    return parser.parse_args()


def load_lm_eval_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("tasks", ["arc_easy", "hellaswag"])
    cfg.setdefault("num_fewshot", 0)
    cfg.setdefault("limit", None)
    cfg.setdefault("seed", 42)
    cfg.setdefault("batch_size", "auto")
    cfg.setdefault("device", "auto")
    cfg.setdefault("dtype", "bfloat16")
    cfg.setdefault("trust_remote_code", True)
    cfg.setdefault("output_dir", str(_DEFAULT_OUTPUT))
    return cfg


def merge_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    config_path = args.config
    if args.profile:
        if config_path:
            raise SystemExit("Pass only one of --profile or --config, not both.")
        config_path = str(config_path_for_profile(args.profile))
    if config_path:
        cfg = load_lm_eval_config(config_path)

    if args.tasks:
        cfg["tasks"] = args.tasks
    if args.num_fewshot is not None:
        cfg["num_fewshot"] = args.num_fewshot
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device
    if args.dtype is not None:
        cfg["dtype"] = args.dtype
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    cfg["preset"] = args.preset
    cfg["model_path"] = args.model
    cfg["adapter_path"] = args.adapter
    cfg["compare_to"] = args.compare_to or cfg.get("compare_to")

    if not cfg.get("experiment_name"):
        if args.experiment_name:
            cfg["experiment_name"] = args.experiment_name
        else:
            tag = args.preset or Path(args.model or "model").name
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg["experiment_name"] = f"{tag}__lm-eval__{ts}"
    elif args.experiment_name:
        cfg["experiment_name"] = args.experiment_name

    return cfg


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _primary_metric(task_metrics: dict[str, Any]) -> tuple[str, float] | None:
    for key in _METRIC_PRIORITY:
        if key in task_metrics and isinstance(task_metrics[key], (int, float)):
            return key, float(task_metrics[key])
    for key, value in task_metrics.items():
        if isinstance(value, (int, float)):
            return key, float(value)
    return None


def write_summary_md(
    path: Path,
    *,
    spec,
    cfg: dict[str, Any],
    results_payload: dict[str, Any],
) -> None:
    lines = [
        "# lm-eval summary",
        "",
        f"- experiment: `{cfg['experiment_name']}`",
        f"- model backend: `{spec.lm_eval_model}`",
        f"- base model: `{spec.base_model}`",
    ]
    if spec.adapter_path:
        lines.append(f"- adapter: `{spec.adapter_path}`")
    lines.extend(
        [
            f"- tasks: {', '.join(cfg['tasks'])}",
            f"- num_fewshot: {cfg.get('num_fewshot')}",
            f"- limit: {cfg.get('limit')}",
            f"- seed: {cfg.get('seed')}",
            "",
            "| task | metric | score |",
            "| --- | --- | ---: |",
        ]
    )

    task_results = results_payload.get("results", {})
    for task, metrics in sorted(task_results.items()):
        picked = _primary_metric(metrics)
        if picked:
            metric_name, score = picked
            lines.append(f"| {task} | {metric_name} | {score:.4f} |")
        else:
            lines.append(f"| {task} | — | — |")

    path.write_text("\n".join(lines) + "\n")


def compare_results(
    baseline_path: Path,
    candidate_path: Path,
    *,
    cfg: dict[str, Any],
) -> str:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())

    warnings: list[str] = []
    for key in ("seed", "limit", "num_fewshot"):
        b_cfg = baseline.get("run_meta", {}).get(key, baseline.get("config", {}).get(key))
        c_cfg = candidate.get("run_meta", {}).get(key, candidate.get("config", {}).get(key))
        if b_cfg != c_cfg and b_cfg is not None and c_cfg is not None:
            warnings.append(f"Mismatch on {key}: baseline={b_cfg!r} candidate={c_cfg!r}")

    b_tasks = set(baseline.get("results", {}))
    c_tasks = set(candidate.get("results", {}))
    shared = sorted(b_tasks & c_tasks)
    if not shared:
        warnings.append("No shared tasks between baseline and candidate.")

    lines = [
        "# lm-eval comparison",
        "",
        f"- baseline: `{baseline_path}`",
        f"- candidate: `{candidate_path}`",
        f"- candidate experiment: `{cfg['experiment_name']}`",
        "",
    ]
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {w}" for w in warnings)
        lines.append("")

    lines.extend(["| task | baseline | candidate | delta |", "| --- | ---: | ---: | ---: |"])
    for task in shared:
        b_metric = _primary_metric(baseline["results"][task])
        c_metric = _primary_metric(candidate["results"][task])
        if not b_metric or not c_metric:
            continue
        _, b_score = b_metric
        _, c_score = c_metric
        delta = c_score - b_score
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {task} | {b_score:.4f} | {c_score:.4f} | {sign}{delta:.4f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    if args.list_profiles or args.list_profiles_all:
        print(
            format_profiles_table(
                include_suites=args.list_profiles_all,
                include_external=args.list_profiles_all,
            )
        )
        return 0

    if args.list_tasks or args.list_tasks_all:
        print(format_lm_eval_tasks(limit=0 if args.list_tasks_all else 80))
        return 0

    cfg = merge_config(args)

    if not cfg.get("preset") and not cfg.get("model_path"):
        print("Error: pass --preset or --model (or set in config).", file=sys.stderr)
        return 1

    spec = resolve_model_spec(
        preset=cfg.get("preset"),
        model_path=cfg.get("model_path"),
        adapter_path=cfg.get("adapter_path"),
        trust_remote_code=cfg.get("trust_remote_code"),
        dtype=cfg.get("dtype"),
        device=cfg.get("device"),
    )

    out_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import lm_eval
    except ImportError as exc:
        print(
            "lm-eval is not installed. Run: uv sync --group lm-eval",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    _ensure_lm_eval_models_registered()

    seed = int(cfg.get("seed", 42))
    model_args = dict(spec.model_args)
    eval_device = cfg.get("device")
    if spec.lm_eval_model == "hf":
        model_args.pop("device", None)
    else:
        eval_device = None

    eval_results = lm_eval.simple_evaluate(
        model=spec.lm_eval_model,
        model_args=model_args,
        tasks=cfg["tasks"],
        num_fewshot=cfg.get("num_fewshot"),
        batch_size=cfg.get("batch_size"),
        device=eval_device,
        limit=cfg.get("limit"),
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        log_samples=False,
    )

    if eval_results is None:
        print("lm-eval returned no results.", file=sys.stderr)
        return 1

    run_meta = {
        "experiment_name": cfg["experiment_name"],
        "preset": spec.preset_key,
        "lm_eval_model": spec.lm_eval_model,
        "base_model": spec.base_model,
        "adapter_path": spec.adapter_path,
        "tasks": cfg["tasks"],
        "num_fewshot": cfg.get("num_fewshot"),
        "limit": cfg.get("limit"),
        "seed": seed,
        "batch_size": cfg.get("batch_size"),
        "device": cfg.get("device"),
        "dtype": cfg.get("dtype"),
        "git_hash": _git_hash(),
    }

    payload = dict(eval_results)
    payload["run_meta"] = run_meta

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, default=str))

    summary_path = out_dir / "summary.md"
    write_summary_md(summary_path, spec=spec, cfg=cfg, results_payload=payload)

    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2))

    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")

    compare_to = cfg.get("compare_to")
    if compare_to:
        compare_path = out_dir / "comparison.md"
        compare_text = compare_results(
            Path(compare_to),
            results_path,
            cfg=cfg,
        )
        compare_path.write_text(compare_text)
        print(f"Wrote {compare_path}")
        print()
        print(compare_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
