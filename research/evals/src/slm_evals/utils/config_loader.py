"""
utils/config_loader.py
───────────────────────
Load experiment config from a YAML file OR build one from CLI args.
"""

from __future__ import annotations
import datetime
from pathlib import Path
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    """Parse a YAML config file into a flat config dict."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    _fill_defaults(cfg)
    return cfg


def build_config_from_args(args) -> dict[str, Any]:
    """Convert argparse Namespace into a config dict."""
    benchmarks = args.benchmarks if args.benchmarks else ["all"]
    cfg: dict[str, Any] = {
        "model_path":    args.model,
        "benchmarks":    benchmarks,
        "max_samples":   args.max_samples,
        "output_dir":    args.output_dir,
        "experiment_name": args.experiment_name,
        "device":        args.device,
        "dtype":         args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature":   args.temperature,
        "benchmark_overrides": {},
    }
    _fill_defaults(cfg)
    return cfg


def _fill_defaults(cfg: dict[str, Any]) -> None:
    """In-place: fill any missing keys with sensible defaults."""
    if not cfg.get("experiment_name"):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = Path(cfg.get("model_path", "unknown")).name
        cfg["experiment_name"] = f"{model_tag}__{ts}"

    cfg.setdefault("device",         "auto")
    cfg.setdefault("dtype",          "bfloat16")
    cfg.setdefault("max_new_tokens",  512)
    cfg.setdefault("temperature",     0.0)
    cfg.setdefault("max_samples",     None)
    cfg.setdefault("output_dir",      "results")
    cfg.setdefault("benchmark_overrides", {})
