"""Shared Modal image, volumes, and command builders for finetune + server apps."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal
import yaml

_file = Path(__file__).resolve()
try:
    LOCAL_REPO_ROOT = _file.parents[2]
except IndexError:
    LOCAL_REPO_ROOT = Path("/repo")

if (_file.parent / "experiments.yaml").is_file():
    EXPERIMENTS_PATH = _file.parent / "experiments.yaml"
else:
    EXPERIMENTS_PATH = Path("/repo/research/modal/experiments.yaml")

_EVAL_PROFILES_REL = "research/evals/configs/eval_profiles.yaml"
if (LOCAL_REPO_ROOT / _EVAL_PROFILES_REL).is_file():
    EVAL_PROFILES_PATH = LOCAL_REPO_ROOT / _EVAL_PROFILES_REL
else:
    EVAL_PROFILES_PATH = Path("/repo") / _EVAL_PROFILES_REL

REPO_ROOT = LOCAL_REPO_ROOT

HF_CACHE_PATH = "/root/.cache/huggingface"
FINETUNE_VOL_PATH = "/vol/finetuned"
LM_EVAL_OUTPUT = f"{FINETUNE_VOL_PATH}/results/lm_eval"
BASE_MODEL_ID = "openbmb/MiniCPM5-1B"

BASELINE_EXPERIMENT = "minicpm5-1b__modal-baseline"
BASELINE_RESULTS_JSON = f"{LM_EVAL_OUTPUT}/{BASELINE_EXPERIMENT}/results.json"

# Metric keys to prefer when picking a task's "primary" score, in priority
# order. Covers lm-eval-harness multiple-choice (acc), generation (exact_match),
# and code (pass@1) tasks so gates and model cards pick a real score, not a stderr.
_METRIC_PRIORITY = (
    "acc,none",
    "acc_norm,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "pass_at_1,create_test",
    "pass_at_1,none",
    "f1,none",
    "bleu,none",
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
finetune_vol = modal.Volume.from_name("slm-finetune", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv", "pyyaml")
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/repo",
        copy=True,
        ignore=[
            ".git/**",
            ".venv/**",
            "models/**",
            "results/**",
            "outputs/**",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            "**/node_modules/**",
        ],
    )
    .run_commands(
        "cd /repo && uv sync --frozen --group finetune --group lm-eval --no-dev"
    )
)

COMMON_ENV = {
    "TRUST_REMOTE_CODE": "true",
    "HF_HOME": HF_CACHE_PATH,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

DEFAULT_GPU = "A10G"
DEFAULT_KEEPALIVE_HOURS = 4.0
DEFAULT_SCALEDOWN_WINDOW = 3600  # max allowed by Modal (1h idle before scale-down)
DEFAULT_WORKER_TIMEOUT = 14400  # 4h per method call


def repo_env() -> dict[str, str]:
    return {**os.environ, **COMMON_ENV}


def reload_volumes() -> None:
    finetune_vol.reload()
    hf_cache_vol.reload()


def commit_volumes() -> None:
    finetune_vol.commit()
    hf_cache_vol.commit()


def load_experiments() -> dict[str, Any]:
    with EXPERIMENTS_PATH.open() as f:
        return yaml.safe_load(f) or {}


def apply_defaults(job: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    return {**defaults, **job}


def build_finetune_cmd(job: dict[str, Any], out_dir: str) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "research/finetune.py",
        "--preset",
        job.get("preset", "minicpm5-1b"),
        "--mode",
        job.get("mode", "lora"),
        "--dataset",
        job["dataset"],
        "--format",
        job["format"],
        "--out",
        out_dir,
    ]
    if job.get("max_steps") is not None:
        cmd.extend(["--max_steps", str(int(job["max_steps"]))])
    if job.get("epochs") is not None:
        cmd.extend(["--epochs", str(job["epochs"])])
    if job.get("dataset_config"):
        cmd.extend(["--dataset-config", job["dataset_config"]])
    if job.get("dataset_split"):
        cmd.extend(["--dataset-split", str(job["dataset_split"])])
    if job.get("max_samples") is not None:
        cmd.extend(["--dataset-max-samples", str(int(job["max_samples"]))])
    return cmd


def build_lm_eval_cmd(
    *,
    experiment_name: str,
    config: str,
    preset: str | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    compare_to: str | None = None,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--package",
        "slm-evals",
        "slm-lm-eval",
        "--config",
        config,
        "--experiment-name",
        experiment_name,
        "--output-dir",
        LM_EVAL_OUTPUT,
    ]
    if preset:
        cmd.extend(["--preset", preset])
    if model_path:
        cmd.extend(["--model", model_path])
    if adapter_path:
        cmd.extend(["--adapter", adapter_path])
    if compare_to:
        cmd.extend(["--compare-to", compare_to])
    return cmd


def prepare_jobs(
    *,
    job: str | None = None,
    max_steps: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = load_experiments()
    defaults = spec.get("defaults", {})
    jobs = spec.get("finetune", [])

    if job:
        jobs = [j for j in jobs if j.get("name") == job]
        if not jobs:
            raise SystemExit(
                f"Unknown job {job!r}; check research/modal/experiments.yaml"
            )

    prepared: list[dict[str, Any]] = []
    for raw in jobs:
        merged = apply_defaults(raw, defaults)
        if max_steps is not None:
            merged["max_steps"] = max_steps
        prepared.append(merged)
    return defaults, prepared
