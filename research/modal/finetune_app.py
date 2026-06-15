"""
Modal GPU pipeline for research/finetune.py + slm-lm-eval.

Run from repo root:
    modal run research/modal/finetune_app.py
    modal run research/modal/finetune_app.py --eval-only
    modal run research/modal/finetune_app.py --job lesson-lora --max-steps 20
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal
import yaml

_file = Path(__file__).resolve()
try:
    _LOCAL_REPO_ROOT = _file.parents[2]
except IndexError:
    _LOCAL_REPO_ROOT = Path("/repo")

if (_file.parent / "experiments.yaml").is_file():
    EXPERIMENTS_PATH = _file.parent / "experiments.yaml"
else:
    EXPERIMENTS_PATH = Path("/repo/research/modal/experiments.yaml")

# Local image build copies this tree into the container at /repo.
REPO_ROOT = _LOCAL_REPO_ROOT

APP_NAME = "slm-finetune-benchmark"
HF_CACHE_PATH = "/root/.cache/huggingface"
FINETUNE_VOL_PATH = "/vol/finetuned"
LM_EVAL_OUTPUT = f"{FINETUNE_VOL_PATH}/results/lm_eval"

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

app = modal.App(APP_NAME, image=image)

COMMON_ENV = {
    "TRUST_REMOTE_CODE": "true",
    "HF_HOME": HF_CACHE_PATH,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _repo_env() -> dict[str, str]:
    return {**os.environ, **COMMON_ENV}


def _load_experiments() -> dict[str, Any]:
    with EXPERIMENTS_PATH.open() as f:
        return yaml.safe_load(f) or {}


def _apply_defaults(job: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = {**defaults, **job}
    return merged


def _build_finetune_cmd(job: dict[str, Any], out_dir: str) -> list[str]:
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


def _build_lm_eval_cmd(
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


BASELINE_EXPERIMENT = "minicpm5-1b__modal-baseline"
BASELINE_RESULTS_JSON = f"{LM_EVAL_OUTPUT}/{BASELINE_EXPERIMENT}/results.json"


def _reload_volumes() -> None:
    """Pick up commits from other containers before reading Volume paths."""
    finetune_vol.reload()
    hf_cache_vol.reload()


@app.function(
    gpu="A10G",
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        FINETUNE_VOL_PATH: finetune_vol,
    },
    secrets=[hf_secret],
    timeout=7200,
)
def finetune_one(job: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune one dataset job; persist adapter to Modal Volume."""
    name = job["name"]
    out_dir = f"{FINETUNE_VOL_PATH}/{name}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = _build_finetune_cmd(job, out_dir)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd="/repo", check=True, env=_repo_env())

    finetune_vol.commit()
    hf_cache_vol.commit()

    results_path = Path(out_dir) / "training_results.json"
    payload = json.loads(results_path.read_text())
    payload["job_name"] = name
    return payload


@app.function(
    gpu="A10G",
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        FINETUNE_VOL_PATH: finetune_vol,
    },
    secrets=[hf_secret],
    timeout=3600,
)
def run_lm_eval(
    *,
    experiment_name: str,
    config: str = "research/evals/configs/lm_eval_smoke.yaml",
    preset: str | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    compare_to: str | None = None,
) -> dict[str, Any]:
    """Run slm-lm-eval on base model or finetuned checkpoint."""
    _reload_volumes()

    if adapter_path:
        adapter_dir = Path(adapter_path)
        adapter_cfg = adapter_dir / "adapter_config.json"
        if not adapter_cfg.is_file():
            raise FileNotFoundError(
                f"LoRA adapter not visible at {adapter_path} "
                f"(missing {adapter_cfg.name}). "
                "If training just finished, retry after volume commit/reload."
            )

    cmd = _build_lm_eval_cmd(
        experiment_name=experiment_name,
        config=config,
        preset=preset,
        model_path=model_path,
        adapter_path=adapter_path,
        compare_to=compare_to,
    )
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/repo", check=False, env=_repo_env())

    finetune_vol.commit()
    hf_cache_vol.commit()

    out_root = Path(LM_EVAL_OUTPUT) / experiment_name
    results_json = out_root / "results.json"
    summary_md = out_root / "summary.md"
    comparison_md = out_root / "comparison.md"

    return {
        "experiment_name": experiment_name,
        "config": config,
        "preset": preset,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "compare_to": compare_to,
        "results_json": str(results_json),
        "summary_md": str(summary_md),
        "comparison_md": str(comparison_md) if comparison_md.is_file() else None,
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0,
    }


@app.local_entrypoint()
def main(
    train: bool = True,
    eval_only: bool = False,
    skip_baseline: bool = False,
    parallel: bool = False,
    job: str | None = None,
    max_steps: int | None = None,
    lm_eval_config: str | None = None,
    baseline_config: str | None = None,
):
    """
    Orchestrate baseline lm-eval, finetune jobs, and post-train evals.

    Examples:
        modal run research/modal/finetune_app.py
        modal run research/modal/finetune_app.py --job lesson-lora --max-steps 20
        modal run research/modal/finetune_app.py --eval-only
    """
    spec = _load_experiments()
    defaults = spec.get("defaults", {})
    jobs = spec.get("finetune", [])

    if job:
        jobs = [j for j in jobs if j.get("name") == job]
        if not jobs:
            raise SystemExit(f"Unknown job {job!r}; check research/modal/experiments.yaml")

    eval_cfg = lm_eval_config or defaults.get(
        "lm_eval_config", "research/evals/configs/lm_eval_smoke.yaml"
    )
    base_cfg = baseline_config or defaults.get(
        "baseline_config", "research/evals/configs/lm_eval_compare_study.yaml"
    )

    prepared: list[dict[str, Any]] = []
    for raw in jobs:
        merged = _apply_defaults(raw, defaults)
        if max_steps is not None:
            merged["max_steps"] = max_steps
        prepared.append(merged)

    baseline_result: dict[str, Any] | None = None
    compare_path: str | None = None

    if not eval_only and not skip_baseline:
        print("--- baseline lm-eval ---")
        baseline_result = run_lm_eval.remote(
            experiment_name=BASELINE_EXPERIMENT,
            config=base_cfg,
            preset=defaults.get("preset", "minicpm5-1b"),
        )
        print(json.dumps(baseline_result, indent=2))
        if not baseline_result.get("ok"):
            print("Warning: baseline lm-eval failed; continuing without compare_to")
        elif baseline_result.get("results_json"):
            compare_path = baseline_result["results_json"]
    elif skip_baseline:
        compare_path = BASELINE_RESULTS_JSON
        print(f"--- skipping baseline; compare_to {compare_path} ---")
    elif eval_only:
        compare_path = BASELINE_RESULTS_JSON
        print(f"--- eval-only; compare_to {compare_path} if present ---")

    train_results: list[dict[str, Any]] = []
    if train and not eval_only:
        print(f"--- finetune ({len(prepared)} jobs, parallel={parallel}) ---")
        if parallel:
            train_results = list(finetune_one.map(prepared))
        else:
            for j in prepared:
                print(f"Training {j['name']}...")
                train_results.append(finetune_one.remote(j))

        for r in train_results:
            print(json.dumps(r, indent=2))

    if eval_only and not prepared:
        raise SystemExit("eval_only requires --job or finetune entries in experiments.yaml")

    eval_targets: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    if eval_only:
        for j in prepared:
            eval_targets.append((j, None))
    else:
        for j, r in zip(prepared, train_results):
            eval_targets.append((j, r))

    print("--- post-train lm-eval ---")
    for j, train_payload in eval_targets:
        job_name = j["name"]
        adapter_path = (
            train_payload["output_dir"]
            if train_payload
            else f"{FINETUNE_VOL_PATH}/{job_name}"
        )
        exp_name = f"{job_name}__modal-lm-eval"
        eval_result = run_lm_eval.remote(
            experiment_name=exp_name,
            config=eval_cfg,
            model_path="openbmb/MiniCPM5-1B",
            adapter_path=adapter_path,
            compare_to=compare_path,
        )
        print(json.dumps(eval_result, indent=2))

    print("\nDone. Pull artifacts with:")
    print(f"  modal volume get slm-finetune lesson-lora ./models/finetuned/lesson-lora")
    print(f"  modal volume get slm-finetune results/lm_eval ./results/lm_eval")
