"""
Modal GPU pipeline for research/finetune.py + slm-lm-eval.

Skill-matrix pipeline: train -> eval -> gate -> publish.
Each job in experiments.yaml fine-tunes one QLoRA adapter for a skill
(math, science, coding, reasoning, teaching, ...), evaluates it against the
matching slm-lm-eval profile vs. a per-profile baseline, checks the result
against `goals`, and (only if the gate passes) publishes the adapter to the
Hugging Face Hub.

Run from repo root:
    modal run research/modal/finetune_app.py
    modal run research/modal/finetune_app.py --eval-only
    modal run research/modal/finetune_app.py --job math-lora --max-steps 20
    modal run research/modal/finetune_app.py --category science
    modal run research/modal/finetune_app.py --no-publish --no-pull
    modal run research/modal/finetune_app.py::publish_only --job math-lora
    modal run research/modal/finetune_app.py::pull --category math
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

_modal_dir = Path(__file__).resolve().parent
if str(_modal_dir) not in sys.path:
    sys.path.insert(0, str(_modal_dir))

from _common import (
    BASE_MODEL_ID,
    FINETUNE_VOL_PATH,
    HF_CACHE_PATH,
    LM_EVAL_OUTPUT,
    build_finetune_cmd,
    build_lm_eval_cmd,
    check_gate_files,
    commit_volumes,
    config_for_profile,
    finetune_vol,
    hf_cache_vol,
    hf_secret,
    image,
    job_gpu,
    load_experiments,
    prepare_jobs,
    publish_adapter_files,
    reload_volumes,
    repo_env,
)

APP_NAME = "slm-finetune-benchmark"

app = modal.App(APP_NAME, image=image)


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

    cmd = build_finetune_cmd(job, out_dir)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd="/repo", check=True, env=repo_env())

    commit_volumes()

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
    reload_volumes()

    if adapter_path:
        adapter_dir = Path(adapter_path)
        adapter_cfg = adapter_dir / "adapter_config.json"
        if not adapter_cfg.is_file():
            raise FileNotFoundError(
                f"LoRA adapter not visible at {adapter_path} "
                f"(missing {adapter_cfg.name}). "
                "If training just finished, retry after volume commit/reload."
            )

    cmd = build_lm_eval_cmd(
        experiment_name=experiment_name,
        config=config,
        preset=preset,
        model_path=model_path,
        adapter_path=adapter_path,
        compare_to=compare_to,
    )
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd="/repo", check=False, env=repo_env())

    commit_volumes()

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
        "ok": proc.returncode == 0 and results_json.is_file(),
    }


@app.function(volumes={FINETUNE_VOL_PATH: finetune_vol}, timeout=300)
def check_gate(
    *,
    candidate_results_path: str,
    baseline_results_path: str | None,
    goals: dict[str, Any],
) -> dict[str, Any]:
    """Check a candidate's lm-eval results against `goals` (Hub publish gate)."""
    reload_volumes()
    return check_gate_files(
        candidate_results_path=candidate_results_path,
        baseline_results_path=baseline_results_path,
        goals=goals,
    )


@app.function(
    volumes={FINETUNE_VOL_PATH: finetune_vol},
    secrets=[hf_secret],
    timeout=900,
)
def publish_adapter(
    *,
    job: dict[str, Any],
    adapter_dir: str,
    gate_result: dict[str, Any],
    candidate_results_path: str,
    baseline_results_path: str | None,
) -> dict[str, Any]:
    """Write a model card and push the adapter to the Hub, but only if the gate passed."""
    reload_volumes()
    return publish_adapter_files(
        job=job,
        adapter_dir=adapter_dir,
        gate_result=gate_result,
        candidate_results_path=candidate_results_path,
        baseline_results_path=baseline_results_path,
    )


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n--- summary ---")
    print(f"{'skill':<18} {'category':<12} {'gate':<6} {'published':<10} hub_repo")
    for row in rows:
        gate = "PASS" if row.get("gate_passed") else "fail"
        published = "yes" if row.get("published") else "no"
        print(
            f"{row['name']:<18} {row.get('category') or '-':<12} {gate:<6} "
            f"{published:<10} {row.get('hub_repo') or '-'}"
        )


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
    defaults, prepared = prepare_jobs(job=job, max_steps=max_steps)

    eval_cfg = lm_eval_config or defaults.get(
        "lm_eval_config", "research/evals/configs/lm_eval_smoke.yaml"
    )
    base_cfg = baseline_config or defaults.get(
        "baseline_config", "research/evals/configs/lm_eval_compare_study.yaml"
    )

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
