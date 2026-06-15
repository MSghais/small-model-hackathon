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
    .pip_install("uv", "pyyaml", "huggingface_hub")
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
    category: str | None = None,
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
    if category:
        jobs = [j for j in jobs if j.get("category") == category]
        if not jobs:
            raise SystemExit(f"No jobs with category {category!r}")

    prepared: list[dict[str, Any]] = []
    for raw in jobs:
        merged = apply_defaults(raw, defaults)
        if max_steps is not None:
            merged["max_steps"] = max_steps
        prepared.append(merged)
    return defaults, prepared


def job_gpu(job: dict[str, Any]) -> str:
    return job.get("gpu") or DEFAULT_GPU


def config_for_profile(profile: str) -> str:
    """Map an eval_profiles.yaml profile name to its config path (relative to repo root)."""
    with EVAL_PROFILES_PATH.open() as f:
        catalog = yaml.safe_load(f) or {}
    meta = (catalog.get("profiles") or {}).get(profile)
    if not meta or not meta.get("config"):
        known = ", ".join(sorted((catalog.get("profiles") or {})))
        raise SystemExit(
            f"Unknown eval_profile {profile!r}; check {_EVAL_PROFILES_REL} (known: {known})"
        )
    return f"research/evals/configs/{meta['config']}"


def primary_metric(task_metrics: dict[str, Any]) -> tuple[str, float] | None:
    """Pick a task's headline (metric_name, score), matching slm_evals summary tables."""
    for key in _METRIC_PRIORITY:
        if key in task_metrics and isinstance(task_metrics[key], (int, float)):
            return key, float(task_metrics[key])
    for key, value in task_metrics.items():
        if "stderr" in key:
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return None


def evaluate_gate(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    goals: dict[str, Any],
) -> dict[str, Any]:
    """Check a candidate's lm-eval results dict against `goals` (Hub publish gate).

    `goals` schema:
        task: <lm-eval task name>       # scored via primary_metric(), same as summary.md
        min_score: <float, optional>    # candidate score must be >= this
        min_improve: <float, optional>  # candidate - baseline must be >= this
        guard_tasks:                     # optional regression guards
          - task: <lm-eval task name>
            max_regress: <float>         # baseline - candidate must be <= this
    """
    cand_tasks = candidate.get("results", {})
    base_tasks = (baseline or {}).get("results", {})

    def _score(tasks: dict[str, Any], task_name: str) -> float | None:
        metrics = tasks.get(task_name)
        if not metrics:
            return None
        picked = primary_metric(metrics)
        return picked[1] if picked else None

    checks: list[dict[str, Any]] = []
    passed = True

    task = goals["task"]
    cand_score = _score(cand_tasks, task)
    base_score = _score(base_tasks, task)

    if goals.get("min_score") is not None:
        ok = cand_score is not None and cand_score >= goals["min_score"]
        checks.append({"check": f"{task} >= {goals['min_score']}", "value": cand_score, "ok": ok})
        passed = passed and ok

    if goals.get("min_improve") is not None:
        delta = (
            cand_score - base_score
            if (cand_score is not None and base_score is not None)
            else None
        )
        ok = delta is not None and delta >= goals["min_improve"]
        checks.append(
            {"check": f"{task} improve >= {goals['min_improve']}", "value": delta, "ok": ok}
        )
        passed = passed and ok

    for guard in goals.get("guard_tasks", []):
        g_task = guard["task"]
        g_cand = _score(cand_tasks, g_task)
        g_base = _score(base_tasks, g_task)
        regress = g_base - g_cand if (g_cand is not None and g_base is not None) else None
        ok = regress is not None and regress <= guard["max_regress"]
        checks.append(
            {"check": f"{g_task} regress <= {guard['max_regress']}", "value": regress, "ok": ok}
        )
        passed = passed and ok

    if not checks:
        passed = False
        checks.append({"check": "goals defined no checks", "value": None, "ok": False})

    return {
        "passed": passed,
        "checks": checks,
        "task": task,
        "candidate_score": cand_score,
        "baseline_score": base_score,
    }


def pull_artifacts(job_name: str, exp_name: str, dest: str = "models/finetuned") -> None:
    """Download an adapter and its lm-eval results from the `slm-finetune` Volume (run locally)."""
    import subprocess

    local_dir = f"{dest}/{job_name}"
    print(f"--- pulling {job_name} -> {local_dir} ---")
    subprocess.run(
        ["modal", "volume", "get", "slm-finetune", job_name, local_dir, "--force"],
        check=False,
    )

    results_dir = f"results/lm_eval/{exp_name}"
    print(f"--- pulling {results_dir} ---")
    subprocess.run(
        ["modal", "volume", "get", "slm-finetune", results_dir, results_dir, "--force"],
        check=False,
    )


def check_gate_files(
    *,
    candidate_results_path: str,
    baseline_results_path: str | None,
    goals: dict[str, Any],
) -> dict[str, Any]:
    """Like evaluate_gate(), but reads results.json files (run inside a volume-mounted function)."""
    cand_path = Path(candidate_results_path)
    if not cand_path.is_file():
        return {"passed": False, "checks": [], "reason": f"missing results file: {cand_path}"}

    candidate = json.loads(cand_path.read_text())
    baseline = None
    if baseline_results_path and Path(baseline_results_path).is_file():
        baseline = json.loads(Path(baseline_results_path).read_text())

    return evaluate_gate(candidate=candidate, baseline=baseline, goals=goals)


def render_model_card(
    *,
    job: dict[str, Any],
    gate_result: dict[str, Any],
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    training_payload: dict[str, Any] | None,
) -> str:
    def _fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:.4f}"

    cand_tasks = candidate.get("results", {})
    base_tasks = (baseline or {}).get("results", {})
    base_model = (training_payload or {}).get("model") or BASE_MODEL_ID

    lines = [
        "---",
        "library_name: peft",
        f"base_model: {base_model}",
        "tags:",
        "  - lora",
        "  - qlora",
        f"  - {job.get('category', 'general')}",
        "---",
        "",
        f"# {job['name']}",
        "",
        f"QLoRA adapter for **{job.get('category', 'general')}**, fine-tuned from "
        f"`{base_model}` on `{job['dataset']}` (format: `{job['format']}`).",
        "",
        "Trained, evaluated, and gated on [Modal](https://modal.com/docs/guide) via "
        "`research/modal/` (app `slm-finetune-benchmark`).",
        "",
        "## Benchmark gate",
        "",
        f"- eval profile: `{job.get('eval_profile')}`",
        f"- gate: {'**PASSED**' if gate_result.get('passed') else '**FAILED**'}",
        "",
        "| check | value | result |",
        "| --- | ---: | --- |",
    ]
    for c in gate_result.get("checks", []):
        lines.append(f"| {c['check']} | {_fmt(c['value'])} | {'pass' if c['ok'] else 'fail'} |")
    if not gate_result.get("checks"):
        lines.append("| — | — | — |")

    lines.extend(
        [
            "",
            "## lm-eval results",
            "",
            "| task | metric | baseline | candidate | delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for task in sorted(set(cand_tasks) | set(base_tasks)):
        c = primary_metric(cand_tasks.get(task, {}))
        b = primary_metric(base_tasks.get(task, {}))
        metric_name = (c or b or (None, None))[0] or "—"
        c_val = c[1] if c else None
        b_val = b[1] if b else None
        delta = c_val - b_val if (c_val is not None and b_val is not None) else None
        sign = "+" if (delta is not None and delta >= 0) else ""
        delta_str = "—" if delta is None else f"{sign}{delta:.4f}"
        lines.append(f"| {task} | {metric_name} | {_fmt(b_val)} | {_fmt(c_val)} | {delta_str} |")

    if training_payload:
        lines.extend(
            [
                "",
                "## Training",
                "",
                f"- dataset: `{training_payload.get('dataset')}`",
                f"- mode: `{training_payload.get('mode')}`",
                f"- samples: {training_payload.get('samples')}",
                f"- final train loss: {training_payload.get('metrics', {}).get('final_train_loss')}",
                f"- eval loss: {training_payload.get('metrics', {}).get('eval_loss')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Load with PEFT",
            "",
            "```python",
            "from peft import PeftModel",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'base = "{base_model}"',
            f'adapter = "{job.get("publish", {}).get("hub_repo", "<hub-repo>")}"',
            "",
            "tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)",
            "model = AutoModelForCausalLM.from_pretrained(",
            '    base, torch_dtype="auto", device_map="auto", trust_remote_code=True',
            ")",
            "model = PeftModel.from_pretrained(model, adapter)",
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def publish_adapter_files(
    *,
    job: dict[str, Any],
    adapter_dir: str,
    gate_result: dict[str, Any],
    candidate_results_path: str,
    baseline_results_path: str | None,
) -> dict[str, Any]:
    """Write a model card and push the adapter to the Hub — only if the gate passed.

    Run inside a function with `finetune_vol` mounted and `hf_secret` set.
    """
    publish_cfg = job.get("publish")
    if not publish_cfg:
        return {"published": False, "reason": "no publish config for this job"}

    if not gate_result.get("passed"):
        return {"published": False, "reason": "gate failed", "gate": gate_result}

    adapter_path = Path(adapter_dir)
    if not adapter_path.is_dir():
        return {"published": False, "reason": f"adapter dir missing: {adapter_dir}"}

    candidate = {}
    cand_path = Path(candidate_results_path)
    if cand_path.is_file():
        candidate = json.loads(cand_path.read_text())

    baseline = None
    if baseline_results_path and Path(baseline_results_path).is_file():
        baseline = json.loads(Path(baseline_results_path).read_text())

    training_payload = None
    training_results_path = adapter_path / "training_results.json"
    if training_results_path.is_file():
        training_payload = json.loads(training_results_path.read_text())

    card = render_model_card(
        job=job,
        gate_result=gate_result,
        candidate=candidate,
        baseline=baseline,
        training_payload=training_payload,
    )
    (adapter_path / "README.md").write_text(card)
    commit_volumes()

    from huggingface_hub import HfApi

    repo_id = publish_cfg["hub_repo"]
    private = publish_cfg.get("private", True)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Publish {job['name']} (gate passed: {gate_result.get('task')})",
    )

    return {"published": True, "repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}"}
