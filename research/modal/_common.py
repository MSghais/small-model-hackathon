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
# Shared general-capability profile for publish gates (limit 100; see compare_study).
GENERAL_EVAL_PROFILE = "compare_study"

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
        "cd /repo && uv sync --frozen --group finetune --group lm-eval --no-dev",
        # lm-eval's ifeval task (instructions profile) needs these, declared via
        # the lm-eval[ifeval] extra but not activated into the project venv by the
        # frozen group sync. Install the lock-pinned versions into /repo/.venv so
        # `uv run slm-lm-eval` can import them.
        "cd /repo && uv pip install langdetect==1.0.9 immutabledict==4.3.1",
    )
)

COMMON_ENV = {
    "TRUST_REMOTE_CODE": "true",
    "HF_HOME": HF_CACHE_PATH,
    # Keep hf-xet logs off the HF cache Volume mount so volume.reload() is not
    # blocked by open log file handles on warm containers.
    "HF_XET_LOG_DEST": "/tmp/xet-logs/",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

DEFAULT_GPU = "A10G"
DEFAULT_KEEPALIVE_HOURS = 4.0
DEFAULT_SCALEDOWN_WINDOW = 3600  # max allowed by Modal (1h idle before scale-down)
DEFAULT_WORKER_TIMEOUT = 14400  # 4h per method call


def repo_env() -> dict[str, str]:
    return {**os.environ, **COMMON_ENV}


def _reload_volume_safe(vol: modal.Volume, *, label: str) -> None:
    """Reload a Volume; skip (with warning) when open files block the operation."""
    try:
        vol.reload()
    except (RuntimeError, modal.exception.ConflictError) as exc:
        if "open files preventing the operation" in str(exc):
            print(f"warning: skipping {label} volume reload ({exc})")
            return
        raise


def reload_finetune_volume() -> None:
    finetune_vol.reload()


def reload_volumes() -> None:
    reload_finetune_volume()
    _reload_volume_safe(hf_cache_vol, label="hf-cache")


def commit_volumes() -> None:
    finetune_vol.commit()
    hf_cache_vol.commit()


def load_experiments() -> dict[str, Any]:
    with EXPERIMENTS_PATH.open() as f:
        return yaml.safe_load(f) or {}


def apply_defaults(job: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    return {**defaults, **job}


# Scalar hyperparameters an experiments.yaml job (or its nested `args:` block)
# may set; each maps 1:1 onto a research/finetune.py flag so any run is tunable
# from config without code changes.
_FINETUNE_FLAGS: dict[str, str] = {
    "model": "--model",
    "lr": "--lr",
    "batch_size": "--batch_size",
    "grad_accum": "--grad_accum",
    "max_len": "--max_len",
    "warmup_ratio": "--warmup_ratio",
    "weight_decay": "--weight_decay",
    "max_grad_norm": "--max_grad_norm",
    "lr_scheduler": "--lr_scheduler",
    "logging_steps": "--logging_steps",
    "eval_steps": "--eval_steps",
    "save_steps": "--save_steps",
    "save_total_limit": "--save_total_limit",
    "early_stopping_patience": "--early_stopping_patience",
    "neftune_noise_alpha": "--neftune_noise_alpha",
    "report_to": "--report_to",
    "seed": "--seed",
    "lora_r": "--lora_r",
    "lora_alpha": "--lora_alpha",
    "lora_dropout": "--lora_dropout",
    "lora_targets": "--lora_targets",
    "val_split": "--val_split",
    "device": "--device",
}


def split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def parse_json_object(value: str | None, *, flag: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{flag} must be a JSON object")
    return parsed


def job_plan_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact, printable description of selected jobs and their eval profile."""
    rows = []
    for job in jobs:
        rows.append(
            {
                "name": job.get("name"),
                "category": job.get("category"),
                "usecase": job.get("usecase") or job.get("use_case"),
                "profile": job.get("eval_profile", "compare_study"),
                "dataset": "mix" if job.get("mix") else job.get("dataset"),
                "mode": job.get("mode", "lora"),
                "max_steps": job.get("max_steps"),
                "max_samples": job.get("max_samples"),
                "publish": bool(job.get("publish")),
            }
        )
    return rows


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
        "--out",
        out_dir,
    ]
    # Dataset: a `mix:` list (skill data + general replay) takes precedence over
    # a single --dataset/--format source.
    if job.get("mix"):
        cmd.extend(["--mix-json", json.dumps(job["mix"])])
    else:
        cmd.extend(["--dataset", job["dataset"], "--format", job["format"]])
        if job.get("dataset_config"):
            cmd.extend(["--dataset-config", job["dataset_config"]])
        if job.get("dataset_split"):
            cmd.extend(["--dataset-split", str(job["dataset_split"])])
        if job.get("max_samples") is not None:
            cmd.extend(["--dataset-max-samples", str(int(job["max_samples"]))])
        # Optional column remap so a dataset's own columns fit the --format
        # (e.g. MetaMathQA query/response -> prompt format).
        for field, col in (job.get("columns") or {}).items():
            cmd.extend([f"--{field}-key", str(col)])

    if job.get("max_steps") is not None:
        cmd.extend(["--max_steps", str(int(job["max_steps"]))])
    if job.get("epochs") is not None:
        cmd.extend(["--epochs", str(job["epochs"])])
    if job.get("mask_prompt") is False:
        cmd.append("--no_mask_prompt")

    # Scalar hyperparameters: top-level keys plus an optional nested `args:` block.
    overrides = {k: job[k] for k in _FINETUNE_FLAGS if k in job}
    overrides.update(job.get("args") or {})
    for key, value in overrides.items():
        flag = _FINETUNE_FLAGS.get(key, f"--{key}")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def build_lm_eval_cmd(
    *,
    experiment_name: str,
    config: str,
    preset: str | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    compare_to: str | None = None,
    tasks: list[str] | None = None,
    limit: int | None = None,
    num_fewshot: int | None = None,
    batch_size: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    seed: int | None = None,
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
    if tasks:
        cmd.append("--tasks")
        cmd.extend(tasks)
    if limit is not None:
        cmd.extend(["--limit", str(int(limit))])
    if num_fewshot is not None:
        cmd.extend(["--num-fewshot", str(int(num_fewshot))])
    if batch_size:
        cmd.extend(["--batch-size", str(batch_size)])
    if device:
        cmd.extend(["--device", str(device)])
    if dtype:
        cmd.extend(["--dtype", str(dtype)])
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    return cmd


def _matches_job_filters(
    job: dict[str, Any],
    *,
    sector: str | None = None,
    usecase: str | None = None,
    profiles: list[str] | None = None,
) -> bool:
    if sector and job.get("sector", job.get("category")) != sector:
        return False
    if usecase:
        values = {
            job.get("usecase"),
            job.get("use_case"),
            job.get("category"),
            job.get("name"),
        }
        values.update(job.get("tags") or [])
        if usecase not in values:
            return False
    if profiles and job.get("eval_profile", "compare_study") not in profiles:
        return False
    return True


def prepare_jobs(
    *,
    job: str | None = None,
    category: str | None = None,
    sector: str | None = None,
    usecase: str | None = None,
    profiles: list[str] | None = None,
    max_steps: int | None = None,
    max_samples: int | None = None,
    finetune_overrides: dict[str, Any] | None = None,
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
    if sector or usecase or profiles:
        jobs = [
            j
            for j in jobs
            if _matches_job_filters(
                j,
                sector=sector,
                usecase=usecase,
                profiles=profiles,
            )
        ]
        if not jobs:
            filters = {
                "sector": sector,
                "usecase": usecase,
                "profiles": profiles,
            }
            raise SystemExit(f"No jobs matched filters: {filters}")

    prepared: list[dict[str, Any]] = []
    for raw in jobs:
        merged = apply_defaults(raw, defaults)
        if max_steps is not None:
            merged["max_steps"] = max_steps
        if max_samples is not None:
            merged["max_samples"] = max_samples
        if finetune_overrides:
            args = {**(merged.get("args") or {})}
            for key, value in finetune_overrides.items():
                if key in _FINETUNE_FLAGS:
                    args[key] = value
                else:
                    merged[key] = value
            if args:
                merged["args"] = args
        prepared.append(merged)
    return defaults, prepared


def job_gpu(job: dict[str, Any]) -> str:
    return job.get("gpu") or DEFAULT_GPU


def job_needs_general_gate(job: dict[str, Any]) -> bool:
    """Publishable jobs run a second general eval and must pass `general_goals`."""
    return bool(job.get("goals") and job.get("publish"))


def general_eval_profile(defaults: dict[str, Any]) -> str:
    return defaults.get("general_eval_profile", GENERAL_EVAL_PROFILE)


def general_goals_for_job(
    job: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any] | None:
    if not job_needs_general_gate(job):
        return None
    goals = job.get("general_goals") or defaults.get("general_goals")
    return goals if goals else None


def baseline_profiles_for_jobs(
    jobs: list[dict[str, Any]], defaults: dict[str, Any]
) -> list[str]:
    profiles = {j.get("eval_profile", "compare_study") for j in jobs}
    if any(job_needs_general_gate(j) for j in jobs):
        profiles.add(general_eval_profile(defaults))
    return sorted(profiles)


def baseline_experiment_name(preset: str, profile: str) -> str:
    """Volume path key for the unfine-tuned base model on a given eval profile."""
    return f"{preset}__baseline__{profile}"


def _load_models_registry() -> dict[str, Any]:
    path = REPO_ROOT / "models.yaml"
    if not path.is_file():
        path = Path("/repo") / "models.yaml"
    if not path.is_file():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def resolve_base_model_id(job: dict[str, Any], defaults: dict[str, Any]) -> str:
    """Hub/path id of the base model this job fine-tunes — used as the eval baseline."""
    explicit = job.get("model") or (job.get("args") or {}).get("model")
    if explicit:
        return str(explicit)
    preset = job.get("preset", defaults.get("preset", "minicpm5-1b"))
    entry = (_load_models_registry().get("models") or {}).get(preset) or {}
    return entry.get("model_id") or BASE_MODEL_ID


def discover_cached_baselines(
    profile_names: list[str],
    *,
    preset: str,
    eval_tasks: list[str] | None = None,
    eval_limit: int | None = None,
    eval_num_fewshot: int | None = None,
    eval_seed: int | None = None,
) -> dict[str, bool]:
    """True per profile when base-model baseline results already exist on the Volume."""
    cached: dict[str, bool] = {}
    for profile in profile_names:
        cached[profile] = baseline_is_cached(
            baseline_experiment_name(preset, profile),
            config_for_profile(profile),
            tasks=eval_tasks,
            limit=eval_limit,
            num_fewshot=eval_num_fewshot,
            seed=eval_seed,
        )
    return cached


def profiles_needing_baseline_run(
    profile_names: list[str],
    cached: dict[str, bool],
    *,
    skip_baseline: bool,
) -> list[str]:
    if skip_baseline:
        return []
    return [profile for profile in profile_names if not cached.get(profile)]


def eval_paths(
    *,
    job_name: str,
    preset: str,
    profile: str,
) -> tuple[str, str, str]:
    """Return (candidate_results_path, baseline_results_path, experiment_name)."""
    exp_name = f"{job_name}__{profile}"
    candidate = f"{LM_EVAL_OUTPUT}/{exp_name}/results.json"
    baseline = f"{LM_EVAL_OUTPUT}/{baseline_experiment_name(preset, profile)}/results.json"
    return candidate, baseline, exp_name


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


def baseline_is_cached(
    experiment_name: str,
    config_path: str,
    *,
    tasks: list[str] | None = None,
    limit: int | None = None,
    num_fewshot: int | None = None,
    seed: int | None = None,
) -> bool:
    """True if a baseline results.json exists AND its run_meta still matches the
    profile config's tasks/limit/num_fewshot. Config changes (e.g. new guard
    tasks or a higher limit) therefore correctly force a fresh baseline."""
    results = Path(LM_EVAL_OUTPUT) / experiment_name / "results.json"
    if not results.is_file():
        return False
    candidates = [Path(config_path)]
    if not Path(config_path).is_absolute():
        candidates += [REPO_ROOT / config_path, Path("/repo") / config_path]
    cfg_file = next((p for p in candidates if p.is_file()), None)
    if cfg_file is None:
        return False
    try:
        meta = json.loads(results.read_text()).get("run_meta", {})
        cfg = yaml.safe_load(cfg_file.read_text()) or {}
    except Exception:
        return False
    expected_tasks = tasks or cfg.get("tasks") or []
    expected_limit = limit if limit is not None else cfg.get("limit")
    expected_fewshot = (
        num_fewshot if num_fewshot is not None else cfg.get("num_fewshot", 0)
    )
    expected_seed = seed if seed is not None else cfg.get("seed")
    same = (
        sorted(meta.get("tasks") or []) == sorted(expected_tasks)
        and meta.get("limit") == expected_limit
        and meta.get("num_fewshot") == expected_fewshot
    )
    if expected_seed is not None:
        same = same and meta.get("seed") == expected_seed
    return same


def evaluate_gate(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    goals: dict[str, Any],
) -> dict[str, Any]:
    """Check a candidate's lm-eval results dict against `goals` (Hub publish gate).

    `goals` schema:
        task: <lm-eval task name, optional when only guard_tasks are set>
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

    task = goals.get("task")
    cand_score = base_score = None
    if task:
        cand_score = _score(cand_tasks, task)
        base_score = _score(base_tasks, task)

    # Tolerance so a score landing exactly on a threshold (e.g. a clean +0.02
    # improvement stored as 0.0199999996) is not rejected by float epsilon.
    eps = 1e-9

    if goals.get("min_score") is not None:
        ok = cand_score is not None and cand_score >= goals["min_score"] - eps
        checks.append({"check": f"{task} >= {goals['min_score']}", "value": cand_score, "ok": ok})
        passed = passed and ok

    if goals.get("min_improve") is not None:
        delta = (
            cand_score - base_score
            if (cand_score is not None and base_score is not None)
            else None
        )
        ok = delta is not None and delta >= goals["min_improve"] - eps
        checks.append(
            {"check": f"{task} improve >= {goals['min_improve']}", "value": delta, "ok": ok}
        )
        passed = passed and ok

    for guard in goals.get("guard_tasks", []):
        g_task = guard["task"]
        g_cand = _score(cand_tasks, g_task)
        g_base = _score(base_tasks, g_task)
        regress = g_base - g_cand if (g_cand is not None and g_base is not None) else None
        ok = regress is not None and regress <= guard["max_regress"] + eps
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
    import shutil
    import subprocess

    def _get(remote: str, parent: str) -> None:
        # For a folder REMOTE_PATH, `modal volume get` expects the *parent*
        # directory as the destination and recreates the folder inside it.
        # Passing the full target path (parent/<name>) raises
        # "[Errno 21] Is a directory". Clear the target first for a clean pull.
        name = remote.rsplit("/", 1)[-1]
        shutil.rmtree(Path(parent) / name, ignore_errors=True)
        Path(parent).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["modal", "volume", "get", "slm-finetune", remote, f"{parent}/", "--force"],
            check=False,
        )

    print(f"--- pulling {job_name} -> {dest}/{job_name} ---")
    _get(job_name, dest)

    exp_dir = f"results/lm_eval/{exp_name}"
    print(f"--- pulling {exp_dir} ---")
    _get(exp_dir, "results/lm_eval")


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


def check_publish_gate_files(
    *,
    skill_candidate_path: str,
    skill_baseline_path: str | None,
    skill_goals: dict[str, Any],
    general_candidate_path: str | None = None,
    general_baseline_path: str | None = None,
    general_goals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate on skill-specific eval plus optional general-capability eval."""
    skill_gate = check_gate_files(
        candidate_results_path=skill_candidate_path,
        baseline_results_path=skill_baseline_path,
        goals=skill_goals,
    )
    general_gate: dict[str, Any] | None = None
    if general_goals:
        if not general_candidate_path:
            general_gate = {
                "passed": False,
                "checks": [
                    {
                        "check": "general eval results missing",
                        "value": None,
                        "ok": False,
                    }
                ],
                "reason": "general candidate results path not provided",
            }
        else:
            general_gate = check_gate_files(
                candidate_results_path=general_candidate_path,
                baseline_results_path=general_baseline_path,
                goals=general_goals,
            )

    passed = skill_gate.get("passed") and (
        general_gate is None or general_gate.get("passed")
    )
    checks = list(skill_gate.get("checks", []))
    if general_gate:
        for check in general_gate.get("checks", []):
            checks.append({**check, "check": f"general: {check['check']}"})

    return {
        "passed": passed,
        "checks": checks,
        "skill": skill_gate,
        "general": general_gate,
        "task": skill_gate.get("task"),
        "candidate_score": skill_gate.get("candidate_score"),
        "baseline_score": skill_gate.get("baseline_score"),
    }


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

    # A job is either a single dataset (`dataset`/`format`) or a `mix:` of sources.
    if job.get("mix"):
        dataset_desc = " + ".join(
            f"`{s.get('dataset', '?')}`" for s in job["mix"]
        )
        format_desc = "mix"
    else:
        dataset_desc = f"`{job.get('dataset', '?')}`"
        format_desc = job.get("format", "?")

    lines = [
        "---",
        "library_name: peft",
        f"base_model: {base_model}",
        "license: apache-2.0",
        "tags:",
        "  - lora",
        "  - qlora",
        "  - build-small-hackathon",
        "  - well-tuned",
        f"  - {job.get('category', 'general')}",
        "---",
        "",
        f"# {job['name']}",
        "",
        f"QLoRA adapter for **{job.get('category', 'general')}**, fine-tuned from "
        f"`{base_model}` on {dataset_desc} (format: `{format_desc}`).",
        "",
        "Trained, evaluated, and gated on [Modal](https://modal.com/docs/guide) via "
        "`research/modal/` (app `slm-finetune-benchmark`).",
        "",
        "## Benchmark gate",
        "",
        f"- skill eval profile: `{job.get('eval_profile')}`",
        f"- gate: {'**PASSED**' if gate_result.get('passed') else '**FAILED**'}",
        "",
    ]

    def _gate_table(section: dict[str, Any] | None, *, prefix: str = "") -> list[str]:
        if not section:
            return []
        out = [
            f"### {prefix}checks".strip(),
            "",
            "| check | value | result |",
            "| --- | ---: | --- |",
        ]
        for c in section.get("checks", []):
            out.append(
                f"| {c['check']} | {_fmt(c['value'])} | {'pass' if c['ok'] else 'fail'} |"
            )
        if not section.get("checks"):
            out.append("| — | — | — |")
        out.append("")
        return out

    skill_section = gate_result.get("skill") or gate_result
    lines.extend(_gate_table(skill_section, prefix="Skill "))
    if gate_result.get("general"):
        gen_profile = job.get("general_eval_profile") or GENERAL_EVAL_PROFILE
        lines.append(f"- general eval profile: `{gen_profile}`")
        lines.append("")
        lines.extend(_gate_table(gate_result["general"], prefix="General "))

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

    repo_ids = [publish_cfg["hub_repo"], *(publish_cfg.get("mirror_repos") or [])]
    private = publish_cfg.get("private", True)

    api = HfApi()
    uploads = []
    for repo_id in dict.fromkeys(repo_ids):
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Publish {job['name']} (gate passed: {gate_result.get('task')})",
        )
        uploads.append({"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}"})

    return {
        "published": True,
        "repo_id": uploads[0]["repo_id"],
        "url": uploads[0]["url"],
        "uploads": uploads,
    }
