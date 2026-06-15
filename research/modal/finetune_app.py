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

# Make `_common` importable both locally (sibling file) and in the Modal
# container, where the entrypoint lands at /root but the repo is baked into the
# image at /repo (see add_local_dir in _common.py).
for _candidate in (Path(__file__).resolve().parent, Path("/repo/research/modal")):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from _common import (  # noqa: E402
    FINETUNE_VOL_PATH,
    HF_CACHE_PATH,
    LM_EVAL_OUTPUT,
    baseline_profiles_for_jobs,
    build_finetune_cmd,
    build_lm_eval_cmd,
    check_gate_files,
    check_publish_gate_files,
    commit_volumes,
    config_for_profile,
    discover_cached_baselines,
    eval_paths,
    finetune_vol,
    general_eval_profile,
    general_goals_for_job,
    hf_cache_vol,
    hf_secret,
    image,
    job_gpu,
    job_plan_rows,
    parse_json_object,
    prepare_jobs,
    profiles_needing_baseline_run,
    resolve_base_model_id,
    split_csv,
    publish_adapter_files,
    pull_artifacts,
    reload_finetune_volume,
    reload_volumes,
    repo_env,
    baseline_experiment_name,
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
    tasks: list[str] | None = None,
    limit: int | None = None,
    num_fewshot: int | None = None,
    batch_size: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run slm-lm-eval on base model or finetuned checkpoint."""
    reload_finetune_volume()

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
        tasks=tasks,
        limit=limit,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        seed=seed,
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
        "tasks": tasks,
        "limit": limit,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": device,
        "dtype": dtype,
        "seed": seed,
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
    general_candidate_results_path: str | None = None,
    general_baseline_results_path: str | None = None,
    general_goals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check skill + general lm-eval results against publish goals."""
    reload_finetune_volume()
    if general_goals:
        return check_publish_gate_files(
            skill_candidate_path=candidate_results_path,
            skill_baseline_path=baseline_results_path,
            skill_goals=goals,
            general_candidate_path=general_candidate_results_path,
            general_baseline_path=general_baseline_results_path,
            general_goals=general_goals,
        )
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
    reload_finetune_volume()
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
    parallel: bool = False,
    job: str | None = None,
    category: str | None = None,
    sector: str | None = None,
    usecase: str | None = None,
    profiles: str | None = None,
    max_steps: int | None = None,
    max_samples: int | None = None,
    finetune_args_json: str | None = None,
    publish: bool = True,
    pull: bool = True,
    plan: bool = False,
    skip_baseline: bool = False,
    eval_tasks: str | None = None,
    eval_limit: int | None = None,
    eval_num_fewshot: int | None = None,
    eval_batch_size: str | None = None,
    eval_device: str | None = None,
    eval_dtype: str | None = None,
    eval_seed: int | None = None,
):
    """
    Skill-matrix pipeline: per-profile baselines -> train -> eval -> gate -> publish -> pull.

    Examples:
        modal run research/modal/finetune_app.py
        modal run research/modal/finetune_app.py --job math-lora --max-steps 20
        modal run research/modal/finetune_app.py --category science
        modal run research/modal/finetune_app.py --eval-only --job math-lora
        modal run research/modal/finetune_app.py --no-publish --no-pull
    """
    defaults, prepared = prepare_jobs(
        job=job,
        category=category,
        sector=sector,
        usecase=usecase,
        profiles=split_csv(profiles),
        max_steps=max_steps,
        max_samples=max_samples,
        finetune_overrides=parse_json_object(
            finetune_args_json, flag="--finetune-args-json"
        ),
    )
    if not prepared:
        raise SystemExit("No matching jobs; check --job/--category and experiments.yaml")
    preset = defaults.get("preset", "minicpm5-1b")
    plan_rows = job_plan_rows(prepared)
    if plan:
        print(json.dumps({"preset": preset, "jobs": plan_rows}, indent=2))
        return

    profile_names = baseline_profiles_for_jobs(prepared, defaults)

    eval_task_list = split_csv(eval_tasks)
    baselines_ok = discover_cached_baselines(
        profile_names,
        preset=preset,
        eval_tasks=eval_task_list,
        eval_limit=eval_limit,
        eval_num_fewshot=eval_num_fewshot,
        eval_seed=eval_seed,
    )
    missing_baselines = profiles_needing_baseline_run(
        profile_names, baselines_ok, skip_baseline=skip_baseline
    )
    if missing_baselines:
        print(f"--- base-model baselines ({', '.join(missing_baselines)}) ---")
        for profile in missing_baselines:
            exp = baseline_experiment_name(preset, profile)
            result = run_lm_eval.remote(
                experiment_name=exp,
                config=config_for_profile(profile),
                preset=preset,
                tasks=eval_task_list,
                limit=eval_limit,
                num_fewshot=eval_num_fewshot,
                batch_size=eval_batch_size,
                device=eval_device,
                dtype=eval_dtype,
                seed=eval_seed,
            )
            print(json.dumps(result, indent=2))
            baselines_ok[profile] = bool(result.get("ok"))
    elif any(baselines_ok.values()):
        cached = [p for p in profile_names if baselines_ok.get(p)]
        print(f"--- base-model baselines: reusing cached ({', '.join(cached)}) ---")

    train_results: dict[str, dict[str, Any]] = {}
    if train and not eval_only:
        print(f"--- finetune ({len(prepared)} job(s), parallel={parallel}) ---")
        if parallel:
            handles = {
                j["name"]: finetune_one.with_options(gpu=job_gpu(j)).spawn(j)
                for j in prepared
            }
            for name, handle in handles.items():
                train_results[name] = handle.get()
                print(json.dumps(train_results[name], indent=2))
        else:
            for j in prepared:
                print(f"Training {j['name']}...")
                result = finetune_one.with_options(gpu=job_gpu(j)).remote(j)
                train_results[j["name"]] = result
                print(json.dumps(result, indent=2))

    print("--- post-train lm-eval / gate / publish ---")
    summary: list[dict[str, Any]] = []
    gen_profile = general_eval_profile(defaults)
    for j in prepared:
        job_name = j["name"]
        profile = j.get("eval_profile", "compare_study")
        train_payload = train_results.get(job_name)
        adapter_path = (
            train_payload["output_dir"] if train_payload else f"{FINETUNE_VOL_PATH}/{job_name}"
        )

        baseline_path = f"{LM_EVAL_OUTPUT}/{baseline_experiment_name(preset, profile)}/results.json"
        compare_to = baseline_path if baselines_ok.get(profile) else None
        base_model_id = resolve_base_model_id(j, defaults)

        exp_name = f"{job_name}__{profile}"
        eval_result = run_lm_eval.remote(
            experiment_name=exp_name,
            config=config_for_profile(profile),
            model_path=base_model_id,
            adapter_path=adapter_path,
            compare_to=compare_to,
            tasks=eval_task_list,
            limit=eval_limit,
            num_fewshot=eval_num_fewshot,
            batch_size=eval_batch_size,
            device=eval_device,
            dtype=eval_dtype,
            seed=eval_seed,
        )
        print(json.dumps(eval_result, indent=2))

        general_goals = general_goals_for_job(j, defaults)
        general_eval_result: dict[str, Any] | None = None
        general_candidate_path: str | None = None
        general_baseline_path: str | None = None
        if general_goals:
            general_baseline_path = (
                f"{LM_EVAL_OUTPUT}/{baseline_experiment_name(preset, gen_profile)}/results.json"
            )
            gen_compare_to = (
                general_baseline_path if baselines_ok.get(gen_profile) else None
            )
            gen_exp_name = f"{job_name}__{gen_profile}"
            general_eval_result = run_lm_eval.remote(
                experiment_name=gen_exp_name,
                config=config_for_profile(gen_profile),
                model_path=base_model_id,
                adapter_path=adapter_path,
                compare_to=gen_compare_to,
                tasks=eval_task_list,
                limit=eval_limit,
                num_fewshot=eval_num_fewshot,
                batch_size=eval_batch_size,
                device=eval_device,
                dtype=eval_dtype,
                seed=eval_seed,
            )
            print(json.dumps(general_eval_result, indent=2))
            general_candidate_path = general_eval_result["results_json"]

        row: dict[str, Any] = {
            "name": job_name,
            "category": j.get("category"),
            "profile": profile,
            "general_profile": gen_profile if general_goals else None,
            "plan": next((p for p in plan_rows if p["name"] == job_name), None),
        }

        gate_result: dict[str, Any] | None = None
        if j.get("goals"):
            skill_ok = bool(eval_result.get("ok"))
            general_ok = (
                not general_goals
                or bool(general_eval_result and general_eval_result.get("ok"))
            )
            if skill_ok and general_ok:
                gate_result = check_gate.remote(
                    candidate_results_path=eval_result["results_json"],
                    baseline_results_path=baseline_path,
                    goals=j["goals"],
                    general_candidate_results_path=general_candidate_path,
                    general_baseline_results_path=general_baseline_path,
                    general_goals=general_goals,
                )
                print(json.dumps(gate_result, indent=2))
            row["gate_passed"] = bool(gate_result and gate_result.get("passed"))

        if j.get("publish"):
            row["hub_repo"] = j["publish"].get("hub_repo")
            if publish and gate_result is not None:
                publish_result = publish_adapter.remote(
                    job=j,
                    adapter_dir=adapter_path,
                    gate_result=gate_result,
                    candidate_results_path=eval_result["results_json"],
                    baseline_results_path=baseline_path,
                )
                print(json.dumps(publish_result, indent=2))
                row["published"] = publish_result.get("published")

        summary.append(row)

        if pull:
            pull_artifacts(job_name, exp_name)
            if general_goals and general_eval_result:
                pull_artifacts(job_name, f"{job_name}__{gen_profile}", dest="models/finetuned")

    _print_summary(summary)


@app.local_entrypoint()
def publish_only(job: str):
    """Re-run the gate and Hub publish for a job using already-computed results (no train/eval)."""
    defaults, prepared = prepare_jobs(job=job)
    j = prepared[0]
    if not j.get("goals"):
        raise SystemExit(f"Job {job!r} has no `goals`; nothing to gate on")
    if not j.get("publish"):
        raise SystemExit(f"Job {job!r} has no `publish` config")

    preset = defaults.get("preset", "minicpm5-1b")
    profile = j.get("eval_profile", "compare_study")
    gen_profile = general_eval_profile(defaults)
    general_goals = general_goals_for_job(j, defaults)
    adapter_path = f"{FINETUNE_VOL_PATH}/{job}"
    candidate_results_path, baseline_results_path, _ = eval_paths(
        job_name=job, preset=preset, profile=profile
    )
    general_candidate_path = None
    general_baseline_path = None
    if general_goals:
        general_candidate_path, general_baseline_path, _ = eval_paths(
            job_name=job, preset=preset, profile=gen_profile
        )

    gate_result = check_gate.remote(
        candidate_results_path=candidate_results_path,
        baseline_results_path=baseline_results_path,
        goals=j["goals"],
        general_candidate_results_path=general_candidate_path,
        general_baseline_results_path=general_baseline_path,
        general_goals=general_goals,
    )
    print(json.dumps(gate_result, indent=2))

    publish_result = publish_adapter.remote(
        job=j,
        adapter_dir=adapter_path,
        gate_result=gate_result,
        candidate_results_path=candidate_results_path,
        baseline_results_path=baseline_results_path,
    )
    print(json.dumps(publish_result, indent=2))


@app.local_entrypoint()
def pull(job: str | None = None, category: str | None = None, dest: str = "models/finetuned"):
    """Download adapters and their lm-eval results from the `slm-finetune` Volume."""
    _, prepared = prepare_jobs(job=job, category=category)
    if not prepared:
        raise SystemExit("No matching jobs; pass --job or --category")

    for j in prepared:
        profile = j.get("eval_profile", "compare_study")
        pull_artifacts(j["name"], f"{j['name']}__{profile}", dest)
