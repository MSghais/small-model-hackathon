from pathlib import Path

import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.modal._common import (  # noqa: E402
    COMMON_ENV,
    baseline_profiles_for_jobs,
    build_finetune_cmd,
    build_lm_eval_cmd,
    check_publish_gate_files,
    evaluate_gate,
    general_goals_for_job,
    prepare_jobs,
    split_csv,
)


def test_build_lm_eval_cmd_accepts_runtime_overrides():
    cmd = build_lm_eval_cmd(
        experiment_name="exp",
        config="cfg.yaml",
        preset="minicpm5-1b",
        tasks=["arc_easy", "hellaswag"],
        limit=5,
        num_fewshot=1,
        batch_size="2",
        device="cuda",
        dtype="float16",
        seed=7,
    )

    assert cmd[-15:] == [
        "--tasks",
        "arc_easy",
        "hellaswag",
        "--limit",
        "5",
        "--num-fewshot",
        "1",
        "--batch-size",
        "2",
        "--device",
        "cuda",
        "--dtype",
        "float16",
        "--seed",
        "7",
    ]


def test_prepare_jobs_filters_and_applies_finetune_overrides():
    _, jobs = prepare_jobs(
        sector="math",
        profiles=["math"],
        max_steps=3,
        max_samples=11,
        finetune_overrides={"lr": 1e-4, "lora_r": 8, "dataset_split": "train[:11]"},
    )

    assert [job["name"] for job in jobs] == ["math-lora"]
    job = jobs[0]
    assert job["max_steps"] == 3
    assert job["max_samples"] == 11
    assert job["dataset_split"] == "train[:11]"
    assert job["args"]["lr"] == 1e-4
    assert job["args"]["lora_r"] == 8

    cmd = build_finetune_cmd(job, "/tmp/out")
    assert "--max_steps" in cmd
    assert cmd[cmd.index("--max_steps") + 1] == "3"
    assert "--lr" in cmd
    assert cmd[cmd.index("--lr") + 1] == "0.0001"
    assert "--lora_r" in cmd
    assert cmd[cmd.index("--lora_r") + 1] == "8"


def test_split_csv_trims_empty_values():
    assert split_csv(" math, science ,,code ") == ["math", "science", "code"]
    assert split_csv(None) is None


def _results(task_scores: dict[str, float]) -> dict:
    return {
        "results": {
            task: {"acc,none": score, "acc_stderr,none": 0.01}
            for task, score in task_scores.items()
        }
    }


def test_evaluate_gate_guard_only_goals():
    candidate = _results({"arc_easy": 0.5, "hellaswag": 0.4})
    baseline = _results({"arc_easy": 0.52, "hellaswag": 0.41})
    goals = {
        "guard_tasks": [
            {"task": "arc_easy", "max_regress": 0.03},
            {"task": "hellaswag", "max_regress": 0.03},
        ]
    }
    gate = evaluate_gate(candidate=candidate, baseline=baseline, goals=goals)
    assert gate["passed"] is True
    assert len(gate["checks"]) == 2


def test_check_publish_gate_requires_both_skill_and_general(tmp_path):
    skill_cand = tmp_path / "skill_cand.json"
    skill_base = tmp_path / "skill_base.json"
    general_cand = tmp_path / "general_cand.json"
    general_base = tmp_path / "general_base.json"
    skill_cand.write_text(
        json.dumps(_results({"gsm8k": 0.4}))
    )
    skill_base.write_text(
        json.dumps(_results({"gsm8k": 0.33}))
    )
    general_cand.write_text(
        json.dumps(_results({"arc_easy": 0.5, "hellaswag": 0.4}))
    )
    general_base.write_text(
        json.dumps(_results({"arc_easy": 0.52, "hellaswag": 0.41}))
    )

    gate = check_publish_gate_files(
        skill_candidate_path=str(skill_cand),
        skill_baseline_path=str(skill_base),
        skill_goals={"task": "gsm8k", "min_improve": 0.02},
        general_candidate_path=str(general_cand),
        general_baseline_path=str(general_base),
        general_goals={
            "guard_tasks": [{"task": "arc_easy", "max_regress": 0.03}]
        },
    )
    assert gate["passed"] is True
    assert gate["skill"]["passed"] is True
    assert gate["general"]["passed"] is True
    assert any(c["check"].startswith("general:") for c in gate["checks"])


def test_baseline_profiles_include_general_for_publishable_jobs():
    _, jobs = prepare_jobs(job="math-lora")
    defaults = {"general_eval_profile": "compare_study", "general_goals": {"guard_tasks": []}}
    profiles = baseline_profiles_for_jobs(jobs, defaults)
    assert "math" in profiles
    assert "compare_study" in profiles


def test_general_goals_only_for_publishable_jobs():
    _, math_jobs = prepare_jobs(job="math-lora")
    _, local_jobs = prepare_jobs(job="alpaca-lora")
    defaults = {"general_goals": {"guard_tasks": [{"task": "piqa", "max_regress": 0.03}]}}
    assert general_goals_for_job(math_jobs[0], defaults) is not None
    assert general_goals_for_job(local_jobs[0], defaults) is None


def test_common_env_redirects_xet_logs_off_hf_cache_volume():
    assert COMMON_ENV["HF_XET_LOG_DEST"] == "/tmp/xet-logs/"
